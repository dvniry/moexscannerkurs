"""MultiScale Hybrid v2 — Factual 1D backbone + OHLC regression head.

Изменения по сравнению с v1:
  1. Визуальный backbone: EfficientNet-B0 (3ch, 224×224) →
     1D Conv backbone (N_channels, W) для многоканальных рядов
  2. Две головы: cls_head (BUY/HOLD/SELL) + reg_head (OHLC × future_bars)
  3. Multi-task loss: α × FocalLoss + (1-α) × HuberLoss
  4. Усиленная регуляризация: dropout 0.3-0.4, Mixup augmentation,
     label smoothing 0.15, SpatialDropout1d
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import json, datetime, math
from ml.config import CFG, SCALES
from ml.candle_render_v2 import N_RENDER_CHANNELS


# ══════════════════════════════════════════════════════════════════
#  Focal Loss
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  SpatialDropout1D
# ══════════════════════════════════════════════════════════════════

class SpatialDropout1d(nn.Module):
    """Дропает целые каналы (а не отдельные элементы). Лучше для 1D Conv."""
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: (B, C, L)
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(
            torch.full((x.shape[0], x.shape[1], 1), 1 - self.p, device=x.device)
        )
        return x * mask / (1 - self.p)


# ══════════════════════════════════════════════════════════════════
#  Mixup augmentation (применяется в training loop)
# ══════════════════════════════════════════════════════════════════

def mixup_data(imgs_dict, num_dict, cls_y, ohlc_y, ctx, alpha=0.2):
    """
    Mixup augmentation для multi-input, multi-output.
    Возвращает смешанные данные + lam + перемешанные targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B = cls_y.size(0)
    perm = torch.randperm(B, device=cls_y.device)

    mixed_imgs = {W: lam * imgs_dict[W] + (1 - lam) * imgs_dict[W][perm]
                  for W in imgs_dict}
    mixed_nums = {W: lam * num_dict[W] + (1 - lam) * num_dict[W][perm]
                  for W in num_dict} if num_dict is not None else None
    mixed_ohlc = lam * ohlc_y + (1 - lam) * ohlc_y[perm]
    mixed_ctx  = lam * ctx + (1 - lam) * ctx[perm] if ctx is not None else None

    return mixed_imgs, mixed_nums, cls_y, cls_y[perm], mixed_ohlc, mixed_ctx, lam


# ══════════════════════════════════════════════════════════════════
#  Focal Loss
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(
            weight=weight, reduction='none', label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ══════════════════════════════════════════════════════════════════
#  1D Conv Backbone — замена EfficientNet для factual рендера
# ══════════════════════════════════════════════════════════════════

class Conv1DBlock(nn.Module):
    """Residual 1D Conv block с BatchNorm + SpatialDropout."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dropout=0.15):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                               padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.SiLU()
        self.drop  = SpatialDropout1d(dropout)
        self.res   = nn.Conv1d(in_ch, out_ch, 1, stride=stride) \
                     if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        return self.act(out + self.res(x))


class FactualBackbone(nn.Module):
    """
    1D CNN backbone для многоканального рендера.
    Вход:  (B, N_RENDER_CHANNELS, W)
    Выход: (B, out_dim)
    """
    def __init__(self, in_channels: int = N_RENDER_CHANNELS,
                 out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            Conv1DBlock(in_channels, 32,  kernel=7, dropout=dropout * 0.5),
            Conv1DBlock(32,          64,  kernel=5, dropout=dropout * 0.7),
            Conv1DBlock(64,          128, kernel=3, dropout=dropout),
            Conv1DBlock(128,         256, kernel=3, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N_RENDER_CHANNELS, W)
        h = self.net(x)          # (B, 256, W')
        h = self.pool(h)         # (B, 256, 1)
        h = h.squeeze(-1)        # (B, 256)
        return self.proj(h)      # (B, out_dim)


# ══════════════════════════════════════════════════════════════════
#  TCN — для коротких масштабов [5, 10]
# ══════════════════════════════════════════════════════════════════

class _TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad       = (kernel - 1) * dilation
        self.pad  = pad
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              padding=pad, dilation=dilation)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch \
                    else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        return self.act(out + self.res(x))


class TCNEncoder(nn.Module):
    """TCN-энкодер для одного масштаба. Вход: (B, C_in, W), выход: (B, out_dim)."""
    def __init__(self, in_channels: int, out_dim: int = 64,
                 dilations: tuple = (1, 2, 4), dropout: float = 0.2):
        super().__init__()
        layers = []
        ch = in_channels
        for d in dilations:
            layers.append(_TemporalBlock(ch, 64, kernel=3, dilation=d,
                                         dropout=dropout))
            ch = 64
        self.net  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.proj(self.pool(self.net(x)).squeeze(-1))


# ══════════════════════════════════════════════════════════════════
#  MultiScaleHybrid v2
#  Factual 1D backbone × 4 масштаба
#  + TCN × [5,10]
#  + BiLSTM × [30]
#  + Cross-scale attention
#  + Dual head: cls (3) + reg (future_bars × 4)
# ══════════════════════════════════════════════════════════════════

SHORT_SCALES = [5, 10]
LONG_SCALES  = [20, 30]


class MultiScaleHybridV2(nn.Module):
    """
    Гибридная архитектура v2:
      - FactualBackbone (1D CNN) на рендерах всех масштабов → 128 × 4
      - TCN на числовых признаках масштабов [5,10]           → 128 × 2
      - BiLSTM на числовых признаках масштаба [30]           → 128 × 1
      - Cross-scale Multi-Head Attention
      - Контекстный вектор ctx (HMM + сектор)
      - Две головы: cls_head (3 класса) + reg_head (future_bars × 4)
    """
    def __init__(self, ctx_dim: int = 0,
                 n_indicator_cols: int = 30,
                 future_bars: int = None,
                 lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        if future_bars is None:
            future_bars = CFG.future_bars
        self.future_bars = future_bars
        n_scales = len(SCALES)

        # ── Factual backbone (общий для всех масштабов) ───────
        self.backbone = FactualBackbone(
            in_channels=N_RENDER_CHANNELS, out_dim=128, dropout=0.3)

        # Per-scale projection (после общего backbone)
        self.scale_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Dropout(0.3))
            for _ in SCALES
        ])

        # ── TCN для коротких масштабов ────────────────────────
        self.tcn_encoders = nn.ModuleList([
            TCNEncoder(in_channels=n_indicator_cols, out_dim=64,
                       dilations=(1, 2, 4), dropout=0.3)
            for _ in SHORT_SCALES
        ])

        # ── BiLSTM для длинных масштабов ──────────────────────
        self.bilstm = nn.LSTM(
            input_size=n_indicator_cols,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.4,
        )
        lstm_out_dim = lstm_hidden * 2  # 256

        # ── Проекции в d=128 ──────────────────────────────────
        self.tcn_proj  = nn.Linear(64,          128)
        self.lstm_proj = nn.Linear(lstm_out_dim, 128)

        # ── Cross-scale Attention ─────────────────────────────
        # 4 (backbone) + 2 (tcn) + 1 (lstm) = 7 токенов × 128
        n_tokens = n_scales + len(SHORT_SCALES) + 1
        self.attn      = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, dropout=0.2, batch_first=True)
        self.attn_norm = nn.LayerNorm(128)

        # ── Общий trunk после attention ───────────────────────
        fused_dim = 128 * n_tokens + ctx_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),       nn.SiLU(), nn.Dropout(0.4),
        )

        # ── Голова классификации ──────────────────────────────
        self.cls_head = nn.Linear(128, 3)

        # ── Голова регрессии OHLC ─────────────────────────────
        # Выход: (future_bars × 4) = 20 значений
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, future_bars * 4),
        )

    def forward(self, imgs_by_scale: dict,
                num_by_scale: dict = None,
                ctx: torch.Tensor = None):
        """
        imgs_by_scale: {W: (B, N_RENDER_CHANNELS, W)} — factual рендеры
        num_by_scale:  {W: (B, W, n_indicator_cols)} — числовые ряды
        ctx:           (B, ctx_dim)

        Возвращает: (cls_logits, ohlc_pred)
          cls_logits: (B, 3)
          ohlc_pred:  (B, future_bars, 4)
        """
        tokens = []

        # ── Factual backbone токены ───────────────────────────
        for i, W in enumerate(SCALES):
            e = self.backbone(imgs_by_scale[W])   # (B, 128)
            tokens.append(self.scale_proj[i](e))  # (B, 128)

        # ── TCN-токены ────────────────────────────────────────
        if num_by_scale is not None:
            for j, W in enumerate(SHORT_SCALES):
                x = num_by_scale[W].permute(0, 2, 1)  # (B, C, W)
                t = self.tcn_encoders[j](x)
                tokens.append(self.tcn_proj(t))

            # ── BiLSTM-токен ──────────────────────────────────
            long_W  = max(LONG_SCALES)
            x_long  = num_by_scale[long_W]
            out, _  = self.bilstm(x_long)
            tokens.append(self.lstm_proj(out[:, -1, :]))
        else:
            B   = next(iter(imgs_by_scale.values())).shape[0]
            dev = next(self.parameters()).device
            for _ in range(len(SHORT_SCALES) + 1):
                tokens.append(torch.zeros(B, 128, device=dev))

        # ── Cross-scale Attention ─────────────────────────────
        seq = torch.stack(tokens, dim=1)          # (B, 7, 128)
        attn_out, _ = self.attn(seq, seq, seq)
        seq   = self.attn_norm(seq + attn_out)    # residual
        fused = seq.flatten(1)                     # (B, 896)

        if ctx is not None:
            fused = torch.cat([fused, ctx], dim=1)

        # ── Trunk → две головы ────────────────────────────────
        h = self.trunk(fused)                      # (B, 128)
        cls_logits = self.cls_head(h)              # (B, 3)
        ohlc_flat  = self.reg_head(h)              # (B, future_bars*4)
        ohlc_pred  = ohlc_flat.view(-1, self.future_bars, 4)  # (B, F, 4)

        return cls_logits, ohlc_pred


# ══════════════════════════════════════════════════════════════════
#  Multi-task Loss
# ══════════════════════════════════════════════════════════════════

class MultiTaskLoss(nn.Module):
    """
    L = α × FocalLoss(cls) + (1-α) × HuberLoss(ohlc_reg)
    α — learnable (log_sigma parametrization для автобалансировки).
    """
    def __init__(self, cls_weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, weight=cls_weight,
                               label_smoothing=label_smoothing)
        self.huber = nn.HuberLoss(delta=0.02)  # δ=2% — small returns
        # Learnable task balance (uncertainty weighting)
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_reg = nn.Parameter(torch.tensor(0.0))

    def forward(self, cls_logits, cls_targets, ohlc_pred, ohlc_targets):
        """
        cls_logits:   (B, 3)
        cls_targets:  (B,)
        ohlc_pred:    (B, F, 4)
        ohlc_targets: (B, F, 4)
        """
        loss_cls = self.focal(cls_logits, cls_targets)
        loss_reg = self.huber(ohlc_pred, ohlc_targets)

        # Uncertainty weighting (Kendall et al., 2018)
        # L = 1/(2σ²) × L_task + log(σ)
        prec_cls = torch.exp(-2 * self.log_sigma_cls)
        prec_reg = torch.exp(-2 * self.log_sigma_reg)

        total = (prec_cls * loss_cls + self.log_sigma_cls +
                 prec_reg * loss_reg + self.log_sigma_reg)

        return total, loss_cls.detach(), loss_reg.detach()


# ══════════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        d = torch.device('cuda')
        p = torch.cuda.get_device_properties(0)
        print(f"  Устройство: {p.name} ({p.total_memory // 1024**2} MB VRAM)")
    else:
        d = torch.device('cpu')
        print("  Устройство: cpu")
    return d


def _collate_v2(batch):
    """Collate для нового формата с OHLC labels."""
    imgs_list, num_list, cls_list, ohlc_list, ctx_list = zip(*batch)
    imgs_batch = {W: torch.stack([x[W] for x in imgs_list]) for W in SCALES}
    num_batch  = {W: torch.stack([x[W] for x in num_list])  for W in SCALES} \
                 if num_list[0] is not None else None
    return (imgs_batch, num_batch,
            torch.tensor(cls_list,  dtype=torch.long),
            torch.stack(ohlc_list),
            torch.stack(ctx_list))


def _make_loader(ds, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, collate_fn=_collate_v2)


# ══════════════════════════════════════════════════════════════════
#  Обучение
# ══════════════════════════════════════════════════════════════════

def _train_loop_v2(model, tr_loader, val_loader, n_val,
                   n_epochs, max_lr, wd, patience_limit,
                   criterion, save_path, phase_name,
                   device, ctx_dim, accum_steps: int = 1,
                   use_mixup: bool = True, mixup_alpha: float = 0.2):
    """
    Цикл обучения v2 с multi-task loss + Mixup augmentation.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=max_lr / 10, weight_decay=wd)
    opt_steps = math.ceil(len(tr_loader) / accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr,
        steps_per_epoch=opt_steps, epochs=n_epochs,
        pct_start=0.2, div_factor=10, final_div_factor=200)

    best_metric, patience = 0.0, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_cls, total_reg = 0.0, 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tr_loader, 1):
            imgs_dict, num_dict, cls_y, ohlc_y, ctx = batch
            imgs = {W: imgs_dict[W].to(device) for W in SCALES}
            cls_y  = cls_y.to(device)
            ohlc_y = ohlc_y.to(device)
            ctx_t  = ctx.to(device) if ctx_dim > 0 else None

            if num_dict is not None:
                nums = {W: num_dict[W].to(device) for W in SCALES}
            else:
                nums = None

            # ── Mixup augmentation ────────────────────────
            if use_mixup and mixup_alpha > 0:
                m_imgs, m_nums, cls_a, cls_b, m_ohlc, m_ctx, lam = \
                    mixup_data(imgs, nums, cls_y, ohlc_y, ctx_t, mixup_alpha)

                if m_nums is not None:
                    cls_logits, ohlc_pred = model(m_imgs, m_nums, m_ctx)
                else:
                    cls_logits, ohlc_pred = model(m_imgs, ctx=m_ctx)

                # Mixup loss: смешиваем cls loss для двух targets
                loss_a, l_cls_a, l_reg = criterion(
                    cls_logits, cls_a, ohlc_pred, m_ohlc)
                loss_b, l_cls_b, _ = criterion(
                    cls_logits, cls_b, ohlc_pred, m_ohlc)
                loss = lam * loss_a + (1 - lam) * loss_b
                l_cls = lam * l_cls_a + (1 - lam) * l_cls_b
            else:
                if nums is not None:
                    cls_logits, ohlc_pred = model(imgs, nums, ctx_t)
                else:
                    cls_logits, ohlc_pred = model(imgs, ctx=ctx_t)

                loss, l_cls, l_reg = criterion(
                    cls_logits, cls_y, ohlc_pred, ohlc_y)

            loss = loss / accum_steps
            loss.backward()

            total_cls += l_cls.item()
            total_reg += l_reg.item()

            if step % accum_steps == 0 or step == len(tr_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # ── Валидация ─────────────────────────────────────────
        model.eval()
        val_cls_correct = 0
        val_reg_loss    = 0.0
        n_batches       = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs_dict, num_dict, cls_y, ohlc_y, ctx = batch
                imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
                cls_y = cls_y.to(device)
                ohlc_y = ohlc_y.to(device)
                ctx_t = ctx.to(device) if ctx_dim > 0 else None

                if num_dict is not None:
                    nums = {W: num_dict[W].to(device) for W in SCALES}
                    cls_logits, ohlc_pred = model(imgs, nums, ctx_t)
                else:
                    cls_logits, ohlc_pred = model(imgs, ctx=ctx_t)

                val_cls_correct += (cls_logits.argmax(1) == cls_y).sum().item()
                val_reg_loss += nn.functional.huber_loss(
                    ohlc_pred, ohlc_y, delta=0.02).item()
                n_batches += 1

        val_acc = val_cls_correct / n_val
        val_reg = val_reg_loss / max(n_batches, 1)

        # Метрика: среднее cls_acc и (1 - reg_loss*100) — оба важны
        val_metric = 0.5 * val_acc + 0.5 * max(0, 1.0 - val_reg * 100)

        n_steps = len(tr_loader)
        print(f"  [{phase_name}] Epoch {epoch:3d}/{n_epochs} | "
              f"cls_loss={total_cls/n_steps:.4f} reg_loss={total_reg/n_steps:.6f} | "
              f"val_acc={val_acc:.4f} val_reg={val_reg:.6f} metric={val_metric:.4f}")

        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Сохранено (metric={val_metric:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  ⏹ Ранняя остановка [{phase_name}]")
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))
    print(f"  └─ Лучший [{phase_name}]: metric={best_metric:.4f}")


# ══════════════════════════════════════════════════════════════════
#  train_multiscale_v2 — стандартный режим (--multiscale)
# ══════════════════════════════════════════════════════════════════

def train_multiscale_v2(tr_ds, y_tr, val_ds, y_val, ctx_dim,
                        save_path='ml/model_multiscale_v2.pt'):
    device  = get_device()
    model   = MultiScaleHybridV2(ctx_dim=ctx_dim).to(device)
    counts  = np.bincount(y_tr, minlength=3)
    weights = torch.tensor(counts.sum() / (3 * counts + 1),
                           dtype=torch.float).to(device)
    criterion = MultiTaskLoss(cls_weight=weights, gamma=2.0,
                              label_smoothing=0.15).to(device)
    tr_loader  = _make_loader(tr_ds, CFG.batch_size, True)
    val_loader = _make_loader(val_ds, CFG.batch_size, False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Параметры: {n_params:,}")
    print(f"  Mixup: alpha=0.2, label_smoothing=0.15")

    # Фаза 1: Pretrain (backbone заморожен — учатся TCN, BiLSTM, heads)
    print("\n  ── Фаза 1: Pretrain (backbone заморожен) ──")
    for p in model.backbone.parameters():
        p.requires_grad = False
    _train_loop_v2(model, tr_loader, val_loader, len(y_val),
                   n_epochs=40, max_lr=1e-3, wd=1e-3, patience_limit=12,
                   criterion=criterion, save_path=save_path,
                   phase_name="F1-pretrain", device=device, ctx_dim=ctx_dim,
                   use_mixup=True, mixup_alpha=0.2)

    # Фаза 2: Fine-tune (backbone разморожен)
    print("\n  ── Фаза 2: Fine-tune (backbone разморожен) ──")
    for p in model.backbone.parameters():
        p.requires_grad = True
    _train_loop_v2(model, tr_loader, val_loader, len(y_val),
                   n_epochs=30, max_lr=5e-5, wd=1e-2, patience_limit=10,
                   criterion=criterion, save_path=save_path,
                   phase_name="F2-finetune", device=device, ctx_dim=ctx_dim,
                   use_mixup=True, mixup_alpha=0.1)
    return model


# ══════════════════════════════════════════════════════════════════
#  train_multiscale_deep_v2 — deep-режим (--deep)
# ══════════════════════════════════════════════════════════════════

def train_multiscale_deep_v2(tr_ds, y_tr, val_ds, y_val, ctx_dim,
                             save_path='ml/model_multiscale_deep_v2.pt'):
    device  = get_device()
    model   = MultiScaleHybridV2(ctx_dim=ctx_dim).to(device)
    counts  = np.bincount(y_tr, minlength=3)
    weights = torch.tensor(counts.sum() / (3 * counts + 1),
                           dtype=torch.float).to(device)
    criterion = MultiTaskLoss(cls_weight=weights, gamma=2.0,
                              label_smoothing=0.15).to(device)
    tr_loader  = _make_loader(tr_ds, CFG.batch_size, True)
    val_loader = _make_loader(val_ds, CFG.batch_size, False)

    # Фаза 1/3: backbone заморожен
    print("\n  ── Фаза 1/3: Pretrain — backbone заморожен ──")
    for p in model.backbone.parameters():
        p.requires_grad = False
    _train_loop_v2(model, tr_loader, val_loader, len(y_val),
                   n_epochs=40, max_lr=1e-3, wd=1e-3, patience_limit=12,
                   criterion=criterion, save_path=save_path,
                   phase_name="F1-pretrain", device=device, ctx_dim=ctx_dim,
                   use_mixup=True, mixup_alpha=0.2)

    # Фаза 2/3: backbone частично разморожен (последние блоки)
    print("\n  ── Фаза 2/3: Fine-tune — часть backbone ──")
    backbone_blocks = list(model.backbone.net.children())
    for block in backbone_blocks[-2:]:
        for p in block.parameters():
            p.requires_grad = True
    _train_loop_v2(model, tr_loader, val_loader, len(y_val),
                   n_epochs=30, max_lr=3e-5, wd=5e-3, patience_limit=12,
                   criterion=criterion, save_path=save_path,
                   phase_name="F2-finetune", device=device, ctx_dim=ctx_dim,
                   use_mixup=True, mixup_alpha=0.15)

    # Фаза 3/3: весь backbone
    print("\n  ── Фаза 3/3: Deep fine-tune — весь backbone ──")
    for p in model.backbone.parameters():
        p.requires_grad = True
    torch.cuda.empty_cache()
    deep_bs = max(8, CFG.batch_size // 4)
    accum   = CFG.batch_size // deep_bs
    tr3     = _make_loader(tr_ds, deep_bs, True)
    val3    = _make_loader(val_ds, deep_bs, False)
    print(f"  (batch={deep_bs}, accum={accum}, effective={deep_bs * accum})")
    _train_loop_v2(model, tr3, val3, len(y_val),
                   n_epochs=20, max_lr=5e-6, wd=1e-2, patience_limit=8,
                   criterion=criterion, save_path=save_path,
                   phase_name="F3-deep", device=device, ctx_dim=ctx_dim,
                   accum_steps=accum, use_mixup=False)
    return model


# ══════════════════════════════════════════════════════════════════
#  Evaluate v2
# ══════════════════════════════════════════════════════════════════

def evaluate_multiscale_v2(model, te_ds, y_test, ctx_dim,
                           save_json: str = 'ml/eval_results_v2.json'):
    device = next(model.parameters()).device
    loader = _make_loader(te_ds, 64, False)
    model.eval()

    all_cls_preds = []
    all_ohlc_preds = []
    all_ohlc_true  = []
    all_probs      = []

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx = batch
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None

            if num_dict is not None:
                nums = {W: num_dict[W].to(device) for W in SCALES}
                cls_logits, ohlc_pred = model(imgs, nums, ctx_t)
            else:
                cls_logits, ohlc_pred = model(imgs, ctx=ctx_t)

            probs = torch.softmax(cls_logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_cls_preds.extend(cls_logits.argmax(1).cpu().numpy())
            all_ohlc_preds.append(ohlc_pred.cpu().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    all_probs      = np.concatenate(all_probs, axis=0)
    all_ohlc_preds = np.concatenate(all_ohlc_preds, axis=0)
    all_ohlc_true  = np.concatenate(all_ohlc_true,  axis=0)

    # ── Classification report ─────────────────────────────────
    report_str  = classification_report(
        y_test, all_cls_preds,
        target_names=['BUY', 'HOLD', 'SELL'], digits=4)
    report_dict = classification_report(
        y_test, all_cls_preds,
        target_names=['BUY', 'HOLD', 'SELL'], output_dict=True)
    print(report_str)

    # ── OHLC regression metrics ───────────────────────────────
    mae_per_channel = np.abs(all_ohlc_preds - all_ohlc_true).mean(axis=(0, 1))
    channel_names = ['ΔOpen', 'ΔHigh', 'ΔLow', 'ΔClose']
    print("  OHLC MAE (средний абсолютный % ошибки):")
    for i, name in enumerate(channel_names):
        print(f"    {name}: {mae_per_channel[i]*100:.3f}%")

    # Общий MAE
    total_mae = np.abs(all_ohlc_preds - all_ohlc_true).mean()
    print(f"  Общий MAE: {total_mae*100:.3f}%")

    # Direction accuracy (совпадение знака ΔClose[-1])
    pred_dir  = np.sign(all_ohlc_preds[:, -1, 3])
    true_dir  = np.sign(all_ohlc_true[:, -1, 3])
    dir_acc   = (pred_dir == true_dir).mean()
    print(f"  Direction accuracy (ΔClose[-1]): {dir_acc:.4f}")

    results = {
        "timestamp":      datetime.datetime.now().isoformat(),
        "cls_accuracy":   float(report_dict['accuracy']),
        "cls_macro_f1":   float(report_dict['macro avg']['f1-score']),
        "ohlc_mae":       float(total_mae),
        "ohlc_mae_per_ch": {n: float(v) for n, v in zip(channel_names, mae_per_channel)},
        "direction_acc":  float(dir_acc),
        "per_class": {
            n: {"precision": float(report_dict[n]['precision']),
                "recall":    float(report_dict[n]['recall']),
                "f1":        float(report_dict[n]['f1-score'])}
            for n in ['BUY', 'HOLD', 'SELL']
        }
    }
    with open(save_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Метрики → {save_json}")
    return results
