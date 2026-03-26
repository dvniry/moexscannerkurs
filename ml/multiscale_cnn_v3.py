"""MultiScale Hybrid v3.2 — фикс SELL-bias, σ_reg коллапса, метрики ранней остановки.

Изменения v3.1 → v3.2:
  1. MultiTaskLossV3: clamp log_sigma чтобы prec не улетал в бесконечность
  2. train_multiscale_v3: убран двойной буст SELL (base_weights[2] *= 2.0)
  3. _train_loop_v3: метрика = 0.3*acc + 0.5*macro_f1 + 0.2*min_class_f1
     — min_class_f1 убивает SELL-only решения
  4. train_multiscale_v3: дифференциальный LR (backbone × 0.1, hourly × 0.3)
  5. Фаза 1: n_epochs=15, max_lr=3e-3, patience=7
     Фаза 2: n_epochs=40, max_lr=3e-4, patience=12
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
import json, datetime, math


from ml.config import CFG, SCALES
from ml.candle_render_v2 import N_RENDER_CHANNELS
from ml.hourly_encoder import (
    HourlyEncoder, N_HOURLY_CHANNELS, N_HOURS_PER_DAY, N_INTRADAY_DAYS
)


# ══════════════════════════════════════════════════════════════════
#  SpatialDropout1D
# ══════════════════════════════════════════════════════════════════


class SpatialDropout1d(nn.Module):
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(
            torch.full((x.shape[0], x.shape[1], 1), 1 - self.p, device=x.device)
        )
        return x * mask / (1 - self.p)


# ══════════════════════════════════════════════════════════════════
#  Mixup augmentation
# ══════════════════════════════════════════════════════════════════


def mixup_data(imgs_dict, num_dict, cls_y, ohlc_y, ctx, alpha=0.2,
               hourly=None):
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
    mixed_hourly = (lam * hourly + (1 - lam) * hourly[perm]) \
                   if hourly is not None else None

    return (mixed_imgs, mixed_nums, cls_y, cls_y[perm],
            mixed_ohlc, mixed_ctx, lam, mixed_hourly)


# ══════════════════════════════════════════════════════════════════
#  Asymmetric Focal Loss
# ══════════════════════════════════════════════════════════════════


class AsymmetricFocalLoss(nn.Module):
    """Focal Loss с разным gamma для каждого класса.

    BUY  (class 0): gamma=2.0 → стандартный
    HOLD (class 1): gamma=1.5 → больше подавление easy samples
    SELL (class 2): gamma=1.5 → менее агрессивное подавление
    """
    def __init__(self, gamma_per_class=(2.0, 1.5, 1.5),
                 weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma_per_class
        self.ce = nn.CrossEntropyLoss(
            weight=weight, reduction='none', label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)

        gamma_t = torch.zeros_like(ce_loss)
        for cls_idx, g in enumerate(self.gamma):
            mask = (targets == cls_idx)
            gamma_t[mask] = g

        return ((1 - pt) ** gamma_t * ce_loss).mean()


# ══════════════════════════════════════════════════════════════════
#  Improved Heads
# ══════════════════════════════════════════════════════════════════


class ImprovedRegHead(nn.Module):
    """Регрессионная голова с temporal decomposition + residual shortcut."""
    def __init__(self, in_dim: int = 128, future_bars: int = 5,
                 hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.future_bars = future_bars

        self.temporal_proj = nn.Sequential(
            nn.Linear(in_dim, hidden * future_bars),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.bar_refine = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),
        )
        self.shortcut = nn.Linear(in_dim, future_bars * 4)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B = x.shape[0]
        temporal = self.temporal_proj(x)
        temporal = temporal.view(B, self.future_bars, -1)
        ohlc_1 = self.bar_refine(temporal).reshape(B, -1)
        ohlc_2 = self.shortcut(x)
        alpha = torch.sigmoid(self.alpha)
        return alpha * ohlc_1 + (1 - alpha) * ohlc_2


class CalibratedClsHead(nn.Module):
    """Классификационная голова с learnable temperature scaling."""
    def __init__(self, in_dim: int = 128, n_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, n_classes),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.head(x)
        T = self.temperature.clamp(min=0.1, max=10.0)
        return logits / T


# ══════════════════════════════════════════════════════════════════
#  1D Conv Backbone
# ══════════════════════════════════════════════════════════════════


class Conv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dropout=0.15):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad)
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
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


# ══════════════════════════════════════════════════════════════════
#  TCN
# ══════════════════════════════════════════════════════════════════


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
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
#  Gated Residual Network (GRN)
# ══════════════════════════════════════════════════════════════════


class GRN(nn.Module):
    """Gated Residual Network — по мотивам Temporal Fusion Transformer."""
    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.elu  = nn.ELU()
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) \
                    if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.drop(h)
        out  = self.fc2(h)
        gate = self.gate(h)
        return self.norm(gate * out + (1 - gate) * skip)


# ══════════════════════════════════════════════════════════════════
#  MultiScaleHybrid v3.2
# ══════════════════════════════════════════════════════════════════


SHORT_SCALES = [5, 10]
LONG_SCALES  = [20, 30]
TRUNK_OUT    = 128


class MultiScaleHybridV3(nn.Module):
    """
    Гибридная архитектура v3.2:
      - FactualBackbone (1D CNN) на рендерах всех масштабов → 128 × 4
      - TCN на числовых признаках масштабов [5,10]           → 128 × 2
      - BiLSTM на числовых признаках масштаба [30]           → 128 × 1
      - HourlyEncoder (часовые свечи 5 дней)                 → 128 × 1
      - Cross-scale MHA (8 токенов × 128)
      - GRN trunk
      - CalibratedClsHead (3 класса) + ImprovedRegHead (OHLC)
    """
    def __init__(self, ctx_dim: int = 0,
                 n_indicator_cols: int = 30,
                 future_bars: int = None,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 use_hourly: bool = True):
        super().__init__()
        if future_bars is None:
            future_bars = CFG.future_bars
        self.future_bars = future_bars
        self.use_hourly  = use_hourly
        n_scales = len(SCALES)

        self.backbone = FactualBackbone(
            in_channels=N_RENDER_CHANNELS, out_dim=TRUNK_OUT, dropout=0.3)

        self.scale_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(TRUNK_OUT, TRUNK_OUT), nn.SiLU(), nn.Dropout(0.3))
            for _ in SCALES
        ])

        self.tcn_encoders = nn.ModuleList([
            TCNEncoder(in_channels=n_indicator_cols, out_dim=64,
                       dilations=(1, 2, 4), dropout=0.3)
            for _ in SHORT_SCALES
        ])

        self.bilstm = nn.LSTM(
            input_size=n_indicator_cols,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.4,
        )
        lstm_out_dim = lstm_hidden * 2

        self.tcn_proj  = nn.Linear(64,           TRUNK_OUT)
        self.lstm_proj = nn.Linear(lstm_out_dim,  TRUNK_OUT)

        if use_hourly:
            self.hourly_enc = HourlyEncoder(
                n_days=N_INTRADAY_DAYS, d_intra=64,
                d_out=TRUNK_OUT, n_heads=2, dropout=0.2)

        n_tokens = n_scales + len(SHORT_SCALES) + 1 + (1 if use_hourly else 0)
        self.attn = nn.MultiheadAttention(
            embed_dim=TRUNK_OUT, num_heads=4, dropout=0.2, batch_first=True)
        self.attn_norm = nn.LayerNorm(TRUNK_OUT)

        fused_dim = TRUNK_OUT * n_tokens + ctx_dim
        self.trunk = GRN(
            input_dim=fused_dim, hidden_dim=256,
            output_dim=TRUNK_OUT, dropout=0.4)

        self.cls_head = CalibratedClsHead(
            in_dim=TRUNK_OUT, n_classes=3, dropout=0.3)
        self.reg_head = ImprovedRegHead(
            in_dim=TRUNK_OUT, future_bars=future_bars,
            hidden=64, dropout=0.2)

    def forward(self, imgs_by_scale: dict,
                num_by_scale: dict = None,
                ctx: torch.Tensor = None,
                hourly: torch.Tensor = None):
        tokens = []

        for i, W in enumerate(SCALES):
            e = self.backbone(imgs_by_scale[W])
            tokens.append(self.scale_proj[i](e))

        if num_by_scale is not None:
            for j, W in enumerate(SHORT_SCALES):
                x = num_by_scale[W].permute(0, 2, 1)
                t = self.tcn_encoders[j](x)
                tokens.append(self.tcn_proj(t))

            long_W = max(LONG_SCALES)
            x_long = num_by_scale[long_W]
            out, _ = self.bilstm(x_long)
            tokens.append(self.lstm_proj(out[:, -1, :]))
        else:
            B   = next(iter(imgs_by_scale.values())).shape[0]
            dev = next(self.parameters()).device
            for _ in range(len(SHORT_SCALES) + 1):
                tokens.append(torch.zeros(B, TRUNK_OUT, device=dev))

        if self.use_hourly and hourly is not None:
            tokens.append(self.hourly_enc(hourly))
        elif self.use_hourly:
            B   = next(iter(imgs_by_scale.values())).shape[0]
            dev = next(self.parameters()).device
            tokens.append(torch.zeros(B, TRUNK_OUT, device=dev))

        seq = torch.stack(tokens, dim=1)
        attn_out, _ = self.attn(seq, seq, seq)
        seq   = self.attn_norm(seq + attn_out)
        fused = seq.flatten(1)

        if ctx is not None:
            fused = torch.cat([fused, ctx], dim=1)

        h = self.trunk(fused)
        cls_logits = self.cls_head(h)
        ohlc_flat  = self.reg_head(h)

        return cls_logits, ohlc_flat


# ══════════════════════════════════════════════════════════════════
#  Multi-task Loss v3.2  — FIX: clamp log_sigma
# ══════════════════════════════════════════════════════════════════


class MultiTaskLossV3(nn.Module):
    """
    L = (1/2σ²_cls) * Focal(cls) + (1/2σ²_reg) * Huber(reg) + log(σ) terms

    v3.2 ФИКС: clamp log_sigma в [-1.5, 2.0] чтобы prec_reg не улетал
    в бесконечность и не давил cls loss.
    """
    def __init__(self, cls_weight=None,
                 gamma_per_class=(2.0, 3.0, 1.0),
                 label_smoothing=0.1,
                 huber_delta=0.02,
                 direction_weight=0.1):
        super().__init__()
        self.focal = AsymmetricFocalLoss(
            gamma_per_class=gamma_per_class,
            weight=cls_weight,
            label_smoothing=label_smoothing)
        self.huber_delta      = huber_delta
        self.direction_weight = direction_weight

        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_reg = nn.Parameter(torch.tensor(0.0))

    def forward(self, cls_logits, cls_targets, ohlc_pred, ohlc_targets):
        loss_cls = self.focal(cls_logits, cls_targets)

        loss_huber = F.huber_loss(
            ohlc_pred, ohlc_targets,
            reduction='mean', delta=self.huber_delta)

        pred_close_last = ohlc_pred[:, -1]
        true_close_last = ohlc_targets[:, -1]
        sign_wrong = (pred_close_last * true_close_last) < 0
        dir_penalty = torch.where(
            sign_wrong,
            (pred_close_last - true_close_last).abs(),
            torch.zeros_like(pred_close_last)
        ).mean()

        loss_reg = loss_huber + self.direction_weight * dir_penalty

        # ── ФИКС v3.2: clamp чтобы prec не улетал ────────────
        log_s_cls = self.log_sigma_cls.clamp(-1.5, 1.0)
        log_s_reg = self.log_sigma_reg.clamp(-1.5, 1.5)
        prec_cls  = torch.exp(-2 * log_s_cls)
        prec_reg  = torch.exp(-2 * log_s_reg)

        total = (prec_cls * loss_cls + log_s_cls +
                 prec_reg * loss_reg + log_s_reg)

        return total, loss_cls.detach(), loss_reg.detach()


# ══════════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════════


def get_device():
    if torch.cuda.is_available():
        d = torch.device('cuda')
        p = torch.cuda.get_device_properties(0)
        mem = getattr(p, 'total_mem', None) or getattr(p, 'total_memory', 0)
        print(f"  Устройство: {p.name} ({mem // 1024**2} MB VRAM)")
    else:
        d = torch.device('cpu')
        print("  Устройство: cpu")
    return d


def _collate_v3(batch):
    items = list(zip(*batch))
    imgs_list, num_list, cls_list, ohlc_list, ctx_list = items[:5]
    hourly_list = items[5] if len(items) > 5 else None

    imgs_batch = {W: torch.stack([x[W] for x in imgs_list]) for W in SCALES}
    num_batch  = {W: torch.stack([x[W] for x in num_list])  for W in SCALES} \
                 if num_list[0] is not None else None

    result = (
        imgs_batch, num_batch,
        torch.tensor(cls_list,  dtype=torch.long),
        torch.stack(ohlc_list),
        torch.stack(ctx_list),
    )

    if hourly_list is not None and hourly_list[0] is not None:
        result = result + (torch.stack(hourly_list),)
    else:
        result = result + (None,)

    return result


def _make_loader_v3(ds, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, collate_fn=_collate_v3)


# ══════════════════════════════════════════════════════════════════
#  Training loop v3.2
# ══════════════════════════════════════════════════════════════════


def _train_loop_v3(model, tr_loader, val_loader, n_val,
                   n_epochs, max_lr, wd, patience_limit,
                   criterion, save_path, phase_name,
                   device, ctx_dim, use_hourly,
                   param_groups: list = None,
                   accum_steps: int = 1,
                   use_mixup: bool = True, mixup_alpha: float = 0.2):
    """
    param_groups: если передан — используется вместо стандартного набора params.
    Позволяет задать дифференциальный LR для backbone/heads.
    """
    if param_groups is not None:
        # Добавляем criterion.parameters() к первой группе (heads lr)
        head_lr = param_groups[-1]['lr']
        crit_params = list(criterion.parameters())
        optimizer = torch.optim.AdamW(
            param_groups + [{'params': crit_params, 'lr': head_lr}],
            weight_decay=wd)
        # max_lr для scheduler — берём максимальный из групп
        sched_max_lr = [g['lr'] for g in param_groups] + [head_lr]
    else:
        all_params = list(filter(lambda p: p.requires_grad, model.parameters())) \
                   + list(criterion.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=max_lr / 10,
                                       weight_decay=wd)
        sched_max_lr = max_lr

    opt_steps = math.ceil(len(tr_loader) / accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=sched_max_lr,
        steps_per_epoch=opt_steps, epochs=n_epochs,
        pct_start=0.2, div_factor=10, final_div_factor=200)

    best_metric, patience = 0.0, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        criterion.train()
        total_cls, total_reg = 0.0, 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tr_loader, 1):
            imgs_dict, num_dict, cls_y, ohlc_y, ctx = batch[:5]
            hourly_data = batch[5] if len(batch) > 5 else None

            imgs   = {W: imgs_dict[W].to(device) for W in SCALES}
            cls_y  = cls_y.to(device)
            ohlc_y = ohlc_y.to(device)
            ctx_t  = ctx.to(device) if ctx_dim > 0 else None
            hourly_t = hourly_data.to(device) \
                       if (use_hourly and hourly_data is not None) else None

            nums = {W: num_dict[W].to(device) for W in SCALES} \
                   if num_dict is not None else None

            if use_mixup and mixup_alpha > 0:
                m_imgs, m_nums, cls_a, cls_b, m_ohlc, m_ctx, lam, m_hourly = \
                    mixup_data(imgs, nums, cls_y, ohlc_y, ctx_t, mixup_alpha,
                               hourly=hourly_t)

                cls_logits, ohlc_pred = model(
                    m_imgs, m_nums, m_ctx, hourly=m_hourly)

                loss_a, l_cls_a, l_reg = criterion(
                    cls_logits, cls_a, ohlc_pred, m_ohlc)
                loss_b, l_cls_b, _ = criterion(
                    cls_logits, cls_b, ohlc_pred, m_ohlc)
                loss  = lam * loss_a + (1 - lam) * loss_b
                l_cls = lam * l_cls_a + (1 - lam) * l_cls_b
            else:
                cls_logits, ohlc_pred = model(
                    imgs, nums, ctx_t, hourly=hourly_t)
                loss, l_cls, l_reg = criterion(
                    cls_logits, cls_y, ohlc_pred, ohlc_y)

            loss = loss / accum_steps
            loss.backward()

            total_cls += l_cls.item()
            total_reg += l_reg.item()

            if step % accum_steps == 0 or step == len(tr_loader):
                trainable = [p for g in optimizer.param_groups
                             for p in g['params'] if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # ── Валидация ─────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_reg_sum = 0.0
        n_batches   = 0
        val_preds   = []
        val_trues   = []

        with torch.no_grad():
            for batch in val_loader:
                imgs_dict, num_dict, cls_y, ohlc_y, ctx = batch[:5]
                hourly_data = batch[5] if len(batch) > 5 else None

                imgs   = {W: imgs_dict[W].to(device) for W in SCALES}
                cls_y  = cls_y.to(device)
                ohlc_y = ohlc_y.to(device)
                ctx_t  = ctx.to(device) if ctx_dim > 0 else None
                hourly_t = hourly_data.to(device) \
                           if (use_hourly and hourly_data is not None) else None

                nums = {W: num_dict[W].to(device) for W in SCALES} \
                       if num_dict is not None else None

                if nums is not None:
                    cls_logits, ohlc_pred = model(imgs, nums, ctx_t,
                                                  hourly=hourly_t)
                else:
                    cls_logits, ohlc_pred = model(imgs, ctx=ctx_t,
                                                  hourly=hourly_t)

                preds = cls_logits.argmax(1)
                val_correct += (preds == cls_y).sum().item()
                val_reg_sum += F.huber_loss(
                    ohlc_pred, ohlc_y, delta=0.02).item()
                n_batches += 1

                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(cls_y.cpu().numpy())

        val_acc      = val_correct / n_val
        val_preds_np = np.array(val_preds)
        val_trues_np = np.array(val_trues)

        macro_f1 = f1_score(val_trues_np, val_preds_np, average='macro',
                            zero_division=0)

        # ── ФИКС v3.2: метрика штрафует за SELL-only решения ──
        buy_f1  = f1_score(val_trues_np == 0, val_preds_np == 0,
                           zero_division=0)
        hold_f1 = f1_score(val_trues_np == 1, val_preds_np == 1,
                           zero_division=0)
        sell_f1 = f1_score(val_trues_np == 2, val_preds_np == 2,
                           zero_division=0)
        min_class_f1 = min(buy_f1, hold_f1, sell_f1)

        val_metric = 0.3 * val_acc + 0.5 * macro_f1 + 0.2 * min_class_f1
        # ───────────────────────────────────────────────────────

        n_steps   = len(tr_loader)
        sigma_cls = torch.exp(criterion.log_sigma_cls).item()
        sigma_reg = torch.exp(criterion.log_sigma_reg).item()

        print(f"  [{phase_name}] Epoch {epoch:3d}/{n_epochs} | "
              f"cls={total_cls/n_steps:.4f} reg={total_reg/n_steps:.6f} | "
              f"val_acc={val_acc:.4f} macro_f1={macro_f1:.4f} "
              f"buy={buy_f1:.3f} hold={hold_f1:.3f} sell={sell_f1:.3f} | "
              f"σ_cls={sigma_cls:.3f} σ_reg={sigma_reg:.3f} | "
              f"metric={val_metric:.4f}")

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
#  train_multiscale_v3  v3.2 — дифференциальный LR + balanced weights
# ══════════════════════════════════════════════════════════════════


def train_multiscale_v3(tr_ds, y_tr, val_ds, y_val, ctx_dim,
                        use_hourly=True,
                        save_path='ml/model_multiscale_v3.pt'):
    device = get_device()
    model  = MultiScaleHybridV3(ctx_dim=ctx_dim,
                                 use_hourly=use_hourly).to(device)
    counts = np.bincount(y_tr, minlength=3)

    # ── ФИКС v3.2: убираем двойной буст SELL ─────────────────
    # Focal loss с gamma уже усиливает редкие классы —
    # дополнительный × 2.0 создавал избыточный SELL-bias
    base_weights = counts.sum() / (3 * counts + 1)
    weights = torch.tensor(base_weights, dtype=torch.float).to(device)
    print(f"  Class weights: BUY={weights[0]:.2f} HOLD={weights[1]:.2f} "
          f"SELL={weights[2]:.2f}")

    criterion = MultiTaskLossV3(
        cls_weight=weights,
        gamma_per_class=(2.0, 1.5, 1.5),
        label_smoothing=0.1,
        huber_delta=0.02,
        direction_weight=0.1).to(device)

    tr_loader  = _make_loader_v3(tr_ds, CFG.batch_size, True)
    val_loader = _make_loader_v3(val_ds, CFG.batch_size, False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Параметры: {n_params:,}")
    print(f"  Mixup: alpha=0.2, label_smoothing=0.1")
    print(f"  Asymmetric Focal: gamma=(BUY=2.0, HOLD=3.0, SELL=1.0)")
    print(f"  Huber delta=0.02, direction_weight=0.1")
    print(f"  Hourly encoder: {'ON' if use_hourly else 'OFF'}")
    print(f"  Метрика: 0.3*acc + 0.5*macro_f1 + 0.2*min_class_f1")

    # ── Фаза 1: Pretrain (backbone заморожен, короткая) ───────
    print("\n  ── Фаза 1: Pretrain (backbone заморожен) ──")
    for p in model.backbone.parameters():
        p.requires_grad = False

    _train_loop_v3(model, tr_loader, val_loader, len(y_val),
                   n_epochs=15, max_lr=3e-3, wd=1e-3, patience_limit=7,
                   criterion=criterion, save_path=save_path,
                   phase_name="F1-pretrain", device=device,
                   ctx_dim=ctx_dim, use_hourly=use_hourly,
                   param_groups=None,
                   use_mixup=True, mixup_alpha=0.2)

    # ── Фаза 2: Fine-tune с дифференциальным LR ───────────────
    print("\n  ── Фаза 2: Fine-tune (дифференциальный LR) ──")
    for p in model.backbone.parameters():
        p.requires_grad = True

    # Backbone учится в 10× медленнее, hourly в 3× медленнее
    backbone_params = list(model.backbone.parameters())
    backbone_ids    = {id(p) for p in backbone_params}

    hourly_params = list(model.hourly_enc.parameters()) \
                    if use_hourly else []
    hourly_ids    = {id(p) for p in hourly_params}

    other_params = [p for p in model.parameters()
                    if p.requires_grad
                    and id(p) not in backbone_ids
                    and id(p) not in hourly_ids]

    max_lr_ft = 3e-4
    param_groups = [
        {'params': backbone_params, 'lr': max_lr_ft * 0.1},
        {'params': hourly_params,   'lr': max_lr_ft * 0.3},
        {'params': other_params,    'lr': max_lr_ft},
    ]
    if not hourly_params:
        param_groups = [
            {'params': backbone_params, 'lr': max_lr_ft * 0.1},
            {'params': other_params,    'lr': max_lr_ft},
        ]

    _train_loop_v3(model, tr_loader, val_loader, len(y_val),
                   n_epochs=40, max_lr=max_lr_ft, wd=1e-2,
                   patience_limit=12,
                   criterion=criterion, save_path=save_path,
                   phase_name="F2-finetune", device=device,
                   ctx_dim=ctx_dim, use_hourly=use_hourly,
                   param_groups=param_groups,
                   use_mixup=True, mixup_alpha=0.1)

    return model


# ══════════════════════════════════════════════════════════════════
#  Evaluate v3
# ══════════════════════════════════════════════════════════════════


def evaluate_multiscale_v3(model, te_ds, y_test, ctx_dim,
                           use_hourly=True,
                           save_json: str = 'ml/eval_results_v3.json'):
    device = next(model.parameters()).device
    loader = _make_loader_v3(te_ds, 64, False)
    model.eval()

    all_cls_preds  = []
    all_ohlc_preds = []
    all_ohlc_true  = []
    all_probs      = []

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx = batch[:5]
            hourly_data = batch[5] if len(batch) > 5 else None

            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            hourly_t = hourly_data.to(device) \
                       if (use_hourly and hourly_data is not None) else None

            nums = {W: num_dict[W].to(device) for W in SCALES} \
                   if num_dict is not None else None

            if nums is not None:
                cls_logits, ohlc_pred = model(imgs, nums, ctx_t,
                                              hourly=hourly_t)
            else:
                cls_logits, ohlc_pred = model(imgs, ctx=ctx_t,
                                              hourly=hourly_t)

            probs = torch.softmax(cls_logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_cls_preds.extend(cls_logits.argmax(1).cpu().numpy())
            all_ohlc_preds.append(ohlc_pred.cpu().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    all_probs      = np.concatenate(all_probs,      axis=0)
    all_ohlc_preds = np.concatenate(all_ohlc_preds, axis=0)
    all_ohlc_true  = np.concatenate(all_ohlc_true,  axis=0)

    n_samples     = len(all_ohlc_preds)
    fb            = CFG.future_bars
    ohlc_pred_5x4 = all_ohlc_preds.reshape(n_samples, fb, 4)
    ohlc_true_5x4 = all_ohlc_true.reshape(n_samples, fb, 4)

    report_str  = classification_report(
        y_test, all_cls_preds,
        target_names=['BUY', 'HOLD', 'SELL'], digits=4)
    report_dict = classification_report(
        y_test, all_cls_preds,
        target_names=['BUY', 'HOLD', 'SELL'], output_dict=True)
    print(report_str)

    mae_per_channel = np.abs(ohlc_pred_5x4 - ohlc_true_5x4).mean(axis=(0, 1))
    channel_names   = ['ΔOpen', 'ΔHigh', 'ΔLow', 'ΔClose']
    print("  OHLC MAE (средний абсолютный % ошибки):")
    for i, name in enumerate(channel_names):
        print(f"    {name}: {mae_per_channel[i]*100:.3f}%")

    total_mae = np.abs(ohlc_pred_5x4 - ohlc_true_5x4).mean()
    print(f"  Общий MAE: {total_mae*100:.3f}%")

    pred_dir = np.sign(ohlc_pred_5x4[:, -1, 3])
    true_dir = np.sign(ohlc_true_5x4[:, -1, 3])
    dir_acc  = (pred_dir == true_dir).mean()
    print(f"  Direction accuracy (ΔClose[-1]): {dir_acc:.4f}")

    results = {
        "timestamp":       datetime.datetime.now().isoformat(),
        "cls_accuracy":    float(report_dict['accuracy']),
        "cls_macro_f1":    float(report_dict['macro avg']['f1-score']),
        "ohlc_mae":        float(total_mae),
        "ohlc_mae_per_ch": {n: float(v)
                            for n, v in zip(channel_names, mae_per_channel)},
        "direction_acc":   float(dir_acc),
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
