"""MultiScale EfficientNet-B0 + TCN (короткие) + BiLSTM (длинные) + TFT-fusion."""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import json, datetime
from ml.config import CFG, SCALES
from torchvision import models


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
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]     # причинная обрезка
        return self.act(out + self.res(x))


class TCNEncoder(nn.Module):
    """
    TCN-энкодер для одного масштаба.
    Вход: (B, C_in, W) — числовые признаки вдоль времени.
    Выход: (B, out_dim).
    """
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

    def forward(self, x):          # x: (B, C_in, W)
        return self.proj(self.pool(self.net(x)).squeeze(-1))  # (B, out_dim)


# ══════════════════════════════════════════════════════════════════
#  EfficientNet backbone (общий для всех масштабов)
# ══════════════════════════════════════════════════════════════════

def _make_backbone():
    _net = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    return nn.Sequential(_net.features, _net.avgpool, nn.Flatten())


# ══════════════════════════════════════════════════════════════════
#  MultiScaleCNN — стандартный режим (--multiscale)
# ══════════════════════════════════════════════════════════════════

class MultiScaleCNN(nn.Module):
    def __init__(self, ctx_dim: int = 0, n_scales: int = len(SCALES)):
        super().__init__()
        self.backbone   = _make_backbone()
        feat_dim        = 1280
        self.scale_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 128), nn.SiLU(), nn.Dropout(0.4))
            for _ in SCALES
        ])
        fused_dim  = 128 * n_scales + ctx_dim
        self.head  = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),       nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(128, 3),
        )

    def forward(self, imgs_by_scale: dict, ctx: torch.Tensor = None):
        feats = [self.scale_proj[i](self.backbone(imgs_by_scale[W]))
                 for i, W in enumerate(SCALES)]
        fused = torch.cat(feats, dim=1)
        if ctx is not None:
            fused = torch.cat([fused, ctx], dim=1)
        return self.head(fused)


# ══════════════════════════════════════════════════════════════════
#  MultiScaleHybrid — deep-режим (--deep / --full)
#  Короткие масштабы [5,10] → TCN
#  Длинные масштабы  [20,30] → BiLSTM
#  EfficientNet CNN  — общий визуальный backbone для всех масштабов
#  Все три потока → cross-scale attention → голова
# ══════════════════════════════════════════════════════════════════

SHORT_SCALES = [5, 10]
LONG_SCALES  = [20, 30]

class MultiScaleHybrid(nn.Module):
    """
    Гибридная архитектура:
      - EfficientNet (2D CNN) на картинках всех масштабов → proj 128
      - TCN на числовых признаках масштабов [5,10]         → 64
      - BiLSTM на числовых признаках масштабов [20,30]     → 128
      - Cross-scale Multi-Head Attention по всем потокам
      - Контекстный вектор ctx (HMM-режим + секторный)
    """
    def __init__(self, ctx_dim: int = 0,
                 n_indicator_cols: int = 28,
                 lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        n_scales = len(SCALES)

        # ── Визуальный backbone (общий) ──────────────────────────
        self.backbone   = _make_backbone()
        self.vis_proj   = nn.ModuleList([
            nn.Sequential(nn.Linear(1280, 128), nn.SiLU(), nn.Dropout(0.3))
            for _ in SCALES
        ])

        # ── TCN для коротких масштабов ───────────────────────────
        self.tcn_encoders = nn.ModuleList([
            TCNEncoder(in_channels=n_indicator_cols, out_dim=64,
                       dilations=(1, 2, 4), dropout=0.2)
            for _ in SHORT_SCALES
        ])

        # ── BiLSTM для длинных масштабов ─────────────────────────
        self.bilstm = nn.LSTM(
            input_size=n_indicator_cols,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        lstm_out_dim = lstm_hidden * 2   # 256

        # ── Проекция всех потоков в общее пространство d=128 ────
        # Визуальные: 128 × 4 масштаба
        # TCN:        64  × 2 → proj 128
        # BiLSTM:     256      → proj 128
        self.tcn_proj   = nn.Linear(64,          128)
        self.lstm_proj  = nn.Linear(lstm_out_dim, 128)

        # ── Cross-scale Multi-Head Attention ─────────────────────
        # Последовательность: 4 (vis) + 2 (tcn) + 1 (lstm) = 7 токенов по 128
        n_tokens    = n_scales + len(SHORT_SCALES) + 1
        self.attn   = nn.MultiheadAttention(embed_dim=128, num_heads=4,
                                             dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(128)

        # ── Финальная голова ────────────────────────────────────
        fused_dim = 128 * n_tokens + ctx_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),       nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(128, 3),
        )

    def forward(self, imgs_by_scale: dict,
                num_by_scale: dict = None,
                ctx: torch.Tensor = None):
        """
        imgs_by_scale: {W: (B,3,224,224)}
        num_by_scale:  {W: (B,W,n_indicator_cols)} — числовые ряды
        ctx:           (B, ctx_dim)
        """
        tokens = []

        # ── Визуальные токены ────────────────────────────────────
        for i, W in enumerate(SCALES):
            e = self.backbone(imgs_by_scale[W])        # (B, 1280)
            tokens.append(self.vis_proj[i](e))         # (B, 128)

        # ── TCN-токены (короткие масштабы) ───────────────────────
        if num_by_scale is not None:
            for j, W in enumerate(SHORT_SCALES):
                x = num_by_scale[W].permute(0, 2, 1)  # (B, C, W)
                t = self.tcn_encoders[j](x)            # (B, 64)
                tokens.append(self.tcn_proj(t))        # (B, 128)

            # ── BiLSTM-токен (длинные масштабы: cat по времени) ─
            # Берём самый длинный масштаб (30) как основной
            long_W = max(LONG_SCALES)
            x_long = num_by_scale[long_W]              # (B, 30, C)
            lstm_out, _ = self.bilstm(x_long)          # (B, 30, 256)
            lstm_last   = lstm_out[:, -1, :]           # (B, 256)
            tokens.append(self.lstm_proj(lstm_last))   # (B, 128)
        else:
            # Fallback: если числовые данные не переданы — нули
            B = next(iter(imgs_by_scale.values())).shape[0]
            dev = next(self.parameters()).device
            for _ in range(len(SHORT_SCALES) + 1):
                tokens.append(torch.zeros(B, 128, device=dev))

        # ── Cross-scale Attention ────────────────────────────────
        seq  = torch.stack(tokens, dim=1)              # (B, n_tokens, 128)
        attn_out, _ = self.attn(seq, seq, seq)         # (B, n_tokens, 128)
        seq  = self.attn_norm(seq + attn_out)          # residual
        fused = seq.flatten(1)                          # (B, n_tokens*128)

        if ctx is not None:
            fused = torch.cat([fused, ctx], dim=1)

        return self.head(fused)


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


def _collate(batch):
    imgs_list, num_list, y_list, ctx_list = zip(*batch)
    imgs_batch = {W: torch.stack([x[W] for x in imgs_list]) for W in SCALES}
    num_batch  = {W: torch.stack([x[W] for x in num_list])  for W in SCALES} \
                 if num_list[0] is not None else None
    return (imgs_batch, num_batch,
            torch.tensor(y_list, dtype=torch.long),
            torch.stack(ctx_list))


def _make_loader(ds, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, collate_fn=_collate)


def _mixup_batch(imgs_dict, y, alpha=0.2):
    if alpha <= 0:
        return imgs_dict, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B   = y.shape[0]
    idx = torch.randperm(B, device=y.device)
    imgs_mixed = {W: lam * imgs_dict[W] + (1 - lam) * imgs_dict[W][idx]
                  for W in imgs_dict}
    return imgs_mixed, y, y[idx], lam


def _forward(model, batch, device, ctx_dim, hybrid=False):
    imgs_dict, num_dict, y, ctx = batch
    imgs = {W: imgs_dict[W].to(device) for W in SCALES}
    y    = y.to(device)
    ctx  = ctx.to(device) if ctx_dim > 0 else None
    if hybrid and num_dict is not None:
        nums = {W: num_dict[W].to(device) for W in SCALES}
        return model(imgs, nums, ctx), y
    return model(imgs, ctx=ctx), y


# ══════════════════════════════════════════════════════════════════
#  Общий цикл обучения
# ══════════════════════════════════════════════════════════════════

def _train_loop(model, tr_loader, val_loader, y_val,
                n_epochs, max_lr, wd, patience_limit,
                criterion, save_path, phase_name,
                device, ctx_dim, hybrid=False, mixup_alpha=0.2):

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=max_lr / 10, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr,
        steps_per_epoch=len(tr_loader), epochs=n_epochs,
        pct_start=0.2, div_factor=10, final_div_factor=200)

    best_acc, patience = 0.0, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tr_loader:
            imgs_dict, num_dict, y, ctx = batch
            imgs_dict = {W: imgs_dict[W].to(device) for W in SCALES}
            y   = y.to(device)
            ctx = ctx.to(device) if ctx_dim > 0 else None

            imgs_m, y_a, y_b, lam = _mixup_batch(imgs_dict, y, mixup_alpha)

            if hybrid and num_dict is not None:
                nums   = {W: num_dict[W].to(device) for W in SCALES}
                logits = model(imgs_m, nums, ctx)
            else:
                logits = model(imgs_m, ctx=ctx)

            loss = (lam * criterion(logits, y_a) +
                    (1 - lam) * criterion(logits, y_b))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                logits, y_v = _forward(model, batch, device, ctx_dim, hybrid)
                correct += (logits.argmax(1) == y_v).sum().item()
        val_acc = correct / len(y_val)
        print(f"  [{phase_name}] Epoch {epoch:3d}/{n_epochs} | "
              f"loss={total_loss/len(tr_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Сохранено (val_acc={val_acc:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  ⏹ Ранняя остановка [{phase_name}]")
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))
    print(f"  └─ Лучший [{phase_name}]: {best_acc:.4f}")


# ══════════════════════════════════════════════════════════════════
#  train_multiscale — стандартный режим (--multiscale)
# ══════════════════════════════════════════════════════════════════

def train_multiscale(tr_ds, y_tr, val_ds, y_val, ctx_dim,
                     save_path='ml/model_multiscale.pt'):
    device    = get_device()
    model     = MultiScaleCNN(ctx_dim=ctx_dim).to(device)
    counts    = np.bincount(y_tr)
    weights   = torch.tensor(counts.sum() / (3 * counts),
                              dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=2.0, weight=weights, label_smoothing=0.1)
    tr_loader = _make_loader(tr_ds, CFG.batch_size, True)
    val_loader= _make_loader(val_ds, CFG.batch_size, False)

    print("\n  ── Фаза 1: Pretrain (backbone заморожен) ──")
    for p in model.backbone[0].parameters():
        p.requires_grad = False
    _train_loop(model, tr_loader, val_loader, y_val,
                n_epochs=30, max_lr=1e-3, wd=1e-4, patience_limit=10,
                criterion=criterion, save_path=save_path,
                phase_name="F1-pretrain", device=device,
                ctx_dim=ctx_dim, hybrid=False, mixup_alpha=0.2)

    print("\n  ── Фаза 2: Fine-tune (features[8] + head) ──")
    for name, p in model.backbone[0].named_parameters():
        if name.split('.')[0] == '8':
            p.requires_grad = True
    _train_loop(model, tr_loader, val_loader, y_val,
                n_epochs=25, max_lr=1e-4, wd=5e-3, patience_limit=8,
                criterion=criterion, save_path=save_path,
                phase_name="F2-finetune", device=device,
                ctx_dim=ctx_dim, hybrid=False, mixup_alpha=0.2)
    return model


# ══════════════════════════════════════════════════════════════════
#  train_multiscale_deep — deep-режим (--deep)
# ══════════════════════════════════════════════════════════════════

def train_multiscale_deep(tr_ds, y_tr, val_ds, y_val, ctx_dim,
                          save_path='ml/model_multiscale_deep.pt'):
    device    = get_device()
    model     = MultiScaleHybrid(ctx_dim=ctx_dim).to(device)
    counts    = np.bincount(y_tr)
    weights   = torch.tensor(counts.sum() / (3 * counts),
                              dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=2.0, weight=weights, label_smoothing=0.05)
    tr_loader = _make_loader(tr_ds, CFG.batch_size, True)
    val_loader= _make_loader(val_ds, CFG.batch_size, False)

    print("\n  ── Фаза 1/3: Pretrain — backbone заморожен ──")
    for p in model.backbone.parameters():
        p.requires_grad = False
    _train_loop(model, tr_loader, val_loader, y_val,
                n_epochs=30, max_lr=1e-3, wd=1e-4, patience_limit=12,
                criterion=criterion, save_path=save_path,
                phase_name="F1-pretrain", device=device,
                ctx_dim=ctx_dim, hybrid=True, mixup_alpha=0.3)

    print("\n  ── Фаза 2/3: Fine-tune — features 6/7/8 ──")
    for name, p in model.backbone[0].named_parameters():
        if name.split('.')[0] in ('6', '7', '8'):
            p.requires_grad = True
    _train_loop(model, tr_loader, val_loader, y_val,
                n_epochs=30, max_lr=3e-5, wd=1e-3, patience_limit=12,
                criterion=criterion, save_path=save_path,
                phase_name="F2-finetune", device=device,
                ctx_dim=ctx_dim, hybrid=True, mixup_alpha=0.3)

    print("\n  ── Фаза 3/3: Deep fine-tune — весь backbone ──")
    for p in model.backbone.parameters():
        p.requires_grad = True
    _train_loop(model, tr_loader, val_loader, y_val,
                n_epochs=20, max_lr=5e-6, wd=5e-3, patience_limit=8,
                criterion=criterion, save_path=save_path,
                phase_name="F3-deep", device=device,
                ctx_dim=ctx_dim, hybrid=True, mixup_alpha=0.1)
    return model


# ══════════════════════════════════════════════════════════════════
#  Evaluate
# ══════════════════════════════════════════════════════════════════

def evaluate_multiscale(model, te_ds, y_test, ctx_dim,
                        save_json: str = 'ml/eval_results.json',
                        confidence_thr: float = 0.0,
                        hybrid: bool = False):
    device = next(model.parameters()).device
    loader = _make_loader(te_ds, 64, False)
    model.eval()

    preds, probs_all = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = _forward(model, batch, device, ctx_dim, hybrid)
            softmax   = torch.softmax(logits, dim=1)
            probs_all.append(softmax.cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())

    probs_all = np.concatenate(probs_all, axis=0)

    if confidence_thr > 0.0:
        max_conf = probs_all.max(axis=1)
        preds    = np.array(preds)
        preds[max_conf < confidence_thr] = 1
        print(f"  Отфильтровано как HOLD (conf < {confidence_thr}): "
              f"{(max_conf < confidence_thr).sum()} / {len(preds)}")

    report_str  = classification_report(
        y_test, preds, target_names=['BUY','HOLD','SELL'], digits=4)
    report_dict = classification_report(
        y_test, preds, target_names=['BUY','HOLD','SELL'], output_dict=True)
    print(report_str)

    print("  Среднее max_confidence:")
    for cls_idx, name in enumerate(['BUY','HOLD','SELL']):
        mask = np.array(y_test) == cls_idx
        if mask.sum() > 0:
            print(f"    {name}: {probs_all[mask].max(axis=1).mean():.3f}")

    results = {
        "timestamp":      datetime.datetime.now().isoformat(),
        "accuracy":       float(report_dict['accuracy']),
        "macro_f1":       float(report_dict['macro avg']['f1-score']),
        "confidence_thr": confidence_thr,
        "per_class": {
            n: {"precision": float(report_dict[n]['precision']),
                "recall":    float(report_dict[n]['recall']),
                "f1":        float(report_dict[n]['f1-score'])}
            for n in ['BUY','HOLD','SELL']
        }
    }
    with open(save_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Метрики → {save_json}")
    return results
