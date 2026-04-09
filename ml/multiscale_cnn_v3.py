# ml/multiscale_cnn_v3.py
"""MultiScale CNN Hybrid v3.10

Изменения v3.10:
- AsymmetricFocalLoss: переписан через F.cross_entropy (stable, logsumexp trick)
  → нет NaN gradients при уверенных неверных предсказаниях
- BiLSTM вход: жёсткий clamp(-8, 8) + nan_to_num перед forward
- CalibratedClsHead: LayerNorm → SiLU → Linear (без dropout, без промежуточного 64)
- NaN-guards в 3 точках forward()
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

try:
    from ml.config import CFG, SCALES
except ImportError:
    from config import CFG, SCALES

TRUNK_OUT = 128   # единая размерность признаков


# ────────────────────────────────────────────────────────────
# Blocks
# ────────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, k, stride=stride,
                      padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            ConvBnAct(c, c, k=3, dilation=dilation),
            nn.Conv1d(c, c, 1, bias=False),
            nn.BatchNorm1d(c),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.body(x))


class SingleScaleBackbone(nn.Module):
    """1D-CNN backbone для одного масштаба.
    Input:  [B, 3, T]   (T = 64 для всех масштабов после resize в датасете)
    Output: [B, TRUNK_OUT]
    """
    def __init__(self, base_ch=64):
        super().__init__()
        self.stem = ConvBnAct(3, base_ch, k=5)
        self.blocks = nn.Sequential(
            ResBlock(base_ch, dilation=1),
            ConvBnAct(base_ch, base_ch*2, k=3, stride=2),
            ResBlock(base_ch*2, dilation=2),
            ConvBnAct(base_ch*2, TRUNK_OUT, k=3, stride=2),
            ResBlock(TRUNK_OUT, dilation=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)   # [B, TRUNK_OUT]


class GRN(nn.Module):
    """Gated Residual Network."""
    def __init__(self, d_in, d_out, dropout=0.3):
        super().__init__()
        self.fc1  = nn.Linear(d_in,  d_out)
        self.fc2  = nn.Linear(d_out, d_out)
        self.gate = nn.Linear(d_in,  d_out)
        self.proj = nn.Linear(d_in,  d_out) if d_in != d_out else nn.Identity()
        self.ln   = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        h = F.gelu(self.fc1(x))
        h = self.drop(self.fc2(h))
        return self.ln(self.proj(x) + g * h)


class BiLSTMBranch(nn.Module):
    """Обрабатывает числовые фичи длинного масштаба (scale=30).
    Input:  [B, T, n_ind]   T = scale=30 шагов, n_ind индикаторов
    Output: [B, TRUNK_OUT]
    """
    def __init__(self, n_ind, hidden=128, num_layers=2):
        super().__init__()
        self.hidden = hidden
        self.bilstm = nn.LSTM(
            input_size=n_ind, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=0.1 if num_layers > 1 else 0.0,
        )
        # 2 directions × 2 layers последних hidden → projection
        self.proj = nn.Sequential(
            nn.Linear(hidden * 4, TRUNK_OUT),
            nn.LayerNorm(TRUNK_OUT),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, T, n_ind]
        # FIX v3.10: жёсткий clamp + nan_to_num перед LSTM
        x = x.clamp(-8.0, 8.0).nan_to_num(nan=0.0, posinf=8.0, neginf=-8.0)
        _, (h_n, _) = self.bilstm(x)
        # h_n: [num_layers*2, B, hidden]
        h = torch.cat([h_n[-2], h_n[-1],
                       h_n[-4] if h_n.shape[0] >= 4 else h_n[0],
                       h_n[-3] if h_n.shape[0] >= 4 else h_n[1]], dim=-1)
        # FIX v3.10: nan_to_num после LSTM
        h = h.nan_to_num(nan=0.0, posinf=10.0, neginf=-10.0)
        return self.proj(h)   # [B, TRUNK_OUT]


class HourlyEncoder(nn.Module):
    """Кодировщик часовых OHLCV данных.
    Input:  [B, n_days, n_hours, n_feats]  например [B, 5, 11, 9]
    Output: [B, TRUNK_OUT]
    """
    def __init__(self, n_feats=9, n_hours=11, n_days=5, out_dim=TRUNK_OUT):
        super().__init__()
        flat_dim = n_days * n_hours * n_feats
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        x = x.nan_to_num(nan=0.0, posinf=5.0, neginf=-5.0)
        return self.net(x)


# ────────────────────────────────────────────────────────────
# Heads
# ────────────────────────────────────────────────────────────

class CalibratedClsHead(nn.Module):
    """Классификационная голова v3.10 — без Dropout.
    Структура: LayerNorm → SiLU → Linear(→3)
    Ровно 4 параметра: head.0.weight, head.0.bias, head.2.weight, head.2.bias
    """
    def __init__(self, in_dim=TRUNK_OUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),   # [0] weight, bias
            nn.SiLU(),              # [1] no params
            nn.Linear(in_dim, 3),   # [2] weight, bias
        )

    def forward(self, x):
        return self.head(x)


class OHLCHead(nn.Module):
    """Регрессионная голова для ΔOHLC.
    n_out = 4 * future_bars (OHLC × число баров вперёд)
    """
    def __init__(self, in_dim=TRUNK_OUT, n_out=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ────────────────────────────────────────────────────────────
# Full Model
# ────────────────────────────────────────────────────────────

class MultiScaleHybridV3(nn.Module):
    def __init__(self, ctx_dim=0, n_indicator_cols=30,
                 future_bars=5, use_hourly=True):
        super().__init__()
        self.use_hourly = use_hourly
        self.ctx_dim    = ctx_dim

        # Backbones — по одному на каждый масштаб
        self.backbones = nn.ModuleDict(
            {str(W): SingleScaleBackbone() for W in SCALES}
        )

        # BiLSTM для длинного масштаба
        self.bilstm_branch = BiLSTMBranch(n_ind=n_indicator_cols)

        # Числовые признаки коротких масштабов → mean по времени + GRN
        # БЫЛО: GRN(n_ind * W) — huge matrix, overfit
        # СТАЛО: mean([B,W,n]) → [B,n] → GRN(n_ind, TRUNK_OUT)
        self.num_grn = nn.ModuleDict(
            {str(W): GRN(n_indicator_cols, TRUNK_OUT)
             for W in SCALES if W < max(SCALES)}
        )

        # Часовой кодировщик
        if use_hourly:
            self.hourly_enc = HourlyEncoder()

        # Контекстный проектор
        if ctx_dim > 0:
            self.ctx_proj = nn.Sequential(
                nn.Linear(ctx_dim, TRUNK_OUT),
                nn.LayerNorm(TRUNK_OUT),
                nn.GELU(),
            )

        # Кол-во потоков для fusion
        n_streams = len(SCALES)                     # backbones
        n_streams += 1                              # bilstm
        n_streams += len([W for W in SCALES if W < max(SCALES)])  # num_grn
        if use_hourly:  n_streams += 1
        if ctx_dim > 0: n_streams += 1

        self.fusion = nn.Sequential(
            GRN(TRUNK_OUT * n_streams, TRUNK_OUT * 2),
            GRN(TRUNK_OUT * 2, TRUNK_OUT),
        )

        self.cls_head  = CalibratedClsHead(TRUNK_OUT)
        self.ohlc_head = OHLCHead(TRUNK_OUT, n_out=4 * future_bars)

        # Сохраняем backbone как property для pretrain MAE
        self.backbone = self.backbones[str(min(SCALES))]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, imgs, nums, ctx=None, hourly=None):
        feats = []

        # 1. CNN backbones (каждый масштаб)
        for W in SCALES:
            x = imgs[W].float()
            feats.append(self.backbones[str(W)](x))

        # 2. BiLSTM (длинный масштаб, max scale)
        long_W = max(SCALES)
        if nums is not None and long_W in nums:
            x_long = nums[long_W].float()
            # nan_to_num + clamp — в BiLSTMBranch.forward()
            lstm_feat = self.bilstm_branch(x_long)
        else:
            lstm_feat = torch.zeros(imgs[min(SCALES)].shape[0], TRUNK_OUT,
                                    device=imgs[min(SCALES)].device)
        feats.append(lstm_feat)

        # 3. GRN для числовых признаков коротких масштабов
        for W in SCALES:
            if W < long_W and nums is not None and W in nums:
                x_num = nums[W].float().nan_to_num(nan=0.0, posinf=5.0, neginf=-5.0)
                # mean по временному измерению → [B, n_ind]
                x_mean = x_num.mean(dim=1)
                feats.append(self.num_grn[str(W)](x_mean))

        # 4. Часовые данные
        if self.use_hourly and hourly is not None:
            feats.append(self.hourly_enc(hourly.float()))

        # 5. Контекст
        if self.ctx_dim > 0 and ctx is not None:
            feats.append(self.ctx_proj(ctx.float()))

        # Fusion
        cat = torch.cat(feats, dim=-1)
        # FIX v3.10: nan_to_num перед fusion
        cat = cat.nan_to_num(nan=0.0, posinf=10.0, neginf=-10.0)
        cat = F.dropout(cat, p=0.2, training=self.training)  # anti-overfit
        h = self.fusion(cat)

        logits = self.cls_head(h)
        ohlc   = self.ohlc_head(h)
        return logits, ohlc


# ────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────

class AsymmetricFocalLoss(nn.Module):
    """Focal CE — v3.10: использует F.cross_entropy (numerically stable).

    БЫЛО (v3.9): ручной log_softmax + gather + smooth_loss
      → NaN grad когда softmax(wrong_class)≈0, grad=1/softmax→inf

    СТАЛО (v3.10): F.cross_entropy с label_smoothing (fused logsumexp kernel)
      → pt = exp(-ce.detach()) без gradient через focal weight
      → нет NaN/inf в backward
    """
    def __init__(self, weight=None, gamma_per_class=(2.0, 2.0, 2.0),
                 label_smoothing=0.15):
        super().__init__()
        if weight is not None:
            self.register_buffer('cls_weight', weight)
        else:
            self.cls_weight = None
        self.gamma = gamma_per_class
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits.float(), targets,
            weight=self.cls_weight,
            label_smoothing=self.ls,
            reduction='none'
        )  # [B]

        # pt через detach — focal weight не участвует в backward
        pt      = torch.exp(-ce.detach()).clamp(0.0, 1.0 - 1e-6)
        gamma_t = torch.tensor(self.gamma, device=logits.device,
                               dtype=logits.dtype)[targets]
        focal_w = (1.0 - pt) ** gamma_t

        return (focal_w * ce).mean()

    # Для совместимости с trainer (criterion.train/eval)
    def extra_repr(self): return f"gamma={self.gamma}, ls={self.ls}"


class MultiTaskLossV3(nn.Module):
    def __init__(self, cls_weight=None, gamma_per_class=(2.0, 2.0, 2.0),
                 label_smoothing=0.1, huber_delta=0.5,
                 direction_weight=0.3, reg_loss_weight=0.1):
        super().__init__()
        self.focal      = AsymmetricFocalLoss(
            weight=cls_weight,
            gamma_per_class=gamma_per_class,
            label_smoothing=label_smoothing,
        )
        self.huber      = nn.HuberLoss(delta=huber_delta)
        self.dir_w      = direction_weight
        self.reg_loss_weight = reg_loss_weight

    def forward(self, logits, cls_y, ohlc_pred, ohlc_true):
        # Classification
        cls_loss = self.focal(logits, cls_y)

        # Regression (ΔOHLC) — slice target до размера pred (датасет может отдавать 4*future_bars)
        n = ohlc_pred.shape[1]
        reg_loss = self.huber(ohlc_pred, ohlc_true[:, :n])

        # Direction consistency (знак ΔClose)
        if self.dir_w > 0 and ohlc_pred.shape[-1] >= 4:
            dir_loss = F.binary_cross_entropy_with_logits(
                ohlc_pred[:, 3],
                (ohlc_true[:, 3] > 0).float(),
            )
            reg_loss = reg_loss + self.dir_w * dir_loss

        total = cls_loss + self.reg_loss_weight * reg_loss
        return total, cls_loss, reg_loss


# ────────────────────────────────────────────────────────────
# Mixup
# ────────────────────────────────────────────────────────────

def mixup_data(imgs, nums, cls_y, ohlc_y, ctx, alpha=0.2, hourly=None):
    if alpha <= 0:
        return imgs, nums, cls_y, cls_y, ohlc_y, ctx, 1.0, hourly
    lam = np.random.beta(alpha, alpha)
    B   = cls_y.shape[0]
    idx = torch.randperm(B, device=cls_y.device)

    m_imgs = {W: lam * imgs[W] + (1-lam) * imgs[W][idx] for W in imgs}
    m_nums = ({W: lam * nums[W] + (1-lam) * nums[W][idx] for W in nums}
              if nums is not None else None)
    m_ohlc = lam * ohlc_y + (1-lam) * ohlc_y[idx]
    m_ctx  = (lam * ctx + (1-lam) * ctx[idx]) if ctx is not None else None
    m_hourly = (lam * hourly + (1-lam) * hourly[idx]) if hourly is not None else None
    return m_imgs, m_nums, cls_y, cls_y[idx], m_ohlc, m_ctx, lam, m_hourly


# ────────────────────────────────────────────────────────────
# DataLoader helper
# ────────────────────────────────────────────────────────────

def _make_loader_v3(dataset, batch_size, shuffle=False, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=shuffle,
    )


# ────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────

def evaluate_multiscale_v3(model, te_ds, y_test, ctx_dim,
                            use_hourly=True, save_json=None):
    from sklearn.metrics import classification_report
    import json

    device = next(model.parameters()).device
    loader = _make_loader_v3(te_ds, batch_size=256, shuffle=False)
    model.eval()

    all_preds, all_trues = [], []
    all_ohlc_pred, all_ohlc_true = [], []

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
            hourly_data = hourly_opt[0] if hourly_opt else None
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            ht    = (hourly_data.to(device)
                     if (use_hourly and hourly_data is not None) else None)
            nums  = ({W: num_dict[W].to(device) for W in SCALES}
                     if num_dict is not None else None)
            lo, op = model(imgs, nums, ctx_t, hourly=ht)
            all_preds.extend(lo.argmax(1).cpu().numpy())
            all_trues.extend(cls_y.numpy())
            all_ohlc_pred.append(op.cpu().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    preds = np.array(all_preds)
    trues = np.array(all_trues)
    ohlc_p = np.concatenate(all_ohlc_pred, 0)
    ohlc_t = np.concatenate(all_ohlc_true, 0)

    print(classification_report(trues, preds, target_names=['UP','FLAT','DOWN'],
                                 digits=4, zero_division=0))

    mae = np.abs(ohlc_p - ohlc_t).mean(0)
    print(f"\n  OHLC MAE:")
    for i, name in enumerate(['ΔOpen','ΔHigh','ΔLow','ΔClose']):
        print(f"    {name}: {mae[i]:.4f}")

    dir_acc = (np.sign(ohlc_p[:, 3]) == np.sign(ohlc_t[:, 3])).mean()
    print(f"  Direction accuracy: {dir_acc:.4f}")

    if save_json:
        from sklearn.metrics import f1_score
        result = {
            'accuracy': float((preds==trues).mean()),
            'macro_f1': float(f1_score(trues, preds, average='macro', zero_division=0)),
            'dir_acc': float(dir_acc),
            'ohlc_mae': mae.tolist(),
        }
        with open(save_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {save_json}")
    return preds, trues
