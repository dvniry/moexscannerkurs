# ml/multiscale_cnn_v3.py
"""MultiScale CNN Hybrid v3.13

Изменения v3.13:
- [2.3] PinballLoss: ΔHigh→q90, ΔLow→q10, ΔOpen→q50, ΔClose→q50
  Заменяет HuberLoss для OHLC — явный учёт асимметрии High/Low
- [2.4] AuxHead: предсказание realized_vol + skew будущих 5 баров
  aux_loss = 0.05 * (mse_vol + mse_skew), отдельная голова от TRUNK_OUT
- n_indicator_cols по умолчанию = 37 (было 30)
- StreamFusion с MHA (из v3.11) — сохранён
- Все NaN-guards из v3.10 — сохранены
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

TRUNK_OUT = 128


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
    def __init__(self, base_ch=64):
        super().__init__()
        self.stem = ConvBnAct(3, base_ch, k=5)
        self.blocks = nn.Sequential(
            ResBlock(base_ch, dilation=1),
            ConvBnAct(base_ch, base_ch * 2, k=3, stride=2),
            ResBlock(base_ch * 2, dilation=2),
            ConvBnAct(base_ch * 2, TRUNK_OUT, k=3, stride=2),
            ResBlock(TRUNK_OUT, dilation=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pool(self.blocks(self.stem(x))).squeeze(-1)


class GRN(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.3):
        super().__init__()
        self.fc1  = nn.Linear(d_in, d_out)
        self.fc2  = nn.Linear(d_out, d_out)
        self.gate = nn.Linear(d_in, d_out)
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.ln   = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        h = self.drop(self.fc2(F.gelu(self.fc1(x))))
        return self.ln(self.proj(x) + g * h)


class BiLSTMBranch(nn.Module):
    def __init__(self, n_ind=37, hidden=128, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=n_ind, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 4, TRUNK_OUT),
            nn.LayerNorm(TRUNK_OUT),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.clamp(-8., 8.).nan_to_num(nan=0., posinf=8., neginf=-8.)
        _, (h_n, _) = self.bilstm(x)
        h = torch.cat([h_n[-2], h_n[-1],
                       h_n[-4] if h_n.shape[0] >= 4 else h_n[0],
                       h_n[-3] if h_n.shape[0] >= 4 else h_n[1]], dim=-1)
        return self.proj(h.nan_to_num(nan=0., posinf=10., neginf=-10.))


class HourlyEncoder(nn.Module):
    def __init__(self, n_feats=9, n_hours=11, n_days=5, out_dim=TRUNK_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(n_days * n_hours * n_feats, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x.nan_to_num(nan=0., posinf=5., neginf=-5.))


class StreamFusion(nn.Module):
    """MHA fusion: [B, n_streams, TRUNK_OUT] → [B, TRUNK_OUT]."""
    def __init__(self, d_model=TRUNK_OUT, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.grn  = GRN(d_model, d_model, dropout=dropout)

    def forward(self, x):
        # x: [B, n_streams, d_model]
        out, _ = self.attn(x, x, x)
        out = self.norm(x + out)          # residual
        return self.grn(out.mean(dim=1))  # [B, d_model]


# ────────────────────────────────────────────────────────────
# Heads
# ────────────────────────────────────────────────────────────

class CalibratedClsHead(nn.Module):
    def __init__(self, in_dim=TRUNK_OUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, 3),
        )

    def forward(self, x): return self.head(x)


class OHLCHead(nn.Module):
    """Предсказывает 4*future_bars квантилей.
    Порядок: [open_q50, high_q90, low_q10, close_q50] × future_bars
    """
    def __init__(self, in_dim=TRUNK_OUT, n_out=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_out),
        )

    def forward(self, x): return self.net(x)


class AuxHead(nn.Module):
    """[2.4] Предсказывает aux_y = [realized_vol*100, skew].
    Отдельная голова — не мешает основной регрессии.
    """
    def __init__(self, in_dim=TRUNK_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),   # [vol_hat, skew_hat]
        )

    def forward(self, x): return self.net(x)


# ────────────────────────────────────────────────────────────
# Full Model
# ────────────────────────────────────────────────────────────

class MultiScaleHybridV3(nn.Module):
    def __init__(self, ctx_dim=0, n_indicator_cols=37,
                 future_bars=5, use_hourly=True):
        super().__init__()
        self.use_hourly = use_hourly
        self.ctx_dim    = ctx_dim

        self.backbones = nn.ModuleDict(
            {str(W): SingleScaleBackbone() for W in SCALES})

        self.bilstm_branch = BiLSTMBranch(n_ind=n_indicator_cols)

        self.num_grn = nn.ModuleDict(
            {str(W): GRN(n_indicator_cols, TRUNK_OUT)
             for W in SCALES if W < max(SCALES)})

        if use_hourly:
            self.hourly_enc = HourlyEncoder()

        if ctx_dim > 0:
            self.ctx_proj = nn.Sequential(
                nn.Linear(ctx_dim, TRUNK_OUT),
                nn.LayerNorm(TRUNK_OUT),
                nn.GELU(),
            )

        n_streams  = len(SCALES)                              # backbones
        n_streams += 1                                        # bilstm
        n_streams += len([W for W in SCALES if W < max(SCALES)])  # num_grn
        if use_hourly:  n_streams += 1
        if ctx_dim > 0: n_streams += 1

        self.stream_proj  = nn.Linear(TRUNK_OUT, TRUNK_OUT)  # каждый поток выравниваем
        self.stream_fusion = StreamFusion(d_model=TRUNK_OUT, n_heads=4, dropout=0.1)

        self.cls_head  = CalibratedClsHead(TRUNK_OUT)
        self.ohlc_head = OHLCHead(TRUNK_OUT, n_out=4 * future_bars)
        self.aux_head  = AuxHead(TRUNK_OUT)   # [2.4]

        self.backbone  = self.backbones[str(min(SCALES))]  # для pretrain MAE
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

        for W in SCALES:
            feats.append(self.backbones[str(W)](imgs[W].float()))

        long_W = max(SCALES)
        if nums is not None and long_W in nums:
            feats.append(self.bilstm_branch(nums[long_W].float()))
        else:
            feats.append(torch.zeros(
                imgs[min(SCALES)].shape[0], TRUNK_OUT,
                device=imgs[min(SCALES)].device))

        for W in SCALES:
            if W < long_W and nums is not None and W in nums:
                x_mean = nums[W].float().nan_to_num(
                    nan=0., posinf=5., neginf=-5.).mean(dim=1)
                feats.append(self.num_grn[str(W)](x_mean))

        if self.use_hourly and hourly is not None:
            feats.append(self.hourly_enc(hourly.float()))

        if self.ctx_dim > 0 and ctx is not None:
            feats.append(self.ctx_proj(ctx.float()))

        # Stack → [B, n_streams, TRUNK_OUT] → MHA fusion
        stacked = torch.stack(feats, dim=1)   # [B, n_streams, TRUNK_OUT]
        stacked = stacked.nan_to_num(nan=0., posinf=10., neginf=-10.)
        stacked = F.dropout(stacked, p=0.1, training=self.training)
        h = self.stream_fusion(stacked)        # [B, TRUNK_OUT]

        logits = self.cls_head(h)
        ohlc   = self.ohlc_head(h)
        aux    = self.aux_head(h)              # [B, 2]
        return logits, ohlc, aux


# ────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma_per_class=(1.5, 3.5, 1.5),
                 label_smoothing=0.15):
        super().__init__()
        if weight is not None:
            self.register_buffer('cls_weight', weight)
        else:
            self.cls_weight = None
        self.gamma = gamma_per_class
        self.ls    = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits.float(), targets,
            weight=self.cls_weight,
            label_smoothing=self.ls,
            reduction='none')
        pt      = torch.exp(-ce.detach()).clamp(0., 1. - 1e-6)
        gamma_t = torch.tensor(self.gamma, device=logits.device,
                               dtype=logits.dtype)[targets]
        return ((1. - pt) ** gamma_t * ce).mean()


class PinballLoss(nn.Module):
    """[2.3] Квантильный loss для OHLC.
    Компоненты: [open_q50, high_q90, low_q10, close_q50] × future_bars
    q50 = MAE, q90/q10 = асимметричный штраф.
    """
    QUANTILES = [0.50, 0.90, 0.10, 0.50]  # open, high, low, close

    def __init__(self, future_bars=5):
        super().__init__()
        self.future_bars = future_bars
        # Расширяем квантили на future_bars
        qs = self.QUANTILES * future_bars          # [q0,q1,q2,q3, q0,q1,q2,q3, ...]
        self.register_buffer('q', torch.tensor(qs, dtype=torch.float32))

    def forward(self, pred, target):
        # pred, target: [B, 4*future_bars]
        n = min(pred.shape[1], target.shape[1])
        p = pred[:, :n].float()
        t = target[:, :n].float()
        q = self.q[:n]
        err = t - p
        loss = torch.where(err >= 0, q * err, (q - 1.) * err)
        return loss.mean()


class AuxLoss(nn.Module):
    """[2.4] MSE для vol и tanh-MSE для skew."""
    def forward(self, pred, target):
        # pred, target: [B, 2]
        vol_loss  = F.mse_loss(pred[:, 0], target[:, 0])
        skew_loss = F.mse_loss(torch.tanh(pred[:, 1] / 3.),
                               torch.tanh(target[:, 1] / 3.))
        return vol_loss + skew_loss


class MultiTaskLossV3(nn.Module):
    def __init__(self, cls_weight=None,
                gamma_per_class=(1.5, 3.5, 1.5),
                label_smoothing=0.05,   # ← синхронизировать
                future_bars=5,
                direction_weight=0.40,  # ← синхронизировать
                reg_loss_weight=0.30,
                aux_loss_weight=0.05):
        super().__init__()
        self.focal    = AsymmetricFocalLoss(
            weight=cls_weight,
            gamma_per_class=gamma_per_class,
            label_smoothing=label_smoothing)
        self.pinball  = PinballLoss(future_bars=future_bars)
        self.aux_loss = AuxLoss()
        self.dir_w    = direction_weight
        self.reg_loss_weight = reg_loss_weight
        self.aux_loss_weight = aux_loss_weight

    def forward(self, logits, cls_y, ohlc_pred, ohlc_true, aux_pred=None, aux_true=None):
        cls_loss  = self.focal(logits, cls_y)

        n         = min(ohlc_pred.shape[1], ohlc_true.shape[1])
        reg_loss  = self.pinball(ohlc_pred, ohlc_true[:, :n])

        # Direction consistency: close bar0 = ohlc[:, 3]
        if self.dir_w > 0 and ohlc_pred.shape[-1] >= 4:
            dir_loss = F.binary_cross_entropy_with_logits(
                ohlc_pred[:, 3],
                (ohlc_true[:, 3] > 0).float())
            reg_loss = reg_loss + self.dir_w * dir_loss

        # [2.4] Aux loss
        if aux_pred is not None and aux_true is not None:
            a_loss = self.aux_loss(aux_pred.float(), aux_true.float())
        else:
            a_loss = torch.tensor(0., device=logits.device)

        total = (cls_loss
                 + self.reg_loss_weight * reg_loss
                 + self.aux_loss_weight * a_loss)
        return total, cls_loss, reg_loss, a_loss


# ────────────────────────────────────────────────────────────
# Mixup
# ────────────────────────────────────────────────────────────

def mixup_data(imgs, nums, cls_y, ohlc_y, ctx, alpha=0.2,
               hourly=None, aux_y=None):
    if alpha <= 0:
        return imgs, nums, cls_y, cls_y, ohlc_y, ctx, 1.0, hourly, aux_y
    lam = np.random.beta(alpha, alpha)
    B   = cls_y.shape[0]
    idx = torch.randperm(B, device=cls_y.device)
    m_imgs   = {W: lam * imgs[W] + (1 - lam) * imgs[W][idx] for W in imgs}
    m_nums   = ({W: lam * nums[W] + (1 - lam) * nums[W][idx] for W in nums}
                if nums is not None else None)
    m_ohlc   = lam * ohlc_y   + (1 - lam) * ohlc_y[idx]
    m_ctx    = (lam * ctx     + (1 - lam) * ctx[idx])    if ctx    is not None else None
    m_hourly = (lam * hourly  + (1 - lam) * hourly[idx]) if hourly is not None else None
    m_aux    = (lam * aux_y   + (1 - lam) * aux_y[idx])  if aux_y  is not None else None
    return m_imgs, m_nums, cls_y, cls_y[idx], m_ohlc, m_ctx, lam, m_hourly, m_aux


# ────────────────────────────────────────────────────────────
# DataLoader
# ────────────────────────────────────────────────────────────

def _make_loader_v3(dataset, batch_size, shuffle=False, num_workers=4, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=shuffle or sampler is not None,
    )


# ────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────

def evaluate_multiscale_v3(model, te_ds, y_test, ctx_dim,
                            use_hourly=True, save_json=None):
    from sklearn.metrics import classification_report
    import json

    device = next(model.parameters()).device
    loader = _make_loader_v3(te_ds, batch_size=256, shuffle=False, num_workers=0) 
    model.eval()

    all_preds, all_trues = [], []
    all_ohlc_pred, all_ohlc_true = [], []

    with torch.no_grad():
        for batch in loader:
            # v3.3: 7-кортеж (imgs, nums, cls_y, ohlc_y, ctx, hourly, aux_y)
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, hourly_data, *_ = batch
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            ht    = (hourly_data.to(device)
                     if use_hourly and hourly_data is not None else None)
            nums  = ({W: num_dict[W].to(device) for W in SCALES}
                     if num_dict is not None else None)
            lo, op, _ = model(imgs, nums, ctx_t, hourly=ht)
            all_preds.extend(lo.argmax(1).cpu().numpy())
            all_trues.extend(cls_y.numpy())
            all_ohlc_pred.append(op.cpu().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    preds  = np.array(all_preds)
    trues  = np.array(all_trues)
    ohlc_p = np.concatenate(all_ohlc_pred, 0)
    ohlc_t = np.concatenate(all_ohlc_true, 0)

    print(classification_report(trues, preds,
                                 target_names=['UP', 'FLAT', 'DOWN'],
                                 digits=4, zero_division=0))

    n_cols = min(ohlc_p.shape[1], 4)
    mae    = np.abs(ohlc_p[:, :n_cols] - ohlc_t[:, :n_cols]).mean(0)
    names  = ['ΔOpen', 'ΔHigh', 'ΔLow', 'ΔClose']
    print("\n  OHLC MAE:")
    for i in range(n_cols):
        print(f"    {names[i]}: {mae[i]:.4f}")

    if ohlc_p.shape[1] >= 4:
        dir_acc = (np.sign(ohlc_p[:, 3]) == np.sign(ohlc_t[:, 3])).mean()
        print(f"  Direction accuracy: {dir_acc:.4f}")

    if save_json:
        from sklearn.metrics import f1_score
        result = {
            'accuracy':  float((preds == trues).mean()),
            'macro_f1':  float(f1_score(trues, preds, average='macro', zero_division=0)),
            'dir_acc':   float(dir_acc) if ohlc_p.shape[1] >= 4 else 0.,
            'ohlc_mae':  mae.tolist(),
        }
        with open(save_json, 'w') as f:
            import json; json.dump(result, f, indent=2)
        print(f"  Saved → {save_json}")
    return preds, trues