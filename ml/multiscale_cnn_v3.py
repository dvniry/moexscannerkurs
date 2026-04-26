# ml/multiscale_cnn_v3.py
"""MultiScale CNN Hybrid v3.19

Изменения v3.19:
- [AUX FIX]  AuxLoss: нормировка vol на VOL_SCALE=0.02 → градиенты сравнимы
- [AUX FIX]  MultiTaskLossV3: direction_weight=0.40 (было 0.80)
             reg_loss_weight=0.20 (было 0.30), aux_loss_weight=0.10 (было 0.05)
- [DIR FIX]  dir_loss: mask.sum() >= 4 для стабильного BCE
- Все улучшения v3.17/v3.18 сохранены
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

try:
    from xlstm import (xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig)
    HAS_XLSTM = True
except (ImportError, OSError, Exception):
    HAS_XLSTM = False

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


class DropPath(nn.Module):
    def __init__(self, drop_rate: float = 0.0):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate <= 0.0:
            return x
        keep  = 1.0 - self.drop_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(
            torch.full(shape, keep, dtype=x.dtype, device=x.device))
        return x * mask / keep


class ResBlock(nn.Module):
    def __init__(self, c, dilation=1, drop_path_rate: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            ConvBnAct(c, c, k=3, dilation=dilation),
            nn.Conv1d(c, c, 1, bias=False),
            nn.BatchNorm1d(c),
        )
        self.act       = nn.GELU()
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        return self.act(x + self.drop_path(self.body(x)))


class SingleScaleBackbone(nn.Module):
    def __init__(self, base_ch=64, in_channels: int = 4):
        super().__init__()
        self.stem   = ConvBnAct(in_channels, base_ch, k=5)
        self.blocks = nn.Sequential(
            ResBlock(base_ch,      dilation=1, drop_path_rate=0.05),
            ConvBnAct(base_ch,     base_ch * 2, k=3, stride=2),
            ResBlock(base_ch * 2,  dilation=2, drop_path_rate=0.10),
            ConvBnAct(base_ch * 2, TRUNK_OUT,   k=3, stride=2),
            ResBlock(TRUNK_OUT,    dilation=1, drop_path_rate=0.15),
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


# ────────────────────────────────────────────────────────────
# SeqStatsPool
# ────────────────────────────────────────────────────────────

class SeqStatsPool(nn.Module):
    """[v3.17] Агрегация [last, mean, std, delta] → GRN → TRUNK_OUT."""
    def __init__(self, n_ind: int, out_dim: int = TRUNK_OUT,
                 dropout: float = 0.2):
        super().__init__()
        self.proj = GRN(4 * n_ind, out_dim, dropout=dropout)

    def forward(self, x):   # [B, T, n_ind]
        x       = x.nan_to_num(nan=0., posinf=5., neginf=-5.)
        x_last  = x[:, -1, :]
        x_mean  = x.mean(dim=1)
        x_std   = x.std(dim=1).clamp(max=10.)
        x_delta = x[:, -1, :] - x[:, 0, :]
        agg     = torch.cat([x_last, x_mean, x_std, x_delta], dim=-1)
        return self.proj(agg)


# ────────────────────────────────────────────────────────────
# WaveletDenoise
# ────────────────────────────────────────────────────────────

class WaveletDenoise(nn.Module):
    def __init__(self, threshold: float = 0.08, levels: int = 1):
        super().__init__()
        self.threshold = threshold
        self.levels    = levels
        sqrt2 = math.sqrt(2)
        lp = torch.tensor([[1 / sqrt2,  1 / sqrt2]]).unsqueeze(0)
        hp = torch.tensor([[1 / sqrt2, -1 / sqrt2]]).unsqueeze(0)
        self.register_buffer('lp', lp)
        self.register_buffer('hp', hp)

    def _dwt(self, x):
        approx = F.conv1d(x, self.lp, stride=2, padding=0)
        detail = F.conv1d(x, self.hp, stride=2, padding=0)
        return approx, detail

    def _idwt(self, approx, detail, out_len):
        lp_t = self.lp.flip(-1)
        hp_t = self.hp.flip(-1)
        rec  = (F.conv_transpose1d(approx, lp_t, stride=2)
                + F.conv_transpose1d(detail, hp_t, stride=2))
        return rec[:, :, :out_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T % 2:
            x = F.pad(x, (0, 0, 0, 1))
        xt      = x.permute(0, 2, 1).reshape(B * C, 1, -1)
        details = []
        cur     = xt
        for _ in range(self.levels):
            cur, det = self._dwt(cur)
            det = torch.sign(det) * F.relu(det.abs() - self.threshold)
            details.append(det)
        rec = cur
        for det in reversed(details):
            rec = self._idwt(rec, det, out_len=rec.shape[-1] * 2)
        rec = rec[:, 0, :T]
        return rec.view(B, C, T).permute(0, 2, 1)


# ────────────────────────────────────────────────────────────
# xLSTMBranch
# ────────────────────────────────────────────────────────────

class xLSTMBranch(nn.Module):
    def __init__(self, n_ind: int = 37, hidden: int = 128,
                 context_length: int = None):
        super().__init__()
        ctx_len        = context_length or max(SCALES)
        self._use_xlstm = HAS_XLSTM

        if HAS_XLSTM:
            self.xlstm = xLSTMBlockStack(xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(),
                context_length=ctx_len,
                num_blocks=2,
                embedding_dim=n_ind,
            ))
            self.proj = nn.Sequential(
                nn.Linear(n_ind, TRUNK_OUT),
                nn.LayerNorm(TRUNK_OUT),
                nn.GELU(),
            )
        else:
            self.bilstm = nn.LSTM(
                input_size=n_ind, hidden_size=hidden, num_layers=2,
                batch_first=True, bidirectional=True, dropout=0.1,
            )
            self.proj = nn.Sequential(
                nn.Linear(hidden * 4, TRUNK_OUT),
                nn.LayerNorm(TRUNK_OUT),
                nn.GELU(),
            )

    def forward(self, x):
        x = x.clamp(-8., 8.).nan_to_num(nan=0., posinf=8., neginf=-8.)
        if self._use_xlstm:
            out = self.xlstm(x)
            h   = out[:, -1]
        else:
            self.bilstm.flatten_parameters()
            _, (h_n, _) = self.bilstm(x)
            h = torch.cat([
                h_n[-2], h_n[-1],
                h_n[-4] if h_n.shape[0] >= 4 else h_n[0],
                h_n[-3] if h_n.shape[0] >= 4 else h_n[1],
            ], dim=-1)
        return self.proj(h.nan_to_num(nan=0., posinf=10., neginf=-10.))


# ────────────────────────────────────────────────────────────
# VSN
# ────────────────────────────────────────────────────────────

class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_streams: int, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.grn_select = GRN(n_streams * d_model, n_streams, dropout=dropout)
        self.grns       = nn.ModuleList(
            [GRN(d_model, d_model, dropout=dropout) for _ in range(n_streams)])
        self.out_grn    = GRN(d_model, d_model, dropout=dropout)

    def forward(self, streams: list) -> torch.Tensor:
        flat    = torch.cat(streams, dim=-1)
        weights = torch.softmax(self.grn_select(flat), dim=-1)
        proc    = torch.stack(
            [g(s) for g, s in zip(self.grns, streams)], dim=1)
        fused   = (proc * weights.unsqueeze(-1)).sum(dim=1)
        return self.out_grn(fused)


# ────────────────────────────────────────────────────────────
# Heads
# ────────────────────────────────────────────────────────────

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


class CalibratedClsHead(nn.Module):
    def __init__(self, in_dim=TRUNK_OUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(in_dim, 3),
        )
        self.log_T = nn.Parameter(torch.zeros(1))   # T=1 изначально

    def forward(self, x):
        T = torch.exp(self.log_T).clamp(0.5, 3.0)
        return self.head(x) / T


class OHLCHeadV2(nn.Module):
    def __init__(self, in_dim=TRUNK_OUT, future_bars=5):
        super().__init__()
        self.future_bars = future_bars
        self.shared      = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.1),
        )
        self.head_oc    = nn.Linear(128, 2 * future_bars)
        self.head_mid   = nn.Linear(128, future_bars)
        self.head_range = nn.Linear(128, future_bars)

    def _init_head(self):
        with torch.no_grad():
            for layer in [self.head_oc, self.head_mid, self.head_range]:
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
            self.head_range.bias.fill_(-2.0)

    def forward(self, x):
        h   = self.shared(x)
        oc  = self.head_oc(h)
        mid = self.head_mid(h)
        rng = F.softplus(self.head_range(h)) + 1e-4
        fb  = self.future_bars
        O   = oc[:, :fb]
        C   = oc[:, fb:]
        H   = mid + rng / 2
        L   = mid - rng / 2
        ohlc = torch.stack([O, H, L, C], dim=-1)
        return ohlc.reshape(x.shape[0], -1)


class OHLCLossV2(nn.Module):
    def __init__(self, weights=(0.5, 1.5, 1.5, 1.0),
                 delta=0.5, constraint_w=0.3):
        super().__init__()
        self.register_buffer('w', torch.tensor(weights, dtype=torch.float32))
        self.delta        = delta
        self.constraint_w = constraint_w

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        B  = pred.shape[0]
        fb = pred.shape[1] // 4
        p  = pred.reshape(B, fb, 4)
        t  = true.reshape(B, fb, 4)

        huber    = F.huber_loss(p, t, reduction='none', delta=self.delta)
        reg_loss = (huber * self.w).mean()

        O, H, L, C = p[:, 0, 0], p[:, 0, 1], p[:, 0, 2], p[:, 0, 3]
        constraint = (F.relu(O - H).mean()
                      + F.relu(C - H).mean()
                      + F.relu(L - O).mean()
                      + F.relu(L - C).mean())
        return reg_loss + self.constraint_w * constraint


class AuxHead(nn.Module):
    def __init__(self, in_dim=TRUNK_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )
        self._init_head()

    def _init_head(self):
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, std=0.01)   # было 0.5
            nn.init.zeros_(self.net[-1].bias)
            nn.init.xavier_uniform_(self.net[0].weight, gain=0.1)  # было 0.5
            nn.init.zeros_(self.net[0].bias)

    def forward(self, x):
        return self.net(x)


class AuxLoss(nn.Module):
    """v3.19.2: возвращаем оригинальную логику + уменьшаем вес aux до 0.01"""
    VOL_SCALE = 0.02

    def forward(self, pred, target):
        pred = pred.clamp(-1.0, 1.0)   # vol и tanh(skew) физически ограничены
        vol_loss = F.mse_loss(pred[:, 0].clamp(0., 0.15),
                              target[:, 0].clamp(0., 0.15))


        # skew: tanh нормировка
        skew_loss = F.mse_loss(
            torch.tanh(pred[:, 1] / 3.),
            torch.tanh(target[:, 1].clamp(-5., 5.) / 3.))

        return vol_loss + skew_loss


class DirectionHead(nn.Module):
    def __init__(self, in_dim=TRUNK_OUT, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _init_head(self):
        with torch.no_grad():
            last = self.net[-1]
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)
            mid = self.net[1]
            nn.init.xavier_normal_(mid.weight, gain=0.3)
            nn.init.zeros_(mid.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ────────────────────────────────────────────────────────────
# Full Model
# ────────────────────────────────────────────────────────────

class MultiScaleHybridV3(nn.Module):
    def __init__(self, ctx_dim=0, n_indicator_cols=37, future_bars=5,
                 use_hourly=True, in_channels=4):
        super().__init__()
        self.use_hourly = use_hourly
        self.ctx_dim    = ctx_dim

        self.backbones = nn.ModuleDict(
            {str(W): SingleScaleBackbone(in_channels=in_channels)
             for W in SCALES})

        self.wavelet    = WaveletDenoise(threshold=0.08, levels=1)
        self.seq_branch = xLSTMBranch(n_ind=n_indicator_cols,
                                       context_length=max(SCALES))

        self.num_stats = nn.ModuleDict(
            {str(W): SeqStatsPool(n_indicator_cols, TRUNK_OUT)
             for W in SCALES if W < max(SCALES)})

        if use_hourly:
            self.hourly_enc = HourlyEncoder()
        if ctx_dim > 0:
            self.ctx_proj = nn.Sequential(
                nn.Linear(ctx_dim, TRUNK_OUT),
                nn.LayerNorm(TRUNK_OUT),
                nn.GELU())

        n_streams  = len(SCALES) + 1   # backbones + seq_branch
        n_streams += len([W for W in SCALES if W < max(SCALES)])  # num_stats
        if use_hourly: n_streams += 1
        if ctx_dim > 0: n_streams += 1

        self.vsn = VariableSelectionNetwork(
            n_streams=n_streams, d_model=TRUNK_OUT, dropout=0.15)

        self.cls_head  = CalibratedClsHead(TRUNK_OUT)
        self.ohlc_head = OHLCHeadV2(TRUNK_OUT, future_bars=future_bars)
        self.aux_head  = AuxHead(TRUNK_OUT)
        self.dir_head  = DirectionHead(TRUNK_OUT)

        # Алиас для совместимости с trainer (backbone_ids)
        self.backbone = self.backbones[str(min(SCALES))]

        self._init_weights()
        self.dir_head._init_head()
        self.ohlc_head._init_head()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
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
            x_long = nums[long_W].float()
            x_long = self.wavelet(x_long)
            feats.append(self.seq_branch(x_long))
        else:
            feats.append(torch.zeros(
                imgs[min(SCALES)].shape[0], TRUNK_OUT,
                device=imgs[min(SCALES)].device))

        for W in SCALES:
            if W < long_W and nums is not None and W in nums:
                feats.append(self.num_stats[str(W)](nums[W].float()))

        if self.use_hourly and hourly is not None:
            feats.append(self.hourly_enc(hourly.float()))
        if self.ctx_dim > 0 and ctx is not None:
            feats.append(self.ctx_proj(ctx.float()))

        feats = [f.nan_to_num(nan=0., posinf=10., neginf=-10.) for f in feats]
        h     = self.vsn(feats)

        logits    = self.cls_head(h)
        ohlc      = self.ohlc_head(h)
        aux       = self.aux_head(h)
        dir_logit = self.dir_head(h)
        return logits, ohlc, aux, dir_logit


# ────────────────────────────────────────────────────────────
# Losses
# ────────────────────────────────────────────────────────────

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma_per_class=(2.0, 1.0, 2.0),
                 label_smoothing=0.05):
        super().__init__()
        if weight is not None:
            self.register_buffer('cls_weight', weight)
        else:
            self.cls_weight = None
        self._max_gamma = list(gamma_per_class)
        self.gamma      = [0.0, 0.0, 0.0]
        self.ls         = label_smoothing

    def set_gamma(self, epoch: int, warmup_epochs: int = 10):
        t          = min(1.0, epoch / max(warmup_epochs, 1))
        self.gamma = [g * t for g in self._max_gamma]

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits.float(), targets,
            weight=self.cls_weight,
            label_smoothing=self.ls,
            reduction='none')
        pt      = torch.exp(-ce.detach()).clamp(0., 1. - 1e-6)
        gamma_t = torch.tensor(
            self.gamma, device=logits.device,
            dtype=logits.dtype)[targets]
        return ((1. - pt) ** gamma_t * ce).mean()


class PinballLoss(nn.Module):
    """Оставлен для совместимости."""
    QUANTILES = [0.50, 0.90, 0.10, 0.50]

    def __init__(self, future_bars=5):
        super().__init__()
        qs = self.QUANTILES * future_bars
        self.register_buffer('q', torch.tensor(qs, dtype=torch.float32))

    def forward(self, pred, target):
        n   = min(pred.shape[1], target.shape[1])
        err = target[:, :n].float() - pred[:, :n].float()
        q   = self.q[:n]
        return torch.where(err >= 0, q * err, (q - 1.) * err).mean()


class MultiTaskLossV3(nn.Module):
    """v3.19: исправленные веса, aux_loss передаётся из trainer."""
    def __init__(self, cls_weight=None,
                 gamma_per_class=(2.0, 1.0, 2.0),
                 label_smoothing=0.05,
                 future_bars=1,
                 huber_delta=0.5,
                 direction_weight=0.40,    # v3.19: было 0.80
                 reg_loss_weight=0.20,     # v3.19: было 0.30
                 aux_loss_weight=0.10,     # v3.19: было 0.05
                 dir_mask_threshold=1e-4):
        super().__init__()
        self.focal = AsymmetricFocalLoss(
            weight=cls_weight,
            gamma_per_class=gamma_per_class,
            label_smoothing=label_smoothing)
        self.ohlc_loss       = OHLCLossV2(delta=huber_delta, constraint_w=0.3)
        self.aux_fn          = AuxLoss()
        self.dir_w           = direction_weight
        self.reg_loss_weight = reg_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.dir_mask_threshold = dir_mask_threshold

    def forward(self, logits, cls_y, ohlc_pred, ohlc_true,
                dir_logit=None, aux_pred=None, aux_true=None):

        cls_loss = self.focal(logits, cls_y)

        n        = min(ohlc_pred.shape[1], ohlc_true.shape[1])
        reg_loss = self.ohlc_loss(ohlc_pred, ohlc_true[:, :n])

        # Direction loss — только UP/DOWN (FLAT игнорируем)
        dir_loss = torch.tensor(0., device=logits.device)
        if self.dir_w > 0 and dir_logit is not None:
            mask = cls_y != 1
            if mask.sum() >= 4:   # минимум 4 сэмпла для стабильного BCE
                dir_target = (cls_y[mask] == 0).float()
                dir_loss   = F.binary_cross_entropy_with_logits(
                    dir_logit[mask], dir_target)

        # Aux loss — vol + skew (теперь реально передаётся из trainer)
        a_loss = torch.tensor(0., device=logits.device)
        if (aux_pred is not None
                and aux_true is not None
                and aux_pred.shape == aux_true.shape
                and aux_pred.shape[-1] >= 2):
            a_loss = self.aux_fn(aux_pred.float(), aux_true.float())

        total = (cls_loss
                 + self.reg_loss_weight * reg_loss
                 + self.dir_w           * dir_loss
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
    m_ohlc   = lam * ohlc_y + (1 - lam) * ohlc_y[idx]
    m_ctx    = (lam * ctx    + (1 - lam) * ctx[idx])    if ctx    is not None else None
    m_hourly = (lam * hourly + (1 - lam) * hourly[idx]) if hourly is not None else None
    m_aux    = (lam * aux_y  + (1 - lam) * aux_y[idx])  if aux_y  is not None else None
    return m_imgs, m_nums, cls_y, cls_y[idx], m_ohlc, m_ctx, lam, m_hourly, m_aux


# ────────────────────────────────────────────────────────────
# DataLoader
# ────────────────────────────────────────────────────────────

def _make_loader_v3(dataset, batch_size, shuffle=False,
                    num_workers=0, sampler=None):
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
        drop_last=(shuffle or sampler is not None),
    )


# ────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────

def evaluate_multiscale_v3(model, te_ds, y_test, ctx_dim,
                            use_hourly=True, save_json=None):
    from sklearn.metrics import classification_report, f1_score
    import json

    device = next(model.parameters()).device
    loader = _make_loader_v3(te_ds, batch_size=256, shuffle=False, num_workers=2)
    model.eval()

    all_preds = []; all_trues = []
    all_ohlc_pred = []; all_ohlc_true = []
    all_dir_prob  = []

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, hourly_data, *_ = batch
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            ht    = (hourly_data.to(device)
                     if use_hourly and hourly_data is not None else None)
            nums  = ({W: num_dict[W].to(device) for W in SCALES}
                     if num_dict is not None else None)
            out = model(imgs, nums, ctx_t, hourly=ht)
            if len(out) == 4:
                lo, op, _, dir_l = out
                all_dir_prob.append(torch.sigmoid(dir_l).cpu().numpy())
            else:
                lo, op, _ = out
                all_dir_prob.append(
                    np.full(lo.shape[0], 0.5, dtype=np.float32))
            all_preds.extend(lo.argmax(1).cpu().numpy())
            all_trues.extend(cls_y.numpy())
            all_ohlc_pred.append(op.cpu().float().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    preds  = np.array(all_preds)
    trues  = np.array(all_trues)
    ohlc_p = np.concatenate(all_ohlc_pred, 0)
    ohlc_t = np.concatenate(all_ohlc_true, 0)
    dir_p  = np.concatenate(all_dir_prob,  0)

    print(classification_report(trues, preds,
                                 target_names=['UP', 'FLAT', 'DOWN'],
                                 digits=4, zero_division=0))

    n_bars = ohlc_p.shape[1] // 4
    print(f'\n  OHLC MAE ({n_bars} bars):')
    print(f'  {"Bar":>4}  {"ΔOpen":>8}  {"ΔHigh":>8}  '
          f'{"ΔLow":>8}  {"ΔClose":>8}')
    for bar in range(n_bars):
        s     = bar * 4
        p_bar = ohlc_p[:, s:s + 4]
        t_bar = ohlc_t[:, s:s + 4]
        if p_bar.shape[1] < 4: break
        mae_b = np.abs(p_bar - t_bar).mean(0)
        print(f'  {bar+1:>4}  {mae_b[0]:>8.4f}  {mae_b[1]:>8.4f}  '
              f'{mae_b[2]:>8.4f}  {mae_b[3]:>8.4f}')

    mask_ud = trues != 1
    dir_cov = float(mask_ud.mean())
    dir_acc = 0.5
    if mask_ud.any():
        dir_target = (trues[mask_ud] == 0).astype(int)
        dir_pred   = (dir_p[mask_ud] > 0.5).astype(int)
        dir_acc    = float((dir_pred == dir_target).mean())
    print(f'\n  dir_acc (dir_head): {dir_acc:.4f}')
    print(f'  Coverage (non-HOLD): {dir_cov:.2%}')
    print(f'  Baseline (always BUY): {(trues[mask_ud]==0).mean():.4f}')

    if ohlc_p.shape[1] >= 4:
        dir_acc_reg = float(
            (np.sign(ohlc_p[:, 3]) == np.sign(ohlc_t[:, 3])).mean())
        print(f'  dir_acc (regression): {dir_acc_reg:.4f}')
        O_p, H_p, L_p, C_p = (ohlc_p[:, 0], ohlc_p[:, 1],
                                ohlc_p[:, 2], ohlc_p[:, 3])
        h_viol = (H_p < np.maximum(O_p, C_p)).mean()
        l_viol = (L_p > np.minimum(O_p, C_p)).mean()
        print(f'  H viol: {h_viol:.2%}  L viol: {l_viol:.2%}')

    if save_json:
        result = {
            'accuracy':     float((preds == trues).mean()),
            'macro_f1':     float(f1_score(trues, preds,
                                            average='macro', zero_division=0)),
            'dir_acc_head': dir_acc,
            'dir_coverage': dir_cov,
            'ohlc_mae':     np.abs(ohlc_p[:, :4] - ohlc_t[:, :4]).mean(0).tolist()
                            if ohlc_p.shape[1] >= 4 else [],
        }
        import json
        with open(save_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  Saved → {save_json}')

    return preds, trues