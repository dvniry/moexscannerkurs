# ml/multiscale_cnn_v3.py  v3.21 (Sprint 1.5: Intraday Feedback Loop)
# forward() + intraday_feats/intraday_mask; DayHourCrossAttention; extremes_head
# backward-compat: intraday_feats=None -> works as v3.20

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

from ml.hourly_encoder import N_HOURS_PER_DAY, N_HOURLY_CHANNELS, N_INTRADAY_DAYS
try:
    from ml.hourly_feedback import HourlyFeedbackEncoder, DayHourCrossAttention
    HAS_INTRADAY_FEEDBACK = True
except ImportError:
    HAS_INTRADAY_FEEDBACK = False
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
    def __init__(self, n_feats=N_HOURLY_CHANNELS, n_hours=N_HOURS_PER_DAY, n_days=N_INTRADAY_DAYS, out_dim=TRUNK_OUT):
        super().__init__()
        self.n_days = n_days
        self.n_hours = n_hours
        self.n_feats = n_feats

        self.slot_net = nn.Sequential(
            nn.Linear(n_feats, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.slot_head = nn.Linear(16, 1)

        self.global_net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(n_days * n_hours * n_feats, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
            x = x.nan_to_num(nan=0., posinf=5., neginf=-5.).float()

            if x.ndim != 4:
                raise ValueError(f"HourlyEncoder: expected 4D input, got {tuple(x.shape)}")

            # Приводим к канонической форме [B, D, C, H] для global_net
            if x.shape[-2] == self.n_feats and x.shape[-1] == self.n_hours:
                x_c_h = x                      # [B, D, C, H]
            elif x.shape[-2] == self.n_hours and x.shape[-1] == self.n_feats:
                x_c_h = x.permute(0, 1, 3, 2).contiguous()   # [B, D, C, H]
            else:
                raise ValueError(
                    f"HourlyEncoder: bad shape {tuple(x.shape)}, "
                    f"expected [..., {self.n_feats}, {self.n_hours}] or "
                    f"[..., {self.n_hours}, {self.n_feats}]"
                )

            g = self.global_net(x_c_h)

            # Для slot_net нужна форма [B, D, H, C]
            slots = x_c_h.permute(0, 1, 3, 2).contiguous()
            sh = self.slot_net(slots)
            intraday = self.slot_head(sh).squeeze(-1)

            return g, intraday


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


class EconomicHeads(nn.Module):
    """Sprint 2: cost-aware головы для DecisionLayer.

    Выходы (dict):
      mfe_mae    [B, 4]: [mfe_long, mae_long, mfe_short, mae_short] >= 0 (Softplus β=1)
      fill_logit [B, 2]: [fill_long, fill_short] — logits, sigmoid в loss
      edge_pred  [B, 2]: [net_edge_long, net_edge_short] — scalar regression

    Размерности голов малые (32-64) — основная работа делается в trunk (TRUNK_OUT=128).

    Инициализация: финальные слои с gain=0.01 → старт почти от 0 (Softplus(0)≈0.69 но
    через линейную проекцию 64→4 с очень малыми весами: выход ≈ 0). Промежуточные слои
    стандартный gain=0.5. Это критично: с gain=0.1 + clamp активации застревали
    на границах clamp и градиенты обнулялись.
    """
    def __init__(self, in_dim: int = TRUNK_OUT, dropout: float = 0.2):
        super().__init__()
        self.mfe_head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 4),
        )
        self.fill_head = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(),
            nn.Linear(32, 2),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self._init_heads()

    def _init_heads(self):
        # B-16 fix: edge_head получает gain=0.1 (вместо 0.01 у mfe/fill),
        # чтобы реально отходить от bias. Прежний gain=0.01 + bias=0.002 +
        # penalty(cost - edge) совместно "приклеивали" edge_pred к ~0.002.
        for module in (self.mfe_head, self.fill_head):
            linears = [m for m in module if isinstance(m, nn.Linear)]
            nn.init.xavier_uniform_(linears[-1].weight, gain=0.01)
            nn.init.zeros_(linears[-1].bias)
            nn.init.xavier_uniform_(linears[0].weight, gain=0.5)
            nn.init.zeros_(linears[0].bias)

        edge_linears = [m for m in self.edge_head if isinstance(m, nn.Linear)]
        # B-16: финальный edge с gain=0.1 — диапазон выходов ~0.05 при std входов ~1
        nn.init.xavier_uniform_(edge_linears[-1].weight, gain=0.1)
        nn.init.zeros_(edge_linears[-1].bias)
        nn.init.xavier_uniform_(edge_linears[0].weight, gain=0.5)
        nn.init.zeros_(edge_linears[0].bias)

        # mfe_head: разные биасы для mfe (cols 0,2) и mae (cols 1,3).
        # mfe_long/short → bias=-3.5: softplus≈0.030 (ожидаемый upside)
        # mae_long/short → bias=-4.5: softplus≈0.011 (ожидаемый drawdown)
        # rr_init = 0.030 / 0.011 ≈ 2.7 > min_rr=0.8 с первого батча,
        # поэтому модель сразу генерирует BUY и SELL, а не только HOLD.
        mfe_last = [m for m in self.mfe_head if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            mfe_last.bias[0] = -3.5   # mfe_long
            mfe_last.bias[1] = -4.5   # mae_long
            mfe_last.bias[2] = -3.5   # mfe_short
            mfe_last.bias[3] = -4.5   # mae_short
        # B-16: edge_bias=0 (было 0.002). Penalty + target пусть сами решают,
        # куда тянуть edge. Прежний bias=0.002 закреплял edge_pred к cost
        # и не давал градиенту работать.
        nn.init.zeros_(edge_linears[-1].bias)

    def forward(self, x: torch.Tensor) -> dict:
        # Softplus β=1 (стандарт): без мёртвой зоны при отрицательных входах
        mfe_raw = self.mfe_head(x)
        mfe = F.softplus(mfe_raw, beta=1.0).clamp(0., 0.30)
        # fill — clamp для численной стабильности sigmoid
        fill = self.fill_head(x).clamp(-15., 15.)
        # edge — широкий safety-clamp: target в диапазоне [-0.05, +0.05],
        # широкий [-0.50, 0.50] не давит на градиент
        edge = self.edge_head(x).clamp(-0.50, 0.50)
        return {"mfe_mae": mfe, "fill_logit": fill, "edge_pred": edge}


class EconomicLoss(nn.Module):
    """Sprint 2: cost-aware loss для EconomicHeads.

    Компоненты:
      l_mfe     — Huber по MFE/MAE long+short, beta=0.01 (масштаб ~1% движения)
      l_fill    — BCE-with-logits по [fill_long, fill_short]
      l_edge    — Huber по net_edge long+short, beta=0.001 (масштаб ~0.1%)
      l_penalty — F.relu(cost - edge_long).mean() + F.relu(cost - edge_short).mean()
                  Штрафует модель за предсказание edge меньше cost: учит осторожности.

    econ_target колонки (см. ECON_COL_NAMES в labels_ohlc):
      0:future_ret  1:mfe_long   2:mae_long   3:mfe_short  4:mae_short
      5:rr_long     6:rr_short   7:fill_long  8:fill_short
      9:net_edge_long  10:net_edge_short
    """
    def __init__(self, cost_roundtrip: float = 0.002,
                 w_mfe: float = 0.5, w_fill: float = 0.3,
                 w_edge: float = 1.0, w_penalty: float = 0.05,
                 edge_beta: float = 0.005):
        # B-16 fix: w_edge 0.5 → 1.0 (×2) — edge регрессия слабый сигнал, нужен больший вес.
        # edge_beta 0.001 → 0.005 (×5) — smooth_l1 c beta=0.001 был слишком плоский
        # в районе target≈0.001, gradient к edge_head был мал.
        # w_penalty=0.05 — оставлен как мягкая регуляризация осторожности.
        super().__init__()
        self.cost_roundtrip = float(cost_roundtrip)
        self.w_mfe = w_mfe
        self.w_fill = w_fill
        self.w_edge = w_edge
        self.w_penalty = w_penalty
        self.edge_beta = edge_beta

    def forward(self, econ_pred: dict, econ_target: torch.Tensor):
        # Таргеты
        mfe_l_t = econ_target[:, 1]
        mae_l_t = econ_target[:, 2]
        mfe_s_t = econ_target[:, 3]
        mae_s_t = econ_target[:, 4]
        fill_l_t = econ_target[:, 7]
        fill_s_t = econ_target[:, 8]
        edge_l_t = econ_target[:, 9]
        edge_s_t = econ_target[:, 10]

        mfe_mae_target = torch.stack([mfe_l_t, mae_l_t, mfe_s_t, mae_s_t], dim=1)
        fill_target    = torch.stack([fill_l_t, fill_s_t], dim=1)
        edge_target    = torch.stack([edge_l_t, edge_s_t], dim=1)

        l_mfe  = F.smooth_l1_loss(econ_pred["mfe_mae"],   mfe_mae_target, beta=0.01)
        l_fill = F.binary_cross_entropy_with_logits(econ_pred["fill_logit"], fill_target)
        l_edge = F.smooth_l1_loss(econ_pred["edge_pred"], edge_target,    beta=self.edge_beta)

        edge_l = econ_pred["edge_pred"][:, 0]
        edge_s = econ_pred["edge_pred"][:, 1]
        # Penalty ограничен сверху одним cost: при edge << 0 не убегает в +∞
        # и не пушит edge_pred в clamp границы.
        l_penalty = (F.relu(self.cost_roundtrip - edge_l).clamp_max(self.cost_roundtrip).mean()
                     + F.relu(self.cost_roundtrip - edge_s).clamp_max(self.cost_roundtrip).mean())

        total = (self.w_mfe     * l_mfe
                 + self.w_fill  * l_fill
                 + self.w_edge  * l_edge
                 + self.w_penalty * l_penalty)

        return total, {
            "mfe":     l_mfe.item(),
            "fill":    l_fill.item(),
            "edge":    l_edge.item(),
            "penalty": l_penalty.item(),
        }


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
# Sprint 8.2: high_first order head
# ────────────────────────────────────────────────────────────

class HighLowOrderHead(nn.Module):
    """P(high reached before low intraday): binary BCE head."""
    def __init__(self, in_dim=TRUNK_OUT, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B] logit


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

        # Sprint 1.5: intraday feedback components
        if HAS_INTRADAY_FEEDBACK:
            self.intraday_enc   = HourlyFeedbackEncoder(out_dim=TRUNK_OUT)
            self.day_hour_attn  = DayHourCrossAttention(d_model=TRUNK_OUT)
        self.extremes_head = nn.Linear(TRUNK_OUT, 2)   # pred [dHigh, dLow]
        self.high_low_head = HighLowOrderHead(TRUNK_OUT)  # Sprint 8.2: P(high before low)
        # Sprint 11.1: quantile head для full OHLC (3 q × 4 channels × fb).
        # Прежде была QuantileExtremesHead (low+high only). Теперь все 4 канала
        # с ordering penalty за перекрытие диапазонов.
        self.quantile_head = QuantileOHLCHead(TRUNK_OUT, future_bars=future_bars)

        self.cls_head  = CalibratedClsHead(TRUNK_OUT)
        self.ohlc_head = OHLCHeadV2(TRUNK_OUT, future_bars=future_bars)
        self.aux_head  = AuxHead(TRUNK_OUT)
        self.dir_head  = DirectionHead(TRUNK_OUT)
        self.econ_heads = EconomicHeads(TRUNK_OUT)   # Sprint 2

        # Алиас для совместимости с trainer (backbone_ids)
        self.backbone = self.backbones[str(min(SCALES))]

        self._init_weights()
        self.dir_head._init_head()
        self.ohlc_head._init_head()
        self.econ_heads._init_heads()   # Sprint 2: после _init_weights — иначе bias=-4 обнулится
        nn.init.xavier_uniform_(self.extremes_head.weight, gain=0.1)
        nn.init.zeros_(self.extremes_head.bias)
        self.quantile_head._init_head()

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

    def forward(self, imgs, nums, ctx=None, hourly=None,
                intraday_feats=None, intraday_mask=None):
        feats = []
        for W in SCALES:
            feats.append(self.backbones[str(W)](imgs[W]))

        long_W = max(SCALES)
        if nums is not None and long_W in nums:
            x_long = self.wavelet(nums[long_W].float())
            feats.append(self.seq_branch(x_long))
        else:
            feats.append(torch.zeros(imgs[min(SCALES)].shape[0], TRUNK_OUT, device=imgs[min(SCALES)].device))

        for W in SCALES:
            if W != long_W and nums is not None and W in nums:
                feats.append(self.num_stats[str(W)](nums[W].float()))

        intraday_pred = None
        if self.use_hourly and hourly is not None:
            hourly_feat, intraday_pred = self.hourly_enc(hourly.float())
            feats.append(hourly_feat)

        if self.ctx_dim > 0 and ctx is not None:
            feats.append(self.ctx_proj(ctx.float()))

        feats = [f.nan_to_num(nan=0., posinf=10., neginf=-10.) for f in feats]
        h = self.vsn(feats)

        # Sprint 1.5: intraday feedback refinement (backward-compat: feats=None -> skip)
        next_hr_pred = None
        if HAS_INTRADAY_FEEDBACK and intraday_feats is not None:
            feats_f = intraday_feats.float().nan_to_num(nan=0.)
            mask_f  = (intraday_mask.float() if intraday_mask is not None
                       else torch.ones(feats_f.shape[:2], device=feats_f.device))
            _, next_hr_pred, all_hidden = self.intraday_enc(feats_f, mask_f)
            all_hidden = all_hidden.nan_to_num(nan=0., posinf=10., neginf=-10.)
            h = self.day_hour_attn(h, all_hidden, mask_f)

        logits    = self.cls_head(h)
        ohlc      = self.ohlc_head(h)
        aux       = self.aux_head(h)
        dir_logit = self.dir_head(h)
        econ      = self.econ_heads(h)
        ext_dhl   = self.extremes_head(h)              # [B, 2]: pred [dHigh, dLow]
        hl_logit  = self.high_low_head(h).unsqueeze(-1) # [B, 1]: P(high before low) logit
        # Sprint 11.1: quantile (low/high) prediction. Конкатенируется в конец
        # extremes для backward-compat — старые consumer'ы берут [:3], новые
        # знают что [3:] это [low_q × fb || high_q × fb] flatten.
        quantile = self.quantile_head(h)               # [B, 6 * fb]
        extremes = torch.cat([ext_dhl, hl_logit, quantile], dim=-1)  # [B, 3 + 6*fb]

        return logits, ohlc, aux, dir_logit, intraday_pred, econ, next_hr_pred, extremes


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


# ────────────────────────────────────────────────────────────
# Sprint 11.1: Quantile heads для full OHLC.
# Идея: вместо точечной OHLC выдаём 3 квантиля для каждой из 4 каналов
# (open, high, low, close). Pinball loss калибрует ширину интервала,
# ordering penalty заставляет диапазоны квантилей не перекрываться:
# range(L) ∪ range(O,C) ∪ range(H) должны идти строго снизу вверх.
#
# Output shape: [B, 4 channels × 3 quantiles × future_bars] = [B, 60] для fb=5
# Layout: O × 3q × fb || H × 3q × fb || L × 3q × fb || C × 3q × fb
#
# Backward-compat: предыдущий формат был [B, 30] (low+high only).
# Старые npz, скрипты, и quantile_eval.py читают первые 30 как high/low.
# Переименовали ключ в npz: quantile_pred теперь shape [N, 60].
# ────────────────────────────────────────────────────────────

class QuantileOHLCHead(nn.Module):
    """3 квантиля × 4 канала (O,H,L,C) × future_bars."""
    QUANTILES = (0.10, 0.50, 0.90)
    CHANNELS  = ("O", "H", "L", "C")

    def __init__(self, in_dim=TRUNK_OUT, future_bars=5):
        super().__init__()
        self.future_bars = future_bars
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.1),
        )
        # 4 отдельные головы — каждая выдаёт 3*fb значений
        self.head_open  = nn.Linear(128, len(self.QUANTILES) * future_bars)
        self.head_high  = nn.Linear(128, len(self.QUANTILES) * future_bars)
        self.head_low   = nn.Linear(128, len(self.QUANTILES) * future_bars)
        self.head_close = nn.Linear(128, len(self.QUANTILES) * future_bars)

    def _init_head(self):
        with torch.no_grad():
            for layer in (self.head_open, self.head_high,
                          self.head_low, self.head_close):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
            # Bias-init для H/L: все outputs смещены к полюсам (направление).
            self.head_low.bias.fill_(-0.3)
            self.head_high.bias.fill_(+0.3)
            # Bias-init для O/C: quantile spread — q10 ниже, q90 выше median.
            # Layout: bias[0:fb]=q10, bias[fb:2fb]=q50, bias[2fb:3fb]=q90
            # SPREAD=0.30 ≈ 1.28*std_OC (std_OC ≈ 0.23 в normalized ATR units)
            _SPREAD = 0.30
            fb = self.future_bars
            for head in (self.head_open, self.head_close):
                head.bias.data[0:fb].fill_(-_SPREAD)
                head.bias.data[fb:2*fb].fill_(0.0)
                head.bias.data[2*fb:3*fb].fill_(+_SPREAD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        # Layout: [B, O || H || L || C] each [3*fb]
        return torch.cat([self.head_open(h),  self.head_high(h),
                          self.head_low(h),   self.head_close(h)], dim=-1)


def split_ohlc_quantiles(pred: torch.Tensor, future_bars: int = 5,
                         n_quantiles: int = 3) -> dict:
    """Разделяет [B, 60] на {O,H,L,C} -> [B, n_q, fb]."""
    chunk = n_quantiles * future_bars
    B = pred.shape[0]
    O = pred[:, 0*chunk:1*chunk].reshape(B, n_quantiles, future_bars)
    H = pred[:, 1*chunk:2*chunk].reshape(B, n_quantiles, future_bars)
    L = pred[:, 2*chunk:3*chunk].reshape(B, n_quantiles, future_bars)
    C = pred[:, 3*chunk:4*chunk].reshape(B, n_quantiles, future_bars)
    return {"O": O, "H": H, "L": L, "C": C}


def pinball_loss_quantile(pred: torch.Tensor, target: torch.Tensor,
                          quantiles=(0.10, 0.50, 0.90)) -> torch.Tensor:
    """Pinball loss: pred [B, len(q) * fb], target [B, fb].

    Узкие интервалы при попадании = низкий loss, широкие = средний loss
    всегда, узкие при промахе = большой штраф (хвостовая регрессия).
    """
    B, total = pred.shape
    n_q  = len(quantiles)
    fb   = total // n_q
    pred_r = pred.reshape(B, n_q, fb)
    target_r = target[:, :fb].unsqueeze(1).expand(B, n_q, fb)
    err = target_r.float() - pred_r.float()
    q   = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).view(1, n_q, 1)
    return torch.where(err >= 0, q * err, (q - 1.) * err).mean()


def ordering_penalty_ohlc(quant_pred: torch.Tensor, future_bars: int = 5,
                          target_O: torch.Tensor | None = None,
                          target_C: torch.Tensor | None = None,
                          body_eps: float = 0.05) -> torch.Tensor:
    """Штраф за перекрытие диапазонов по схеме L < {O,C} < H (всегда)
    плюс направленное body O↔C (когда переданы таргеты).

    Безусловная часть:
        L_q90 ≤ O_q10  (low's upper tail ≤ open's lower tail — нет overlap)
        L_q90 ≤ C_q10
        O_q90 ≤ H_q10
        C_q90 ≤ H_q10
        + intra-channel monotonicity q10 ≤ q50 ≤ q90.

    Body O↔C (опц., 2026-05-09): O и C могут идти в любом порядке,
    поэтому жёсткий запрет overlap некорректен. Решение — supervised:
      • bull bar  (target_C > target_O + eps):  штраф если O_q90 > C_q10
      • bear bar  (target_C < target_O − eps):  штраф если C_q90 > O_q10
      • doji      (|ΔOC| ≤ eps):                штрафа нет (модель имеет право
                                                на overlap при неопределённости)
    eps берём в нормированных единицах ATR — 0.05 ≈ 5% от ATR-расстояния.
    """
    parts = split_ohlc_quantiles(quant_pred, future_bars=future_bars, n_quantiles=3)
    O, H, L, C = parts["O"], parts["H"], parts["L"], parts["C"]
    # Q index: 0=q10, 1=q50, 2=q90

    p1 = F.relu(L[:, 2, :] - O[:, 0, :])
    p2 = F.relu(L[:, 2, :] - C[:, 0, :])
    p3 = F.relu(O[:, 2, :] - H[:, 0, :])
    p4 = F.relu(C[:, 2, :] - H[:, 0, :])
    intra = sum(F.relu(t[:, 0, :] - t[:, 1, :]).mean()
                + F.relu(t[:, 1, :] - t[:, 2, :]).mean()
                for t in (O, H, L, C))

    body_pen = torch.tensor(0., device=quant_pred.device, dtype=quant_pred.dtype)
    if target_O is not None and target_C is not None:
        n = min(target_O.shape[1], target_C.shape[1], O.shape[2])
        body = target_C[:, :n].float() - target_O[:, :n].float()
        bull_mask = (body >  body_eps).to(quant_pred.dtype)   # [B, n]
        bear_mask = (body < -body_eps).to(quant_pred.dtype)
        # bull: O_q90 ≤ C_q10  (тело снизу-вверх, нет overlap)
        overlap_bull = F.relu(O[:, 2, :n] - C[:, 0, :n]) * bull_mask
        # bear: C_q90 ≤ O_q10
        overlap_bear = F.relu(C[:, 2, :n] - O[:, 0, :n]) * bear_mask
        # нормируем на число активных bar'ов, чтобы вес не зависел от долю doji
        denom = (bull_mask.sum() + bear_mask.sum()).clamp_min(1.0)
        body_pen = (overlap_bull.sum() + overlap_bear.sum()) / denom

    return p1.mean() + p2.mean() + p3.mean() + p4.mean() + intra + body_pen


class MultiTaskLossV3(nn.Module):
    """v3.19: исправленные веса, aux_loss передаётся из trainer.

    Sprint 11.1: optional quantile_loss_weight для pinball loss на high/low
    quantile heads (см. QuantileExtremesHead). Активируется передачей
    quantile_pred + ohlc_true в forward.
    """
    def __init__(self, cls_weight=None,
                 gamma_per_class=(2.0, 1.0, 2.0),
                 label_smoothing=0.05,
                 future_bars=1,
                 huber_delta=0.5,
                 direction_weight=0.40,    # v3.19: было 0.80
                 reg_loss_weight=0.20,     # v3.19: было 0.30
                 aux_loss_weight=0.10,     # v3.19: было 0.05
                 hl_quantile_loss_weight=0.05,  # Sprint 11.2 — pinball для H и L каналов
                                                # (H/L coverage 79%/76% при 0.025 эфф. ранее)
                 oc_quantile_loss_weight=0.15,  # Sprint 11.2 — pinball для O и C каналов (3×)
                                                # O/C collapse: coverage 2.5%/0.9% при равных
                                                # весах → нужен 3× gradient. Сумма 0.05+0.15=0.20
                                                # × 2 каналов каждый = 0.40 total (как было 4×0.10)
                 ordering_loss_weight=0.03,    # Sprint 11.1 — штраф за overlap L<O,C<H
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
        self.hl_q_w          = hl_quantile_loss_weight
        self.oc_q_w          = oc_quantile_loss_weight
        self.ordering_w      = ordering_loss_weight
        self.future_bars     = future_bars
        self.dir_mask_threshold = dir_mask_threshold

    def forward(self, logits, cls_y, ohlc_pred, ohlc_true,
                dir_logit=None, aux_pred=None, aux_true=None,
                quantile_pred=None):

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

        # Sprint 11.1 — Pinball loss на full OHLC quantile head + ordering penalty.
        # quantile_pred shape [B, 12*fb]: [O × 3q × fb || H × 3q × fb || L × 3q × fb || C × 3q × fb]
        # ohlc_true shape [B, 4*fb]: [O,H,L,C] per bar (row-major)
        q_loss = torch.tensor(0., device=logits.device)
        ord_loss = torch.tensor(0., device=logits.device)
        if (self.hl_q_w > 0 or self.oc_q_w > 0 or self.ordering_w > 0) and quantile_pred is not None:
            B = ohlc_true.shape[0]
            fb_true = ohlc_true.shape[1] // 4
            ohlc_3d = ohlc_true.reshape(B, fb_true, 4)
            target_O = ohlc_3d[:, :, 0]  # [B, fb]
            target_H = ohlc_3d[:, :, 1]
            target_L = ohlc_3d[:, :, 2]
            target_C = ohlc_3d[:, :, 3]
            chunk = quantile_pred.shape[1] // 4   # 3*fb (3 quantiles × fb)
            pred_O = quantile_pred[:, 0*chunk:1*chunk]
            pred_H = quantile_pred[:, 1*chunk:2*chunk]
            pred_L = quantile_pred[:, 2*chunk:3*chunk]
            pred_C = quantile_pred[:, 3*chunk:4*chunk]
            if self.hl_q_w > 0 or self.oc_q_w > 0:
                # Sprint 11.2: O/C получают 3× вес vs H/L — лечит O/C coverage collapse.
                # Сумма весов идентична старому 4×0.10: 2×0.15 + 2×0.05 = 0.40
                q_loss = (self.oc_q_w * pinball_loss_quantile(pred_O, target_O)
                          + self.hl_q_w * pinball_loss_quantile(pred_H, target_H)
                          + self.hl_q_w * pinball_loss_quantile(pred_L, target_L)
                          + self.oc_q_w * pinball_loss_quantile(pred_C, target_C))
            if self.ordering_w > 0:
                ord_loss = ordering_penalty_ohlc(
                    quantile_pred, future_bars=fb_true,
                    target_O=target_O, target_C=target_C,
                )

        total = (cls_loss
                 + self.reg_loss_weight * reg_loss
                 + self.dir_w           * dir_loss
                 + self.aux_loss_weight * a_loss
                 + q_loss                           # уже взвешен per-channel
                 + self.ordering_w      * ord_loss)

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
    loader = _make_loader_v3(te_ds, batch_size=512, shuffle=False, num_workers=2)
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
            n_out = len(out) if isinstance(out, (tuple, list)) else 1
            if n_out >= 4:
                lo, op = out[0], out[1]
                dir_l = out[3]
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