# ml/hourly_feedback.py
"""Sprint 1.5: Intraday Feedback Loop.

Компоненты:
  HourlyFeedbackEncoder  — LSTM по часам текущего дня T0
  DayHourCrossAttention  — уточнение дневного признака через часовые (gated cross-attn)
  IntradayConsistencyLoss — next_hour_loss + extremes_loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.hourly_encoder import N_HOURS_PER_DAY
from ml.dataset_v3 import N_INTRADAY_FEATS, INTRADAY_N_COLS

TRUNK_OUT = 128   # должно совпадать с multiscale_cnn_v3.TRUNK_OUT
_LSTM_HIDDEN = 128


class HourlyFeedbackEncoder(nn.Module):
    """LSTM по часам текущего дня T0.

    Вход:
        feats [B, T, N_INTRADAY_FEATS]  — нормированные часовые OHLCV+meta
        mask  [B, T]                    — 1.0 для известных часов, 0.0 иначе

    Выход:
        hour_ctx     [B, TRUNK_OUT]     — контекстный вектор всего дня
        next_hr_pred [B, T, 5]          — предсказание OHLCV следующего часа
        all_hidden   [B, T, H]          — все hidden states → для cross-attention
    """

    def __init__(self, in_feats: int = N_INTRADAY_FEATS,
                 hidden: int = _LSTM_HIDDEN, out_dim: int = TRUNK_OUT):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(in_feats, hidden, batch_first=True)
        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.next_hour_head = nn.Linear(hidden, 5)   # OHLCV следующего часа

        nn.init.xavier_uniform_(self.next_hour_head.weight, gain=0.1)
        nn.init.zeros_(self.next_hour_head.bias)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor):
        """
        feats: [B, T, F]
        mask:  [B, T]  (float, 1=known)
        """
        B, T, _ = feats.shape

        # Зануляем padding-часы через маску
        feats = feats * mask.unsqueeze(-1)

        # Вычисляем длины последовательностей (количество известных часов)
        lengths = mask.sum(dim=1).long().clamp(min=1)   # [B]

        # LSTM forward (без pack — на GPU разница мала, зато проще)
        all_h, (h_n, _) = self.lstm(feats)              # all_h: [B, T, H]

        # Зануляем hidden states для padding-часов
        all_h = all_h * mask.unsqueeze(-1)

        # Берём last known hidden state для контекста
        idx = (lengths - 1).clamp(0, T - 1)             # [B]
        last_h = all_h[torch.arange(B, device=feats.device), idx]  # [B, H]

        hour_ctx    = self.ctx_proj(last_h)              # [B, TRUNK_OUT]
        next_hr_pred = self.next_hour_head(all_h)        # [B, T, 5]

        return hour_ctx, next_hr_pred, all_h


class DayHourCrossAttention(nn.Module):
    """Cross-attention: дневной признак уточняется через часовые hidden states.

    Q = day_feat [B, D]       из VSN
    K = V = hour_hidden [B, T, D]  из HourlyFeedbackEncoder

    Gate с bias=-2: на старте часовые влияют ~12%, обучаются постепенно.
    Возвращает: h_refined = h + gate * cross_attn_output
    """

    def __init__(self, d_model: int = TRUNK_OUT, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          batch_first=True, dropout=0.1)
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate_proj.bias, -2.0)   # sigmoid(-2) ≈ 0.12
        self.norm = nn.LayerNorm(d_model)

    def forward(self, day_feat: torch.Tensor,
                hour_hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        day_feat:    [B, D]
        hour_hidden: [B, T, D]
        mask:        [B, T]  (1=known, 0=padding)
        Returns:     [B, D]  refined day feature
        """
        q = day_feat.unsqueeze(1)                       # [B, 1, D]

        # key_padding_mask: True = IGNORE (логика PyTorch)
        key_pad = (mask == 0)                           # [B, T]

        # Если все позиции замаскированы — пропускаем attention
        all_masked = key_pad.all(dim=1, keepdim=True)  # [B, 1]

        # Временно снимаем маску там где всё замаскировано, чтобы не получить NaN
        key_pad_safe = key_pad & ~all_masked

        attn_out, _ = self.attn(q, hour_hidden, hour_hidden,
                                key_padding_mask=key_pad_safe)  # [B, 1, D]
        attn_out = attn_out.squeeze(1)                          # [B, D]

        gate = torch.sigmoid(self.gate_proj(day_feat))          # [B, 1]
        h_refined = day_feat + gate * attn_out
        h_refined = self.norm(h_refined)

        return h_refined


class IntradayConsistencyLoss(nn.Module):
    """Три компонента:

    next_hour_loss: MSE между предсказанием OHLCV часа t+1 и фактическим [t+1].
    extremes_loss:  Huber между предсказанием [dHigh, dLow] и intraday_extremes[:2].
    high_low_loss:  BCE на high_first_label (extremes[:,2]); маска -1 = unknown.
    """

    def __init__(self, next_hour_weight: float = 0.5,
                 extremes_weight: float = 0.5,
                 high_low_weight: float = 0.2):   # Sprint 8.2
        super().__init__()
        self.w_next   = next_hour_weight
        self.w_ext    = extremes_weight
        self.w_hl     = high_low_weight

    def forward(self,
                next_hr_pred: torch.Tensor,    # [B, T, 5]
                feats_actual: torch.Tensor,     # [B, T, F>=5]
                extremes_pred: torch.Tensor,   # [B, 2] or [B, 3]
                extremes_true: torch.Tensor,   # [B, 2] or [B, 3]
                mask: torch.Tensor             # [B, T]
                ) -> torch.Tensor:

        loss = torch.tensor(0.0, device=next_hr_pred.device)

        # ── next_hour_loss ───────────────────────────────────────────
        if next_hr_pred is not None and mask.sum() > 0:
            pred_slice   = next_hr_pred[:, :-1, :]
            target_slice = feats_actual[:, 1:, :5]
            mask_slice   = mask[:, 1:]
            valid = mask_slice > 0.5
            if valid.sum() > 0:
                p = pred_slice[valid.unsqueeze(-1).expand_as(pred_slice)]
                t = target_slice[valid.unsqueeze(-1).expand_as(target_slice)]
                loss = loss + self.w_next * F.mse_loss(p, t)

        # ── extremes_loss ────────────────────────────────────────────
        if extremes_pred is not None and extremes_true is not None:
            loss = loss + self.w_ext * F.huber_loss(
                extremes_pred[:, :2], extremes_true[:, :2], delta=1.0)

            # Sprint 8.2: high_first BCE — skip where label == -1 (unknown)
            if (self.w_hl > 0
                    and extremes_pred.shape[-1] >= 3
                    and extremes_true.shape[-1] >= 3):
                hl_logit = extremes_pred[:, 2]
                hl_label = extremes_true[:, 2]
                valid_hl = hl_label >= 0.0  # -1 = unknown (no hourly data)
                if valid_hl.sum() >= 4:
                    loss = loss + self.w_hl * F.binary_cross_entropy_with_logits(
                        hl_logit[valid_hl], hl_label[valid_hl])

        return loss
