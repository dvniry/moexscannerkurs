"""Часовой энкодер — агрегация внутридневных свечей для каждого дня.

Тактика: для каждой дневной свечи подгружаем часовые свечи торговой сессии
(MOEX: 10:00-18:50 → ~9 часовых свечей), рендерим факты → токен.

Это даёт модели:
  - Внутридневную микроструктуру (где был Low, где High)
  - Volume Profile (распределение объёма по времени)
  - Паттерны открытия/закрытия
  - VWAP отклонение
  - Cumulative Delta (давление покупателей vs продавцов)

Архитектура:
  HourlyEncoder:
    Вход:  (B, N_days, N_hours, N_features)  — последние N_days дней × N_hours часов
    1) Per-day CNN: 1D Conv по N_hours → (B, N_days, D_intra)
    2) Cross-day Attention: (B, N_days, D_intra) → (B, D_out)
    Выход: один токен (B, D_out) для подачи в cross-scale attention

Каналы часового рендера (per hour candle):
  0: Open  (norm to daily range)
  1: High  (norm)
  2: Low   (norm)
  3: Close (norm)
  4: Volume (norm to daily max)
  5: VWAP deviation (close - vwap) / atr
  6: Body ratio (close-open) / (high-low)  — форма свечи
  7: Upper wick ratio
  8: Lower wick ratio
  9: Cumulative return from day open
 10: Hour position (0..1) — positional encoding
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

N_HOURLY_CHANNELS = 11   # количество каналов часового рендера
N_HOURS_PER_DAY   = 9    # MOEX: 10:00-18:00 → 9 часовых свечей (макс)
N_INTRADAY_DAYS   = 5    # последние 5 дней часовых данных


# ══════════════════════════════════════════════════════════════════
#  Рендер часовых свечей → тензор
# ══════════════════════════════════════════════════════════════════

def render_hourly_candles(hourly_df: pd.DataFrame,
                          daily_close: float,
                          daily_high: float,
                          daily_low: float) -> np.ndarray:
    """
    Рендерит часовые свечи одного дня в тензор.

    Вход:
      hourly_df:  DataFrame с колонками open/high/low/close/volume
                  (часовые свечи одного торгового дня)
      daily_close, daily_high, daily_low:  дневные значения для нормировки

    Выход:
      np.ndarray shape (N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype float32
      Если свечей < N_HOURS_PER_DAY — паддинг нулями слева (early hours missing)
    """
    o = hourly_df['open'].values.astype(np.float64)
    h = hourly_df['high'].values.astype(np.float64)
    l = hourly_df['low'].values.astype(np.float64)
    c = hourly_df['close'].values.astype(np.float64)
    v = hourly_df['volume'].values.astype(np.float64)
    n = len(o)

    if n == 0:
        return np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype=np.float32)

    # ── Нормировка по дневному диапазону ──────────────────────
    price_rng = daily_high - daily_low + 1e-9
    def norm_p(arr):
        return ((arr - daily_low) / price_rng).astype(np.float32)

    # ── VWAP ─────────────────────────────────────────────────
    typical_price = (h + l + c) / 3.0
    cum_tp_vol = np.cumsum(typical_price * v)
    cum_vol    = np.cumsum(v) + 1e-9
    vwap       = cum_tp_vol / cum_vol
    atr_proxy  = price_rng  # используем дневной диапазон как proxy ATR
    vwap_dev   = ((c - vwap) / atr_proxy).astype(np.float32)

    # ── Body / Wick ratios ───────────────────────────────────
    candle_range = h - l + 1e-9
    body_ratio   = ((c - o) / candle_range).astype(np.float32)
    upper_wick   = ((h - np.maximum(o, c)) / candle_range).astype(np.float32)
    lower_wick   = ((np.minimum(o, c) - l) / candle_range).astype(np.float32)

    # ── Cumulative return from day open ──────────────────────
    day_open   = o[0]
    cum_ret    = ((c - day_open) / (day_open + 1e-9)).astype(np.float32)

    # ── Hour position (positional encoding) ──────────────────
    hour_pos = np.linspace(0, 1, n).astype(np.float32)

    # ── Volume norm ──────────────────────────────────────────
    vol_norm = (v / (v.max() + 1e-9)).astype(np.float32)

    # ── Stack channels ───────────────────────────────────────
    channels = np.stack([
        norm_p(o),       # 0: Open
        norm_p(h),       # 1: High
        norm_p(l),       # 2: Low
        norm_p(c),       # 3: Close
        vol_norm,        # 4: Volume
        vwap_dev,        # 5: VWAP deviation
        body_ratio,      # 6: Body ratio
        upper_wick,      # 7: Upper wick
        lower_wick,      # 8: Lower wick
        cum_ret,         # 9: Cumulative return
        hour_pos,        # 10: Hour position
    ], axis=0)  # (11, n)

    # ── Pad to N_HOURS_PER_DAY ───────────────────────────────
    if n < N_HOURS_PER_DAY:
        pad = np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY - n), dtype=np.float32)
        channels = np.concatenate([pad, channels], axis=1)
    elif n > N_HOURS_PER_DAY:
        channels = channels[:, -N_HOURS_PER_DAY:]

    return channels.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
#  PyTorch модули
# ══════════════════════════════════════════════════════════════════

class IntraDayConv(nn.Module):
    """1D CNN для одного дня часовых свечей.

    Вход:  (B, N_HOURLY_CHANNELS, N_HOURS_PER_DAY)
    Выход: (B, D_intra)
    """
    def __init__(self, in_ch: int = N_HOURLY_CHANNELS,
                 d_intra: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(64, d_intra),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (B, C, T) where T = N_HOURS_PER_DAY
        h = self.net(x)        # (B, 64, T')
        h = self.pool(h)       # (B, 64, 1)
        h = h.squeeze(-1)      # (B, 64)
        return self.proj(h)    # (B, D_intra)


class HourlyEncoder(nn.Module):
    """Мультитаймфрейм энкодер: N_days дней часовых свечей → один токен.

    Архитектура:
      1. IntraDayConv × N_days → (B, N_days, D_intra) — per-day embeddings
      2. Self-Attention across days → context-aware embeddings
      3. Weighted pool → (B, D_out)

    Параметры:
      n_days:    сколько последних дней часовых данных использовать (default 5)
      d_intra:   размерность per-day embedding (default 64)
      d_out:     размерность выходного токена (default 128)
      n_heads:   количество голов attention (default 2)
      dropout:   dropout rate (default 0.2)
    """
    def __init__(self, n_days: int = N_INTRADAY_DAYS,
                 d_intra: int = 64, d_out: int = 128,
                 n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_days  = n_days
        self.d_out   = d_out

        # Общий per-day encoder (shared weights)
        self.day_conv = IntraDayConv(
            in_ch=N_HOURLY_CHANNELS, d_intra=d_intra, dropout=dropout)

        # Positional embedding для дней (день 0 = самый старый)
        self.day_pos_embed = nn.Embedding(n_days, d_intra)

        # Cross-day attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_intra, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_intra)

        # Projection → output dim
        self.proj = nn.Sequential(
            nn.Linear(d_intra * n_days, d_out),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, hourly_data: torch.Tensor) -> torch.Tensor:
        """
        hourly_data: (B, N_days, N_HOURLY_CHANNELS, N_HOURS_PER_DAY)
        Возвращает:  (B, D_out)
        """
        B, D, C, T = hourly_data.shape
        assert D == self.n_days, f"Expected {self.n_days} days, got {D}"

        # ── Per-day encoding ─────────────────────────────────
        # Reshape: (B * N_days, C, T) → pass through shared CNN
        x = hourly_data.view(B * D, C, T)
        day_emb = self.day_conv(x)       # (B * D, d_intra)
        day_emb = day_emb.view(B, D, -1) # (B, N_days, d_intra)

        # Add positional encoding
        pos_ids = torch.arange(D, device=hourly_data.device)
        day_emb = day_emb + self.day_pos_embed(pos_ids).unsqueeze(0)

        # ── Cross-day attention ──────────────────────────────
        attn_out, _ = self.attn(day_emb, day_emb, day_emb)
        day_emb = self.attn_norm(day_emb + attn_out)  # residual

        # ── Flatten + project ────────────────────────────────
        fused = day_emb.flatten(1)       # (B, N_days * d_intra)
        return self.proj(fused)          # (B, D_out)
