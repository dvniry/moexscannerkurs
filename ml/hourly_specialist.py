"""Sprint 4: HourlySpecialist — модель для часовых баров.

v1 (2026-05-01): откат от v2 (углубление 2L×96 не дало прироста, val_acc 0.5378→0.5389
                  при ×2.2 параметров — модель упёрлась в потолок данных, не capacity).
  Сохранены полезные части v2:
  - input_dropout 0.1 (random feature-mask на тренинге) — регуляризация
  - vol_head с реальным таргетом обучается через trainer (B-3)
Архитектура: CNN(scales=[5,10,45]) + BiLSTM(64) → dir_head + vol_head
Параметров: ~280k (намеренно маленькая, ~8× меньше V3)
Вход:  [B, 45, 37]  — 45 часовых баров, 37 индикаторов
Выход: {dir_logit [B], vol_pred [B]}
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.hourly_only_dataset import HOURLY_WINDOW, N_HOURLY_FEAT


class HourlySpecialist(nn.Module):
    """BiLSTM + multi-scale CNN для часовых баров MOEX."""

    def __init__(
        self,
        n_feat:        int   = N_HOURLY_FEAT,    # 37
        window:        int   = HOURLY_WINDOW,    # 45
        cnn_ch:        int   = 32,
        lstm_hid:      int   = 64,
        proj_dim:      int   = 64,
        dropout:       float = 0.3,
        input_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_feat          = n_feat
        self.window          = window
        self.lstm_hid        = lstm_hid
        self.input_dropout_p = input_dropout

        # ── Feature projection ──────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(n_feat, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # ── CNN branches (работаем по временному измерению) ─────
        self.cnn5  = nn.Sequential(
            nn.Conv1d(proj_dim, cnn_ch, kernel_size=5,  padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),   # → [B, cnn_ch, 1]
        )
        self.cnn10 = nn.Sequential(
            nn.Conv1d(proj_dim, cnn_ch, kernel_size=10, padding=5),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cnn_all = nn.Sequential(
            nn.Conv1d(proj_dim, cnn_ch, kernel_size=window, padding=0),
            nn.GELU(),
            # kernel_size=window → output length = 1, no pooling needed
        )

        cnn_out_dim = 3 * cnn_ch   # concat трёх веток
        self.cnn_fusion = nn.Sequential(
            nn.Linear(cnn_out_dim, proj_dim),
            nn.GELU(),
        )

        # ── BiLSTM branch (1 layer, hidden=64) ──────────────────
        self.lstm = nn.LSTM(
            input_size    = proj_dim,
            hidden_size   = lstm_hid,
            num_layers    = 1,
            batch_first   = True,
            bidirectional = True,
        )
        lstm_out_dim = 2 * lstm_hid   # bidirectional → concat

        # ── Fusion ───────────────────────────────────────────────
        fusion_in = proj_dim + lstm_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Output heads ─────────────────────────────────────────
        self.dir_head = nn.Linear(128, 1)
        self.vol_head = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Небольшое смещение dir_head к нейтральному
        nn.init.constant_(self.dir_head.bias, 0.0)
        nn.init.constant_(self.vol_head.bias, -2.0)   # начинаем с низкой вол

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: [B, T=45, F=37]
        Returns:
            dir_logit: [B]   — P(UP) = sigmoid(dir_logit)
            vol_pred:  [B]   — realized vol proxy (mean of range_norm в окне)
        """
        # B-3 + v2: input dropout — random feature-mask на тренинге.
        # Маскирует случайные feature-каналы в [0.85, 1.0] коэффициентом.
        if self.training and self.input_dropout_p > 0.0:
            mask = torch.empty(x.shape[0], 1, x.shape[2], device=x.device).uniform_(0.0, 1.0)
            mask = (mask > self.input_dropout_p).float()
            x = x * mask

        # Projection: [B, T, proj_dim]
        h = self.proj(x)

        # ── CNN branch (k=5, 10, window=45) ────────────────────
        h_t = h.transpose(1, 2)                   # [B, proj_dim, T]
        c5   = self.cnn5(h_t).squeeze(-1)         # [B, cnn_ch]
        c10  = self.cnn10(h_t).squeeze(-1)        # [B, cnn_ch]
        call = self.cnn_all(h_t).squeeze(-1)      # [B, cnn_ch]
        cnn_out = self.cnn_fusion(
            torch.cat([c5, c10, call], dim=-1))   # [B, proj_dim]

        # ── BiLSTM branch (1 layer) ────────────────────────────
        lstm_out, (h_n, _) = self.lstm(h)         # h_n: [num_layers*2, B, lstm_hid]
        # last hidden state from both directions (1 layer → h_n[0]=fwd, h_n[1]=bwd)
        h_fwd = h_n[0]
        h_bwd = h_n[1]
        lstm_feat = torch.cat([h_fwd, h_bwd], dim=-1)   # [B, 2*lstm_hid]

        # ── Fusion ───────────────────────────────────────────────
        feat = self.fusion(torch.cat([cnn_out, lstm_feat], dim=-1))  # [B, 128]

        dir_logit = self.dir_head(feat).squeeze(-1)   # [B]
        vol_pred  = self.vol_head(feat).squeeze(-1)   # [B]

        return {
            "dir_logit": dir_logit,
            "vol_pred":  vol_pred,
        }

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> dict:
        """Инференс: возвращает вероятности."""
        out = self.forward(x)
        return {
            "dir_prob":  torch.sigmoid(out["dir_logit"]),
            "vol_proxy": out["vol_pred"],
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_hourly_specialist(**kwargs) -> HourlySpecialist:
    """Фабрика с параметрами по умолчанию."""
    model = HourlySpecialist(**kwargs)
    n = model.count_params()
    print(f"  HourlySpecialist: {n:,} параметров")
    return model


if __name__ == "__main__":
    # Smoke-test
    model = build_hourly_specialist()
    x = torch.randn(4, HOURLY_WINDOW, N_HOURLY_FEAT)
    out = model(x)
    print(f"dir_logit: {out['dir_logit'].shape}  {out['dir_logit']}")
    print(f"vol_pred:  {out['vol_pred'].shape}   {out['vol_pred']}")
