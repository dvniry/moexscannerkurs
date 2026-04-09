# ml/config.py
"""Гиперпараметры и константы — v3.2 (consolidated)."""
from dataclasses import dataclass, field
from typing import List, Optional

SCALES = [5, 10, 20, 30]

SECTOR_CONTEXT = {
    "ROSN": ["IMOEX", "LCOc1"], "LKOH": ["IMOEX", "LCOc1"],
    "NVTK": ["IMOEX", "LCOc1"], "TATN": ["IMOEX", "LCOc1"],
    "SNGS": ["IMOEX", "LCOc1"], "AFLT": ["IMOEX", "LCOc1"],
    "GAZP": ["IMOEX", "LCOc1"], "FLOT": ["IMOEX", "LCOc1"],
    "GMKN": ["IMOEX", "XAU"],  "MAGN": ["IMOEX", "LCOc1"],
    "NLMK": ["IMOEX", "LCOc1"], "CHMF": ["IMOEX", "LCOc1"],
    "RUAL": ["IMOEX", "XAU"],  "PLZL": ["IMOEX", "XAU"],
    "ALRS": ["IMOEX", "XAU"],  "UGLD": ["IMOEX", "XAU"],
    "SELG": ["IMOEX", "XAU"],  "SBER": ["IMOEX", "RVI"],
    "VTBR": ["IMOEX", "RVI"],  "CBOM": ["IMOEX", "RVI"],
    "SVCB": ["IMOEX", "RVI"],  "BSPB": ["IMOEX", "RVI"],
    "MOEX": ["IMOEX", "RVI"],  "T":    ["IMOEX", "RVI"],
    "MTSS": ["IMOEX", "RVI"],  "YDEX": ["IMOEX", "RVI"],
    "RTKM": ["IMOEX", "RVI"],  "VKCO": ["IMOEX", "RVI"],
    "POSI": ["IMOEX", "RVI"],  "HEAD": ["IMOEX", "RVI"],
    "MGNT": ["IMOEX", "RVI"],  "FIVE": ["IMOEX", "RVI"],
    "OZON": ["IMOEX", "RVI"],  "PHOR": ["IMOEX", "RVI"],
    "IRAO": ["IMOEX", "RVI"],  "PIKK": ["IMOEX", "RVI"],
    "AFKS": ["IMOEX", "RVI"],
    "__default__": ["IMOEX", "RVI"],
    "TRNFP": ["IMOEX", "LCOc1"],
    "BANEP": ["IMOEX", "LCOc1"],
    "SELG":  ["IMOEX", "XAU"],
    "UGLD":  ["IMOEX", "XAU"],
    "TCSG":  ["IMOEX", "RVI"],
    "HYDR":  ["IMOEX", "RVI"],
    "FEES":  ["IMOEX", "RVI"],
    "MSNG":  ["IMOEX", "RVI"],
    "UPRO":  ["IMOEX", "RVI"],
    "FIXP":  ["IMOEX", "RVI"],
    "LENT":  ["IMOEX", "RVI"],
    "DSKY":  ["IMOEX", "RVI"],
    "SMLT":  ["IMOEX", "RVI"],
    "CIAN":  ["IMOEX", "RVI"],
}


@dataclass
class MLConfig:
    # ── Данные ──────────────────────────────────────────────────
    tickers:    list  = None
    interval:   str   = "1d"
    days_back:  int   = 3650  
    future_bars: int  = 5
    profit_thr: float = 0.010
    loss_thr:   float = -0.010

    # ── Комиссия брокера ────────────────────────────────────────
    broker_commission: float = 0.0005
    min_net_profit:    float = 0.0030

    # ── Image encoder ───────────────────────────────────────────
    img_size: int = 64
    window:   int = 15

    # ── MLP ─────────────────────────────────────────────────────
    mlp_hidden:  list  = None
    mlp_dropout: float = 0.3
    mlp_lr:      float = 1e-3

    # ── CNN / MultiScale ────────────────────────────────────────
    cnn_backbone:    str   = "resnet18"
    cnn_lr:          float = 1e-4   # используется как base_lr в trainer
    cnn_finetune_lr: float = 5e-5
    scale_dim:       int   = 64
    lstm_hidden:     int   = 128
    market_dim:      int   = 16

    # ── Обучение ────────────────────────────────────────────────
    batch_size:  int   = 64
    epochs_pre:  int   = 50    # полное обучение (full phase)
    epochs_fine: int   = 25    # finetune фаза (frozen backbone)
    val_split:   float = 0.2
    seed:        int   = 42

    # ── Loss weights ────────────────────────────────────────────
    ohlc_loss_weight: float = 0.3  # вес регрессионного loss (OHLC)

    # ── Адаптивные пороги v3.1 ──────────────────────────────────
    use_adaptive_threshold: bool  = True
    adaptive_k:             float = 0.5   # порог = k * ATR_ratio
    adaptive_min_thr:       float = 0.004

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                # Уже есть (35 тикеров)
                "SBER", "GAZP", "LKOH", "GMKN", "NVTK",
                "ROSN", "TATN", "MGNT", "MTSS", "T",
                "ALRS", "PLZL", "SNGS", "VTBR", "AFLT",
                "MAGN", "NLMK", "CHMF", "PHOR", "CBOM",
                "YDEX", "MOEX", "IRAO", "PIKK", "RTKM",
                "RUAL", "OZON", "FIVE", "HEAD", "FLOT",
                "SVCB", "AFKS", "BSPB", "VKCO", "POSI",
                # ── Новые: нефть/газ ──────────────────────────────────
                "TRNFP",   # Транснефть-ап (высоколиквидна)
                "BANEP",   # Башнефть-ап
                # ── Новые: металлы/горнодобыча ───────────────────────
                "SELG",    # Селигдар (золото)
                "UGLD",    # ЮГК
                # ── Новые: банки/финансы ─────────────────────────────
                "TCSG",    # ТКС (много данных, высокая ликвидность)
                # ── Новые: энергетика ────────────────────────────────
                "HYDR",    # РусГидро
                "FEES",    # ФСК-Россети
                "MSNG",    # Мосэнерго
                "UPRO",    # Юнипро
                # ── Новые: ритейл/потребсектор ───────────────────────
                "FIXP",    # Fix Price
                "LENT",    # Лента
                "DSKY",    # Детский мир (если ещё торгуется)
                # ── Новые: IT/прочее ─────────────────────────────────
                "SMLT",    # Самолёт (девелопер)
                "CIAN",    # ЦИАН
            ]
        if self.mlp_hidden is None:
            self.mlp_hidden = [128, 64, 32]

    # ── Алиасы для обратной совместимости ───────────────────────
    @property
    def lr(self) -> float:
        """Алиас → cnn_lr (base learning rate)."""
        return self.cnn_lr

    @property
    def epochs(self) -> int:
        """Алиас → epochs_pre (full training epochs)."""
        return self.epochs_pre

    @property
    def effective_profit_thr(self) -> float:
        return self.profit_thr + 2 * self.broker_commission + self.min_net_profit


CFG = MLConfig()
