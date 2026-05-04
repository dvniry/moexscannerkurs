# ml/config.py
"""Гиперпараметры и константы — v3.4

Изменения v3.4:
- label_atr_k: float = 0.7 — коэффициент ATR-адаптивного порога разметки.
  Порог = atr_k * ATR(14) / close (в долях цены).
  Заменяет фиксированные profit_thr/loss_thr в build_labels_atr().
"""
from dataclasses import dataclass
from typing import Optional

SCALES = [5, 10, 20, 30]

SECTOR_CONTEXT = {
    # ── Нефть и газ ─────────────────────────────────────────────
    "ROSN":  ["IMOEX", "LCOc1"],
    "LKOH":  ["IMOEX", "LCOc1"],
    "NVTK":  ["IMOEX", "LCOc1"],
    "TATN":  ["IMOEX", "LCOc1"],
    "TATNP": ["IMOEX", "LCOc1"],
    "SIBN":  ["IMOEX", "LCOc1"],
    "SNGS":  ["IMOEX", "LCOc1"],
    "AFLT":  ["IMOEX", "LCOc1"],
    "GAZP":  ["IMOEX", "LCOc1"],
    "FLOT":  ["IMOEX", "LCOc1"],
    "TRNFP": ["IMOEX", "LCOc1"],
    "BANEP": ["IMOEX", "LCOc1"],
    # ── Металлы и горнодобыча ────────────────────────────────────
    "GMKN":  ["IMOEX", "XAU"],
    "MAGN":  ["IMOEX", "LCOc1"],
    "NLMK":  ["IMOEX", "LCOc1"],
    "CHMF":  ["IMOEX", "LCOc1"],
    "RUAL":  ["IMOEX", "XAU"],
    "ENPG":  ["IMOEX", "XAU"],
    "PLZL":  ["IMOEX", "XAU"],
    "ALRS":  ["IMOEX", "XAU"],
    "UGLD":  ["IMOEX", "XAU"],
    "SELG":  ["IMOEX", "XAU"],
    "MTLR":  ["IMOEX", "LCOc1"],
    "RASP":  ["IMOEX", "LCOc1"],
    "TRMK":  ["IMOEX", "LCOc1"],
    # ── Банки и финансы ──────────────────────────────────────────
    "SBER":  ["IMOEX", "RVI"],
    "SBERP": ["IMOEX", "RVI"],
    "VTBR":  ["IMOEX", "RVI"],
    "T":     ["IMOEX", "RVI"],
    "CBOM":  ["IMOEX", "RVI"],
    "BSPB":  ["IMOEX", "RVI"],
    "MOEX":  ["IMOEX", "RVI"],
    # ── Технологии и телеком ─────────────────────────────────────
    "MTSS":  ["IMOEX", "RVI"],
    "YDEX":  ["IMOEX", "RVI"],
    "RTKM":  ["IMOEX", "RVI"],
    "VKCO":  ["IMOEX", "RVI"],
    "HEAD":  ["IMOEX", "RVI"],
    # ── Ритейл и потребсектор ────────────────────────────────────
    "MGNT":  ["IMOEX", "RVI"],
    "OZON":  ["IMOEX", "RVI"],
    "PHOR":  ["IMOEX", "RVI"],
    "AFKS":  ["IMOEX", "RVI"],
    "LENT":  ["IMOEX", "RVI"],
    # ── Недвижимость ─────────────────────────────────────────────
    "LSRG":  ["IMOEX", "RVI"],
    # ── Прочее / холдинги ────────────────────────────────────────
    "IRAO":  ["IMOEX", "RVI"],
    "PIKK":  ["IMOEX", "RVI"],
    "SMLT":  ["IMOEX", "RVI"],
    # ── Транспорт ────────────────────────────────────────────────
    "NMTP":  ["IMOEX", "LCOc1"],
    "FESH":  ["IMOEX", "LCOc1"],
    # ── Энергетика ───────────────────────────────────────────────
    "HYDR":  ["IMOEX", "RVI"],
    "FEES":  ["IMOEX", "RVI"],
    "MSNG":  ["IMOEX", "RVI"],
    "UPRO":  ["IMOEX", "RVI"],
    "OGKB":  ["IMOEX", "RVI"],
    # ── Химия и удобрения ────────────────────────────────────────
    "NKNC":  ["IMOEX", "LCOc1"],
    "KAZT":  ["IMOEX", "LCOc1"],
    # ── Fallback ─────────────────────────────────────────────────
    "__default__": ["IMOEX", "RVI"],
}

@dataclass
class MLConfig:
    # ── Данные ──────────────────────────────────────────────────
    tickers: list = None
    interval: str = "1d"
    days_back: int = 3650
    future_bars: int = 5            # Sprint 2: было 1, поднято для MFE/MAE/edge
    profit_thr: float = 0.010
    loss_thr: float = -0.010

    # ── Комиссия брокера ────────────────────────────────────────
    broker_commission: float = 0.0005
    min_net_profit: float = 0.0030

    # ── Sprint 2: торговые издержки для cost-aware обучения ────
    econ_commission: float = 0.0005   # = broker_commission, одна сторона
    econ_slippage:   float = 0.0003
    econ_spread:     float = 0.0002

    # ── Image encoder ───────────────────────────────────────────
    img_size: int = 64
    window: int = 15

    # ── MLP ─────────────────────────────────────────────────────
    mlp_hidden: list = None
    mlp_dropout: float = 0.3
    mlp_lr: float = 1e-3

    # ── CNN / MultiScale ────────────────────────────────────────
    cnn_backbone: str = "resnet18"
    cnn_lr: float = 1e-4
    cnn_finetune_lr: float = 5e-5
    scale_dim: int = 64
    lstm_hidden: int = 128
    market_dim: int = 16

    # ── Обучение ────────────────────────────────────────────────
    batch_size: int = 64
    epochs_pre: int = 50
    epochs_fine: int = 25
    val_split: float = 0.2
    seed: int = 42

    # ── Loss weights ────────────────────────────────────────────
    ohlc_loss_weight: float = 0.3

    # ── Адаптивные пороги v3.1 ──────────────────────────────────
    use_adaptive_threshold: bool = True
    adaptive_k: float = 0.5
    adaptive_min_thr: float = 0.004

    # ── ATR-адаптивная разметка v3.4 ────────────────────────────
    label_atr_k: float = 0.3  # NEW: порог = atr_k * ATR(14) / close

    # ── Sprint 7: Multi-objective val_metric (#10) ───────────────
    # val_metric = dir_acc + alpha*dec_hit + beta*max(sharpe_proxy, 0)
    # Подобраны через sprint7_sweep.py; beta держим низким —
    # sharpe_proxy шумный на небольшом val-сете.
    val_metric_alpha: float = 0.20   # вес hit_rate (торговая точность)
    val_metric_beta:  float = 0.10   # вес sharpe_proxy (торговое качество)

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                # ── Банки и финансы (7) ──────────────────────────────
                "SBER", "SBERP", "VTBR", "T", "CBOM", "BSPB", "MOEX",
                # ── Нефть и газ (12) ──────────────────────────────────
                "GAZP", "LKOH", "NVTK", "ROSN", "TATN", "TATNP",
                "SIBN", "SNGS", "AFLT", "TRNFP", "BANEP", "FLOT",
                # ── Металлы и горнодобыча (13) ────────────────────────
                "GMKN", "MAGN", "NLMK", "CHMF", "RUAL", "ENPG",
                "PLZL", "ALRS", "UGLD", "SELG", "MTLR", "RASP",
                "TRMK",
                # ── Технологии и телеком (5) ──────────────────────────
                "YDEX", "MTSS", "RTKM", "VKCO", "HEAD",
                # ── Ритейл и потребсектор (5) ─────────────────────────
                "MGNT", "OZON", "PHOR", "LENT", "AFKS",
                # ── Недвижимость (2) ──────────────────────────────────
                "PIKK", "LSRG",
                # ── Прочее / холдинги (2) ─────────────────────────────
                "IRAO", "SMLT",
                # ── Транспорт (2) ─────────────────────────────────────
                "NMTP", "FESH",
                # ── Энергетика (5) ────────────────────────────────────
                "HYDR", "FEES", "MSNG", "UPRO", "OGKB",
                # ── Химия и удобрения (2) ─────────────────────────────
                "NKNC", "KAZT",
            ]
        if self.mlp_hidden is None:
            self.mlp_hidden = [128, 64, 32]

    # ── Алиасы для обратной совместимости ───────────────────────
    @property
    def lr(self) -> float:
        return self.cnn_lr

    @property
    def epochs(self) -> int:
        return self.epochs_pre

    @property
    def effective_profit_thr(self) -> float:
        return self.profit_thr + 2 * self.broker_commission + self.min_net_profit
    
    @property
    def daysback(self) -> int:
        return self.days_back

    @property
    def batchsize(self) -> int:
        return self.batch_size

    @property
    def futurebars(self) -> int:
        return self.future_bars

    @property
    def labelatrk(self) -> float:
        return self.label_atr_k

CFG = MLConfig()