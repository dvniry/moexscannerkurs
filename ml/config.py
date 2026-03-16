"""Гиперпараметры и константы."""
from dataclasses import dataclass

SCALES = [5, 10, 20, 30]

# Реальные тикеры индикативов из T-Bank API
SECTOR_CONTEXT = {
    # Нефтяники → Brent + волатильность рынка
    "ROSN": ["IMOEX", "LCOc1"],   # LCOc1 = Нефть Brent
    "LKOH": ["IMOEX", "LCOc1"],
    "NVTK": ["IMOEX", "LCOc1"],
    "TATN": ["IMOEX", "LCOc1"],
    "SNGS": ["IMOEX", "LCOc1"],
    "AFLT": ["IMOEX", "LCOc1"],   # авиация зависит от Brent
    # Металлурги → никель/медь + рынок
    "GMKN": ["IMOEX", "XAU"],      # Nl пустой → золото как прокси металлов
    "MAGN": ["IMOEX", "LCOc1"],    # Co пустой → нефть как прокси промышленности
    "NLMK": ["IMOEX", "LCOc1"],
    "CHMF": ["IMOEX", "LCOc1"],
    # Золото/алмазы → XAU + палладий
    "PLZL": ["IMOEX", "XAU"],     # XAU = Золото
    "ALRS": ["IMOEX", "XAU"],
    # Финансы/телеком/ритейл → только IMOEX + RVI
    "SBER": ["IMOEX", "RVI"],     # RVI = волатильность MOEX
    "VTBR": ["IMOEX", "RVI"],
    "CBOM": ["IMOEX", "RVI"],
    "GAZP": ["IMOEX", "LCOc1"],
    "MTSS": ["IMOEX", "RVI"],
    "MGNT": ["IMOEX", "RVI"],
    "PHOR": ["IMOEX", "RVI"],
    "T":    ["IMOEX", "RVI"],
    "__default__": ["IMOEX", "RVI"],
}



@dataclass
class MLConfig:
    # Данные
    tickers:      list  = None
    interval:     str   = "1d"
    days_back:    int   = 730
    future_bars:  int   = 5
    profit_thr:   float = 0.010
    loss_thr:     float = -0.010

    # Комиссия брокера
    broker_commission: float = 0.0005
    min_net_profit:    float = 0.0030

    # Image encoder
    img_size:     int   = 64
    window:       int   = 15

    # MLP
    mlp_hidden:   list  = None
    mlp_dropout:  float = 0.3
    mlp_lr:       float = 1e-3

    # CNN / MultiScale
    cnn_backbone:    str   = "resnet18"
    cnn_lr:          float = 1e-4
    cnn_finetune_lr: float = 5e-5
    scale_dim:       int   = 64
    lstm_hidden:     int   = 128
    market_dim:      int   = 16

    # Обучение
    batch_size:   int   = 256   
    epochs_pre:   int   = 50   
    epochs_fine:  int   = 25
    val_split:    float = 0.2
    seed:         int   = 42

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                "SBER", "GAZP", "LKOH", "GMKN", "NVTK",
                "ROSN", "TATN", "MGNT", "MTSS", "T",
                "ALRS", "PLZL", "SNGS", "VTBR", "AFLT",
                "MAGN", "NLMK", "CHMF", "PHOR", "CBOM",
            ]
        if self.mlp_hidden is None:
            self.mlp_hidden = [128, 64, 32]

    @property
    def effective_profit_thr(self) -> float:
        return self.profit_thr + 2 * self.broker_commission + self.min_net_profit


CFG = MLConfig()
