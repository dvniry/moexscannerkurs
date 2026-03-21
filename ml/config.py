"""Гиперпараметры и константы."""
from dataclasses import dataclass

SCALES = [5, 10, 20, 30]

# Реальные тикеры индикативов из T-Bank API
SECTOR_CONTEXT = {
    # Нефтяники → Brent + волатильность рынка
    "ROSN": ["IMOEX", "LCOc1"],
    "LKOH": ["IMOEX", "LCOc1"],
    "NVTK": ["IMOEX", "LCOc1"],
    "TATN": ["IMOEX", "LCOc1"],
    "SNGS": ["IMOEX", "LCOc1"],
    "AFLT": ["IMOEX", "LCOc1"],
    "GAZP": ["IMOEX", "LCOc1"],
    "FLOT": ["IMOEX", "LCOc1"],   # Совкомфлот
    # Металлурги / промышленность
    "GMKN": ["IMOEX", "XAU"],
    "MAGN": ["IMOEX", "LCOc1"],
    "NLMK": ["IMOEX", "LCOc1"],
    "CHMF": ["IMOEX", "LCOc1"],
    "RUAL": ["IMOEX", "XAU"],     # Русал
    # Золото / алмазы
    "PLZL": ["IMOEX", "XAU"],
    "ALRS": ["IMOEX", "XAU"],
    "UGLD": ["IMOEX", "XAU"],     # ЮГК
    "SELG": ["IMOEX", "XAU"],     # Селигдар
    # Финансы
    "SBER": ["IMOEX", "RVI"],
    "VTBR": ["IMOEX", "RVI"],
    "CBOM": ["IMOEX", "RVI"],
    "SVCB": ["IMOEX", "RVI"],     # Совкомбанк
    "BSPB": ["IMOEX", "RVI"],     # БСП
    "MOEX": ["IMOEX", "RVI"],     # МосБиржа
    "T":    ["IMOEX", "RVI"],
    # Телеком / IT
    "MTSS": ["IMOEX", "RVI"],
    "YDEX": ["IMOEX", "RVI"],     # Яндекс
    "RTKM": ["IMOEX", "RVI"],     # Ростелеком
    "VKCO": ["IMOEX", "RVI"],     # VK
    "POSI": ["IMOEX", "RVI"],     # Positive Technologies
    "HEAD": ["IMOEX", "RVI"],     # HeadHunter
    # Ритейл / потребительский
    "MGNT": ["IMOEX", "RVI"],
    "FIVE": ["IMOEX", "RVI"],     # X5 Retail Group
    "OZON": ["IMOEX", "RVI"],     # Ozon
    # Энергетика / химия
    "PHOR": ["IMOEX", "RVI"],
    "IRAO": ["IMOEX", "RVI"],     # Интер РАО
    "PIKK": ["IMOEX", "RVI"],     # ПИК
    "AFKS": ["IMOEX", "RVI"],     # АФК Система
    "__default__": ["IMOEX", "RVI"],
}



@dataclass
class MLConfig:
    # Данные
    tickers:      list  = None
    interval:     str   = "1d"
    days_back:    int   = 1825    # 5 лет
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
    batch_size:   int   = 64   
    epochs_pre:   int   = 50   
    epochs_fine:  int   = 25
    val_split:    float = 0.2
    seed:         int   = 42

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                # Оригинальные 20
                "SBER", "GAZP", "LKOH", "GMKN", "NVTK",
                "ROSN", "TATN", "MGNT", "MTSS", "T",
                "ALRS", "PLZL", "SNGS", "VTBR", "AFLT",
                "MAGN", "NLMK", "CHMF", "PHOR", "CBOM",
                # Новые 15 (ликвидные из IMOEX)
                "YDEX", "MOEX", "IRAO", "PIKK", "RTKM",
                "RUAL", "OZON", "FIVE", "HEAD", "FLOT",
                "SVCB", "AFKS", "BSPB", "VKCO", "POSI",
            ]
        if self.mlp_hidden is None:
            self.mlp_hidden = [128, 64, 32]

    @property
    def effective_profit_thr(self) -> float:
        return self.profit_thr + 2 * self.broker_commission + self.min_net_profit


CFG = MLConfig()
