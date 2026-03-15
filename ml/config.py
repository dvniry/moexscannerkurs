"""Гиперпараметры и константы."""
from dataclasses import dataclass, field

SCALES = [5, 10, 20, 30]   # масштабы окон для мультимасштабной CNN

@dataclass
class MLConfig:
    # Данные
    tickers:      list  = None
    interval:     str   = "1d"
    days_back:    int   = 730
    future_bars:  int   = 5
    profit_thr:   float = 0.005    # +0.5% → BUY
    loss_thr:     float = -0.005   # -0.5% → SELL

    # Image encoder
    img_size:     int   = 64       # 64×64 пикселей
    window:       int   = 15       # окно для MLP/старой CNN

    # MLP
    mlp_hidden:   list  = None
    mlp_dropout:  float = 0.3
    mlp_lr:       float = 1e-3

    # CNN / MultiScale
    cnn_backbone:    str   = "resnet18"
    cnn_lr:          float = 1e-4
    cnn_finetune_lr: float = 1e-5
    scale_dim:       int   = 64    # размерность энкодера каждого масштаба
    lstm_hidden:     int   = 128

    # Обучение
    batch_size:   int   = 64
    epochs_pre:   int   = 30
    epochs_fine:  int   = 10
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
            self.mlp_hidden = [64, 32]

CFG = MLConfig()
