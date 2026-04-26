"""FastAPI сервер для интеграции с MT5.
Запуск: python ml/mt5_server.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json

from ml.config import CFG, SCALES
from ml.dataset_v3 import INDICATOR_COLS, add_indicators
from ml.multiscale_cnn_v3 import MultiScaleHybridV3, _make_loader_v3
from ml.candle_render_v2 import render_candles as _render_candles_orig
import pandas as pd

app = FastAPI(title="ML Trading Signal Server")

# ── Глобальное состояние ─────────────────────────────────────
MODELS      = []   # список загруженных моделей ансамбля
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CTX_DIM     = 0
N_IND       = len(INDICATOR_COLS)
USE_HOURLY  = True   # для инференса в реальном времени отключаем


def load_ensemble(
    save_dir:   str = 'ml/ensemble',
    seeds:      list = [42, 123, 7],
    ctx_dim:    int = 0,
    use_hourly: bool = True,   # ← добавили параметр
):
    """Загружает все модели ансамбля."""
    global MODELS, CTX_DIM, USE_HOURLY
    CTX_DIM    = ctx_dim
    USE_HOURLY = use_hourly
    MODELS     = []

    for seed in seeds:
        path = os.path.join(save_dir, f'model_seed{seed}.pt')
        if not os.path.exists(path):
            print(f'  [WARN] Модель не найдена: {path}')
            continue

        # Загружаем checkpoint чтобы определить архитектуру
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        
        # Определяем n_streams из checkpoint автоматически
        # vsn.grn_select.fc1.weight имеет shape [n_streams, n_streams*TRUNK_OUT]
        vsn_key = 'vsn.grn_select.fc1.weight'
        if vsn_key in ckpt:
            n_streams_ckpt = ckpt[vsn_key].shape[0]
            print(f'  Checkpoint n_streams={n_streams_ckpt}')
        
        # Определяем use_hourly из наличия hourly_enc ключей
        has_hourly = any('hourly_enc' in k for k in ckpt.keys())
        
        model = MultiScaleHybridV3(
            ctx_dim          = ctx_dim,
            n_indicator_cols = N_IND,
            future_bars      = CFG.future_bars,
            use_hourly       = has_hourly,   # ← берём из checkpoint
        ).to(DEVICE)

        model.load_state_dict(ckpt, strict=True)
        model.eval()
        MODELS.append({'model': model, 'use_hourly': has_hourly})
        print(f'  Загружена модель seed={seed} '
              f'use_hourly={has_hourly}: {path}')

    print(f'  Ансамбль: {len(MODELS)} моделей на {DEVICE}')


# ── Схемы запроса/ответа ─────────────────────────────────────
class CandleData(BaseModel):
    time:   str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


class PredictRequest(BaseModel):
    ticker:       str
    candles:      List[CandleData]   # последние N дневных свечей (минимум 60)
    atr_ratio:    Optional[float] = None  # если известен


class PredictResponse(BaseModel):
    ticker:       str
    dir_prob:     float        # P(UP) из dir_head [0,1]
    cls_probs:    List[float]  # [p_up, p_flat, p_down]
    ohlc_pred:    List[float]  # [dO%, dH%, dL%, dC%] в % от цены
    signal:       float        # композитный сигнал [-1, 1]
    confidence:   float        # уверенность (max cls_prob)
    direction:    str          # "BUY" / "HOLD" / "SELL"
    atr_ratio:    float        # использованный ATR/close


# ── Препроцессинг ─────────────────────────────────────────────
def _candles_to_df(candles: List[CandleData]) -> pd.DataFrame:
    """Конвертирует список свечей в DataFrame."""
    records = []
    for c in candles:
        records.append({
            'time':   pd.Timestamp(c.time),
            'open':   c.open,
            'high':   c.high,
            'low':    c.low,
            'close':  c.close,
            'volume': c.volume,
        })
    df = pd.DataFrame(records).set_index('time')
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _compute_atr_ratio(df: pd.DataFrame, period: int = 14) -> float:
    """Вычисляет ATR(14)/close для последнего бара."""
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values

    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(
            h[i] - l[i],
            abs(h[i] - c[i-1]),
            abs(l[i] - c[i-1]),
        )

    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
    last_close = max(c[-1], 1e-9)
    return float(atr[-1] / last_close)


def _build_imgs(df: pd.DataFrame) -> dict:
    """Строит рендер свечей для каждого масштаба."""
    from ml.dataset_v3 import render_candles, _hwc_to_cw

    imgs = {}
    for W in SCALES:
        if len(df) < W:
            # Паддинг если данных меньше чем нужно
            pad = pd.DataFrame(
                [df.iloc[0].to_dict()] * (W - len(df)),
                columns=df.columns,
            )
            df_w = pd.concat([pad, df]).tail(W)
        else:
            df_w = df.tail(W)

        img = _hwc_to_cw(render_candles(df_w))  # [C, W]
        imgs[W] = torch.tensor(img).float().unsqueeze(0).to(DEVICE)  # [1, C, W]

    return imgs


def _build_nums(df: pd.DataFrame) -> dict:
    """Строит числовые признаки (индикаторы)."""
    from sklearn.preprocessing import RobustScaler

    df_ind = add_indicators(df.copy()).fillna(0)
    num_arr = df_ind[INDICATOR_COLS].values.astype(np.float32)
    num_arr = np.nan_to_num(num_arr, nan=0., posinf=5., neginf=-5.)

    # Нормализация на доступных данных
    scaler = RobustScaler()
    scaler.fit(num_arr)
    num_norm = np.clip(scaler.transform(num_arr), -10., 10.).astype(np.float32)

    nums = {}
    for W in SCALES:
        if len(num_norm) >= W:
            window = num_norm[-W:]
        else:
            pad = np.zeros((W - len(num_norm), num_norm.shape[1]), dtype=np.float32)
            window = np.concatenate([pad, num_norm])

        nums[W] = torch.tensor(window).float().unsqueeze(0).to(DEVICE)  # [1, W, n_ind]

    return nums


# ── Инференс ──────────────────────────────────────────────────
def _run_ensemble(imgs: dict, nums: dict) -> dict:
    """Запускает ансамбль и усредняет предсказания."""
    if not MODELS:
        raise RuntimeError("Модели не загружены!")

    all_cls_logits = []
    all_dir_probs  = []
    all_ohlc_preds = []

    with torch.no_grad():
        for entry in MODELS:
            model      = entry['model']
            use_hourly = entry['use_hourly']

            # Часовые данные — нули если нет реальных
            if use_hourly:
                from ml.hourly_encoder import N_HOURLY_CHANNELS, N_HOURS_PER_DAY, N_INTRADAY_DAYS
                hourly_t = torch.zeros(
                    1, N_INTRADAY_DAYS, N_HOURLY_CHANNELS, N_HOURS_PER_DAY,
                    device=DEVICE)
            else:
                hourly_t = None

            lo, op, _, dir_l = model(
                imgs, nums, ctx=None, hourly=hourly_t)

            all_cls_logits.append(
                torch.softmax(lo, dim=1).cpu().numpy())
            all_dir_probs.append(
                torch.sigmoid(dir_l).cpu().numpy())
            all_ohlc_preds.append(
                op.cpu().numpy())

    cls_probs = np.mean(all_cls_logits, axis=0)[0]
    dir_prob  = float(np.mean(all_dir_probs))
    ohlc_pred = np.mean(all_ohlc_preds, axis=0)[0]

    return {
        'cls_probs': cls_probs,
        'dir_prob':  dir_prob,
        'ohlc_pred': ohlc_pred,
    }

def _composite_signal(dir_prob: float, cls_probs: np.ndarray,
                      ohlc_pred_pct: np.ndarray) -> float:
    """Композитный сигнал [-1, 1]."""
    p_buy  = float(cls_probs[0])
    p_sell = float(cls_probs[2])
    delta_c = float(np.clip(ohlc_pred_pct[3], -0.10, 0.10))

    signal = (
          0.50 * (dir_prob - 0.5) * 2
        + 0.30 * (p_buy - p_sell)
        + 0.20 * np.sign(delta_c) * abs(delta_c) / 0.10
    )
    return float(np.clip(signal, -1., 1.))


# ── Эндпоинты ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":      "ok",
        "models":      len(MODELS),
        "device":      str(DEVICE),
        "scales":      SCALES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not MODELS:
        raise HTTPException(503, "Модели не загружены")

    if len(req.candles) < max(SCALES):
        raise HTTPException(
            400,
            f"Нужно минимум {max(SCALES)} свечей, получено {len(req.candles)}"
        )

    try:
        # DataFrame
        df = _candles_to_df(req.candles)

        # ATR ratio
        atr_ratio = (req.atr_ratio
                     if req.atr_ratio is not None
                     else _compute_atr_ratio(df))
        atr_ratio = float(np.clip(atr_ratio, 0.001, 0.15))

        # Фичи
        imgs = _build_imgs(df)
        nums = _build_nums(df)

        # Инференс
        result    = _run_ensemble(imgs, nums)
        cls_probs = result['cls_probs']
        dir_prob  = result['dir_prob']
        ohlc_norm = result['ohlc_pred'][:4]   # берём только bar1

        # Денормализация OHLC
        norm_factor  = atr_ratio * np.sqrt(CFG.future_bars)
        ohlc_pct     = ohlc_norm * norm_factor   # в долях цены

        # Сигнал
        signal    = _composite_signal(dir_prob, cls_probs, ohlc_pct)
        confidence = float(cls_probs.max())

        # Направление
        cls_idx   = int(cls_probs.argmax())
        direction = ['BUY', 'HOLD', 'SELL'][cls_idx]

        return PredictResponse(
            ticker     = req.ticker,
            dir_prob   = round(dir_prob, 4),
            cls_probs  = [round(float(p), 4) for p in cls_probs],
            ohlc_pred  = [round(float(v * 100), 4) for v in ohlc_pct],
            signal     = round(signal, 4),
            confidence = round(confidence, 4),
            direction  = direction,
            atr_ratio  = round(atr_ratio, 5),
        )

    except Exception as e:
        raise HTTPException(500, f"Ошибка инференса: {str(e)}")


@app.get("/models")
def get_models():
    return {
        "count":    len(MODELS),
        "device":   str(DEVICE),
        "n_ind":    N_IND,
        "scales":   SCALES,
        "ctx_dim":  CTX_DIM,
    }


# ── Запуск ────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir',   default='ml/ensemble')
    parser.add_argument('--seeds',      type=int, nargs='+', default=[42, 123, 7])
    parser.add_argument('--ctx-dim',    type=int, default=21)
    parser.add_argument('--no-hourly',  action='store_true')   # ← добавили
    parser.add_argument('--host',       default='127.0.0.1')
    parser.add_argument('--port',       type=int, default=8765)
    args = parser.parse_args()

    print(f'  Загрузка ансамбля...')
    load_ensemble(
        args.save_dir,
        args.seeds,
        args.ctx_dim,
        use_hourly=not args.no_hourly,
    )

    print(f'  Сервер: http://{args.host}:{args.port}')
    print(f'  Docs:   http://{args.host}:{args.port}/docs')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')