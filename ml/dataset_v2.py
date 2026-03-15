"""Мультимасштабный датасет — рендер свечей в 4 масштабах."""
import sys, os

_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert

import numpy as np
import pandas as pd
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.config        import CFG, SCALES
from ml.candle_render import render_candles
from ml.dataset       import add_indicators, label_candles


def build_multiscale_dataset(
    df: pd.DataFrame,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Скользящее окно stride=1 для каждого масштаба.
    Одни и те же свечи попадают в разные масштабы — расширение датасета.
    """
    df     = add_indicators(df.copy()).dropna()
    labels = label_candles(df)
    W_max  = max(SCALES)

    scale_data = {W: [] for W in SCALES}
    y_list     = []

    for i in range(W_max, len(df) - CFG.future_bars):
        for W in SCALES:
            window = df.iloc[i - W : i]
            scale_data[W].append(render_candles(window))
        y_list.append(labels.iloc[i])

    return (
        {W: np.array(scale_data[W], dtype=np.float32) for W in SCALES},
        np.array(y_list, dtype=np.int64),
    )


def build_full_multiscale_dataset() -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    from api.routes.candles import get_client
    client     = get_client()
    all_scales = {W: [] for W in SCALES}
    all_y      = []

    for ticker in CFG.tickers:
        print(f"  Загружаем {ticker}...")
        try:
            figi = client.find_figi(ticker)
            if not figi:
                print(f"  {ticker}: не найден")
                continue
            df = client.get_candles(
                figi=figi, interval=CFG.interval, days_back=CFG.days_back)
            if df is None or df.empty:
                continue
            imgs, y = build_multiscale_dataset(df)
            if len(y) == 0:
                continue
            for W in SCALES:
                all_scales[W].append(imgs[W])
            all_y.append(y)
            print(f"  {ticker}: {len(y)} сэмплов")
        except Exception as e:
            print(f"  {ticker}: ошибка — {e}")

    if not all_y:
        raise RuntimeError("Не удалось загрузить данные.")

    return (
        {W: np.concatenate(all_scales[W]) for W in SCALES},
        np.concatenate(all_y),
    )
