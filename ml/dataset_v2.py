"""Мультимасштабный датасет с рыночным и отраслевым контекстом."""
import sys, os, hashlib

_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.config         import CFG, SCALES
from ml.candle_render  import render_candles
from ml.dataset        import add_indicators, label_candles, load_imoex
from ml.context_loader import load_context_series, build_context_features

def _cache_path(ticker: str, W: int) -> str:
    return f"ml/cache/imgs_{ticker}_{W}.npy"

def build_multiscale_dataset(
    df, imoex=None, context=None,
    ticker: str = "unknown",
) -> Tuple[Dict[int, np.ndarray], np.ndarray, Optional[np.ndarray]]:

    df     = add_indicators(df.copy(), imoex).dropna()
    labels = label_candles(df)
    W_max  = max(SCALES)

    # Проверить кэш
    cache_ok = all(os.path.exists(_cache_path(ticker, W)) for W in SCALES)
    label_path = f"ml/cache/labels_{ticker}.npy"

    if cache_ok and os.path.exists(label_path):
        print(f"    Кэш найден для {ticker}")
        imgs = {W: np.load(_cache_path(ticker, W)) for W in SCALES}
        y    = np.load(label_path)
        # ctx всё равно считаем заново (быстро)
        ctx_list = []
        if context is not None:
            for i in range(W_max, len(df) - CFG.future_bars):
                ctx_list.append(context[i])
        ctx = np.array(ctx_list, dtype=np.float32) if ctx_list else None
        return imgs, y, ctx

    # Строим с нуля
    os.makedirs("ml/cache", exist_ok=True)
    scale_data = {W: [] for W in SCALES}
    ctx_list   = []
    y_list     = []

    total = len(df) - W_max - CFG.future_bars
    for i, idx in enumerate(range(W_max, len(df) - CFG.future_bars)):
        if i % 100 == 0:
            print(f"    {ticker}: рендер {i}/{total}", end='\r')
        for W in SCALES:
            window = df.iloc[idx - W : idx]
            scale_data[W].append(render_candles(window))
        if context is not None:
            ctx_list.append(context[idx])
        y_list.append(labels.iloc[idx])

    print()
    imgs = {W: np.array(scale_data[W], dtype=np.float32) for W in SCALES}
    y    = np.array(y_list, dtype=np.int64)
    ctx  = np.array(ctx_list, dtype=np.float32) if ctx_list else None

    # Сохранить кэш
    for W in SCALES:
        np.save(_cache_path(ticker, W), imgs[W])
    np.save(label_path, y)

    return imgs, y, ctx


def build_full_multiscale_dataset():
    """Собрать полный датасет по всем тикерам с контекстом."""
    from api.routes.candles import get_client
    client = get_client()

    print("  Загружаем IMOEX...")
    imoex = load_imoex()

    all_scales  = {W: [] for W in SCALES}
    all_y       = []
    all_ctx     = []
    ctx_dim     = None

    for ticker in CFG.tickers:
        print(f"  Загружаем {ticker}...")
        try:
            figi = client.find_figi(ticker)
            if not figi:
                print(f"  {ticker}: не найден")
                continue

            df = client.get_candles(figi=figi, interval=CFG.interval,
                                    days_back=CFG.days_back)
            if df is None or df.empty:
                continue

            # Загрузить отраслевой контекст
            print(f"    Контекст для {ticker}...")
            ctx_series = load_context_series(ticker)

            # Передаём ticker чтобы context_loader знал ожидаемую размерность
            ctx_feats = build_context_features(
                ctx_series, df.index, ticker=ticker
            ) if ctx_series is not None else None

            imgs, y, ctx = build_multiscale_dataset(df, imoex, ctx_feats, ticker=ticker)

            if len(y) == 0:
                continue

            for W in SCALES:
                all_scales[W].append(imgs[W])
            all_y.append(y)

            if ctx is not None:
                # Определяем эталонный ctx_dim по первому успешному тикеру
                if ctx_dim is None:
                    ctx_dim = ctx.shape[1]

                # Выравниваем размерность: паддинг или обрезка
                if ctx.shape[1] < ctx_dim:
                    pad = np.zeros((ctx.shape[0], ctx_dim - ctx.shape[1]),
                                   dtype=np.float32)
                    ctx = np.concatenate([ctx, pad], axis=1)
                elif ctx.shape[1] > ctx_dim:
                    ctx = ctx[:, :ctx_dim]

                all_ctx.append(ctx)
            elif ctx_dim is not None:
                # ctx не получен, но другие тикеры имеют контекст → нули
                all_ctx.append(
                    np.zeros((len(y), ctx_dim), dtype=np.float32)
                )

            print(f"  {ticker}: {len(y)} сэмплов")

        except Exception as e:
            print(f"  {ticker}: ошибка — {e}")
            import traceback; traceback.print_exc()

    if not all_y:
        raise RuntimeError("Не удалось загрузить данные.")

    imgs_out = {W: np.concatenate(all_scales[W]) for W in SCALES}
    y_out    = np.concatenate(all_y)
    ctx_out  = np.concatenate(all_ctx) if all_ctx else None

    return imgs_out, y_out, ctx_out, ctx_dim


