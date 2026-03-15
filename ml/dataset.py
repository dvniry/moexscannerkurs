"""Загрузка данных MOEX и формирование датасета для MLP."""
import sys, os

_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert

import numpy as np
import pandas as pd
from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.config import CFG


# ── Технические индикаторы ────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float)

    df['ema9']     = c.ewm(span=9).mean()
    df['ema21']    = c.ewm(span=21).mean()
    df['ema50']    = c.ewm(span=50).mean()
    df['macd']     = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df['macd_sig'] = df['macd'].ewm(span=9).mean()

    delta          = c.diff()
    gain           = delta.clip(lower=0).rolling(14).mean()
    loss           = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi']      = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    mid            = c.rolling(20).mean()
    std            = c.rolling(20).std()
    df['bb_upper'] = mid + 2 * std
    df['bb_lower'] = mid - 2 * std
    df['bb_pct']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)

    tr             = pd.concat([h - l, (h - c.shift()).abs(),
                                (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr']      = tr.rolling(14).mean()
    df['vol_ma']   = v.rolling(20).mean()
    df['vol_ratio']= v / (df['vol_ma'] + 1e-9)
    df['roc5']     = c.pct_change(5)
    df['roc10']    = c.pct_change(10)

    low14          = l.rolling(14).min()
    high14         = h.rolling(14).max()
    df['stoch']    = (c - low14) / (high14 - low14 + 1e-9) * 100

    return df


INDICATOR_COLS = [
    'ema9', 'ema21', 'ema50', 'macd', 'macd_sig',
    'rsi', 'bb_upper', 'bb_lower', 'bb_pct', 'atr',
    'vol_ratio', 'roc5', 'roc10', 'stoch', 'close',
]


# ── Разметка ──────────────────────────────────────────────

def label_candles(df: pd.DataFrame) -> pd.Series:
    close   = df['close'].astype(float)
    future  = close.shift(-CFG.future_bars)
    returns = (future - close) / close
    labels  = np.ones(len(df), dtype=int)
    labels[returns >  CFG.profit_thr] = 0   # BUY
    labels[returns <  CFG.loss_thr  ] = 2   # SELL
    return pd.Series(labels, index=df.index)


# ── Нормализация ──────────────────────────────────────────

def normalize_window(window: pd.DataFrame) -> np.ndarray:
    arr  = window[INDICATOR_COLS].values.astype(float)
    min_ = arr.min(axis=0, keepdims=True)
    max_ = arr.max(axis=0, keepdims=True)
    return (arr - min_) / (max_ - min_ + 1e-9)


# ── Сборка датасета (MLP) ─────────────────────────────────

def build_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df     = add_indicators(df.copy()).dropna()
    labels = label_candles(df)
    W      = CFG.window

    X_flat, X_img, y = [], [], []
    for i in range(W, len(df) - CFG.future_bars):
        window = df.iloc[i - W : i]
        norm   = normalize_window(window)
        X_flat.append(norm.flatten())
        X_img.append(norm)
        y.append(labels.iloc[i])

    return (
        np.array(X_flat, dtype=np.float32),
        np.array(X_img,  dtype=np.float32),
        np.array(y,      dtype=np.int64),
    )


# ── Загрузка данных ───────────────────────────────────────

def load_ticker_data(ticker: str) -> pd.DataFrame:
    from api.routes.candles import get_client
    client = get_client()
    figi   = client.find_figi(ticker)
    if not figi:
        raise ValueError(f"Тикер '{ticker}' не найден")
    df = client.get_candles(figi=figi, interval=CFG.interval, days_back=CFG.days_back)
    if df is None or df.empty:
        raise ValueError(f"Нет данных для {ticker}")
    return df


def build_full_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_flat, all_img, all_y = [], [], []
    for ticker in CFG.tickers:
        print(f"  Загружаем {ticker}...")
        try:
            df = load_ticker_data(ticker)
            X_flat, X_img, y = build_dataset(df)
            if len(y) == 0:
                continue
            all_flat.append(X_flat)
            all_img.append(X_img)
            all_y.append(y)
            print(f"  {ticker}: {len(y)} сэмплов")
        except Exception as e:
            print(f"  {ticker}: ошибка — {e}")

    if not all_flat:
        raise RuntimeError("Не удалось загрузить ни одного тикера.")

    return (
        np.concatenate(all_flat),
        np.concatenate(all_img),
        np.concatenate(all_y),
    )


def class_distribution(y: np.ndarray):
    total = len(y)
    for label, name in {0: "BUY", 1: "HOLD", 2: "SELL"}.items():
        count = int((y == label).sum())
        print(f"  {name:4s}: {count:5d} ({count/total*100:.1f}%)")
