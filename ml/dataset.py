"""MLP датасет — индикаторы + RS + календарь."""
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

def add_indicators(df: pd.DataFrame,
                   imoex: pd.DataFrame = None) -> pd.DataFrame:
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float)

    # Трендовые
    df['ema9']     = c.ewm(span=9).mean()
    df['ema21']    = c.ewm(span=21).mean()
    df['ema50']    = c.ewm(span=50).mean()
    df['macd']     = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df['macd_sig'] = df['macd'].ewm(span=9).mean()

    # RSI
    delta          = c.diff()
    gain           = delta.clip(lower=0).rolling(14).mean()
    loss_s         = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi']      = 100 - (100 / (1 + gain / loss_s.replace(0, 1e-9)))

    # Bollinger
    mid            = c.rolling(20).mean()
    std            = c.rolling(20).std()
    df['bb_upper'] = mid + 2 * std
    df['bb_lower'] = mid - 2 * std
    df['bb_pct']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)

    # ATR
    tr             = pd.concat([h - l, (h - c.shift()).abs(),
                                (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr']      = tr.rolling(14).mean()

    # Объём
    df['vol_ma']   = v.rolling(20).mean()
    df['vol_ratio']= v / (df['vol_ma'] + 1e-9)

    # Моментум
    df['roc5']     = c.pct_change(5)
    df['roc10']    = c.pct_change(10)

    # Stochastic
    low14          = l.rolling(14).min()
    high14         = h.rolling(14).max()
    df['stoch']    = (c - low14) / (high14 - low14 + 1e-9) * 100

    # ── Относительная сила vs IMOEX ───────────────────────
    if imoex is not None:
        idx_c = imoex['close'].astype(float).reindex(df.index, method='ffill')
        ticker_ret5  = c.pct_change(5)
        ticker_ret20 = c.pct_change(20)
        imoex_ret5   = idx_c.pct_change(5)
        imoex_ret20  = idx_c.pct_change(20)
        df['rs_5d']  = ticker_ret5  / (imoex_ret5.abs()  + 1e-9)
        df['rs_20d'] = ticker_ret20 / (imoex_ret20.abs() + 1e-9)
        df['imoex_ret5']  = imoex_ret5
        df['imoex_ret20'] = imoex_ret20
        df['imoex_vol20'] = idx_c.pct_change().rolling(20).std()
    else:
        df['rs_5d']       = 0.0
        df['rs_20d']      = 0.0
        df['imoex_ret5']  = 0.0
        df['imoex_ret20'] = 0.0
        df['imoex_vol20'] = 0.0

    # ── Календарные признаки ──────────────────────────────
    if hasattr(df.index, 'dayofweek'):
        df['day_of_week'] = df.index.dayofweek / 4.0
        df['month']       = df.index.month / 12.0
        df['is_monday']   = (df.index.dayofweek == 0).astype(float)
        df['is_friday']   = (df.index.dayofweek == 4).astype(float)
    else:
        df['day_of_week'] = 0.5
        df['month']       = 0.5
        df['is_monday']   = 0.0
        df['is_friday']   = 0.0

    return df


# 15 базовых + 5 RS/IMOEX + 4 календарных = 24 признака
INDICATOR_COLS = [
    'ema9', 'ema21', 'ema50', 'macd', 'macd_sig',
    'rsi', 'bb_upper', 'bb_lower', 'bb_pct', 'atr',
    'vol_ratio', 'roc5', 'roc10', 'stoch', 'close',
    'rs_5d', 'rs_20d',
    'imoex_ret5', 'imoex_ret20', 'imoex_vol20',
    'day_of_week', 'month', 'is_monday', 'is_friday',
]


# ── Разметка с учётом комиссии ────────────────────────────

def label_candles(df: pd.DataFrame) -> pd.Series:
    """
    Экономически обоснованная разметка.
    HOLD = движение не окупает комиссию брокера.
    """
    close    = df['close'].astype(float)
    future   = close.shift(-CFG.future_bars)
    returns  = (future - close) / close
    # Вычесть 2 комиссии (покупка + продажа)
    net_ret  = returns - 2 * CFG.broker_commission
    thr      = CFG.effective_profit_thr

    labels   = np.ones(len(df), dtype=int)   # HOLD
    labels[net_ret >  thr] = 0               # BUY
    labels[net_ret < -thr] = 2               # SELL
    return pd.Series(labels, index=df.index)


# ── Нормализация окна ─────────────────────────────────────

def normalize_window(window: pd.DataFrame) -> np.ndarray:
    arr  = window[INDICATOR_COLS].values.astype(float)
    min_ = arr.min(axis=0, keepdims=True)
    max_ = arr.max(axis=0, keepdims=True)
    return (arr - min_) / (max_ - min_ + 1e-9)


# ── Сборка датасета (MLP) ─────────────────────────────────

def build_dataset(df: pd.DataFrame,
                  imoex: pd.DataFrame = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df     = add_indicators(df.copy(), imoex).dropna()
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
    df = client.get_candles(figi=figi, interval=CFG.interval,
                            days_back=CFG.days_back)
    if df is None or df.empty:
        raise ValueError(f"Нет данных для {ticker}")
    return df


def load_imoex() -> pd.DataFrame | None:
    from api.routes.candles import get_client
    client = get_client()
    try:
        uid = client.find_indicative_uid("IMOEX")
        if uid:
            df = client.get_candles_by_uid(
                uid=uid, interval=CFG.interval, days_back=CFG.days_back)
            if df is not None and not df.empty:
                print(f"  IMOEX загружен: {len(df)} свечей")
                return df
    except Exception as e:
        print(f"  [WARN] IMOEX: {e}")
    print("  [WARN] IMOEX не загружен — RS признаки будут нулями")
    return None




def build_full_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("  Загружаем IMOEX...")
    imoex    = load_imoex()

    all_flat, all_img, all_y = [], [], []
    for ticker in CFG.tickers:
        print(f"  Загружаем {ticker}...")
        try:
            df = load_ticker_data(ticker)
            X_flat, X_img, y = build_dataset(df, imoex)
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
