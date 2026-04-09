# ml/context_loader.py
"""Контекст v3.1: расширенный набор признаков (21 dim) + честный HMM.

Было:  3 feat × 2 symbols + 3 HMM = 9 dim
Стало: 7 feat × 2 symbols + 3 HMM + 4 market = 21 dim

Новые признаки:
  Per-symbol (×2):
    ret5, ret20, vol20           (было)
    + ret1:   однодневный return (моментум)
    + ret60:  квартальный return (тренд)
    + zscore: (close - SMA50) / std50 (mean reversion)
    + rsi14:  RSI нормализованный (0..1)

  Market-wide (×1):
    breadth:   доля дней с положительным IMOEX return за 20 дней
    trend:     знак SMA50 slope (наклон тренда)
    vol_regime: текущая vol20 / среднеисторическая vol20
    momentum:  IMOEX ret20 / vol20 (risk-adjusted momentum)
"""
import numpy as np
import pandas as pd
from ml.config import CFG, SECTOR_CONTEXT

_uid_cache: dict = {}


def _get_uid(client, ticker: str) -> str | None:
    if ticker in _uid_cache:
        return _uid_cache[ticker]
    uid = client.find_indicative_uid(ticker)
    if uid:
        _uid_cache[ticker] = uid
    return uid


def _load_indicative_chunked(client, sym: str) -> 'pd.DataFrame | None':
    """Загружает индикативный инструмент чанками по 365 дней."""
    import time as _time
    from datetime import timedelta
    from t_tech.invest import Client, CandleInterval
    from t_tech.invest.utils import now as _now

    uid = _get_uid(client, sym)
    if not uid:
        print(f"    [WARN] {sym}: uid не найден")
        return None

    TARGET     = "invest-public-api.tbank.ru:443"
    CHUNK_DAYS = 365
    all_frames = []
    end        = _now()
    remaining  = CFG.days_back
    chunk_num  = 0

    while remaining > 0:
        chunk = min(remaining, CHUNK_DAYS)
        start = end - timedelta(days=chunk)
        try:
            with Client(client.token, target=TARGET) as api:
                candles = api.market_data.get_candles(
                    instrument_id=uid,
                    from_=start,
                    to=end,
                    interval=CandleInterval.CANDLE_INTERVAL_DAY,
                ).candles
            if candles:
                df_chunk = pd.DataFrame({
                    "time":  [c.time            for c in candles],
                    "close": [client._q(c.close) for c in candles],
                }).set_index("time")
                all_frames.append(df_chunk)
            chunk_num += 1
        except Exception as e:
            print(f"    [WARN] {sym} chunk {start.date()}→{end.date()}: {e}")

        end        = start
        remaining -= chunk
        _time.sleep(0.1)

    if not all_frames:
        return None

    result = pd.concat(all_frames).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    print(f"    Контекст {sym}: {len(result)} свечей ({chunk_num} чанков)")
    return result['close'].rename(sym)


def load_context_series(ticker: str) -> 'pd.DataFrame | None':
    from api.routes.candles import get_client
    client  = get_client()
    symbols = SECTOR_CONTEXT.get(ticker, SECTOR_CONTEXT["__default__"])
    frames  = {}

    for sym in symbols:
        try:
            series = _load_indicative_chunked(client, sym)
            if series is not None:
                frames[sym] = series
        except Exception as e:
            print(f"    [WARN] {sym}: {e}")

    return pd.DataFrame(frames) if frames else None


# ── HMM-режим рынка (без look-ahead) ──────────────────────────

def _hmm_regime(close_series: pd.Series,
                n_states: int = 3,
                train_end_idx: int = None) -> np.ndarray:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  [WARN] hmmlearn не установлен")
        return np.zeros((len(close_series), n_states), dtype=np.float32)

    rets = close_series.pct_change().fillna(0).values.reshape(-1, 1)
    N = len(rets)

    if train_end_idx is None:
        train_end_idx = int(N * 0.70)
    train_end_idx = max(train_end_idx, 100)
    train_end_idx = min(train_end_idx, N)

    rets_train = rets[:train_end_idx]

    model = GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=500, random_state=42)
    model.fit(rets_train)

    train_states = model.predict(rets_train)
    means_train = [
        rets_train[train_states == s].mean()
        if (train_states == s).sum() > 0 else 0.0
        for s in range(n_states)
    ]
    order = np.argsort(means_train)
    remap = {old: new for new, old in enumerate(order)}

    all_states = _forward_only_decode(model, rets, n_states)
    remapped = np.array([remap[s] for s in all_states])

    one_hot = np.zeros((N, n_states), dtype=np.float32)
    one_hot[np.arange(N), remapped] = 1.0

    for s_name, s_id in [("bear", 0), ("side", 1), ("bull", 2)]:
        pct = (remapped == s_id).sum() / N * 100
        print(f"  HMM '{s_name}': {pct:.1f}%")

    return one_hot


def _forward_only_decode(model, observations, n_states):
    from scipy.stats import multivariate_normal

    N = len(observations)
    states = np.zeros(N, dtype=np.int32)

    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat  = np.log(model.transmat_ + 1e-300)

    log_emission = np.zeros((N, n_states))
    for s in range(n_states):
        rv = multivariate_normal(
            mean=model.means_[s].flatten(),
            cov=model.covars_[s].squeeze())
        log_emission[:, s] = rv.logpdf(observations.reshape(N, -1))

    alpha = np.full((N, n_states), -np.inf)
    alpha[0] = log_startprob + log_emission[0]
    states[0] = np.argmax(alpha[0])

    for t in range(1, N):
        for s in range(n_states):
            alpha[t, s] = log_emission[t, s] + _logsumexp(
                alpha[t-1] + log_transmat[:, s])
        states[t] = np.argmax(alpha[t])

    return states


def _logsumexp(arr):
    m = arr.max()
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.sum(np.exp(arr - m)))


# ── Расширенные признаки ──────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return rs / (1 + rs)  # уже нормализован 0..1


def _build_symbol_features(series: pd.Series,
                            date_index: pd.Index) -> pd.DataFrame:
    """7 признаков для одного контекстного символа."""
    c = series.reindex(date_index).ffill()
    name = series.name if hasattr(series, 'name') else 'sym'

    sma50 = c.rolling(50, min_periods=10).mean()
    std50 = c.rolling(50, min_periods=10).std()

    feats = pd.DataFrame({
        f'{name}_ret1':   c.pct_change(1),
        f'{name}_ret5':   c.pct_change(5),
        f'{name}_ret20':  c.pct_change(20),
        f'{name}_ret60':  c.pct_change(60),
        f'{name}_vol20':  c.pct_change().rolling(20).std(),
        f'{name}_zscore': (c - sma50) / (std50 + 1e-9),
        f'{name}_rsi14':  _compute_rsi(c, 14),
    }, index=date_index)

    return feats


def _build_market_features(imoex_close: pd.Series,
                            date_index: pd.Index) -> pd.DataFrame:
    """4 рыночных признака (market-wide)."""
    c = imoex_close.reindex(date_index).ffill().bfill()
    ret1 = c.pct_change()
    ret20 = c.pct_change(20)
    vol20 = ret1.rolling(20).std()
    sma50 = c.rolling(50, min_periods=10).mean()

    # Breadth: доля положительных дней за 20 дней
    pos_days = (ret1 > 0).astype(float).rolling(20).mean()

    # Slope SMA50: направление тренда
    sma_slope = sma50.pct_change(5)

    # Vol regime: текущая vol / средняя vol
    vol_mean = vol20.expanding(min_periods=50).mean()
    vol_regime = vol20 / (vol_mean + 1e-9)

    # Risk-adjusted momentum
    momentum = ret20 / (vol20 + 1e-9)

    return pd.DataFrame({
        'mkt_breadth':    pos_days,
        'mkt_trend':      sma_slope,
        'mkt_vol_regime': vol_regime,
        'mkt_momentum':   momentum,
    }, index=date_index)


def build_context_features(
    ctx: pd.DataFrame,
    date_index: pd.Index,
    ticker: str = None,
    imoex_close: pd.Series = None,
    train_end_idx: int = None,
) -> np.ndarray | None:

    expected_dim = get_context_dim(ticker) if ticker else None
    n_symbols = len(SECTOR_CONTEXT.get(ticker, SECTOR_CONTEXT["__default__"]))
    features_per_symbol = 7  # НОВОЕ: было 3

    # ── Ценовые признаки (7 × n_symbols) ─────────────────────
    if ctx is None or ctx.empty:
        price_feats = np.zeros(
            (len(date_index), n_symbols * features_per_symbol),
            dtype=np.float32)
    else:
        feat_frames = []
        for col in ctx.columns:
            feat_frames.append(
                _build_symbol_features(ctx[col], date_index))
        feat_df = pd.concat(feat_frames, axis=1).fillna(0.0)
        price_feats = feat_df.values.astype(np.float32)

        # Паддинг если символов меньше ожидаемого
        expected_price = n_symbols * features_per_symbol
        if price_feats.shape[1] < expected_price:
            pad = np.zeros(
                (price_feats.shape[0], expected_price - price_feats.shape[1]),
                dtype=np.float32)
            price_feats = np.concatenate([price_feats, pad], axis=1)

    # ── Market-wide фичи (4) ─────────────────────────────────
    if imoex_close is not None:
        imoex_aligned = imoex_close.reindex(date_index).ffill().bfill()
        mkt_df = _build_market_features(imoex_aligned, date_index)
        mkt_feats = mkt_df.fillna(0.0).values.astype(np.float32)
    else:
        mkt_feats = np.zeros((len(date_index), 4), dtype=np.float32)

    # ── HMM (3) ───────────────────────────────────────────────
    if imoex_close is not None:
        imoex_aligned = imoex_close.reindex(date_index).ffill().bfill()
        hmm_train_end = train_end_idx or int(len(imoex_aligned) * 0.70)
        hmm_feats = _hmm_regime(imoex_aligned, n_states=3,
                                 train_end_idx=hmm_train_end)
    else:
        hmm_feats = np.zeros((len(date_index), 3), dtype=np.float32)

    # ── Склеиваем ─────────────────────────────────────────────
    # price_feats: (N, 7*n_sym)
    # mkt_feats:   (N, 4)
    # hmm_feats:   (N, 3)
    arr = np.concatenate([price_feats, mkt_feats, hmm_feats], axis=1)

    # Z-score нормализация (только числовые, HMM оставляем)
    n_numeric = price_feats.shape[1] + mkt_feats.shape[1]
    norm_end = train_end_idx if train_end_idx else int(len(arr) * 0.70)
    if n_numeric > 0 and norm_end > 10:
        mu  = arr[:norm_end, :n_numeric].mean(axis=0, keepdims=True)
        sig = arr[:norm_end, :n_numeric].std(axis=0, keepdims=True) + 1e-9
        arr[:, :n_numeric] = (arr[:, :n_numeric] - mu) / sig

    # ── Clamp экстремальных значений ─────────────────────────
    arr[:, :n_numeric] = np.clip(arr[:, :n_numeric], -5.0, 5.0)

    # ── GUARD: заменяем любые NaN/Inf нулями ─────────────────
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr.astype(np.float32)


def get_context_dim(ticker: str) -> int:
    """7 признаков × n_symbols + 4 market + 3 HMM."""
    n = len(SECTOR_CONTEXT.get(ticker, SECTOR_CONTEXT["__default__"]))
    return n * 7 + 4 + 3   # было n*3 + 3 = 9, стало n*7 + 7 = 21