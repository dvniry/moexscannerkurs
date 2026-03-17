"""Загрузка рыночного и отраслевого контекста + HMM-режим рынка."""
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


def load_context_series(ticker: str) -> pd.DataFrame | None:
    from api.routes.candles import get_client
    client  = get_client()
    symbols = SECTOR_CONTEXT.get(ticker, SECTOR_CONTEXT["__default__"])
    frames  = {}

    for sym in symbols:
        try:
            uid = _get_uid(client, sym)
            if not uid:
                print(f"    [WARN] {sym}: uid не найден")
                continue
            df = client.get_candles_by_uid(
                uid=uid, interval=CFG.interval, days_back=CFG.days_back)
            if df is not None and not df.empty:
                frames[sym] = df['close'].astype(float)
                print(f"    Контекст {sym}: {len(df)} свечей")
            else:
                print(f"    [WARN] {sym}: пустые данные")
        except Exception as e:
            print(f"    [WARN] {sym}: {e}")

    return pd.DataFrame(frames) if frames else None


# ── HMM-режим рынка ───────────────────────────────────────────────

def _hmm_regime(close_series: pd.Series, n_states: int = 3) -> np.ndarray:
    """
    Определяет рыночный режим IMOEX через Gaussian HMM.
    Возвращает one-hot (N, 3): бычий / боковик / медвежий.
    Состояния автоматически сортируются по средней доходности:
      0 = медвежий (lowest mean return)
      1 = боковик  (middle)
      2 = бычий    (highest mean return)
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  [WARN] hmmlearn не установлен: pip install hmmlearn")
        return np.zeros((len(close_series), n_states), dtype=np.float32)

    rets  = close_series.pct_change().fillna(0).values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=500, random_state=42)
    model.fit(rets)
    raw_states = model.predict(rets)

    # Сортируем состояния по средней доходности (стабильные индексы)
    means      = [rets[raw_states == s].mean() for s in range(n_states)]
    order      = np.argsort(means)           # 0=медвеж, 1=боковик, 2=бычий
    remap      = {old: new for new, old in enumerate(order)}
    states     = np.array([remap[s] for s in raw_states])

    one_hot = np.zeros((len(states), n_states), dtype=np.float32)
    one_hot[np.arange(len(states)), states] = 1.0
    return one_hot


def build_context_features(
    ctx: pd.DataFrame,
    date_index: pd.Index,
    ticker: str = None,
    imoex_close: pd.Series = None,   # ← передаём для HMM
) -> np.ndarray | None:

    expected_dim = get_context_dim(ticker) if ticker else None

    # ── Ценовые признаки контекстных инструментов ─────────────
    if ctx is None or ctx.empty:
        price_feats = np.zeros(
            (len(date_index), expected_dim - 3 if expected_dim else 0),
            dtype=np.float32)
    else:
        feats = {}
        for col in ctx.columns:
            c = ctx[col].reindex(date_index).ffill()
            feats[f'{col}_ret5']  = c.pct_change(5)
            feats[f'{col}_ret20'] = c.pct_change(20)
            feats[f'{col}_vol20'] = c.pct_change().rolling(20).std()
        feat_df     = pd.DataFrame(feats, index=date_index).fillna(0.0)
        price_feats = feat_df.values.astype(np.float32)

        if expected_dim and price_feats.shape[1] < expected_dim - 3:
            pad         = np.zeros((price_feats.shape[0],
                                    expected_dim - 3 - price_feats.shape[1]),
                                   dtype=np.float32)
            price_feats = np.concatenate([price_feats, pad], axis=1)

    # ── HMM-режим по IMOEX ────────────────────────────────────
    if imoex_close is not None:
        imoex_aligned = imoex_close.reindex(date_index).ffill().bfill()
        hmm_raw       = _hmm_regime(imoex_aligned, n_states=3)  # (N_imoex, 3)
        # hmm_raw уже выровнен по date_index через imoex_aligned
        hmm_feats     = hmm_raw
    else:
        hmm_feats = np.zeros((len(date_index), 3), dtype=np.float32)

    arr = np.concatenate([price_feats, hmm_feats], axis=1)   # (N, ctx_dim+3)

    # Z-score нормализация (только ценовые, HMM и так 0/1)
    n_price = price_feats.shape[1]
    mu      = arr[:, :n_price].mean(axis=0, keepdims=True)
    sig     = arr[:, :n_price].std(axis=0,  keepdims=True) + 1e-9
    arr[:, :n_price] = (arr[:, :n_price] - mu) / sig

    return arr.astype(np.float32)


def get_context_dim(ticker: str) -> int:
    """3 признака × n_symbols + 3 HMM-состояния."""
    n = len(SECTOR_CONTEXT.get(ticker, SECTOR_CONTEXT["__default__"]))
    return n * 3 + 3    # было n*3, теперь +3 для HMM
