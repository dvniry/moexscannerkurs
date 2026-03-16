"""Загрузка рыночного и отраслевого контекста через Indicatives API."""
import numpy as np
import pandas as pd
from ml.config import CFG, SECTOR_CONTEXT

# Кэш uid: ticker → uid (заполняется один раз)
_uid_cache: dict = {}


def _get_uid(client, ticker: str) -> str | None:
    """Получить UID индикатива с кэшированием."""
    if ticker in _uid_cache:
        return _uid_cache[ticker]
    uid = client.find_indicative_uid(ticker)
    if uid:
        _uid_cache[ticker] = uid
    return uid


def load_context_series(ticker: str) -> pd.DataFrame | None:
    """
    Загрузить контекстные инструменты для тикера.
    Все инструменты в SECTOR_CONTEXT — индикативы (IMOEX, LCOc1, Nl и т.д.)
    Возвращает DataFrame с колонками close по каждому инструменту.
    """
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
                uid=uid,
                interval=CFG.interval,
                days_back=CFG.days_back,
            )
            if df is not None and not df.empty:
                frames[sym] = df['close'].astype(float)
                print(f"    Контекст {sym}: {len(df)} свечей")
            else:
                print(f"    [WARN] {sym}: пустые данные")
        except Exception as e:
            print(f"    [WARN] {sym}: {e}")

    return pd.DataFrame(frames) if frames else None


def build_context_features(
    ctx:        pd.DataFrame,
    date_index: pd.Index,
    ticker:     str = None,          # <-- добавить аргумент
) -> np.ndarray | None:
    expected_dim = get_context_dim(ticker) if ticker else None

    if ctx is None or ctx.empty:
        if expected_dim:
            return np.zeros((len(date_index), expected_dim), dtype=np.float32)
        return None

    feats = {}
    for col in ctx.columns:
        c = ctx[col].reindex(date_index, method='ffill')
        feats[f'{col}_ret5']  = c.pct_change(5)
        feats[f'{col}_ret20'] = c.pct_change(20)
        feats[f'{col}_vol20'] = c.pct_change().rolling(20).std()

    feat_df = pd.DataFrame(feats, index=date_index).fillna(0.0)
    arr = feat_df.values.astype(float)

    # Паддинг нулями если часть символов не загрузилась
    if expected_dim and arr.shape[1] < expected_dim:
        pad = np.zeros((arr.shape[0], expected_dim - arr.shape[1]), dtype=float)
        arr = np.concatenate([arr, pad], axis=1)

    mu  = arr.mean(axis=0, keepdims=True)
    sig = arr.std(axis=0, keepdims=True) + 1e-9
    return ((arr - mu) / sig).astype(np.float32)



def get_context_dim(ticker: str) -> int:
    """3 признака × n_symbols."""
    n = len(SECTOR_CONTEXT.get(ticker, SECTOR_CONTEXT["__default__"]))
    return n * 3   # всегда 6 (2 символа × 3 признака)
