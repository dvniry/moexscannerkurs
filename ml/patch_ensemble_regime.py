"""Sprint 5 / Idea #3: добавляет HMM regime tag в существующий ensemble_predictions.npz
без перетренировки V3.

HMM regime хранится в последних 3 столбцах ctx (one-hot: bear=0, side=1, bull=2).
Для каждого test-сэмпла берём `ctx_ticker[local_idx, -3:]` и argmax → regime ID.

Реконструируем ticker_lengths напрямую из cache_v3 без вызова
build_full_multiscale_dataset_v3 (избегаем зависимость от litestar).

Использование:
    py -m ml.patch_ensemble_regime
"""
from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from ml.config import CFG

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_v3")
REGIME_NAMES = {0: "bear", 1: "side", 2: "bull"}


def _ctx_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"ctx_{ticker}.npy")


def _cls_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"cls_{ticker}.npy")


def _load_ticker_lengths() -> dict[str, int]:
    """Реконструируем {ticker: n_samples} из cls_*.npy файлов.
    Порядок тикеров — алфавитный (как при load в dataset_v3 по CFG.tickers).
    """
    out = {}
    for ticker in CFG.tickers:
        p = _cls_path(ticker)
        if os.path.exists(p):
            arr = np.load(p, mmap_mode="r")
            out[ticker] = int(len(arr))
    return out


def _temporal_split_simple(
    ticker_lengths: dict[str, int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    purge_bars: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Реплика temporal_split из dataset_v3 (per-ticker time-based).
    Возвращает global indices в том же порядке, что dataset.records.
    """
    idx_tr, idx_val, idx_test = [], [], []
    g = 0
    for ticker in ticker_lengths:
        n = ticker_lengths[ticker]
        n_test = int(n * test_ratio)
        n_val  = int(n * val_ratio)
        n_tr   = n - n_test - n_val - purge_bars * 2

        if n_tr <= 0:
            g += n
            continue

        # train: [0, n_tr)
        # purge: [n_tr, n_tr + purge_bars)
        # val:   [n_tr + purge_bars, n_tr + purge_bars + n_val)
        # purge: [..., ... + purge_bars)
        # test:  последние n_test
        tr_end  = n_tr
        val_st  = tr_end + purge_bars
        val_end = val_st + n_val
        test_st = val_end + purge_bars

        idx_tr.extend(range(g, g + tr_end))
        idx_val.extend(range(g + val_st, g + val_end))
        idx_test.extend(range(g + test_st, g + n))

        g += n

    return np.array(idx_tr), np.array(idx_val), np.array(idx_test)


def _build_records(ticker_lengths: dict[str, int]) -> list[tuple[str, int]]:
    """Собираем records [(ticker, local_idx), ...] в том же порядке."""
    records = []
    for ticker, n in ticker_lengths.items():
        for li in range(n):
            records.append((ticker, li))
    return records


def main():
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден.")
        return 1

    existing = dict(np.load(NPZ_PATH, allow_pickle=True))
    n_test = len(existing["dir_prob"])
    print(f"Текущий npz: {n_test} test-сэмплов")

    print(f"\nРеконструируем ticker_lengths из {CACHE_DIR}...")
    ticker_lengths = _load_ticker_lengths()
    print(f"  Найдено тикеров: {len(ticker_lengths)}")
    total_samples = sum(ticker_lengths.values())
    print(f"  Всего сэмплов: {total_samples}")

    _, _, idx_test = _temporal_split_simple(
        ticker_lengths,
        val_ratio=0.15, test_ratio=0.15,
        purge_bars=CFG.future_bars,
    )
    print(f"  idx_test: {len(idx_test)} (ожидаем {n_test})")

    if len(idx_test) != n_test:
        # Возможно, тикеров стало больше/меньше или CFG.tickers изменён.
        # Используем test_tickers из npz (B-13) для прямого matching.
        if "test_tickers" in existing and "test_dates" in existing:
            print(f"\n  Длина idx_test не совпадает — используем test_tickers + test_dates как ключ")
            return _patch_via_tickers_dates(existing, n_test)
        print(f"ERROR: длины не совпадают и нет test_tickers/test_dates fallback.")
        return 2

    records = _build_records(ticker_lengths)

    # Кэшируем ctx по тикерам для скорости
    ctx_cache: dict[str, np.ndarray | None] = {}

    def _get_ctx(ticker: str) -> np.ndarray | None:
        if ticker not in ctx_cache:
            cp = _ctx_path(ticker)
            if os.path.exists(cp):
                ctx_cache[ticker] = np.load(cp)
            else:
                ctx_cache[ticker] = None
        return ctx_cache[ticker]

    print(f"\nИзвлекаем regime для {n_test} test-сэмплов...")
    regime_arr = np.full(n_test, -1, dtype=np.int8)
    n_unknown = 0
    for out_idx, global_idx in enumerate(idx_test):
        ticker, local_idx = records[int(global_idx)]
        ctx = _get_ctx(ticker)
        if ctx is None or local_idx >= len(ctx) or ctx.shape[1] < 3:
            n_unknown += 1
            continue
        hmm_one_hot = ctx[local_idx, -3:]
        if hmm_one_hot.sum() < 1e-6:
            n_unknown += 1
            continue
        regime_arr[out_idx] = int(np.argmax(hmm_one_hot))

    _print_dist(regime_arr, n_unknown)

    existing["test_regime"] = regime_arr
    np.savez(NPZ_PATH, **existing)
    print(f"\n✓ Сохранено: {NPZ_PATH}")
    return 0


def _patch_via_tickers_dates(existing: dict, n_test: int) -> int:
    """Fallback: используем test_tickers + test_dates из npz для прямого matching
    с per-ticker dates_*.npy и ctx_*.npy.
    """
    test_tickers = np.array([str(t) for t in existing["test_tickers"]])
    test_dates   = np.array([str(d) for d in existing["test_dates"]])

    ctx_cache: dict[str, np.ndarray | None] = {}
    dates_cache: dict[str, np.ndarray | None] = {}

    def _get(p: str, cache: dict):
        if p not in cache:
            cache[p] = np.load(p) if os.path.exists(p) else None
        return cache[p]

    regime_arr = np.full(n_test, -1, dtype=np.int8)
    n_unknown = 0
    for i in range(n_test):
        ticker = test_tickers[i]
        date_s = test_dates[i]
        ctx_p = _ctx_path(ticker)
        dates_p = os.path.join(CACHE_DIR, f"dates_{ticker}.npy")

        ctx = _get(ctx_p, ctx_cache)
        dates = _get(dates_p, dates_cache)
        if ctx is None or dates is None or ctx.shape[1] < 3:
            n_unknown += 1
            continue

        # Найти local_idx по дате
        match = np.where(dates == date_s)[0]
        if len(match) == 0:
            n_unknown += 1
            continue
        local_idx = int(match[0])
        if local_idx >= len(ctx):
            n_unknown += 1
            continue

        hmm_one_hot = ctx[local_idx, -3:]
        if hmm_one_hot.sum() < 1e-6:
            n_unknown += 1
            continue
        regime_arr[i] = int(np.argmax(hmm_one_hot))

    _print_dist(regime_arr, n_unknown)
    existing["test_regime"] = regime_arr
    np.savez(NPZ_PATH, **existing)
    print(f"\n✓ Сохранено (fallback path): {NPZ_PATH}")
    return 0


def _print_dist(regime_arr: np.ndarray, n_unknown: int):
    n_known = int((regime_arr >= 0).sum())
    print(f"\nРаспределение regime по test-сэмплам:")
    for rid in (0, 1, 2):
        n_r = int((regime_arr == rid).sum())
        pct = n_r / max(n_known, 1) * 100
        print(f"  {REGIME_NAMES[rid]:>4} (={rid}): {n_r:>6}  ({pct:>5.1f}%)")
    if n_unknown > 0:
        print(f"  unknown:    {n_unknown}  (ctx/dates отсутствуют)")


if __name__ == "__main__":
    sys.exit(main())
