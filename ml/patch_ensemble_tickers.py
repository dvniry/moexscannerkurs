"""B-13: одноразовый patch — добавляет test_tickers в существующий
ensemble_predictions.npz без перетренировки V3.

Использование:
    py -m ml.patch_ensemble_tickers
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from ml.config import CFG
from ml.dataset_v3 import build_full_multiscale_dataset_v3, temporal_split

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")


def main():
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден. Сначала: py -m ml.trainer_v3_ensemble")
        return 1

    existing = dict(np.load(NPZ_PATH, allow_pickle=True))
    n_test = len(existing["dir_prob"])
    print(f"Текущий npz: {n_test} test-сэмплов")

    if "test_tickers" in existing and len(existing["test_tickers"]) == n_test:
        print(f"test_tickers уже присутствуют (N={len(existing['test_tickers'])}). Ничего не делаю.")
        return 0

    print("Восстанавливаем test_tickers через build_full_multiscale_dataset_v3 + temporal_split...")
    dataset, _, _, ticker_lengths = build_full_multiscale_dataset_v3(force_rebuild=False)
    _, _, idx_test = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.15,
        purge_bars=CFG.future_bars,
    )
    if len(idx_test) != n_test:
        print(f"ERROR: длина idx_test={len(idx_test)} != n_test={n_test}. "
              f"Возможно, кэш изменился — перетренируйте V3.")
        return 2

    test_tickers = np.array(
        [dataset.records[int(i)][0] for i in idx_test],
        dtype="U16",
    )
    existing["test_tickers"] = test_tickers
    np.savez(NPZ_PATH, **existing)
    print(f"✓ Добавлены test_tickers ({len(test_tickers)}) в {NPZ_PATH}")
    print(f"  Уникальных тикеров: {len(np.unique(test_tickers))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
