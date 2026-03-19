"""Точка входа v2: python -m ml.trainer_v2 [--multiscale|--deep]

Новая версия:
  - Factual рендер (многоканальный, без RGB)
  - OHLC regression + classification (multi-task)
  - 1D Conv backbone вместо EfficientNet
  - Temporal split с purge gap (без data leakage)
"""
import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'
import sys
import numpy as np

_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.config import CFG
from ml.dataset import class_distribution


def run_multiscale_training_v2(mode: str = "standard",
                                force_rebuild: bool = False):
    """
    mode: 'standard' (--multiscale) | 'deep' (--deep)
    force_rebuild: перестроить кэш (нужно при смене days_back или тикеров)
    """
    label = {
        "standard": "Hybrid v2 (1D-CNN+TCN+BiLSTM+Attention) dual-head 2-phase",
        "deep":     "Deep Hybrid v2 (1D-CNN+TCN+BiLSTM+Attention) dual-head 3-phase",
    }
    print("=" * 60 + f"\nMultiScale v2 [{label[mode]}]\n" + "=" * 60)

    from ml.dataset_v2_ohlc import (
        build_full_multiscale_dataset_v2,
        temporal_split,
    )
    from ml.multiscale_cnn_v2 import (
        train_multiscale_v2, train_multiscale_deep_v2,
        evaluate_multiscale_v2,
    )
    from torch.utils.data import Subset

    if force_rebuild:
        print("  ⚠ REBUILD: перестроение кэша (days_back/тикеры изменились)")
    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v2(
        force_rebuild=force_rebuild)

    print(f"\nВсего сэмплов: {len(y_all)}")
    print(f"Контекст: dim={ctx_dim}")
    class_distribution(y_all)

    # ── Temporal split с purge gap ────────────────────────────
    idx_tr, idx_val, idx_test = temporal_split(
        ticker_lengths,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_bars=CFG.future_bars,
    )

    y_tr   = y_all[idx_tr]
    y_val  = y_all[idx_val]
    y_test = y_all[idx_test]

    tr_ds  = Subset(dataset, idx_tr.tolist())
    val_ds = Subset(dataset, idx_val.tolist())
    te_ds  = Subset(dataset, idx_test.tolist())

    print(f"\n  TEMPORAL SPLIT (purge={CFG.future_bars} bars):")
    print(f"  Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"  Train classes:")
    class_distribution(y_tr)
    print(f"  Val classes:")
    class_distribution(y_val)
    print(f"  Test classes:")
    class_distribution(y_test)

    if mode == "deep":
        save_path = 'ml/model_multiscale_deep_v2.pt'
        model = train_multiscale_deep_v2(
            tr_ds, y_tr, val_ds, y_val, ctx_dim, save_path=save_path)
    else:
        save_path = 'ml/model_multiscale_v2.pt'
        model = train_multiscale_v2(
            tr_ds, y_tr, val_ds, y_val, ctx_dim, save_path=save_path)

    print("\n" + "=" * 60 + "\nОценка\n" + "=" * 60)
    evaluate_multiscale_v2(
        model, te_ds, y_test, ctx_dim,
        save_json=save_path.replace('.pt', '_eval.json'))


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    if "--deep" in sys.argv:
        run_multiscale_training_v2(mode="deep", force_rebuild=rebuild)
    else:
        run_multiscale_training_v2(mode="standard", force_rebuild=rebuild)
