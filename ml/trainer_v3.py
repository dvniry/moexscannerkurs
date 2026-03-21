"""Точка входа v3: python -m ml.trainer_v3 [--multiscale] [--rebuild] [--no-hourly]

Новая версия:
  - Часовой слой (HourlyEncoder) — мультитаймфрейм
  - Asymmetric Focal Loss — чиним SELL recall
  - GRN trunk — гейтовое слияние
  - 8 токенов в cross-scale attention (+ hourly)
  - Temporal split с purge gap
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


def run_multiscale_training_v3(force_rebuild: bool = False,
                                use_hourly: bool = True):
    """
    Тренировка модели v3 (multiscale + hourly encoder).
    """
    label = "Hybrid v3 (1D-CNN+TCN+BiLSTM+HourlyEnc+Attention) dual-head 2-phase"
    if not use_hourly:
        label = "Hybrid v3 (1D-CNN+TCN+BiLSTM+Attention) NO hourly, 2-phase"
    print("=" * 60 + f"\nMultiScale v3 [{label}]\n" + "=" * 60)

    from ml.dataset_v3 import (
        build_full_multiscale_dataset_v3,
        temporal_split,
    )
    from ml.multiscale_cnn_v3 import (
        train_multiscale_v3,
        evaluate_multiscale_v3,
    )
    from torch.utils.data import Subset

    if force_rebuild:
        print("  ⚠ REBUILD: перестроение кэша (hourly/days_back изменились)")
    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=force_rebuild, use_hourly=use_hourly)

    print(f"\nВсего сэмплов: {len(y_all)}")
    print(f"Контекст: dim={ctx_dim}")
    print(f"Hourly encoder: {'ON' if use_hourly else 'OFF'}")
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

    save_path = 'ml/model_multiscale_v3.pt'
    model = train_multiscale_v3(
        tr_ds, y_tr, val_ds, y_val, ctx_dim,
        use_hourly=use_hourly, save_path=save_path)

    print("\n" + "=" * 60 + "\nОценка\n" + "=" * 60)
    evaluate_multiscale_v3(
        model, te_ds, y_test, ctx_dim,
        use_hourly=use_hourly,
        save_json=save_path.replace('.pt', '_eval.json'))


if __name__ == "__main__":
    rebuild    = "--rebuild" in sys.argv
    no_hourly  = "--no-hourly" in sys.argv
    use_hourly = not no_hourly
    run_multiscale_training_v3(force_rebuild=rebuild, use_hourly=use_hourly)
