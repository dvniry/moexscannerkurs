"""Бэктест-модуль v3: с поддержкой hourly encoder.

Запуск:
  python -m ml.backtest_v3 --model ml/model_multiscale_v3.pt
  python -m ml.backtest_v3 --model ml/model_multiscale_v3.pt --no-hourly

Изменения v3:
  1. Загрузка и передача hourly данных в модель
  2. Всё остальное — как в v2 (direction-aware r/r, overlap tracking, etc.)
"""
import os, sys, json, argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import List
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.config import CFG, SCALES
from ml.labels_ohlc import ohlc_to_strategy_features
# Импортируем всё из backtest v2 кроме run_backtest
from ml.backtest import (
    StrategyConfig, Trade, direction_risk_reward,
    Backtester, buy_hold_benchmark,
)


def run_backtest_v3(model_path: str = 'ml/model_multiscale_v3.pt',
                    save_json: str = None,
                    use_hourly: bool = True):
    """Полный пайплайн v3: загрузка данных → inference → бэктест."""
    from ml.dataset_v3 import (
        build_full_multiscale_dataset_v3,
        temporal_split,
    )
    from ml.multiscale_cnn_v3 import MultiScaleHybridV3, _collate_v3
    from ml.dataset import class_distribution
    from torch.utils.data import Subset, DataLoader

    print("=" * 60)
    print("  BACKTEST v3 — загрузка данных и модели")
    print("=" * 60)

    # ── Загрузка данных ───────────────────────────────────────
    dataset, y_all, ctx_dim, ticker_lengths = \
        build_full_multiscale_dataset_v3(use_hourly=use_hourly)
    print(f"\n  Всего сэмплов: {len(y_all)}, ctx_dim={ctx_dim}")

    _, _, idx_test = temporal_split(
        ticker_lengths,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_bars=CFG.future_bars,
    )
    y_test = y_all[idx_test]

    te_ds = Subset(dataset, idx_test.tolist())
    print(f"  Test set (temporal): {len(y_test)} сэмплов")
    class_distribution(y_test)

    # ── Загрузка модели ───────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MultiScaleHybridV3(ctx_dim=ctx_dim,
                                 use_hourly=use_hourly).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Модель загружена: {model_path}")
    print(f"  Hourly encoder: {'ON' if use_hourly else 'OFF'}")

    # ── Inference ─────────────────────────────────────────────
    loader = DataLoader(te_ds, batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=True,
                        collate_fn=_collate_v3)

    all_probs      = []
    all_ohlc_pred  = []
    all_ohlc_true  = []

    print("  Inference на test set...")
    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx = batch[:5]
            hourly_data = batch[5] if len(batch) > 5 else None

            imgs = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            hourly_t = hourly_data.to(device) \
                       if (use_hourly and hourly_data is not None) else None

            if num_dict is not None:
                nums = {W: num_dict[W].to(device) for W in SCALES}
                cls_logits, ohlc_pred = model(imgs, nums, ctx_t,
                                              hourly=hourly_t)
            else:
                cls_logits, ohlc_pred = model(imgs, ctx=ctx_t,
                                              hourly=hourly_t)

            probs = torch.softmax(cls_logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_ohlc_pred.append(ohlc_pred.cpu().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    all_probs     = np.concatenate(all_probs, axis=0)
    all_ohlc_pred = np.concatenate(all_ohlc_pred, axis=0)
    all_ohlc_true = np.concatenate(all_ohlc_true, axis=0)

    print(f"  Inference done: {len(all_probs)} сэмплов")

    # ── Бэктест с разными конфигурациями ─────────────────────
    configs = {
        "conservative": StrategyConfig(
            confidence_threshold=0.55,
            min_risk_reward=1.5,
            min_expected_move=0.008,
            risk_per_trade=0.01,
            max_open_positions=3,
        ),
        "balanced": StrategyConfig(
            confidence_threshold=0.45,
            min_risk_reward=1.2,
            min_expected_move=0.005,
            risk_per_trade=0.015,
            max_open_positions=5,
        ),
        "aggressive": StrategyConfig(
            confidence_threshold=0.38,
            min_risk_reward=0.8,
            min_expected_move=0.003,
            risk_per_trade=0.02,
            max_open_positions=8,
        ),
    }

    all_results = {}
    for name, cfg in configs.items():
        print(f"\n  ── Стратегия: {name.upper()} ──")
        bt = Backtester(cfg)
        results = bt.run(all_probs, all_ohlc_pred, all_ohlc_true, y_test)
        bt.print_report(results)
        all_results[name] = results

    # ── Benchmark ─────────────────────────────────────────────
    print(f"\n  ── BENCHMARK: Buy & Hold ──")
    bh = buy_hold_benchmark(all_ohlc_true)
    print(f"  Total Return: {bh['total_return']:+.2f}%")
    print(f"  Mean sample return: {bh['mean_return']:+.4f}%")
    print(f"  Positive samples: {bh['pct_positive']:.1f}%")
    print(f"  Max Drawdown: {bh['max_dd_pct']:.2f}%")
    all_results["benchmark_buy_hold"] = bh

    # ── Сохранение ────────────────────────────────────────────
    if save_json is None:
        save_json = model_path.replace('.pt', '_backtest.json')
    with open(save_json, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Результаты → {save_json}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ml/model_multiscale_v3.pt')
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-hourly', action='store_true')
    args = parser.parse_args()
    run_backtest_v3(args.model, args.output, use_hourly=not args.no_hourly)
