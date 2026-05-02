"""DecisionLayer threshold sweep — без перетренировки.

Перебирает (min_edge_ratio × min_dir_prob × min_sell_dir_prob) на текущем
ensemble_predictions.npz и считает coverage / hit_rate / expectancy для каждого
сочетания. Помогает выбрать оптимальные пороги без полного backtest.

Метрики:
  coverage     = доля BUY+SELL (не HOLD)
  hit_rate     = BUY→UP / SELL→DOWN среди не-HOLD
  expectancy   = средний gross-PnL по входам в долях цены (грубая оценка)
                 BUY: real_C  - 2*fee
                 SELL: -real_C - 2*fee
                 (без TP/SL, чисто close-to-close)

Запуск:
    py -m ml.decision_sweep                        # быстрый scan по edge_ratio
    py -m ml.decision_sweep --grid                 # 2D grid edge_ratio × dir_prob
    py -m ml.decision_sweep --predictions PATH     # custom путь
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from ml.decision_layer import DecisionLayer, costs_from_config, SIG_BUY, SIG_HOLD, SIG_SELL

DEFAULT_NPZ = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
FEE = 0.001  # one-way (учтён 2× в expectancy)


def evaluate_thresholds(
    dir_prob:  np.ndarray,
    mfe_mae:   np.ndarray,
    fill_prob: np.ndarray,
    edge_pred: np.ndarray,
    y_test:    np.ndarray,     # 0=UP, 1=FLAT, 2=DOWN
    ohlc_test: np.ndarray,     # [N, 4*future_bars] — z-нормализованные (ATR units!)
    atr_ratio: np.ndarray,     # B-17: ATR(14)/close для денормализации в доли цены
    *,
    min_edge_ratio:    float,
    min_dir_prob:      float,
    min_sell_dir_prob: float,
    min_fill_prob:     float = 0.40,
    min_rr:            float = 1.2,
) -> dict:
    """Применяет DecisionLayer с заданными порогами и возвращает метрики."""
    dl = DecisionLayer(
        costs=costs_from_config(),
        min_edge_ratio    = min_edge_ratio,
        min_dir_prob      = min_dir_prob,
        min_sell_dir_prob = min_sell_dir_prob,
        min_fill_prob     = min_fill_prob,
        min_rr            = min_rr,
    )
    out = dl.decide_numpy(
        dir_prob  = dir_prob,
        mfe_mae   = mfe_mae,
        fill_prob = fill_prob,
        edge_pred = edge_pred,
    )
    sig = out["signal"]
    n_total = len(sig)
    n_buy   = int((sig == SIG_BUY ).sum())
    n_sell  = int((sig == SIG_SELL).sum())
    n_hold  = int((sig == SIG_HOLD).sum())
    n_act   = n_buy + n_sell

    # Hit rate
    is_buy  = (sig == SIG_BUY)
    is_sell = (sig == SIG_SELL)
    hit = np.zeros_like(sig, dtype=bool)
    hit[is_buy]  = (y_test[is_buy]  == 0)
    hit[is_sell] = (y_test[is_sell] == 2)
    hit_rate = float(hit[is_buy | is_sell].mean()) if n_act > 0 else float("nan")

    # B-17 fix: ohlc_test хранится в ATR-нормализованных единицах.
    # Реальные доли цены = ohlc_test × atr_ratio (как в backtest).
    real_C = ohlc_test[:, 3] * atr_ratio    # ΔC bar1 в долях цены
    pnl = np.zeros_like(real_C)
    pnl[is_buy]  =  real_C[is_buy]  - 2 * FEE
    pnl[is_sell] = -real_C[is_sell] - 2 * FEE
    exp_ud = float(pnl[is_buy | is_sell].mean()) if n_act > 0 else float("nan")
    win_rate = float((pnl[is_buy | is_sell] > 0).mean()) if n_act > 0 else float("nan")

    return {
        "coverage":  n_act / max(n_total, 1),
        "hit_rate":  hit_rate,
        "expectancy_pct": exp_ud * 100,
        "win_rate":  win_rate,
        "n_buy":     n_buy,
        "n_sell":    n_sell,
        "n_hold":    n_hold,
    }


def sweep_edge_ratio(data: dict, dir_prob_value: float, sell_value: float):
    """Скан по min_edge_ratio при фиксированных dir/sell порогах."""
    print(f"\n{'='*88}")
    print(f"  Sweep min_edge_ratio  (min_dir_prob={dir_prob_value}  min_sell={sell_value})")
    print(f"{'='*88}")
    print(f"  {'edge_r':>7} | {'cov':>6} | {'BUY':>5} | {'SELL':>5} | "
          f"{'hit':>6} | {'win':>6} | {'exp%':>7}")
    print(f"  {'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")

    for er in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]:
        m = evaluate_thresholds(
            **data,
            min_edge_ratio    = er,
            min_dir_prob      = dir_prob_value,
            min_sell_dir_prob = sell_value,
        )
        hit = f"{m['hit_rate']:.4f}" if not np.isnan(m['hit_rate']) else "  n/a "
        win = f"{m['win_rate']:.4f}" if not np.isnan(m['win_rate']) else "  n/a "
        exp = f"{m['expectancy_pct']:+.3f}" if not np.isnan(m['expectancy_pct']) else "  n/a "
        print(f"  {er:>7.1f} | {m['coverage']:>6.2%} | "
              f"{m['n_buy']:>5d} | {m['n_sell']:>5d} | "
              f"{hit:>6} | {win:>6} | {exp:>7}")


def sweep_grid(data: dict):
    """2D grid: min_edge_ratio × min_dir_prob (sell симметричный = 1 - dir_prob)."""
    print(f"\n{'='*88}")
    print(f"  Sweep grid: min_edge_ratio × min_dir_prob  (min_sell = 1 - dir_prob + 0.05)")
    print(f"{'='*88}")
    print(f"  Метрика: hit_rate (coverage в скобках)\n")

    edge_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    dir_vals  = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    # header
    header = "  edge\\dir |"
    for d in dir_vals:
        header += f"  {d:.2f}        |"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for er in edge_vals:
        row = f"  {er:>4.1f}     |"
        for d in dir_vals:
            sell = max(1.0 - d + 0.05, 0.55)
            m = evaluate_thresholds(
                **data,
                min_edge_ratio    = er,
                min_dir_prob      = d,
                min_sell_dir_prob = sell,
            )
            cov = m["coverage"]
            hit = m["hit_rate"]
            if np.isnan(hit) or cov < 0.005:
                cell = "  n/a       "
            else:
                cell = f" {hit:.3f}({cov:.1%})"
            row += f" {cell:>13}|"
        print(row)


def sweep_by_regime(data: dict, regime: np.ndarray):
    """Sprint 5: per-regime sweep — отдельные пороги для bear/side/bull.
    Цель: найти, есть ли значимая разница между режимами.
    """
    regime_names = {0: "bear", 1: "side", 2: "bull"}
    print(f"\n{'='*88}")
    print(f"  Per-regime sweep (B-15 baseline: edge=5.0, dir=0.75, sell=0.55)")
    print(f"{'='*88}")

    for rid in (0, 1, 2):
        mask = (regime == rid)
        n_r = int(mask.sum())
        if n_r < 100:
            print(f"\n  [{regime_names[rid]} regime — {n_r} сэмплов: пропускаем (слишком мало)]")
            continue
        print(f"\n  ── regime={regime_names[rid]} (={rid})  N={n_r} ──")

        sub = {
            "dir_prob":  data["dir_prob"][mask],
            "mfe_mae":   data["mfe_mae"][mask],
            "fill_prob": data["fill_prob"][mask],
            "edge_pred": data["edge_pred"][mask],
            "y_test":    data["y_test"][mask],
            "ohlc_test": data["ohlc_test"][mask],
            "atr_ratio": data["atr_ratio"][mask],
        }
        print(f"  {'edge_r':>7} | {'dir_p':>6} | {'sell_p':>6} | {'cov':>6} | "
              f"{'BUY':>5} | {'SELL':>5} | {'hit':>6} | {'win':>6} | {'exp%':>7}")
        print(f"  {'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")

        # B-15 baseline + альтернативы для каждого режима
        configs = [
            (5.0, 0.75, 0.55),   # B-15 default
            (3.0, 0.65, 0.60),   # softer
            (4.0, 0.70, 0.55),   # middle
            (6.0, 0.65, 0.50),   # stricter
        ]
        for er, dp, sp in configs:
            m = evaluate_thresholds(
                **sub, min_edge_ratio=er, min_dir_prob=dp, min_sell_dir_prob=sp,
            )
            hit = f"{m['hit_rate']:.4f}" if not np.isnan(m['hit_rate']) else "  n/a "
            win = f"{m['win_rate']:.4f}" if not np.isnan(m['win_rate']) else "  n/a "
            exp = f"{m['expectancy_pct']:+.3f}" if not np.isnan(m['expectancy_pct']) else "  n/a "
            print(f"  {er:>7.1f} | {dp:>6.2f} | {sp:>6.2f} | {m['coverage']:>6.2%} | "
                  f"{m['n_buy']:>5d} | {m['n_sell']:>5d} | {hit:>6} | {win:>6} | {exp:>7}")

        # Поиск best для этого режима
        best = None
        for er in np.arange(2.0, 8.1, 0.5):
            for dp in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                sp = max(1.0 - dp + 0.05, 0.50)
                m = evaluate_thresholds(
                    **sub, min_edge_ratio=float(er), min_dir_prob=dp, min_sell_dir_prob=sp,
                )
                if m["coverage"] >= 0.03 and not np.isnan(m["expectancy_pct"]):
                    if best is None or m["expectancy_pct"] > best[0]:
                        best = (m["expectancy_pct"], er, dp, sp, m)
        if best:
            ex, er, dp, sp, m = best
            print(f"\n  ★ best for {regime_names[rid]}: edge={er:.1f} dir={dp:.2f} sell={sp:.2f} "
                  f"→ cov={m['coverage']:.2%} hit={m['hit_rate']:.4f} "
                  f"win={m['win_rate']:.4f} exp%={ex:+.3f}")


def find_best(data: dict):
    """Ищем порог с максимальной expectancy при coverage ≥ 5%."""
    print(f"\n{'='*88}")
    print(f"  Поиск оптимума: max(expectancy) при coverage ∈ [5%, 50%]")
    print(f"{'='*88}")
    best = []
    for er in np.arange(1.0, 6.1, 0.5):
        for d in [0.55, 0.60, 0.65, 0.70, 0.75]:
            sell = max(1.0 - d + 0.05, 0.55)
            m = evaluate_thresholds(
                **data,
                min_edge_ratio    = float(er),
                min_dir_prob      = d,
                min_sell_dir_prob = sell,
            )
            if 0.05 <= m["coverage"] <= 0.50 and not np.isnan(m["expectancy_pct"]):
                best.append((m["expectancy_pct"], m["hit_rate"], m["coverage"], er, d, sell, m))

    if not best:
        print("  Нет порогов в диапазоне coverage 5–50%. Снизьте edge_ratio или dir_prob.")
        return

    best.sort(key=lambda t: t[0], reverse=True)
    print(f"\n  Top-10 по expectancy_pct (BUY+SELL trades):\n")
    print(f"  {'rank':>4} | {'edge_r':>7} | {'dir_p':>6} | {'sell_p':>6} | "
          f"{'cov':>6} | {'BUY':>5} | {'SELL':>5} | {'hit':>6} | {'win':>6} | {'exp%':>7}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")
    for i, (ex, hit, cov, er, d, sell, m) in enumerate(best[:10], 1):
        print(f"  {i:>4} | {er:>7.1f} | {d:>6.2f} | {sell:>6.2f} | "
              f"{cov:>6.2%} | {m['n_buy']:>5d} | {m['n_sell']:>5d} | "
              f"{hit:>.4f} | {m['win_rate']:>.4f} | {ex:>+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_NPZ)
    parser.add_argument("--grid", action="store_true",
                        help="2D grid edge_ratio × dir_prob")
    parser.add_argument("--best", action="store_true",
                        help="Поиск оптимума по expectancy")
    parser.add_argument("--by-regime", action="store_true",
                        help="Sprint 5: per-regime sweep (bear/side/bull)")
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        print(f"ERROR: {args.predictions} не найден. Сначала: py -m ml.trainer_v3_ensemble")
        return 1

    print(f"\n  Загружаем {args.predictions} ...")
    npz = np.load(args.predictions, allow_pickle=True)

    required = ["dir_prob", "mfe_mae_pred", "fill_prob", "edge_pred",
                "y_test", "ohlc_test", "atr_ratio"]
    missing = [k for k in required if k not in npz.files]
    if missing:
        print(f"ERROR: в npz отсутствуют ключи: {missing}")
        return 2

    data = dict(
        dir_prob  = npz["dir_prob"].astype(np.float32),
        mfe_mae   = npz["mfe_mae_pred"].astype(np.float32),
        fill_prob = npz["fill_prob"].astype(np.float32),
        edge_pred = npz["edge_pred"].astype(np.float32),
        y_test    = npz["y_test"].astype(np.int8),
        ohlc_test = npz["ohlc_test"].astype(np.float32),
        atr_ratio = npz["atr_ratio"].astype(np.float32),   # B-17
    )
    n = len(data["dir_prob"])
    cls_counts = {int(c): int((data["y_test"] == c).sum()) for c in (0, 1, 2)}
    print(f"  N={n}  UP={cls_counts.get(0,0)}  FLAT={cls_counts.get(1,0)}  DOWN={cls_counts.get(2,0)}")
    print(f"  dir_prob:  mean={data['dir_prob'].mean():.4f}  std={data['dir_prob'].std():.4f}")
    print(f"  edge_pred: mean={data['edge_pred'].mean():+.5f}  std={data['edge_pred'].std():.5f}")

    # Базовый scan по edge_ratio при текущих порогах (0.70 / 0.85)
    sweep_edge_ratio(data, dir_prob_value=0.70, sell_value=0.85)

    # Дополнительно — более мягкий dir_prob 0.60
    sweep_edge_ratio(data, dir_prob_value=0.60, sell_value=0.60)

    if args.grid:
        sweep_grid(data)

    if args.by_regime:
        if "test_regime" not in npz.files:
            print(f"\n  [WARN] test_regime отсутствует в npz. Запустите: py -m ml.patch_ensemble_regime")
        else:
            regime = npz["test_regime"].astype(np.int8)
            sweep_by_regime(data, regime)

    if args.best or True:   # всегда показываем top-10 — это самое полезное
        find_best(data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
