"""ml/walk_forward_d3.py — Sprint 11.1: Walk-forward с D-execution modes.

Зачем: in-sample backtest показал D3_coin_flip +0.254%/trade (Total +0.25%).
Нужно проверить, что прибыль не train leakage.

Алгоритм:
  1. Сортируем test sample'ы по дате
  2. Делим на 5 фолдов (time-series split, без shuffle)
  3. На каждом fold:
     - decision_signal уже посчитан в npz (Platt + blacklist applied)
     - Применяем simulate_strategy с D2/D3 mode на test window fold
     - Собираем trades, считаем exp%/Sharpe/win
  4. Aggregate: mean ± std по 5 фолдам

D3 non-deterministic (RNG): прогоняем 5 seeds и усредняем чтобы убрать шум.

Запуск:
    py -m ml.walk_forward_d3                 # D2+D3 на 5 фолдах
    py -m ml.walk_forward_d3 --folds 5 --seeds 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from ml.decision_layer import SIG_BUY, SIG_HOLD, SIG_SELL
from ml.backtest_strategy import simulate_strategy

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
FEE = 0.001


def fold_metrics(trades: list[dict], fee: float = FEE) -> dict:
    """gross_pnl уже в той же единице что real_C (нормированной через ATR).
    Считаем чистый PnL после двусторонней комиссии 2*fee.
    """
    if not trades:
        return {"n": 0, "win": np.nan, "exp_pct": np.nan, "sharpe": np.nan, "total": 0.0}
    pnl = np.array([t["gross_pnl"] - 2 * fee for t in trades])
    sharpe = float(pnl.mean() / (pnl.std(ddof=1) + 1e-12) * np.sqrt(252)) if len(pnl) > 1 else np.nan
    return {
        "n":       len(pnl),
        "win":     float((pnl > 0).mean()),
        "exp_pct": float(pnl.mean() * 100),
        "sharpe":  sharpe,
        "total":   float(pnl.sum() * 100),
    }


def run_fold(data: dict, idx: np.ndarray, mode: str, conflict: str,
             rng_seed: int, future_bars: int = 5) -> dict:
    """Запускает simulate_strategy на подвыборке.

    NB: backtest_strategy.py применяет norm_factor = atr_ratio*sqrt(fb) к
    ohlc_pred/ohlc_true перед simulate_strategy. Делаем то же самое чтобы
    gross_pnl был в долях цены (FEE=0.001 = 0.1% совпадает).
    """
    norm = (data["atr_ratio"][idx] * np.sqrt(future_bars))[:, None]
    ohlc_pred_pct = data["ohlc_pred"][idx] * norm
    ohlc_true_pct = data["ohlc_test"][idx] * norm

    state = np.random.get_state()
    np.random.seed(rng_seed)
    try:
        trades, diag = simulate_strategy(
            p_dir         = data["dir_prob"][idx],
            cls_probs     = data["cls_probs"][idx],
            ohlc_pred_raw = ohlc_pred_pct,
            ohlc_true_raw = ohlc_true_pct,
            y_true        = data["y_test"][idx],
            entry_mode    = mode,
            conflict_mode = conflict,
            signal_threshold = 0.25,
            entry_depth   = 0.5,
            tp_mult       = 1.5,
            sl_mult       = 1.0,
        )
    finally:
        np.random.set_state(state)
    return fold_metrics(trades)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Сколько RNG seeds для D3 (усреднение non-determinism)")
    args = parser.parse_args()

    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден"); sys.exit(1)

    npz  = np.load(NPZ_PATH, allow_pickle=False)
    data = {k: npz[k] for k in npz.files}
    N    = len(data["dir_prob"])

    # Sprint 10 B: используем dir_prob_platt (если есть) — это то что decision_layer видит
    if "dir_prob_platt" in data:
        data["dir_prob"] = data["dir_prob_platt"]
        print("  [B] dir_prob ← dir_prob_platt (Sprint 10 B)")

    # ── Time-series split по датам ────────────────────────────────────────
    dates = data["test_dates"]
    sort_idx = np.argsort(dates)   # хронологический порядок

    fold_size = N // args.folds
    folds = [sort_idx[i * fold_size : (i + 1) * fold_size] for i in range(args.folds)]
    # Последний fold забирает остаток
    folds[-1] = sort_idx[(args.folds - 1) * fold_size :]

    print(f"\n  Walk-forward D-execution  (N={N}, folds={args.folds}, seeds={args.seeds})")
    print(f"  ──────────────────────────────────────────────────────────────")
    print(f"  fold sizes: {[len(f) for f in folds]}")

    modes = [
        ("A_market",      "market",      "bm_formula"),  # baseline
        ("D2_bm_formula", "limit_tp_sl", "bm_formula"),  # детерминированный
        ("D3_coin_flip",  "limit_tp_sl", "coin_flip"),   # non-det RNG
    ]

    print(f"\n  ──── Per-fold breakdown ────")
    fold_results = {name: [] for name, _, _ in modes}

    for fi, fold_idx in enumerate(folds):
        fold_dates = data["test_dates"][fold_idx]
        d_min = str(min(fold_dates)); d_max = str(max(fold_dates))
        print(f"\n  Fold {fi + 1}/{args.folds}  {d_min} → {d_max}  N={len(fold_idx)}")
        print(f"     mode             | n   | win    | exp%   | sharpe  | total%")
        print(f"     ─────────────────┼─────┼────────┼────────┼─────────┼───────")
        for name, mode, conflict in modes:
            # Усредняем по seeds для D3 (coin_flip), для D2 один seed достаточно
            seeds = list(range(args.seeds)) if conflict == "coin_flip" else [42]
            seed_results = [run_fold(data, fold_idx, mode, conflict, s) for s in seeds]

            avg = {k: np.mean([r[k] for r in seed_results if not np.isnan(r[k])])
                   if any(not np.isnan(r[k]) for r in seed_results) else np.nan
                   for k in ("n", "win", "exp_pct", "sharpe", "total")}
            avg["n"] = int(np.mean([r["n"] for r in seed_results]))   # cast
            std_exp = float(np.std([r["exp_pct"] for r in seed_results if not np.isnan(r["exp_pct"])])) \
                      if len(seeds) > 1 else 0.0

            tag = "" if len(seeds) == 1 else f" (±{std_exp:.3f}, {len(seeds)}×)"
            print(f"     {name:<16} | {avg['n']:>3d} | {avg['win']:.4f} | "
                  f"{avg['exp_pct']:+6.3f} | {avg['sharpe']:+7.2f} | {avg['total']:+6.2f}{tag}")
            fold_results[name].append(avg)

    # ── Aggregate ─────────────────────────────────────────────────────────
    print(f"\n  ──── Aggregate (mean ± std по {args.folds} фолдам) ────")
    print(f"     mode             | n_avg  | win        | exp%        | sharpe     | total%")
    print(f"     ─────────────────┼────────┼────────────┼─────────────┼────────────┼───────")
    for name, _, _ in modes:
        rs = fold_results[name]
        agg = {}
        for k in ("n", "win", "exp_pct", "sharpe", "total"):
            vals = [r[k] for r in rs if not np.isnan(r[k])]
            agg[k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)

        print(f"     {name:<16} | {agg['n'][0]:>5.0f}  | "
              f"{agg['win'][0]:.3f}±{agg['win'][1]:.3f} | "
              f"{agg['exp_pct'][0]:+6.3f}±{agg['exp_pct'][1]:.3f} | "
              f"{agg['sharpe'][0]:+5.2f}±{agg['sharpe'][1]:.2f} | "
              f"{agg['total'][0]:+5.2f}±{agg['total'][1]:.2f}")

    # ── Profit interpretation ────────────────────────────────────────────
    print(f"\n  Интерпретация:")
    print(f"  - exp% > 0  → стратегия покрывает 2*fee = {2*FEE*100:.2f}% costs")
    print(f"  - sharpe > 0.5 → значимая прибыль с поправкой на риск")
    print(f"  - Если D2/D3 in-sample +0.25% но per-fold mean -0.X% → train leakage / regime drift")
    print(f"  - Если D2/D3 in-sample ≈ per-fold mean → прибыль реальна")


if __name__ == "__main__":
    main()
