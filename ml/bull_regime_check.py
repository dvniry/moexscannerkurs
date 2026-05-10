"""Sprint 9.5 — re-test stability of `bull=OFF` insight from Sprint 5.

В Sprint 5 sweep на in-sample data показал, что в bull-режиме модель убыточна:
    bear (8% времени):  +0.375%/trade
    side (44%):         +0.006%/trade
    bull (48%):         −0.190%/trade — отключаем

Sprint 6 walk-forward подтвердил: bull=OFF устойчив на всех 5 фолдах.

Sprint 9.5 (этот скрипт): пересчитывает per-regime expectancy на ПОСЛЕДНЕМ окне
test-выборки (последние ~30 дней) — чтобы убедиться, что инсайт не изменился
после: (a) добавления B-25 нормализации h_vol, (b) calibration v3 артефактов,
(c) свежих рыночных условий.

Запуск:
    py -m ml.bull_regime_check                     # последние 30 дней
    py -m ml.bull_regime_check --window-days 60    # настраиваемое окно
    py -m ml.bull_regime_check --by-regime --grid  # полный sweep по порогам
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from ml.decision_layer import (
    DecisionLayer, RegimeAwareDecisionLayer, costs_from_config,
    SIG_BUY, SIG_HOLD, SIG_SELL,
)

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
REGIME_NAMES = {0: "bear", 1: "side", 2: "bull", -1: "unknown"}
FEE = 0.001


def _expectancy(sig: np.ndarray, ohlc: np.ndarray, atr_ratio: np.ndarray) -> dict:
    """Net expectancy in % of price (close-to-close, after 2*FEE)."""
    is_buy  = (sig == SIG_BUY)
    is_sell = (sig == SIG_SELL)
    n_act = int((is_buy | is_sell).sum())
    if n_act == 0:
        return dict(n=0, exp=float("nan"), win=float("nan"), sharpe=float("nan"))
    real_C = ohlc[:, 3] * atr_ratio  # B-17 denormalize
    pnl = np.zeros_like(real_C)
    pnl[is_buy]  =  real_C[is_buy]  - 2 * FEE
    pnl[is_sell] = -real_C[is_sell] - 2 * FEE
    sub = pnl[is_buy | is_sell]
    sharpe = float(sub.mean() / (sub.std(ddof=1) + 1e-12) * np.sqrt(252)) if len(sub) > 1 else float("nan")
    return dict(n=n_act, exp=float(sub.mean() * 100), win=float((sub > 0).mean()), sharpe=sharpe)


def _grid_search_per_regime(data: dict, mask: np.ndarray) -> dict:
    """Прямой grid-sweep по edge/dir в подвыборке data[mask] для каждого режима."""
    sub = {k: v[mask] for k, v in data.items()}
    regime = sub["regime"]
    out = {}
    for rid in (0, 1, 2):
        rmask = (regime == rid)
        n_r = int(rmask.sum())
        if n_r < 30:
            out[rid] = dict(n=n_r, best=None, best_exp=None, best_thr=None)
            continue
        sub_r = {k: v[rmask] for k, v in sub.items()}
        best = None  # (exp, n_act, er, dp, sp)
        for er in np.arange(2.0, 9.1, 1.0):
            for dp in (0.55, 0.65, 0.75, 0.80):
                sp = max(1.0 - dp + 0.05, 0.50)
                dl = DecisionLayer(costs=costs_from_config(),
                                   min_edge_ratio=float(er),
                                   min_dir_prob=dp,
                                   min_sell_dir_prob=sp)
                o = dl.decide_numpy(
                    dir_prob  = sub_r["dir_prob"],
                    mfe_mae   = sub_r["mfe_mae"],
                    fill_prob = sub_r["fill_prob"],
                    edge_pred = sub_r["edge_pred"],
                )
                m = _expectancy(o["signal"], sub_r["ohlc_test"], sub_r["atr_ratio"])
                if m["n"] >= 5 and not np.isnan(m["exp"]):
                    if best is None or m["exp"] > best[0]:
                        best = (m["exp"], m["n"], er, dp, sp)
        if best is None:
            out[rid] = dict(n=n_r, best=None, best_exp=None, best_thr=None)
        else:
            ex, n_act, er, dp, sp = best
            out[rid] = dict(
                n=n_r, n_act=n_act, best_exp=float(ex),
                best_thr=dict(min_edge_ratio=float(er),
                              min_dir_prob=dp, min_sell_dir_prob=sp),
            )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window-days", type=int, default=30,
                   help="Окно для проверки в днях (по test_dates)")
    p.add_argument("--grid", action="store_true",
                   help="Прямой sweep edge×dir на per-regime подвыборках")
    args = p.parse_args()

    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден"); sys.exit(1)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    if "test_regime" not in npz.files:
        print("ERROR: test_regime отсутствует. Запустите: py -m ml.patch_ensemble_regime")
        sys.exit(1)

    data = dict(
        dir_prob  = npz["dir_prob"].astype(np.float32),
        mfe_mae   = npz["mfe_mae_pred"].astype(np.float32),
        fill_prob = npz["fill_prob"].astype(np.float32),
        edge_pred = npz["edge_pred"].astype(np.float32),
        y_test    = npz["y_test"].astype(np.int8),
        ohlc_test = npz["ohlc_test"].astype(np.float32),
        atr_ratio = npz["atr_ratio"].astype(np.float32),
        regime    = npz["test_regime"].astype(np.int8),
    )
    n = len(data["dir_prob"])
    dates = np.array([str(d)[:10] for d in npz["test_dates"]])

    # Sort by date
    order = np.argsort(dates)
    for k in data:
        data[k] = data[k][order]
    dates = dates[order]

    last_date = dates[-1]
    cutoff = (np.datetime64(last_date) - np.timedelta64(args.window_days, 'D')).astype(str)
    mask_recent = (dates >= cutoff)
    n_recent = int(mask_recent.sum())

    print(f"\n  Sprint 9.5 — bull regime stability check")
    print(f"  ────────────────────────────────────────")
    print(f"  Total samples in test:  {n}")
    print(f"  Window (последние {args.window_days} дн): {cutoff} → {last_date}  (N={n_recent})")
    if n_recent < 50:
        print(f"  ⚠️  Окно слишком узкое (<50 сэмплов) — расширьте --window-days")
        sys.exit(0)

    # ── Распределение по регимам в свежем окне ──
    regime_recent = data["regime"][mask_recent]
    print(f"\n  Распределение режимов в свежем окне:")
    for rid in (0, 1, 2):
        m = (regime_recent == rid)
        pct = float(m.mean() * 100)
        print(f"    {REGIME_NAMES[rid]:<6}: {int(m.sum()):>4d} ({pct:>5.1f}%)")

    # ── Per-regime evaluation: Sprint 5 default thresholds (bull=OFF) ──
    print(f"\n  ── Sprint 5 default thresholds на свежем окне ──")
    rdl = RegimeAwareDecisionLayer(costs=costs_from_config())
    sub = {k: v[mask_recent] for k, v in data.items()}
    out = rdl.decide_numpy(
        dir_prob  = sub["dir_prob"],
        mfe_mae   = sub["mfe_mae"],
        fill_prob = sub["fill_prob"],
        edge_pred = sub["edge_pred"],
        regime    = sub["regime"],
    )
    sig = out["signal"]
    print(f"  {'regime':<8} | {'N':>5} | {'BUY':>4} | {'SELL':>4} | {'cov':>6} | "
          f"{'exp%':>7} | {'win':>6} | {'sharpe':>7}")
    for rid in (0, 1, 2):
        m = (sub["regime"] == rid)
        if not m.any():
            continue
        sig_r = sig[m]
        n_buy  = int((sig_r == SIG_BUY).sum())
        n_sell = int((sig_r == SIG_SELL).sum())
        ex = _expectancy(sig_r, sub["ohlc_test"][m], sub["atr_ratio"][m])
        cov = (n_buy + n_sell) / max(int(m.sum()), 1)
        exp_s = f"{ex['exp']:+.3f}" if not np.isnan(ex["exp"]) else "  n/a"
        win_s = f"{ex['win']:.4f}"  if not np.isnan(ex["win"]) else "  n/a"
        sh_s  = f"{ex['sharpe']:+.2f}" if not np.isnan(ex["sharpe"]) else "  n/a"
        print(f"  {REGIME_NAMES[rid]:<8} | {int(m.sum()):>5d} | {n_buy:>4d} | {n_sell:>4d} | "
              f"{cov:>6.2%} | {exp_s:>7} | {win_s:>6} | {sh_s:>7}")

    # ── Grid sweep, если запрошено ──
    if args.grid:
        print(f"\n  ── Прямой grid sweep edge×dir на свежем окне ──")
        best = _grid_search_per_regime(data, mask_recent)
        for rid in (0, 1, 2):
            r = best[rid]
            if r["best_exp"] is None:
                print(f"    {REGIME_NAMES[rid]:<6}: n={r['n']:>4d}  best — нет валидных порогов "
                      f"(возможно, n_act < 5 на любых)")
            else:
                t = r["best_thr"]
                marker = "  ✅ прибыльно" if r["best_exp"] > 0 else "  🚫 убыточно (рекомендуется OFF)"
                print(f"    {REGIME_NAMES[rid]:<6}: n={r['n']:>4d}  best_exp={r['best_exp']:+.3f}%/trade  "
                      f"n_act={r['n_act']}  thr e={t['min_edge_ratio']:.0f}/d={t['min_dir_prob']:.2f}/s={t['min_sell_dir_prob']:.2f}{marker}")

        # Conclusion
        bull_data = best.get(2, {})
        if bull_data.get("best_exp") is not None:
            if bull_data["best_exp"] <= 0:
                print(f"\n  ✅ ВЫВОД: bull=OFF УСТОЙЧИВ на свежих данных "
                      f"(best bull exp={bull_data['best_exp']:+.3f}%, всё ещё убыточно).")
            else:
                print(f"\n  ⚠️  ВЫВОД: bull стал прибыльным ({bull_data['best_exp']:+.3f}%/trade)!")
                print(f"     Sprint 5 инсайт устарел — стоит пересчитать default thresholds.")
                print(f"     Команда: py -m ml.decision_sweep --by-regime")
        else:
            print(f"\n  ⚠️  В bull-режиме нет валидных порогов в свежем окне. Возможно, режим почти не встречался.")


if __name__ == "__main__":
    main()
