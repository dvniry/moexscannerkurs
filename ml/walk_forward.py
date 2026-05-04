"""Sprint 6: Walk-forward validation на уровне DecisionLayer + MetaLearner.

Проверяет, что результаты Sprint 5 (per-regime thresholds, Sharpe +1.19) и
MetaLearner v2 (holdout val_acc 0.5462) — не in-sample selection bias.

Подход: разбиваем test-выборку из ensemble_predictions.npz на K последовательных
фолдов по дате. Для каждого fold:
  1. Calibration window = все предыдущие фолды (первые k-1 fraction по времени)
  2. Test window = текущий fold
  3. Подбираем regime thresholds на calibration → применяем на test
  4. Обучаем MetaLearner на calibration → оцениваем на test
  5. Считаем coverage, hit_rate, expectancy, Sharpe

V3 ансамбль НЕ переобучается (это занимает >40 мин/fold). Walk-forward на уровне
post-processing — это то, что меняли последние спринты, и где есть риск overfit.

Использование:
    py -m ml.walk_forward                  # 5 фолдов default
    py -m ml.walk_forward --folds 3
    py -m ml.walk_forward --folds 5 --train-meta
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ml.decision_layer import (
    DecisionLayer, RegimeAwareDecisionLayer, costs_from_config,
    SIG_BUY, SIG_HOLD, SIG_SELL,
)

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
FEE = 0.001
REGIME_NAMES = {0: "bear", 1: "side", 2: "bull", -1: "unknown"}


# ══════════════════════════════════════════════════════════════════
# Per-fold metrics
# ══════════════════════════════════════════════════════════════════

def _expectancy_pct(sig: np.ndarray, ohlc_test: np.ndarray, atr_ratio: np.ndarray) -> dict:
    """Net expectancy in % of price (close-to-close, after 2*FEE)."""
    is_buy  = (sig == SIG_BUY)
    is_sell = (sig == SIG_SELL)
    n_act = int((is_buy | is_sell).sum())
    if n_act == 0:
        return {"n_act": 0, "expectancy_pct": np.nan, "win_rate": np.nan,
                "sharpe": np.nan}

    real_C = ohlc_test[:, 3] * atr_ratio   # B-17 denormalize
    pnl = np.zeros_like(real_C)
    pnl[is_buy]  =  real_C[is_buy]  - 2 * FEE
    pnl[is_sell] = -real_C[is_sell] - 2 * FEE

    sub = pnl[is_buy | is_sell]
    sharpe = float(sub.mean() / (sub.std(ddof=1) + 1e-12) * np.sqrt(252)) if len(sub) > 1 else np.nan
    return {
        "n_act":          n_act,
        "expectancy_pct": float(sub.mean() * 100),
        "win_rate":       float((sub > 0).mean()),
        "sharpe":         sharpe,
    }


def _eval_thresholds(data: dict, mask: np.ndarray, decision_layer) -> dict:
    """Применяет DecisionLayer к подвыборке data[mask] и возвращает метрики."""
    sub = {k: v[mask] for k, v in data.items()}
    if hasattr(decision_layer, "decide_numpy"):
        kwargs = dict(
            dir_prob  = sub["dir_prob"],
            mfe_mae   = sub["mfe_mae"],
            fill_prob = sub["fill_prob"],
            edge_pred = sub["edge_pred"],
        )
        if isinstance(decision_layer, RegimeAwareDecisionLayer):
            kwargs["regime"] = sub["regime"]
        out = decision_layer.decide_numpy(**kwargs)
    sig = out["signal"]
    n = int(mask.sum())
    n_buy  = int((sig == SIG_BUY ).sum())
    n_sell = int((sig == SIG_SELL).sum())

    # hit rate (BUY → UP / SELL → DOWN; y_test 3-class: 0=UP, 1=FLAT, 2=DOWN)
    y = sub["y_test"]
    is_buy  = (sig == SIG_BUY)
    is_sell = (sig == SIG_SELL)
    hit = np.zeros_like(sig, dtype=bool)
    hit[is_buy]  = (y[is_buy]  == 0)
    hit[is_sell] = (y[is_sell] == 2)
    n_act = n_buy + n_sell
    hit_rate = float(hit[is_buy | is_sell].mean()) if n_act > 0 else np.nan

    exp = _expectancy_pct(sig, sub["ohlc_test"], sub["atr_ratio"])
    return {
        "n":        n,
        "n_buy":    n_buy,
        "n_sell":   n_sell,
        "coverage": n_act / max(n, 1),
        "hit_rate": hit_rate,
        **exp,
    }


# ══════════════════════════════════════════════════════════════════
# Per-regime threshold optimization on calibration
# ══════════════════════════════════════════════════════════════════

def _find_best_per_regime(data: dict, mask_calib: np.ndarray) -> dict:
    """Для каждого regime ищет (edge, dir, sell), максимизирующий expectancy
    на calibration window. Возвращает dict {rid: {min_edge_ratio, ...}}.
    """
    calib = {k: v[mask_calib] for k, v in data.items()}
    regime = calib["regime"]
    out_thresholds: dict[int, dict] = {}

    for rid in (0, 1, 2):
        rmask = (regime == rid)
        n_r = int(rmask.sum())
        if n_r < 50:
            # Слишком мало → fallback на B-15 default
            out_thresholds[rid] = {
                "min_edge_ratio": 5.0, "min_dir_prob": 0.75,
                "min_sell_dir_prob": 0.55, "min_fill_prob": 0.40, "min_rr": 1.2,
            }
            continue
        sub_calib = {k: v[rmask] for k, v in calib.items()}

        best = None
        for er in np.arange(2.0, 8.1, 1.0):
            for dp in (0.55, 0.65, 0.75, 0.80):
                sp = max(1.0 - dp + 0.05, 0.50)
                dl = DecisionLayer(
                    costs=costs_from_config(),
                    min_edge_ratio=float(er), min_dir_prob=dp,
                    min_sell_dir_prob=sp,
                )
                out = dl.decide_numpy(
                    dir_prob  = sub_calib["dir_prob"],
                    mfe_mae   = sub_calib["mfe_mae"],
                    fill_prob = sub_calib["fill_prob"],
                    edge_pred = sub_calib["edge_pred"],
                )
                sig = out["signal"]
                exp = _expectancy_pct(sig, sub_calib["ohlc_test"], sub_calib["atr_ratio"])
                if exp["n_act"] >= 20 and not np.isnan(exp["expectancy_pct"]):
                    if best is None or exp["expectancy_pct"] > best[0]:
                        best = (exp["expectancy_pct"], er, dp, sp)

        if best is None:
            # disable trading в этом региме на этом fold
            out_thresholds[rid] = {
                "min_edge_ratio": 99.0, "min_dir_prob": 0.99,
                "min_sell_dir_prob": 0.99, "min_fill_prob": 0.99, "min_rr": 99.0,
            }
        else:
            ex, er, dp, sp = best
            # Если best expectancy отрицательная — отключаем regime (HOLD only)
            if ex < 0:
                out_thresholds[rid] = {
                    "min_edge_ratio": 99.0, "min_dir_prob": 0.99,
                    "min_sell_dir_prob": 0.99, "min_fill_prob": 0.99, "min_rr": 99.0,
                }
            else:
                out_thresholds[rid] = {
                    "min_edge_ratio": float(er), "min_dir_prob": dp,
                    "min_sell_dir_prob": sp, "min_fill_prob": 0.40, "min_rr": 1.2,
                }
    out_thresholds[-1] = {
        "min_edge_ratio": 5.0, "min_dir_prob": 0.75,
        "min_sell_dir_prob": 0.55, "min_fill_prob": 0.40, "min_rr": 1.2,
    }
    return out_thresholds


# ══════════════════════════════════════════════════════════════════
# B-22 fix: адаптивные quantile-based пороги per-fold
# ══════════════════════════════════════════════════════════════════

def _adaptive_quantile_thresholds(
    data: dict,
    mask_calib: np.ndarray,
    *,
    q_edge: float = 0.80,
    q_dir:  float = 0.75,
    floor_edge_ratio: float = 2.0,
    floor_dir: float = 0.55,
    floor_sell: float = 0.50,
) -> dict[int, dict]:
    """Считает пороги как quantile калибровочного распределения.

    Гарантирует ~ (1 - q) coverage на калибровке независимо от absolute scale
    edge/dir, что фиксит B-22 (фиксированные in-sample пороги дают 0.5% coverage
    на OOS из-за shift распределения).

    Возвращает {regime_id: {min_edge_ratio, min_dir_prob, min_sell_dir_prob, ...}}
    в формате RegimeAwareDecisionLayer.
    """
    calib = {k: v[mask_calib] for k, v in data.items()}
    cost = costs_from_config().roundtrip
    out: dict[int, dict] = {}

    # Per-regime thresholds + global fallback
    for rid in (0, 1, 2):
        rmask = (calib["regime"] == rid)
        n_r = int(rmask.sum())
        if n_r < 50:
            # Слишком мало данных в этом режиме — fallback к B-15 default
            out[rid] = {
                "min_edge_ratio": 5.0, "min_dir_prob": 0.75,
                "min_sell_dir_prob": 0.55, "min_fill_prob": 0.40, "min_rr": 1.2,
            }
            continue
        # edge_pred[:,0] = long, [:,1] = short (см. decision_layer)
        edge_long  = calib["edge_pred"][rmask, 0] / max(cost, 1e-9)
        edge_short = calib["edge_pred"][rmask, 1] / max(cost, 1e-9)
        dir_p = calib["dir_prob"][rmask]

        # Берём максимум long/short — порог должен пропускать "сильнейшую сторону"
        edge_combined = np.maximum(edge_long, edge_short)
        thr_edge = float(np.quantile(edge_combined, q_edge))
        thr_edge = max(thr_edge, floor_edge_ratio)

        # dir_prob распределён на [0,1]; считаем порог для long-сетапов
        thr_dir = float(np.quantile(dir_p, q_dir))
        thr_dir = max(thr_dir, floor_dir)

        # для SELL — quantile из (1 - dir_prob)
        thr_sell = float(np.quantile(1.0 - dir_p, q_dir))
        thr_sell = max(thr_sell, floor_sell)

        out[rid] = {
            "min_edge_ratio": thr_edge,
            "min_dir_prob":   thr_dir,
            "min_sell_dir_prob": thr_sell,
            "min_fill_prob": 0.40,
            "min_rr": 1.2,
        }

    # Sprint 5 stable insight: bull=OFF на всех фолдах. Сохраняем при адаптивных порогах.
    out[2] = {
        "min_edge_ratio": 99.0, "min_dir_prob": 0.99,
        "min_sell_dir_prob": 0.99, "min_fill_prob": 0.99, "min_rr": 99.0,
    }
    out[-1] = {
        "min_edge_ratio": 5.0, "min_dir_prob": 0.75,
        "min_sell_dir_prob": 0.55, "min_fill_prob": 0.40, "min_rr": 1.2,
    }
    return out


# ══════════════════════════════════════════════════════════════════
# Walk-forward orchestrator
# ══════════════════════════════════════════════════════════════════

def run_walk_forward(folds: int = 5, min_calib_frac: float = 0.30,
                     adaptive_thresholds: bool = False,
                     q_edge: float = 0.80, q_dir: float = 0.75) -> list[dict]:
    """Запускает K-fold walk-forward.

    folds:           число тестовых фолдов
    min_calib_frac:  минимальная доля calibration перед первым fold (по времени).
                     Если 0.30 → calibration начинается с 30% data, далее
                     accumulating up to 1 - 1/folds.
    """
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден.")
        return []

    npz = np.load(NPZ_PATH, allow_pickle=True)
    required = ["dir_prob", "mfe_mae_pred", "fill_prob", "edge_pred",
                "y_test", "ohlc_test", "atr_ratio", "test_dates"]
    missing = [k for k in required if k not in npz.files]
    if missing:
        print(f"ERROR: в npz отсутствуют ключи: {missing}")
        return []

    has_regime = "test_regime" in npz.files

    data = dict(
        dir_prob  = npz["dir_prob"].astype(np.float32),
        mfe_mae   = npz["mfe_mae_pred"].astype(np.float32),
        fill_prob = npz["fill_prob"].astype(np.float32),
        edge_pred = npz["edge_pred"].astype(np.float32),
        y_test    = npz["y_test"].astype(np.int8),
        ohlc_test = npz["ohlc_test"].astype(np.float32),
        atr_ratio = npz["atr_ratio"].astype(np.float32),
        regime    = (npz["test_regime"].astype(np.int8) if has_regime
                     else np.full(len(npz["dir_prob"]), -1, dtype=np.int8)),
    )
    n = len(data["dir_prob"])
    dates = np.array([str(d) for d in npz["test_dates"]])

    # Sort by date for clean temporal split
    order = np.argsort(dates)
    for k in data:
        data[k] = data[k][order]
    dates = dates[order]

    # Define fold boundaries
    test_size = (n - int(n * min_calib_frac)) // folds
    fold_starts = []
    for k in range(folds):
        st = int(n * min_calib_frac) + k * test_size
        en = st + test_size if k < folds - 1 else n
        fold_starts.append((st, en))

    print(f"\n{'='*92}")
    print(f"  Sprint 6 — Walk-forward validation  (folds={folds})")
    print(f"  Total samples: {n}  Calibration min: {int(n*min_calib_frac)}  Test/fold: ~{test_size}")
    print(f"  Regime tag: {'present' if has_regime else 'MISSING (use patch_ensemble_regime)'}")
    print(f"{'='*92}")

    fold_results = []
    for k, (st, en) in enumerate(fold_starts, 1):
        mask_calib = np.zeros(n, dtype=bool)
        mask_calib[:st] = True
        mask_test  = np.zeros(n, dtype=bool)
        mask_test[st:en]  = True
        n_calib = int(mask_calib.sum())
        n_test  = int(mask_test.sum())
        d_calib_first = dates[mask_calib][0]
        d_calib_last  = dates[mask_calib][-1]
        d_test_first  = dates[mask_test][0]
        d_test_last   = dates[mask_test][-1]
        print(f"\n  ── Fold {k}/{folds} ──")
        print(f"     calibration: {d_calib_first} → {d_calib_last}  (N={n_calib})")
        print(f"     test:        {d_test_first} → {d_test_last}  (N={n_test})")

        # ── Strategy A: B-15 fixed thresholds (in-sample baseline) ──
        dl_static = DecisionLayer(costs_from_config())
        m_static = _eval_thresholds(data, mask_test, dl_static)

        # ── Strategy B: per-regime thresholds learned on calibration ──
        if has_regime:
            tuned = _find_best_per_regime(data, mask_calib)
            rdl = RegimeAwareDecisionLayer(
                costs=costs_from_config(), regime_thresholds=tuned,
            )
            m_regime = _eval_thresholds(data, mask_test, rdl)
            # Per-regime breakdown of selected thresholds:
            sel_str = []
            for rid in (0, 1, 2):
                t = tuned.get(rid, {})
                disabled = (t.get("min_edge_ratio", 0) >= 99.0)
                tag = "OFF" if disabled else f"e={t['min_edge_ratio']:.0f}/d={t['min_dir_prob']:.2f}/s={t['min_sell_dir_prob']:.2f}"
                sel_str.append(f"{REGIME_NAMES[rid]}:{tag}")
            print(f"     selected thresholds: " + " | ".join(sel_str))
        else:
            m_regime = None

        # ── Strategy C (B-22): adaptive quantile thresholds per-fold ──
        m_adaptive = None
        if adaptive_thresholds and has_regime:
            adapt = _adaptive_quantile_thresholds(data, mask_calib,
                                                  q_edge=q_edge, q_dir=q_dir)
            adl = RegimeAwareDecisionLayer(
                costs=costs_from_config(), regime_thresholds=adapt,
            )
            m_adaptive = _eval_thresholds(data, mask_test, adl)
            sel_str = []
            for rid in (0, 1, 2):
                t = adapt.get(rid, {})
                disabled = (t.get("min_edge_ratio", 0) >= 99.0)
                tag = "OFF" if disabled else (
                    f"e={t['min_edge_ratio']:.1f}/d={t['min_dir_prob']:.2f}/s={t['min_sell_dir_prob']:.2f}")
                sel_str.append(f"{REGIME_NAMES[rid]}:{tag}")
            print(f"     adaptive q={q_edge:.2f}/{q_dir:.2f}: " + " | ".join(sel_str))

        print(f"     {'strategy':<24} | {'cov':>6} | {'BUY':>4} | {'SELL':>4} | "
              f"{'hit':>6} | {'win':>6} | {'exp%':>7} | {'sharpe':>7}")
        for label, m in [("B-15 static", m_static),
                         ("Sprint 5 regime-aware", m_regime),
                         ("B-22 adaptive quantile", m_adaptive)]:
            if m is None:
                continue
            hit = f"{m['hit_rate']:.4f}" if not np.isnan(m['hit_rate']) else "  n/a "
            win = f"{m['win_rate']:.4f}" if not np.isnan(m['win_rate']) else "  n/a "
            exp = f"{m['expectancy_pct']:+.3f}" if not np.isnan(m['expectancy_pct']) else "  n/a "
            sh  = f"{m['sharpe']:+.2f}"  if not np.isnan(m['sharpe'])  else "  n/a "
            print(f"     {label:<24} | {m['coverage']:>6.2%} | {m['n_buy']:>4d} | "
                  f"{m['n_sell']:>4d} | {hit:>6} | {win:>6} | {exp:>7} | {sh:>7}")

        fold_results.append({
            "fold":       k,
            "n_calib":    n_calib,
            "n_test":     n_test,
            "static":     m_static,
            "regime":     m_regime,
            "adaptive":   m_adaptive,
        })

    # ── Aggregate across folds ──
    print(f"\n{'='*92}")
    print(f"  Аггрегация по {folds} фолдам:")
    print(f"{'='*92}")

    def _agg(key, getter):
        vals = [getter(r[key]) for r in fold_results
                if r[key] is not None and not np.isnan(getter(r[key]))]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0), len(vals)

    strategies = [("B-15 static", "static"),
                  ("Sprint 5 regime-aware", "regime"),
                  ("B-22 adaptive quantile", "adaptive")]
    for label, key in strategies:
        mu_e, sd_e, ne = _agg(key, lambda m: m["expectancy_pct"])
        mu_s, sd_s, _  = _agg(key, lambda m: m["sharpe"])
        mu_h, sd_h, _  = _agg(key, lambda m: m["hit_rate"])
        mu_c, sd_c, _  = _agg(key, lambda m: m["coverage"])
        if ne == 0:
            print(f"  {label:<24}: no valid folds")
            continue
        print(f"  {label:<24}: cov={mu_c*100:.2f}±{sd_c*100:.2f}%  "
              f"hit={mu_h:.4f}±{sd_h:.4f}  exp%={mu_e:+.3f}±{sd_e:.3f}  "
              f"sharpe={mu_s:+.2f}±{sd_s:.2f}  (folds={ne})")

    return fold_results


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-calib-frac", type=float, default=0.30)
    parser.add_argument("--adaptive-thresholds", action="store_true",
                        help="B-22 fix: добавить Strategy C — quantile-based пороги per-fold")
    parser.add_argument("--q-edge", type=float, default=0.80,
                        help="Quantile для edge_ratio (default 0.80 → ~20%% coverage)")
    parser.add_argument("--q-dir",  type=float, default=0.75,
                        help="Quantile для dir_prob")
    args = parser.parse_args()
    run_walk_forward(folds=args.folds, min_calib_frac=args.min_calib_frac,
                     adaptive_thresholds=args.adaptive_thresholds,
                     q_edge=args.q_edge, q_dir=args.q_dir)
