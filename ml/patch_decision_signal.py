"""B-15 / Sprint 5: переапплицирует DecisionLayer на существующем
ensemble_predictions.npz без перетренировки V3.

Режимы:
  default       — единые пороги B-15 (edge=5.0, dir=0.75, sell=0.55)
  --regime-aware — per-regime пороги (требует test_regime в npz, см. patch_ensemble_regime)

Использование:
    py -m ml.patch_decision_signal
    py -m ml.patch_decision_signal --regime-aware
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


def _stats(sig: np.ndarray, y: np.ndarray, label: str):
    n_buy  = int((sig == SIG_BUY ).sum())
    n_hold = int((sig == SIG_HOLD).sum())
    n_sell = int((sig == SIG_SELL).sum())
    print(f"\n  {label}:")
    print(f"    BUY={n_buy}  HOLD={n_hold}  SELL={n_sell}  cov={(n_buy+n_sell)/len(sig):.2%}")
    is_buy  = (sig == SIG_BUY)
    is_sell = (sig == SIG_SELL)
    hit = np.zeros_like(sig, dtype=bool)
    hit[is_buy]  = (y[is_buy]  == 0)
    hit[is_sell] = (y[is_sell] == 2)
    n_act = int((is_buy | is_sell).sum())
    if n_act > 0:
        hit_rate = float(hit[is_buy | is_sell].mean())
        print(f"    Hit rate: {hit_rate:.4f}  (N={n_act})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime-aware", action="store_true",
                        help="Применить RegimeAwareDecisionLayer (требует test_regime в npz)")
    args = parser.parse_args()

    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден.")
        return 1

    existing = dict(np.load(NPZ_PATH, allow_pickle=True))

    if "decision_signal" not in existing:
        print(f"ERROR: decision_signal отсутствует — нет EconomicHeads. Перетренируйте V3.")
        return 2

    old_sig = existing["decision_signal"]
    y       = existing["y_test"]
    _stats(old_sig, y, "Старый decision_signal")

    if args.regime_aware:
        if "test_regime" not in existing:
            print(f"\nERROR: test_regime отсутствует. Сначала: py -m ml.patch_ensemble_regime")
            return 3

        regime = existing["test_regime"].astype(np.int8)
        rdl = RegimeAwareDecisionLayer(costs_from_config())

        print(f"\nПрименяем RegimeAwareDecisionLayer (Sprint 5):")
        for rid, params in rdl.regime_thresholds.items():
            n_r = int((regime == rid).sum())
            if n_r == 0 and rid != -1:
                continue
            name = rdl.REGIME_NAMES[rid]
            disabled = (params["min_edge_ratio"] >= 99.0)
            tag = "  [DISABLED]" if disabled else ""
            print(f"  {name:>7}: edge={params['min_edge_ratio']:.1f} "
                  f"dir={params['min_dir_prob']:.2f} sell={params['min_sell_dir_prob']:.2f}  "
                  f"N={n_r}{tag}")

        out = rdl.decide_numpy(
            dir_prob  = existing["dir_prob"].astype(np.float32),
            mfe_mae   = existing["mfe_mae_pred"].astype(np.float32),
            fill_prob = existing["fill_prob"].astype(np.float32),
            edge_pred = existing["edge_pred"].astype(np.float32),
            regime    = regime,
        )
        new_sig  = out["signal"]
        new_conf = out["confidence"]

        # Per-regime breakdown
        cov_per = rdl.coverage_per_regime(new_sig, regime)
        print(f"\nCoverage per regime:")
        for name, m in cov_per.items():
            print(f"  {name:>7}: BUY={m['buy']:>4} SELL={m['sell']:>4} "
                  f"cov={m['coverage']:.2%}  (N={m['n']})")
    else:
        dl = DecisionLayer(costs_from_config())
        print(f"\nПрименяем DecisionLayer с порогами B-15:")
        print(f"  edge={dl.min_edge_ratio}  dir={dl.min_dir_prob}  sell={dl.min_sell_dir_prob}  "
              f"fill={dl.min_fill_prob}  rr={dl.min_rr}")

        out = dl.decide_numpy(
            dir_prob  = existing["dir_prob"].astype(np.float32),
            mfe_mae   = existing["mfe_mae_pred"].astype(np.float32),
            fill_prob = existing["fill_prob"].astype(np.float32),
            edge_pred = existing["edge_pred"].astype(np.float32),
        )
        new_sig  = out["signal"]
        new_conf = out["confidence"]

    _stats(new_sig, y, "Новый decision_signal")

    existing["decision_signal"]     = new_sig
    existing["decision_confidence"] = new_conf
    np.savez(NPZ_PATH, **existing)
    print(f"\n✓ Сохранено: {NPZ_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
