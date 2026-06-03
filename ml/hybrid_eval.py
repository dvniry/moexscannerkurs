"""Sprint 11.3 — Hybrid E2 (meta bear-only) + D7c (chronos LONG-inverted) overlap analysis.

Два независимых profitable signals найдены на 6-м ребилде:
  - E2_meta_bear_only: 127 trades, win 55.12%, total +1.11%, Sharpe +1.25
  - D7c_chronos_invLONG: 48 trades, win 50%, total +0.11%, Sharpe +0.14

Этот скрипт:
  1. Извлекает trade indices обоих source
  2. Считает overlap (intersection vs symmetric diff vs union)
  3. Backtest: union (any signal), intersection (both agree)
  4. Печатает comparison table

Все trades — SHORT (E2 в bear regime даёт только SELL; D7c после inversion даёт только SHORT).
Direction совпадает → hybrid тривиально через объединение индексов.

Запуск:
    py -m ml.hybrid_eval
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
FUTURE_BARS = 5
FEE = 0.001


def _pnl_short(real_C_pct: float) -> float:
    """SHORT PnL = -ΔC. Возвращает gross (без вычитания fees)."""
    return -real_C_pct


def _normalize_pnl(arr_pct: np.ndarray) -> dict:
    """Стат по списку trades (gross % уже)."""
    if len(arr_pct) == 0:
        return {"n": 0, "win": 0.0, "gross": 0.0, "net": 0.0, "total": 0.0,
                "sharpe": 0.0, "best": 0.0, "worst": 0.0}
    net = arr_pct - 2 * FEE
    return {
        "n": len(arr_pct),
        "win": float((arr_pct > 0).mean()),
        "gross": float(arr_pct.mean()) * 100,                # %
        "net": float(net.mean()) * 100,
        "total": float(net.sum()) * 100,                      # cumulative %
        "sharpe": float(net.mean() / max(net.std(ddof=1), 1e-9))
                  if len(net) > 1 else 0.0,
        "best": float(arr_pct.max()) * 100,
        "worst": float(arr_pct.min()) * 100,
    }


def extract_e2_indices(data: dict) -> np.ndarray:
    """E2: decision_signal == 2 (SELL) AND not bull regime (bull=OFF)."""
    sig = data["decision_signal"]
    regime = data.get("test_regime", None)
    mask = (sig == 2)
    if regime is not None:
        mask &= (regime != 2)   # bull=OFF
    return np.where(mask)[0]


def extract_d7c_indices(data: dict, fb: int = FUTURE_BARS) -> np.ndarray:
    """D7c: chronos band non-straddles-zero, last bar (t+fb−1).

    Все trades direction = SHORT после LONG→SHORT инверсии (см. backtest D7c).
    """
    if "chronos_close_q10" not in data:
        return np.array([], dtype=np.int64)
    q10 = data["chronos_close_q10"]
    q90 = data["chronos_close_q90"]
    has = data["has_chronos"].astype(bool)
    # Используем t+fb−1 (последний бар горизонта) — как в simulate_chronos_strategy
    last = fb - 1
    q10_last = q10[:, last]
    q90_last = q90[:, last]
    valid = has & np.isfinite(q10_last) & np.isfinite(q90_last)
    band_confident = (q10_last > 0) | (q90_last < 0)
    return np.where(valid & band_confident)[0]


def compute_pnl_short(indices: np.ndarray, real_C_pct: np.ndarray) -> np.ndarray:
    """Gross PnL для SHORT на этих индексах. real_C_pct — % изменение close за горизонт."""
    return -real_C_pct[indices]


def print_stats(label: str, stats: dict) -> None:
    if stats["n"] == 0:
        print(f"  {label:<35} N=0 (empty)")
        return
    print(f"  {label:<35} N={stats['n']:>4d}  win%={stats['win']*100:5.2f}  "
          f"gross={stats['gross']:+6.3f}%  net={stats['net']:+6.3f}%  "
          f"total={stats['total']:+6.2f}%  Sharpe={stats['sharpe']:+5.2f}")


def main():
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден"); sys.exit(1)
    d = dict(np.load(NPZ_PATH, allow_pickle=True))

    N = len(d["dir_prob"])
    fb = FUTURE_BARS
    # ohlc_test row-major [N, fb*4] → C-канал last bar
    ohlc_3d = d["ohlc_test"].reshape(N, fb, 4)
    atr = d["atr_ratio"]
    # ΔC за горизонт t+fb−1 в %
    real_C_atr = ohlc_3d[:, fb - 1, 3]
    real_C_pct = real_C_atr * atr * np.sqrt(fb)    # ATR-norm → %

    # ── Indices ────────────────────────────────────────────────
    idx_e2  = extract_e2_indices(d)
    idx_d7c = extract_d7c_indices(d, fb=fb)
    set_e2  = set(int(i) for i in idx_e2)
    set_d7c = set(int(i) for i in idx_d7c)

    common = sorted(set_e2 & set_d7c)
    only_e2 = sorted(set_e2 - set_d7c)
    only_d7c = sorted(set_d7c - set_e2)
    union = sorted(set_e2 | set_d7c)

    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  Hybrid E2 + D7c — Sprint 11.3 overlap analysis")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  N_total samples: {N}")
    print(f"\n  E2 (meta bear-only SELL): {len(set_e2)} trades")
    print(f"  D7c (chronos LONG-inverted SHORT): {len(set_d7c)} trades")
    print(f"  Both directions = SHORT → объединение тривиально\n")

    print(f"  Overlap matrix:")
    print(f"    Intersection (E2 AND D7c): {len(common):3d}")
    print(f"    Only E2 (not D7c):         {len(only_e2):3d}")
    print(f"    Only D7c (not E2):         {len(only_d7c):3d}")
    print(f"    Union (E2 OR D7c):         {len(union):3d}")
    if len(set_e2):
        print(f"    D7c overlap rate (D7c∩E2/E2): {len(common)/len(set_e2)*100:.1f}%")
    if len(set_d7c):
        print(f"    E2 overlap rate (D7c∩E2/D7c): {len(common)/len(set_d7c)*100:.1f}%")

    # ── Backtest каждой подгруппы ─────────────────────────────
    print(f"\n  ── PnL по подгруппам (все SHORT, gross = -ΔC за t+fb) ──")
    def _arr(xs):
        return np.array(list(xs), dtype=np.int64) if xs else np.array([], dtype=np.int64)
    pnl_e2   = compute_pnl_short(_arr(sorted(set_e2)),   real_C_pct)
    pnl_d7c  = compute_pnl_short(_arr(sorted(set_d7c)),  real_C_pct)
    pnl_comm = compute_pnl_short(_arr(common),           real_C_pct)
    pnl_o_e2 = compute_pnl_short(_arr(only_e2),          real_C_pct)
    pnl_o_d7c= compute_pnl_short(_arr(only_d7c),         real_C_pct)
    pnl_un   = compute_pnl_short(_arr(union),            real_C_pct)

    print_stats("E2 (all):",                _normalize_pnl(pnl_e2))
    print_stats("D7c (all):",               _normalize_pnl(pnl_d7c))
    print_stats("Intersection (E2 AND D7c):", _normalize_pnl(pnl_comm))
    print_stats("Only E2 (E2 \\ D7c):",     _normalize_pnl(pnl_o_e2))
    print_stats("Only D7c (D7c \\ E2):",    _normalize_pnl(pnl_o_d7c))
    print(f"  " + "─" * 100)
    print_stats("UNION (E2 OR D7c) ⭐:",   _normalize_pnl(pnl_un))

    # ── Per-ticker breakdown union ────────────────────────────
    print(f"\n  ── Top-10 тикеров по union total% (где торгует hybrid) ──")
    tickers = d["test_tickers"]
    union_arr = _arr(union)
    union_tickers = tickers[union_arr]
    union_pnl_net = pnl_un - 2 * FEE
    from collections import defaultdict
    by_t: dict[str, list[float]] = defaultdict(list)
    for t, p in zip(union_tickers, union_pnl_net):
        by_t[str(t)].append(float(p))
    rows = []
    for t, ps in by_t.items():
        arr = np.array(ps)
        rows.append((t, len(arr), float(arr.mean()) * 100, float(arr.sum()) * 100,
                     float((arr > 0).mean()) * 100))
    rows.sort(key=lambda r: -r[3])   # sort by total desc
    print(f"  {'ticker':<8} {'N':>4}  {'net%':>7}  {'total%':>8}  {'win%':>6}")
    for r in rows[:10]:
        print(f"  {r[0]:<8} {r[1]:>4d}  {r[2]:>+6.3f}  {r[3]:>+7.2f}  {r[4]:>5.1f}")

    # ── Вердикт ────────────────────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  ВЕРДИКТ")
    print(f"══════════════════════════════════════════════════════════════════")
    e2_stats = _normalize_pnl(pnl_e2)
    union_stats = _normalize_pnl(pnl_un)
    diff_total = union_stats["total"] - e2_stats["total"]
    diff_n     = union_stats["n"] - e2_stats["n"]
    print(f"  E2 alone:   N={e2_stats['n']}, total {e2_stats['total']:+.2f}%, "
          f"Sharpe {e2_stats['sharpe']:+.2f}")
    print(f"  Union:      N={union_stats['n']}, total {union_stats['total']:+.2f}%, "
          f"Sharpe {union_stats['sharpe']:+.2f}")
    print(f"  Δ Union−E2: N{diff_n:+d}, total {diff_total:+.2f}pp  "
          f"{'⭐ hybrid лучше' if diff_total > 0 else 'E2 alone лучше'}")


if __name__ == "__main__":
    main()
