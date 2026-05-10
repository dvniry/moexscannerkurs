"""Sprint 9.5 (Sprint 7 polish + Sprint 10 B): Brier + reliability per ticker.

Brier score = mean((p_pred - y_true)^2) — ниже лучше; идеальная модель = 0.
Reliability bins: разбиваем prob на 10 bins → для каждого считаем (mean_pred, mean_actual).
Идеальная калибровка: точки лежат на диагонали y=x.

Источник (ensemble_predictions.npz):
  - dir_prob              raw логит-вероятности UP
  - dir_prob_calibrated   после temperature scaling (Sprint 7)
  - dir_prob_platt        после Platt scaling (Sprint 10 B) — лечит асимметрию
  - test_tickers, y_test (0=UP, 1=FLAT, 2=DOWN — бинаризуем UP=1)

Запуск:
    py -m ml.reliability_report                  # global + top-5 / bottom-5 для всех доступных
    py -m ml.reliability_report --per-ticker     # все тикеры по отдельности
    py -m ml.reliability_report --csv out.csv    # экспорт в CSV
    py -m ml.reliability_report --source platt   # per-ticker таблица только по platt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Brier score для бинарной задачи: mean((p - y)^2)."""
    return float(np.mean((p - y) ** 2))


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Разбивает prob на n_bins, считает (mean_pred, mean_actual, count) на bin.

    ECE (Expected Calibration Error) = sum(w_i * |mean_pred_i - mean_actual_i|),
    где w_i = bin_count_i / N. Идеал = 0.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, n_bins - 1)
    rows = []
    n_total = len(p)
    ece = 0.0
    for b in range(n_bins):
        m = (bin_idx == b)
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({
                "bin": b, "lo": bins[b], "hi": bins[b + 1],
                "n": 0, "mean_pred": np.nan, "mean_actual": np.nan,
            })
            continue
        mp = float(p[m].mean())
        ma = float(y[m].mean())
        rows.append({
            "bin": b, "lo": bins[b], "hi": bins[b + 1],
            "n": cnt, "mean_pred": mp, "mean_actual": ma,
        })
        ece += (cnt / n_total) * abs(mp - ma)
    return {"bins": rows, "ece": float(ece)}


def render_ascii_reliability(rel: dict, label: str) -> None:
    """ASCII-диаграмма надёжности — для тех у кого нет matplotlib под рукой."""
    print(f"\n  ── Reliability diagram: {label}  (ECE={rel['ece']*100:.2f}%) ──")
    print(f"  bin   range          | n      | pred   | actual | bias    | bar")
    for r in rel["bins"]:
        if r["n"] == 0:
            continue
        bias = r["mean_actual"] - r["mean_pred"]
        # Bar: визуализация bias (-/+ от диагонали)
        bar_pos = int(round(bias * 30))
        if bar_pos > 0:
            bar = " " * 15 + "│" + "▰" * min(bar_pos, 14)
        elif bar_pos < 0:
            bar = " " * (15 + bar_pos) + "▰" * (-bar_pos) + "│"
        else:
            bar = " " * 15 + "│"
        print(f"  {r['bin']:<3d}  [{r['lo']:.2f},{r['hi']:.2f}] | "
              f"{r['n']:>5d}  | {r['mean_pred']:.4f} | {r['mean_actual']:.4f} | "
              f"{bias:+.4f} | {bar}")


def per_ticker_summary(p: np.ndarray, y: np.ndarray, tickers: np.ndarray) -> list[dict]:
    """Собирает {ticker, n, brier, ece} per ticker."""
    rows = []
    for t in np.unique(tickers):
        m = (tickers == t)
        if m.sum() < 10:
            continue
        p_t = p[m]; y_t = y[m]
        rel = reliability_bins(p_t, y_t, n_bins=10)
        rows.append({
            "ticker": str(t),
            "n":      int(m.sum()),
            "brier":  brier_score(p_t, y_t),
            "ece":    rel["ece"],
            "mean_pred":   float(p_t.mean()),
            "mean_actual": float(y_t.mean()),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-ticker", action="store_true",
                   help="Reliability + Brier для всех тикеров (default — top/bottom 5)")
    p.add_argument("--csv", metavar="PATH",
                   help="Экспорт per-ticker таблицы в CSV")
    p.add_argument("--no-calibrated", action="store_true",
                   help="Не считать temperature/Platt — только raw")
    p.add_argument("--source", choices=["raw", "calibrated", "platt", "auto"],
                   default="auto",
                   help="Какой источник использовать для per-ticker таблицы. "
                        "auto: platt > calibrated > raw (по доступности)")
    args = p.parse_args()

    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден"); sys.exit(1)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    dir_prob = npz["dir_prob"].astype(np.float32)
    y_test   = npz["y_test"].astype(np.int8)
    tickers  = (np.array([str(t) for t in npz["test_tickers"]])
                if "test_tickers" in npz.files else
                np.full(len(dir_prob), "_unk_", dtype="U16"))

    # Бинаризация: UP=1, иначе=0 (FLAT и DOWN объединены, как в DecisionLayer)
    y_bin = (y_test == 0).astype(np.float32)

    has_calib = ("dir_prob_calibrated" in npz.files) and (not args.no_calibrated)
    has_platt = ("dir_prob_platt"      in npz.files) and (not args.no_calibrated)
    dir_calib = (npz["dir_prob_calibrated"].astype(np.float32)
                 if has_calib else None)
    dir_platt = (npz["dir_prob_platt"].astype(np.float32)
                 if has_platt else None)

    print(f"\n  Sprint 7 + Sprint 10 B — Brier + Reliability")
    print(f"  ─────────────────────────────────────────────")
    print(f"  N={len(dir_prob)}  Тикеров: {len(np.unique(tickers))}")
    print(f"  UP rate (y=1):  {y_bin.mean():.4f}")
    print(f"  Calibrated (T): {'yes' if has_calib else 'нет'}")
    print(f"  Platt (a,b):    {'yes' if has_platt else 'нет'}")

    # ── Глобальные метрики ──
    print(f"\n  ── GLOBAL ──")
    brier_raw = brier_score(dir_prob, y_bin)
    rel_raw   = reliability_bins(dir_prob, y_bin)
    print(f"  RAW dir_prob:        Brier={brier_raw:.4f}  ECE={rel_raw['ece']*100:.2f}%")
    rel_cal = rel_platt = None
    if has_calib:
        brier_cal = brier_score(dir_calib, y_bin)
        rel_cal   = reliability_bins(dir_calib, y_bin)
        print(f"  TEMPERATURE dir_prob_calibrated: "
              f"Brier={brier_cal:.4f}  ECE={rel_cal['ece']*100:.2f}%  "
              f"(Δ ECE vs raw: {(rel_raw['ece']-rel_cal['ece'])*100:+.2f}pp)")
    if has_platt:
        brier_pla = brier_score(dir_platt, y_bin)
        rel_platt = reliability_bins(dir_platt, y_bin)
        print(f"  PLATT       dir_prob_platt:      "
              f"Brier={brier_pla:.4f}  ECE={rel_platt['ece']*100:.2f}%  "
              f"(Δ ECE vs raw: {(rel_raw['ece']-rel_platt['ece'])*100:+.2f}pp)")

    render_ascii_reliability(rel_raw, "RAW")
    if rel_cal is not None:
        render_ascii_reliability(rel_cal, "TEMPERATURE")
    if rel_platt is not None:
        render_ascii_reliability(rel_platt, "PLATT")

    # ── Per-ticker ──
    print(f"\n  ── PER-TICKER ──")
    if args.source == "auto":
        if has_platt:
            src_label, src_p = "platt", dir_platt
        elif has_calib:
            src_label, src_p = "temperature", dir_calib
        else:
            src_label, src_p = "raw", dir_prob
    elif args.source == "platt":
        if not has_platt:
            print("  [WARN] dir_prob_platt отсутствует, fallback на raw")
            src_label, src_p = "raw", dir_prob
        else:
            src_label, src_p = "platt", dir_platt
    elif args.source == "calibrated":
        if not has_calib:
            print("  [WARN] dir_prob_calibrated отсутствует, fallback на raw")
            src_label, src_p = "raw", dir_prob
        else:
            src_label, src_p = "temperature", dir_calib
    else:
        src_label, src_p = "raw", dir_prob
    print(f"  Источник для per-ticker: {src_label}")
    rows = per_ticker_summary(src_p, y_bin, tickers)

    rows_brier_sorted = sorted(rows, key=lambda r: r["brier"])
    print(f"\n  Top-5 best calibrated ({src_label} dir_prob, lowest Brier):")
    print(f"  {'ticker':<8} {'n':>5} {'brier':>8} {'ece%':>7} {'mean_p':>8} {'mean_y':>8}")
    for r in rows_brier_sorted[:5]:
        print(f"  {r['ticker']:<8} {r['n']:>5d} {r['brier']:>8.4f} {r['ece']*100:>7.2f} "
              f"{r['mean_pred']:>8.4f} {r['mean_actual']:>8.4f}")
    print(f"\n  Worst-5 calibrated ({src_label} dir_prob, highest Brier):")
    print(f"  {'ticker':<8} {'n':>5} {'brier':>8} {'ece%':>7} {'mean_p':>8} {'mean_y':>8}")
    for r in rows_brier_sorted[-5:]:
        print(f"  {r['ticker']:<8} {r['n']:>5d} {r['brier']:>8.4f} {r['ece']*100:>7.2f} "
              f"{r['mean_pred']:>8.4f} {r['mean_actual']:>8.4f}")

    if args.per_ticker:
        print(f"\n  ── ALL TICKERS ──")
        print(f"  {'ticker':<8} {'n':>5} {'brier':>8} {'ece%':>7} {'mean_p':>8} {'mean_y':>8}")
        for r in sorted(rows, key=lambda x: x["ticker"]):
            print(f"  {r['ticker']:<8} {r['n']:>5d} {r['brier']:>8.4f} {r['ece']*100:>7.2f} "
                  f"{r['mean_pred']:>8.4f} {r['mean_actual']:>8.4f}")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["ticker", "n", "brier", "ece",
                                                     "mean_pred", "mean_actual"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\n  ✓ CSV сохранён: {args.csv}")


if __name__ == "__main__":
    main()
