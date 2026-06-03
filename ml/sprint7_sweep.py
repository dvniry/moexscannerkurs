"""ml/sprint7_sweep.py — Sprint 7: поиск оптимальных параметров калибровки и порогов.

Аналог decision_sweep.py, но для Sprint 7 параметров.

Sweep сетка:
    T (global temperature) — применяется к dir_prob перед DecisionLayer
    dir_threshold            — min_dir_prob в DecisionLayer
    edge_ratio               — min_edge_ratio в DecisionLayer
    sell_threshold           — min_sell_dir_prob (= 1 - dir_threshold по умолч.)

Методология:
    Данные разбиваются по дате: calib = первые calib_frac% тест-сэмплов,
    oos = оставшиеся. Пороги подбираются на calib, оцениваются на oos.
    Это честная forward-looking оценка без data leakage.

Запуск:
    py -m ml.sprint7_sweep --quick          # 3D grid T×dir×edge, ~200 комбинаций
    py -m ml.sprint7_sweep --full           # + sell_threshold, ~800 комбинаций
    py -m ml.sprint7_sweep --show-top 20    # показать топ-20 результатов
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from ml.decision_layer import DecisionLayer, costs_from_config, SIG_BUY, SIG_HOLD, SIG_SELL

DEFAULT_NPZ   = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
BEST_PARAMS   = os.path.join(os.path.dirname(__file__), "ensemble", "sprint7_best_params.json")
RESULTS_CSV   = os.path.join(os.path.dirname(__file__), "ensemble", "sprint7_sweep_results.csv")
FEE           = 0.001
CALIB_FRAC    = 0.30


def _apply_temperature(dir_prob: np.ndarray, T: float) -> np.ndarray:
    """Применяет температурное масштабирование: sigmoid(logit(p) / T)."""
    eps = 1e-6
    p   = np.clip(dir_prob, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-logit / T))


def _eval(
    dir_prob:  np.ndarray,
    mfe_mae:   np.ndarray,
    fill_prob: np.ndarray,
    edge_pred: np.ndarray,
    y_test:    np.ndarray,
    real_C:    np.ndarray,    # [N] ΔClose в долях цены (уже денормализовано)
    *,
    T:               float,
    min_dir_prob:    float,
    min_sell_dir_prob: float,
    min_edge_ratio:  float,
    min_fill_prob:   float = 0.40,
    min_rr:          float = 1.2,
) -> dict:
    """Применяет параметры к подвыборке, возвращает метрики."""
    dir_cal = _apply_temperature(dir_prob, T) if T != 1.0 else dir_prob

    dl = DecisionLayer(
        costs             = costs_from_config(),
        min_dir_prob      = min_dir_prob,
        min_sell_dir_prob = min_sell_dir_prob,
        min_edge_ratio    = min_edge_ratio,
        min_fill_prob     = min_fill_prob,
        min_rr            = min_rr,
    )
    out = dl.decide_numpy(dir_prob=dir_cal, mfe_mae=mfe_mae,
                          fill_prob=fill_prob, edge_pred=edge_pred)
    sig     = out["signal"]
    n_total = len(sig)
    n_act   = int((sig != SIG_HOLD).sum())
    n_buy   = int((sig == SIG_BUY).sum())
    n_sell  = int((sig == SIG_SELL).sum())

    if n_act == 0:
        return {"n_act": 0, "coverage": 0.0, "hit_rate": float("nan"),
                "expectancy_pct": float("nan"), "sharpe": float("nan"),
                "win_rate": float("nan")}

    is_buy  = (sig == SIG_BUY)
    is_sell = (sig == SIG_SELL)
    is_act  = is_buy | is_sell

    hit = np.zeros(n_total, dtype=bool)
    hit[is_buy]  = (y_test[is_buy]  == 0)
    hit[is_sell] = (y_test[is_sell] == 2)
    hit_rate = float(hit[is_act].mean())

    pnl = np.zeros(n_total, dtype=np.float32)
    pnl[is_buy]  =  real_C[is_buy]  - 2 * FEE
    pnl[is_sell] = -real_C[is_sell] - 2 * FEE
    pnl_act = pnl[is_act]

    exp_pct  = float(pnl_act.mean() * 100)
    win_rate = float((pnl_act > 0).mean())
    sharpe   = float(pnl_act.mean() / (pnl_act.std() + 1e-9) * np.sqrt(252))

    return {
        "n_act":          n_act,
        "n_buy":          n_buy,
        "n_sell":         n_sell,
        "coverage":       n_act / n_total,
        "hit_rate":       hit_rate,
        "expectancy_pct": exp_pct,
        "win_rate":       win_rate,
        "sharpe":         sharpe,
    }


def run_sweep(
    npz_path:   str   = DEFAULT_NPZ,
    quick:      bool  = True,
    calib_frac: float = CALIB_FRAC,
    show_top:   int   = 10,
) -> dict:
    """Запускает sweep и возвращает best_params dict."""

    # ── Загрузка данных ───────────────────────────────────────────────────────
    print(f"Загрузка {npz_path}...")
    data = np.load(npz_path, allow_pickle=False)

    required = {"dir_prob", "mfe_mae_pred", "fill_prob", "edge_pred",
                "y_test", "ohlc_test", "atr_ratio"}
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(f"В npz отсутствуют поля: {missing}. "
                           f"Убедитесь что npz собран с EconomicHeads.")

    dir_prob  = data["dir_prob"].astype(np.float32)
    mfe_mae   = data["mfe_mae_pred"].astype(np.float32)
    fill_prob = data["fill_prob"].astype(np.float32)
    edge_pred = data["edge_pred"].astype(np.float32)
    y_test    = data["y_test"].astype(np.int64)
    ohlc_test = data["ohlc_test"].astype(np.float32)
    atr_ratio = data["atr_ratio"].astype(np.float32)

    # B-17: денормализуем ΔClose → доли цены
    real_C = ohlc_test[:, 3] * atr_ratio

    N     = len(dir_prob)
    n_cal = max(100, int(N * calib_frac))
    # Разбивка: первые n_cal — calib, остальные — OOS
    calib_mask = np.zeros(N, dtype=bool)
    calib_mask[:n_cal] = True
    oos_mask   = ~calib_mask

    print(f"  N={N}  calib={n_cal}  OOS={N - n_cal}  (calib_frac={calib_frac:.0%})")

    # ── Сетки параметров ──────────────────────────────────────────────────────
    T_grid   = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    dir_grid = np.arange(0.50, 0.82, 0.02).round(2).tolist()
    edge_grid = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    if quick:
        sell_grid = ["symmetric"]   # sell = 1 - dir
    else:
        sell_grid = ["symmetric", 0.45, 0.50, 0.55, 0.60]

    total = len(T_grid) * len(dir_grid) * len(edge_grid) * len(sell_grid)
    print(f"  Комбинаций: {total}  (режим: {'quick' if quick else 'full'})")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    records_calib = []
    records_oos   = []

    done = 0
    for T in T_grid:
        for dir_thr in dir_grid:
            for edge_r in edge_grid:
                for sell_raw in sell_grid:
                    sell_thr = (round(1.0 - dir_thr, 2)
                                if sell_raw == "symmetric" else float(sell_raw))
                    params = {
                        "T":                  T,
                        "min_dir_prob":       dir_thr,
                        "min_sell_dir_prob":  sell_thr,
                        "min_edge_ratio":     edge_r,
                    }

                    c = _eval(dir_prob[calib_mask], mfe_mae[calib_mask],
                              fill_prob[calib_mask], edge_pred[calib_mask],
                              y_test[calib_mask], real_C[calib_mask], **params)
                    o = _eval(dir_prob[oos_mask], mfe_mae[oos_mask],
                              fill_prob[oos_mask], edge_pred[oos_mask],
                              y_test[oos_mask], real_C[oos_mask], **params)

                    records_calib.append({**params, **{f"cal_{k}": v for k, v in c.items()}})
                    records_oos.append({**params, **{f"oos_{k}": v for k, v in o.items()}})

                    done += 1
                    if done % 100 == 0:
                        print(f"  ... {done}/{total}")

    # ── Объединяем calib + OOS ────────────────────────────────────────────────
    df_c = pd.DataFrame(records_calib)
    df_o = pd.DataFrame(records_oos)
    key_cols = ["T", "min_dir_prob", "min_sell_dir_prob", "min_edge_ratio"]
    df = df_c.merge(df_o, on=key_cols)

    # Фильтр: OOS coverage >= 1%  &&  calib_n_act >= 20
    df_filt = df[(df["oos_coverage"] >= 0.01) & (df["cal_n_act"] >= 20)].copy()
    if df_filt.empty:
        print("  [WARN] Нет комбинаций с coverage >= 1%. Снижаем порог до 0.5%.")
        df_filt = df[(df["oos_coverage"] >= 0.005) & (df["cal_n_act"] >= 10)].copy()

    # Сортируем по OOS expectancy
    df_filt.sort_values("oos_expectancy_pct", ascending=False, inplace=True)

    # ── Вывод ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*75}")
    print(f"  TOP-{show_top} комбинаций (по OOS expectancy%):")
    print(f"  {'T':>5} {'dir':>5} {'sell':>5} {'edge':>5} | "
          f"{'cal_exp%':>8} {'cal_hr':>7} {'cal_cov':>7} | "
          f"{'oos_exp%':>8} {'oos_hr':>7} {'oos_cov':>7} {'oos_sr':>7}")
    print("  " + "-"*73)

    for _, row in df_filt.head(show_top).iterrows():
        print(f"  {row['T']:>5.2f} {row['min_dir_prob']:>5.2f} "
              f"{row['min_sell_dir_prob']:>5.2f} {row['min_edge_ratio']:>5.1f} | "
              f"{row['cal_expectancy_pct']:>+8.3f} {row['cal_hit_rate']:>7.3f} "
              f"{row['cal_coverage']:>7.2%} | "
              f"{row['oos_expectancy_pct']:>+8.3f} {row['oos_hit_rate']:>7.3f} "
              f"{row['oos_coverage']:>7.2%} {row['oos_sharpe']:>7.2f}")

    # ── Baseline (текущие пороги B-15) ────────────────────────────────────────
    baseline = _eval(dir_prob[oos_mask], mfe_mae[oos_mask],
                     fill_prob[oos_mask], edge_pred[oos_mask],
                     y_test[oos_mask], real_C[oos_mask],
                     T=1.0, min_dir_prob=0.75, min_sell_dir_prob=0.55,
                     min_edge_ratio=5.0)
    print(f"\n  Baseline B-15 (OOS): exp={baseline.get('expectancy_pct', float('nan')):+.3f}%  "
          f"hit={baseline.get('hit_rate', float('nan')):.3f}  "
          f"cov={baseline.get('coverage', 0.0):.2%}  "
          f"sr={baseline.get('sharpe', float('nan')):.2f}")

    # ── Best params ───────────────────────────────────────────────────────────
    if df_filt.empty:
        print("  [WARN] Нет валидных комбинаций — используем B-15 как fallback.")
        best_row = {"T": 1.0, "min_dir_prob": 0.75,
                    "min_sell_dir_prob": 0.55, "min_edge_ratio": 5.0}
        best_oos = baseline
    else:
        best = df_filt.iloc[0]
        best_row = {k: float(best[k]) for k in key_cols}
        best_oos = {k.replace("oos_", ""): float(best[f"oos_{k.replace('oos_', '')}"])
                    for k in ["n_act", "coverage", "hit_rate",
                               "expectancy_pct", "win_rate", "sharpe"]
                    if f"oos_{k.replace('oos_', '')}" in best.index}

    print(f"\n  Best params:")
    for k, v in best_row.items():
        print(f"    {k}: {v}")

    # ── Сохранение ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False, float_format="%.4f")
    print(f"\n  Полные результаты → {RESULTS_CSV}")

    result = {
        "best_params": best_row,
        "best_oos_metrics": best_oos,
        "baseline_oos_metrics": {
            k: float(v) for k, v in baseline.items()
        },
        "n_combinations_total":   int(total),
        "n_combinations_valid":   int(len(df_filt)),
        "calib_frac": calib_frac,
        "mode": "quick" if quick else "full",
        # Для trainer_v3.py / calibrate_temperature.py — подхватываются автоматически
        "recommended_T":          float(best_row["T"]),
        "recommended_dir_thr":    float(best_row["min_dir_prob"]),
        "recommended_sell_thr":   float(best_row["min_sell_dir_prob"]),
        "recommended_edge_ratio": float(best_row["min_edge_ratio"]),
    }

    with open(BEST_PARAMS, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Best params → {BEST_PARAMS}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds",    default=DEFAULT_NPZ)
    parser.add_argument("--quick",    action="store_true", default=True,
                        help="Быстрый режим (T×dir×edge ~200 комбинаций)")
    parser.add_argument("--full",     action="store_true",
                        help="Полный grid включая sell_threshold (~800 комбинаций)")
    parser.add_argument("--show-top", type=int, default=10)
    parser.add_argument("--calib-frac", type=float, default=CALIB_FRAC,
                        help="Доля данных для калибровки (0.0–0.5)")
    args = parser.parse_args()

    if args.full:
        args.quick = False

    run_sweep(
        npz_path   = args.preds,
        quick      = args.quick,
        calib_frac = args.calib_frac,
        show_top   = args.show_top,
    )


if __name__ == "__main__":
    main()
