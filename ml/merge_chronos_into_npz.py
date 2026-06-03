"""Sprint 11.3 — мерж Chronos quantile predictions в ensemble_predictions.npz.

Берёт `chronos_close_quantiles.npz` (выход `chronos_quantile_pred.py`) и записывает
3 квантиля close-канала по всем сэмплам ensemble_predictions.npz через valid_idx
align. Для сэмплов без Chronos-прогноза → NaN (consumer должен fallback'аться).

Ключи, которые пишутся (все в ATR-norm, layout [N_total, fb=5]):
  chronos_close_q10  — q_0.10 (pessimistic ΔC за горизонт)
  chronos_close_q50  — q_0.50 (median ΔC)
  chronos_close_q90  — q_0.90 (optimistic ΔC)
  has_chronos        — bool [N_total]: True там, где есть прогноз

Запуск:
    py -m ml.merge_chronos_into_npz             # мерж + статистика
    py -m ml.merge_chronos_into_npz --inspect   # показать coverage + bias
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

NPZ_PATH     = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
CHRONOS_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "chronos_close_quantiles.npz")


def merge(npz_path: str = NPZ_PATH, chronos_path: str = CHRONOS_PATH) -> dict:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    if not os.path.exists(chronos_path):
        raise FileNotFoundError(chronos_path)

    existing = dict(np.load(npz_path, allow_pickle=True))
    chr_npz  = np.load(chronos_path, allow_pickle=True)

    N_total = len(existing["dir_prob"])
    fb      = int(chr_npz["pred_len"])
    valid_idx = chr_npz["valid_idx"]
    rel_atr   = chr_npz["chronos_close_rel_atr"]   # [N_valid, 3, fb]

    # Поддержка нескольких форматов quantiles из chronos POC
    quants    = chr_npz["quantiles"]
    q10_idx   = int(np.argmin(np.abs(quants - 0.10)))
    q50_idx   = int(np.argmin(np.abs(quants - 0.50)))
    q90_idx   = int(np.argmin(np.abs(quants - 0.90)))

    q10 = np.full((N_total, fb), np.nan, dtype=np.float32)
    q50 = np.full((N_total, fb), np.nan, dtype=np.float32)
    q90 = np.full((N_total, fb), np.nan, dtype=np.float32)
    has_chronos = np.zeros(N_total, dtype=bool)

    q10[valid_idx] = rel_atr[:, q10_idx, :].astype(np.float32)
    q50[valid_idx] = rel_atr[:, q50_idx, :].astype(np.float32)
    q90[valid_idx] = rel_atr[:, q90_idx, :].astype(np.float32)
    has_chronos[valid_idx] = True

    existing["chronos_close_q10"] = q10
    existing["chronos_close_q50"] = q50
    existing["chronos_close_q90"] = q90
    existing["has_chronos"]       = has_chronos
    existing["chronos_model"]     = np.array(str(chr_npz["model_name"]))

    np.savez(npz_path, **existing)
    return {
        "N_total":  N_total,
        "N_chronos": len(valid_idx),
        "coverage": len(valid_idx) / max(N_total, 1),
        "fb":       fb,
        "model":    str(chr_npz["model_name"]),
    }


def inspect(npz_path: str = NPZ_PATH) -> None:
    if not os.path.exists(npz_path):
        print(f"Не найден {npz_path}"); return
    d = np.load(npz_path, allow_pickle=True)
    if "chronos_close_q50" not in d.files:
        print("chronos_close_q* не записан. Запусти: py -m ml.merge_chronos_into_npz")
        return

    has = d["has_chronos"].astype(bool)
    q10 = d["chronos_close_q10"]
    q50 = d["chronos_close_q50"]
    q90 = d["chronos_close_q90"]
    N = len(has)
    n_v = int(has.sum())
    print(f"\nChronos coverage: {n_v}/{N} ({n_v/N*100:.1f}%)")
    print(f"Model: {str(d['chronos_model'])}")
    print(f"fb={q50.shape[1]}")

    if n_v == 0:
        return

    # Actuals для diagnostics
    ohlc = d["ohlc_test"]
    fb = q50.shape[1]
    ohlc_3d = ohlc.reshape(N, fb, 4)
    actual_C = ohlc_3d[has, :, 3]
    # Coverage [q10, q90] per bar
    cov = ((actual_C >= q10[has]) & (actual_C <= q90[has])).mean(axis=0) * 100
    bias = (q50[has] - actual_C).mean(axis=0)
    width = (q90[has] - q10[has]).mean(axis=0)

    print(f"\nPer-bar diagnostics (на valid сэмплах N={n_v}):")
    print(f"  {'bar':>6}  {'coverage':>9}  {'bias q50':>9}  {'width':>7}")
    for i in range(fb):
        print(f"  t+{i+1:<4}  {cov[i]:>7.2f}%  {bias[i]:>+7.3f}    {width[i]:>5.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true", help="показать coverage и bias")
    p.add_argument("--npz", default=NPZ_PATH)
    p.add_argument("--chronos", default=CHRONOS_PATH)
    args = p.parse_args()

    if args.inspect:
        inspect(args.npz); return

    print(f"Мерж {args.chronos} → {args.npz}...")
    stats = merge(args.npz, args.chronos)
    print(f"  N_total:   {stats['N_total']}")
    print(f"  Chronos:   {stats['N_chronos']} (coverage={stats['coverage']*100:.1f}%)")
    print(f"  Missing:   {stats['N_total'] - stats['N_chronos']} (NaN — consumer fallback)")
    print(f"  fb:        {stats['fb']}, model: {stats['model']}")
    print(f"\n✅ Записаны ключи: chronos_close_q10, _q50, _q90, has_chronos")
    print(f"\nДля диагностики: py -m ml.merge_chronos_into_npz --inspect")


if __name__ == "__main__":
    main()
