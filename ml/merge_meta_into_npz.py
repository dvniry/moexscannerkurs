"""Sprint 11.2 — мерж meta_dir_prob (V3) в ensemble_predictions.npz.

После каждого ребилда MetaLearner v3 даёт holdout accuracy выше raw V3 backbone
(на 6-м ребилде: 0.5551 holdout vs 0.5273 raw test = +2.78pp lift). Decision_layer
видит только raw `dir_prob`/`dir_prob_platt`, не meta. Этот скрипт переносит
meta-предсказания в основной npz по (date, ticker) join.

Что записывается:
  ensemble_predictions.npz ← key 'meta_dir_prob' [N]: P(UP) от MetaV3
                          ← key 'meta_dir_logit' [N]: raw logit (для re-calibration)
                          ← key 'has_meta'   [N] bool: True если есть meta-предсказание
                                                       (иначе fallback на dir_prob_platt)

Для сэмплов без meta (например, нет hourly данных в данный день) используется
NaN в meta_dir_prob; consumer должен fallback'аться на dir_prob_platt.

Запуск:
    py -m ml.merge_meta_into_npz             # мерж + статистика
    py -m ml.merge_meta_into_npz --inspect   # показать coverage + распределения
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from ml.meta_ensemble import (
    META_FEAT_PATH_V3, META_MODEL_PATH_V3, MetaLearnerV3,
)

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")


def _norm_date(arr) -> np.ndarray:
    """Приводит даты к строкам формата YYYY-MM-DD (10 chars) для join."""
    return np.array([str(d)[:10] for d in arr], dtype="U10")


def compute_meta_preds() -> dict:
    """Загружает meta_features_v3 + meta_learner_v3, считает P(UP) для всех сэмплов."""
    for p in [META_FEAT_PATH_V3, META_MODEL_PATH_V3]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Не найден {p}. Запусти: py -m ml.meta_ensemble --version v3")

    data = np.load(META_FEAT_PATH_V3, allow_pickle=True)
    X     = data["X"]
    dates = _norm_date(data["dates"])
    tickers = np.array([str(t) for t in data["tickers"]], dtype="U16")

    ckpt = torch.load(META_MODEL_PATH_V3, map_location="cpu", weights_only=True)
    n_feat = ckpt.get("n_feat", X.shape[1])
    hidden = ckpt.get("hidden", 128)
    model = MetaLearnerV3(n_feat=n_feat, hidden=hidden)
    model.load_state_dict(ckpt["state"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float32)
        probs  = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    return {
        "dates":   dates,
        "tickers": tickers,
        "logit":   logits,
        "prob":    probs,
    }


def merge_into_npz(meta: dict, npz_path: str = NPZ_PATH) -> dict:
    """Записывает meta_dir_prob/meta_dir_logit/has_meta в ensemble_predictions.npz."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Не найден {npz_path}. Запусти retrain_all сначала.")

    existing = dict(np.load(npz_path, allow_pickle=True))
    N = len(existing["dir_prob"])

    if "test_dates" not in existing or "test_tickers" not in existing:
        raise RuntimeError(
            "В ensemble_predictions.npz отсутствуют test_dates/test_tickers — "
            "нужны для (date,ticker) join. Перетренируй ансамбль."
        )

    test_dates   = _norm_date(existing["test_dates"])
    test_tickers = np.array([str(t) for t in existing["test_tickers"]], dtype="U16")

    # Index meta-preds by (date, ticker)
    key_to_idx: dict[tuple[str, str], int] = {}
    for i, (d, t) in enumerate(zip(meta["dates"], meta["tickers"])):
        key_to_idx[(d, t)] = i

    meta_prob  = np.full(N, np.nan, dtype=np.float32)
    meta_logit = np.full(N, np.nan, dtype=np.float32)
    has_meta   = np.zeros(N, dtype=bool)

    matched = 0
    for i in range(N):
        key = (test_dates[i], test_tickers[i])
        j = key_to_idx.get(key)
        if j is not None:
            meta_prob[i]  = meta["prob"][j]
            meta_logit[i] = meta["logit"][j]
            has_meta[i]   = True
            matched += 1

    existing["meta_dir_prob"]  = meta_prob
    existing["meta_dir_logit"] = meta_logit
    existing["has_meta"]       = has_meta

    np.savez(npz_path, **existing)
    return {
        "N_total":   N,
        "N_matched": matched,
        "N_missing": N - matched,
        "coverage":  matched / max(N, 1),
        "mean_p":    float(np.nanmean(meta_prob)),
        "std_p":     float(np.nanstd(meta_prob)),
    }


def inspect(npz_path: str = NPZ_PATH) -> None:
    """Печатает распределение meta vs raw на пересечении."""
    if not os.path.exists(npz_path):
        print(f"Не найден {npz_path}"); return
    d = np.load(npz_path, allow_pickle=True)
    if "meta_dir_prob" not in d.files:
        print("meta_dir_prob ещё не записан. Запусти: py -m ml.merge_meta_into_npz")
        return

    p_meta = d["meta_dir_prob"]
    p_raw  = d["dir_prob_platt"] if "dir_prob_platt" in d.files else d["dir_prob"]
    has    = d["has_meta"].astype(bool)
    y      = d["y_test"]

    print(f"\nMeta coverage: {int(has.sum())}/{len(has)} ({has.mean()*100:.1f}%)")
    print(f"  meta_dir_prob: mean={np.nanmean(p_meta):.4f}  std={np.nanstd(p_meta):.4f}")
    print(f"  raw  dir_prob: mean={p_raw.mean():.4f}  std={p_raw.std():.4f}")

    if has.any():
        idx = np.where(has)[0]
        y_h = y[idx]
        is_up = (y_h == 0).astype(int)
        is_dn = (y_h == 2).astype(int)
        # Бинарная задача direction (UP vs DOWN), FLAT исключён
        mask = (is_up + is_dn) > 0
        if mask.sum() > 0:
            up_true = is_up[mask]
            meta_pred = (p_meta[idx][mask] >= 0.5).astype(int)
            raw_pred  = (p_raw[idx][mask]  >= 0.5).astype(int)
            print(f"\nНа intersection (N={int(mask.sum())}, UP={int(up_true.sum())}):")
            print(f"  meta acc: {(meta_pred == up_true).mean():.4f}")
            print(f"  raw  acc: {(raw_pred  == up_true).mean():.4f}")
            print(f"  Δ:        {((meta_pred == up_true).mean() - (raw_pred == up_true).mean())*100:+.2f}pp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true",
                   help="Показать coverage + сравнение acc meta vs raw")
    p.add_argument("--npz", default=NPZ_PATH)
    args = p.parse_args()

    if args.inspect:
        inspect(args.npz)
        return

    print(f"Вычисляем meta_dir_prob из {META_FEAT_PATH_V3}...")
    meta = compute_meta_preds()
    print(f"  Готово: N={len(meta['prob'])}, mean(p)={meta['prob'].mean():.4f}")

    print(f"\nМерж в {args.npz}...")
    stats = merge_into_npz(meta, args.npz)
    print(f"  matched: {stats['N_matched']}/{stats['N_total']} "
          f"(coverage={stats['coverage']*100:.1f}%)")
    print(f"  missing: {stats['N_missing']} (fallback на dir_prob_platt)")
    print(f"  meta_dir_prob mean={stats['mean_p']:.4f}  std={stats['std_p']:.4f}")
    print(f"\n✅ Сохранено в {args.npz}")
    print(f"\nДля анализа: py -m ml.merge_meta_into_npz --inspect")


if __name__ == "__main__":
    main()
