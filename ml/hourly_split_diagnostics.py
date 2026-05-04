"""B-24: диагностика анонимного `test_acc > val_acc` на HourlySpec.

Гипотеза в plan.md:
    Test acc 0.5475 систематически выше val acc 0.5378 на ~1pp на всех 3 seeds.
    В нормально стратифицированном split разница должна быть ±0.5pp.

Этот скрипт НЕ перетренировывает модель (это занимает ~7 мин). Вместо этого
анализирует существующий `hourly_all_predictions.npz`, в котором есть
predictions + h_split метки + dates + tickers. На основе разделения
по фолдам внутри val/test даёт более полную картину:
  1. Acc per (split, sub-window по дате)
  2. Class balance (UP/DOWN) per split
  3. Concentration of "easy" дат (где >70% сэмплов correct)

Запуск:
    py -m ml.hourly_split_diagnostics
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "hourly_all_predictions.npz")


def main():
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден.")
        print("Сгенерируйте: py -m ml.trainer_hourly")
        sys.exit(1)

    d = np.load(NPZ_PATH, allow_pickle=True)
    print(f"\n  HourlySpec all_predictions.npz")
    print(f"  Keys: {list(d.files)}")

    dir_prob = d["dir_prob"].astype(np.float32)
    y_true   = d["y_true"].astype(np.int8)
    splits   = np.array([str(s) for s in d["split"]])
    dates    = np.array([str(x)[:10] for x in d["dates"]])
    has_tickers = "tickers" in d.files
    tickers  = (np.array([str(t) for t in d["tickers"]])
                if has_tickers else np.full(len(y_true), "_unk_", dtype="U16"))

    n = len(y_true)
    pred = (dir_prob >= 0.5).astype(np.int8)

    print(f"  N total: {n}")
    sorted_dates = np.sort(dates)
    print(f"  Date range: {sorted_dates[0]} → {sorted_dates[-1]}")
    print(f"  Tickers in file: {'yes' if has_tickers else 'no'} "
          f"(unique={len(np.unique(tickers))})")
    print()

    # 1. Per-split overall acc + class balance
    print(f"  ── Per-split overall ──")
    print(f"  {'split':<8} {'N':>7} {'UP%':>6} {'DOWN%':>6} {'acc':>7} {'baseline':>9}")
    for s in ("train", "val", "test"):
        mask = (splits == s)
        if not mask.any():
            continue
        y_s    = y_true[mask]
        pr_s   = pred[mask]
        n_s    = int(mask.sum())
        up_pct = float((y_s == 1).mean() * 100)
        dn_pct = 100 - up_pct
        acc    = float((pr_s == y_s).mean())
        baseline = max(up_pct, dn_pct) / 100.0
        print(f"  {s:<8} {n_s:>7d} {up_pct:>5.1f}% {dn_pct:>5.1f}% {acc:>7.4f} {baseline:>9.4f}")

    # 2. Per-split, K-window по дате внутри split
    print(f"\n  ── В пределах val/test: разбиваем на 5 окон по дате ──")
    for s in ("val", "test"):
        mask = (splits == s)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        ds  = dates[idx]
        ys  = y_true[idx]
        ps  = pred[idx]
        order = np.argsort(ds)
        ys = ys[order]; ps = ps[order]; ds = ds[order]
        n_s = len(ys)
        chunk = max(1, n_s // 5)
        print(f"\n  Split={s}  (N={n_s})")
        accs = []
        for k in range(5):
            st = k * chunk
            en = (k + 1) * chunk if k < 4 else n_s
            ys_k = ys[st:en]; ps_k = ps[st:en]
            acc_k = float((ps_k == ys_k).mean())
            up_pct = float((ys_k == 1).mean() * 100)
            d0 = ds[st]; d1 = ds[en - 1]
            print(f"    win{k+1}  {d0} → {d1}  N={en-st:>5d}  "
                  f"UP%={up_pct:>5.1f}  acc={acc_k:.4f}")
            accs.append(acc_k)
        mu = float(np.mean(accs))
        sd = float(np.std(accs, ddof=1))
        print(f"    Σ:  mean={mu:.4f}  std={sd:.4f}  range=[{min(accs):.4f}, {max(accs):.4f}]")

    # 3. Diagnostic conclusion
    val_mask  = (splits == "val")
    test_mask = (splits == "test")
    val_acc   = float((pred[val_mask]  == y_true[val_mask]).mean())  if val_mask.any()  else float('nan')
    test_acc  = float((pred[test_mask] == y_true[test_mask]).mean()) if test_mask.any() else float('nan')
    val_up    = float((y_true[val_mask]  == 1).mean()) if val_mask.any() else 0.5
    test_up   = float((y_true[test_mask] == 1).mean()) if test_mask.any() else 0.5
    val_baseline  = max(val_up, 1 - val_up)
    test_baseline = max(test_up, 1 - test_up)
    val_lift  = val_acc  - val_baseline
    test_lift = test_acc - test_baseline

    print(f"\n  ── Сводка B-24 ──")
    print(f"  val:  acc={val_acc:.4f}  baseline={val_baseline:.4f}  lift={val_lift:+.4f}")
    print(f"  test: acc={test_acc:.4f}  baseline={test_baseline:.4f}  lift={test_lift:+.4f}")
    print()
    if abs(val_lift - test_lift) < 0.005:
        print(f"  ✅ Lift over baseline сопоставим (Δ={(test_lift-val_lift)*100:+.2f}pp).")
        print(f"     Test > val acc объясняется разницей в class balance, не leakage.")
    else:
        print(f"  ⚠️  Lift разный: val={val_lift*100:+.2f}pp, test={test_lift*100:+.2f}pp")
        print(f"     Δ_lift={(test_lift-val_lift)*100:+.2f}pp — возможна нестратифицированная разделка")
        print(f"     или разные режимы между периодами (HMM volatility shift).")


if __name__ == "__main__":
    main()
