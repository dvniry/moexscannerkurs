"""ml/quantile_eval.py — Sprint 11.1 evaluation: качество quantile predictions.

Метрики:
  1. Coverage — доля сэмплов где actual H/L попадает в [q_0.10, q_0.90]
  2. Sharpness — средняя ширина интервала (q_0.90 − q_0.10)
  3. Median bias — bias q_0.50 vs actual H/L
  4. Per-bar trend — растёт ли неопределённость с горизонтом
  5. Pinball loss — raw качество per-quantile

Layout quantile_pred [N, 30]:
  [0:15]   = low quantiles  (q=0.10 [0:5], q=0.50 [5:10], q=0.90 [10:15])
  [15:30]  = high quantiles (q=0.10 [15:20], q=0.50 [20:25], q=0.90 [25:30])
ohlc_test [N, 20]: 5 future bars × [O, H, L, C] (row-major).

Запуск:
    py -m ml.quantile_eval                  # глобальная статистика
    py -m ml.quantile_eval --examples 5     # + 5 примеров с ground truth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
QUANTILES = (0.10, 0.50, 0.90)
FUTURE_BARS = 5


def split_quantiles(qpred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Разделяет quantile_pred на low [N, 3, fb] и high [N, 3, fb].

    [N, 30] (legacy 2-channel L/H): qpred[:, :15] = L, qpred[:, 15:30] = H.
    [N, 60] (новый 4-channel OHLC): layout [O || H || L || C], каждый 3*fb.
                                    Возвращаем только L и H для функций
                                    которые работают с extremes (coverage etc).
    """
    N = qpred.shape[0]
    n_q = len(QUANTILES)
    chunk = n_q * FUTURE_BARS
    if qpred.shape[1] >= 4 * chunk:
        # OHLC: extract H (chunk 1) and L (chunk 2)
        high = qpred[:, 1*chunk:2*chunk].reshape(N, n_q, FUTURE_BARS)
        low  = qpred[:, 2*chunk:3*chunk].reshape(N, n_q, FUTURE_BARS)
    else:
        # Legacy: layout [L || H]
        half = qpred.shape[1] // 2
        low  = qpred[:, :half].reshape(N, n_q, FUTURE_BARS)
        high = qpred[:, half:].reshape(N, n_q, FUTURE_BARS)
    return low, high


def split_quantiles_ohlc(qpred: np.ndarray) -> dict | None:
    """Только для нового формата [N, 60] → {O,H,L,C}: [N, 3, fb]. None если legacy."""
    n_q = len(QUANTILES)
    chunk = n_q * FUTURE_BARS
    if qpred.shape[1] < 4 * chunk:
        return None
    N = qpred.shape[0]
    return {
        "O": qpred[:, 0*chunk:1*chunk].reshape(N, n_q, FUTURE_BARS),
        "H": qpred[:, 1*chunk:2*chunk].reshape(N, n_q, FUTURE_BARS),
        "L": qpred[:, 2*chunk:3*chunk].reshape(N, n_q, FUTURE_BARS),
        "C": qpred[:, 3*chunk:4*chunk].reshape(N, n_q, FUTURE_BARS),
    }


def split_ohlc(ohlc: np.ndarray) -> dict:
    """OHLC [N, 20] → {O,H,L,C} с shape [N, fb] каждый."""
    N = ohlc.shape[0]
    arr = ohlc.reshape(N, FUTURE_BARS, 4)
    return {"O": arr[:, :, 0], "H": arr[:, :, 1],
            "L": arr[:, :, 2], "C": arr[:, :, 3]}


def pinball(pred: np.ndarray, target: np.ndarray, q: float) -> float:
    err = target - pred
    return float(np.where(err >= 0, q * err, (q - 1) * err).mean())


def render_range_chart(actual_h: float, actual_l: float,
                       q_high: tuple, q_low: tuple,
                       width: int = 60, lo: float = -2.5, hi: float = +2.5) -> list[str]:
    """Возвращает строки ASCII-чарта для одного бара.

    actual_l, actual_h — фактические low/high (нормированные через ATR).
    q_low  = (q10, q50, q90) для low
    q_high = (q10, q50, q90) для high
    Шкала [lo, hi] ATR; вне диапазона — clamp с пометкой ⟨ ⟩.
    """
    def col(v: float) -> int:
        c = int(round((v - lo) / (hi - lo) * (width - 1)))
        return max(0, min(width - 1, c))

    def out_of_range(v: float) -> bool:
        return v < lo or v > hi

    # Шкала
    scale = [' '] * width
    for tick in (-2, -1, 0, 1, 2):
        scale[col(float(tick))] = '┼'
    scale_line = ''.join(scale)

    # Predicted LOW range: [q10 .. q90] заливаем "═", q50 = "│", q10 = "[", q90 = "]"
    pred_low = [' '] * width
    cl10, cl50, cl90 = col(q_low[0]), col(q_low[1]), col(q_low[2])
    for c in range(min(cl10, cl90), max(cl10, cl90) + 1):
        pred_low[c] = '═'
    pred_low[cl10] = '['
    pred_low[cl90] = ']'
    pred_low[cl50] = '│'
    pred_low_line = ''.join(pred_low)

    # Predicted HIGH range: то же, но "(" ")" + "│"
    pred_high = [' '] * width
    ch10, ch50, ch90 = col(q_high[0]), col(q_high[1]), col(q_high[2])
    for c in range(min(ch10, ch90), max(ch10, ch90) + 1):
        pred_high[c] = '═'
    pred_high[ch10] = '('
    pred_high[ch90] = ')'
    pred_high[ch50] = '│'
    pred_high_line = ''.join(pred_high)

    # Actual: точки L и H, соединённые "─"
    actual = [' '] * width
    la, ha = col(actual_l), col(actual_h)
    for c in range(min(la, ha), max(la, ha) + 1):
        actual[c] = '─'
    actual[la] = '●' if not out_of_range(actual_l) else '◀'
    actual[ha] = '○' if not out_of_range(actual_h) else '▶'
    actual_line = ''.join(actual)

    return scale_line, pred_low_line, pred_high_line, actual_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=3,
                        help="Сколько примеров предсказаний показать (random)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual", action="store_true",
                        help="ASCII-чарт: actual слева, predictions справа")
    args = parser.parse_args()

    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден"); sys.exit(1)

    data = np.load(NPZ_PATH, allow_pickle=False)
    if "quantile_pred" not in data.files:
        print(f"ERROR: quantile_pred отсутствует. "
              f"Ребилд V3 ансамбля: py -m ml.retrain_all --rebuild ensemble")
        sys.exit(2)

    qpred  = data["quantile_pred"]                # [N, 30]
    ohlc_t = data["ohlc_test"]                    # [N, 20]
    tickers = data["test_tickers"]
    dates   = data["test_dates"]
    y_test  = data["y_test"]
    atr     = data["atr_ratio"]                   # [N]
    dir_prob_platt = data.get("dir_prob_platt", data["dir_prob"])

    low_q, high_q = split_quantiles(qpred)        # [N, 3, fb]
    ohlc = split_ohlc(ohlc_t)
    H_true = ohlc["H"]                            # [N, fb]
    L_true = ohlc["L"]                            # [N, fb]

    print(f"\n  Sprint 11.1 — Quantile Predictions Evaluation")
    print(f"  ──────────────────────────────────────────────")
    print(f"  N={len(qpred)}  fb={FUTURE_BARS}  quantiles={QUANTILES}")
    print(f"  ohlc_test нормирован через ATR (mean={ohlc_t.mean():+.4f}, std={ohlc_t.std():.4f})")

    # ── 1. Coverage ──
    print(f"\n  ── 1. COVERAGE (доля actual в [q_0.10, q_0.90]) ──")
    high_cover_per_bar = []
    low_cover_per_bar  = []
    for b in range(FUTURE_BARS):
        h_in = ((H_true[:, b] >= high_q[:, 0, b]) & (H_true[:, b] <= high_q[:, 2, b])).mean()
        l_in = ((L_true[:, b] >= low_q[:,  0, b]) & (L_true[:, b] <= low_q[:,  2, b])).mean()
        high_cover_per_bar.append(h_in)
        low_cover_per_bar.append(l_in)
    print(f"  bar:        " + "  ".join(f"   t+{b+1}" for b in range(FUTURE_BARS)))
    print(f"  high cov:   " + "  ".join(f"{c*100:6.2f}%" for c in high_cover_per_bar))
    print(f"  low cov:    " + "  ".join(f"{c*100:6.2f}%" for c in low_cover_per_bar))
    print(f"  Идеальное coverage = 80%. Если выше — модель слишком осторожна (широкие интервалы)")
    print(f"  ниже 80% — модель overconfident.")
    print(f"\n  Avg high coverage: {np.mean(high_cover_per_bar)*100:.2f}%")
    print(f"  Avg low  coverage: {np.mean(low_cover_per_bar)*100:.2f}%")

    # ── 2. Sharpness ──
    print(f"\n  ── 2. SHARPNESS (средняя ширина интервала q_0.90 − q_0.10) ──")
    high_width = (high_q[:, 2, :] - high_q[:, 0, :])    # [N, fb]
    low_width  = (low_q[:,  2, :] - low_q[:,  0, :])
    print(f"  bar:           " + "  ".join(f"   t+{b+1}" for b in range(FUTURE_BARS)))
    print(f"  high width:    " + "  ".join(f"{high_width[:, b].mean():6.3f}" for b in range(FUTURE_BARS)))
    print(f"  low  width:    " + "  ".join(f"{low_width[:, b].mean():6.3f}"  for b in range(FUTURE_BARS)))
    print(f"  Ширина растёт с горизонтом → модель учла растущую неопределённость.")

    # ── 3. Median bias ──
    print(f"\n  ── 3. MEDIAN BIAS (q_0.50 − actual) ──")
    high_bias = (high_q[:, 1, :] - H_true).mean(axis=0)
    low_bias  = (low_q[:,  1, :] - L_true).mean(axis=0)
    print(f"  bar:           " + "  ".join(f"   t+{b+1}" for b in range(FUTURE_BARS)))
    print(f"  high bias:     " + "  ".join(f"{high_bias[b]:+6.3f}" for b in range(FUTURE_BARS)))
    print(f"  low  bias:     " + "  ".join(f"{low_bias[b]:+6.3f}"  for b in range(FUTURE_BARS)))
    print(f"  ~0 = калиброван, +X = модель завышает, −X = занижает.")

    # ── 4. Pinball loss per quantile ──
    print(f"\n  ── 4. PINBALL LOSS per quantile (ниже = лучше) ──")
    print(f"  side  q=0.10  q=0.50  q=0.90")
    for label, pred, target in [("low",  low_q,  L_true),
                                  ("high", high_q, H_true)]:
        losses = [pinball(pred[:, qi, :], target, q) for qi, q in enumerate(QUANTILES)]
        print(f"  {label:<5}  " + "  ".join(f"{l:.4f}" for l in losses))

    # ── 5. Connection: quantile width vs dir_prob_platt ──
    print(f"\n  ── 5. WIDTH × CONFIDENCE (узкие интервалы = уверенная модель?) ──")
    # Используем bar 1 (ближайший)
    width_b1 = (high_q[:, 2, 0] - low_q[:, 0, 0])  # full range t+1
    # binарным по dir_prob: confident (|p - 0.5| > 0.2) vs uncertain
    confident_mask = np.abs(dir_prob_platt - 0.5) > 0.10
    print(f"  Confident (|p-0.5|>0.10): N={confident_mask.sum()}  "
          f"avg width t+1 = {width_b1[confident_mask].mean():.3f}")
    print(f"  Uncertain (|p-0.5|≤0.10): N={(~confident_mask).sum()}  "
          f"avg width t+1 = {width_b1[~confident_mask].mean():.3f}")

    # ── 6. Examples ──
    if args.examples > 0:
        print(f"\n  ── 6. ПРИМЕРЫ ПРЕДСКАЗАНИЙ ({args.examples} случайных) ──")
        rng = np.random.default_rng(args.seed)
        idx_pool = rng.choice(len(qpred), size=args.examples, replace=False)
        for i, idx in enumerate(idx_pool):
            t = str(tickers[idx])
            d = str(dates[idx])
            y = int(y_test[idx])
            cls_name = {0: "UP", 1: "FLAT", 2: "DOWN"}.get(y, "?")
            p_up = float(dir_prob_platt[idx])
            atr_r = float(atr[idx])
            ohlc_actual = ohlc_t[idx].reshape(FUTURE_BARS, 4)  # [bar, OHLC]

            print(f"\n  [{i+1}/{args.examples}]  {t}  date={d}  y_true={cls_name}  "
                  f"dir_prob_platt={p_up:.3f}  atr={atr_r:.4f}")

            if args.visual:
                # ── Visual layout: левая колонка ACTUAL, правая PREDICTED ──
                col_w_l = 50
                col_w_r = 78
                print(f"  {'╶─ ACTUAL OHLC '.ljust(col_w_l, '─')}  "
                      f"{'╶─ PREDICTED QUANTILES '.ljust(col_w_r, '─')}")
                print(f"  {'bar  O       H       L       C'.ljust(col_w_l)}  "
                      f"{'bar  q_low (0.10/0.50/0.90)        q_high (0.10/0.50/0.90)'.ljust(col_w_r)}")
                for b in range(FUTURE_BARS):
                    O, H, L, C = ohlc_actual[b]
                    ah, al = H, L
                    hq10, hq50, hq90 = high_q[idx, :, b]
                    lq10, lq50, lq90 = low_q[idx, :, b]
                    h_in = "✓" if (hq10 <= ah <= hq90) else "✗"
                    l_in = "✓" if (lq10 <= al <= lq90) else "✗"
                    candle = (f"t+{b+1}  {O:+6.3f}  {H:+6.3f}  {L:+6.3f}  {C:+6.3f}").ljust(col_w_l)
                    pred   = (f"t+{b+1}  {lq10:+6.3f}/{lq50:+6.3f}/{lq90:+6.3f} {l_in}  "
                              f"{hq10:+6.3f}/{hq50:+6.3f}/{hq90:+6.3f} {h_in}").ljust(col_w_r)
                    print(f"  {candle}  {pred}")

                # Range-chart внизу: actual + predicted в одной шкале -2.5..+2.5 ATR
                print(f"\n  RANGE VIEW (-2.5 ATR ╶── 0 ──╴ +2.5 ATR)  "
                      f"[ ] q_low [0.10..0.90]   ( ) q_high [0.10..0.90]   "
                      f"│ median   ●○ actual L/H")
                for b in range(FUTURE_BARS):
                    H, L = ohlc_actual[b][1], ohlc_actual[b][2]
                    scale, plow, phigh, act = render_range_chart(
                        actual_h=H, actual_l=L,
                        q_high=tuple(high_q[idx, :, b]),
                        q_low =tuple(low_q[idx,  :, b]),
                    )
                    print(f"  t+{b+1}  pred-L:  {plow}")
                    print(f"        pred-H:  {phigh}")
                    print(f"        actual:  {act}")
                    if b == 0:
                        print(f"                 {scale}    [-2  -1   0  +1  +2]")
                    print()
            else:
                # Table-only mode (старый формат)
                print(f"     bar     actual_H  q_high(0.10/0.50/0.90)     actual_L  q_low(0.10/0.50/0.90)")
                for b in range(FUTURE_BARS):
                    ah = H_true[idx, b]
                    al = L_true[idx, b]
                    hq10, hq50, hq90 = high_q[idx, :, b]
                    lq10, lq50, lq90 = low_q[idx, :, b]
                    h_in = "✓" if (hq10 <= ah <= hq90) else "✗"
                    l_in = "✓" if (lq10 <= al <= lq90) else "✗"
                    print(f"     t+{b+1}    {ah:+7.3f} {h_in}   "
                          f"[{hq10:+6.3f}, {hq50:+6.3f}, {hq90:+6.3f}]    "
                          f"{al:+7.3f} {l_in}   [{lq10:+6.3f}, {lq50:+6.3f}, {lq90:+6.3f}]")
                # Проверка консистентности: q_low_0.10 < q_low_0.50 < q_low_0.90 (monotone?)
                mono_low  = all(low_q[idx, 0, b]  <= low_q[idx, 1, b]  <= low_q[idx, 2, b]  for b in range(FUTURE_BARS))
                mono_high = all(high_q[idx, 0, b] <= high_q[idx, 1, b] <= high_q[idx, 2, b] for b in range(FUTURE_BARS))
                print(f"     monotone:  low={'✓' if mono_low else '✗ КРОССОВЕР'}  "
                      f"high={'✓' if mono_high else '✗ КРОССОВЕР'}")

    # ── 7. Глобальный quantile crossover check ──
    print(f"\n  ── 7. QUANTILE CROSSOVER (нарушение монотонности — штраф для модели) ──")
    low_cross  = ((low_q[:,  0, :] > low_q[:,  1, :]) | (low_q[:,  1, :] > low_q[:,  2, :])).any(axis=1).mean()
    high_cross = ((high_q[:, 0, :] > high_q[:, 1, :]) | (high_q[:, 1, :] > high_q[:, 2, :])).any(axis=1).mean()
    print(f"  Доля сэмплов с крестом low:  {low_cross*100:.2f}%")
    print(f"  Доля сэмплов с крестом high: {high_cross*100:.2f}%")
    print(f"  Идеал: 0%. Большие значения → нужно добавить monotone constraint в loss.")

    # ── 8. Полнота ширины: q_high_0.50 > q_low_0.50? ──
    print(f"\n  ── 8. SANITY: high_q_0.50 > low_q_0.50 (high всегда выше low)? ──")
    valid_order = (high_q[:, 1, :] > low_q[:, 1, :]).all(axis=1).mean()
    print(f"  {valid_order*100:.2f}% сэмплов имеют корректный порядок high > low по медиане.")

    # ── 9. OHLC ordering check (только для нового 4-channel формата) ──
    parts = split_quantiles_ohlc(qpred)
    if parts is not None:
        O, H, L, C = parts["O"], parts["H"], parts["L"], parts["C"]
        print(f"\n  ── 9. OHLC ORDERING (новый 4-channel формат) ──")
        print(f"  Идеал: range(L) < range(O,C) < range(H) — диапазоны не перекрываются.")

        # Strict overlap: L_q90 ≤ O_q10, L_q90 ≤ C_q10, O_q90 ≤ H_q10, C_q90 ≤ H_q10
        overlap_LO = ((L[:, 2, :] > O[:, 0, :])).any(axis=1).mean()
        overlap_LC = ((L[:, 2, :] > C[:, 0, :])).any(axis=1).mean()
        overlap_OH = ((O[:, 2, :] > H[:, 0, :])).any(axis=1).mean()
        overlap_CH = ((C[:, 2, :] > H[:, 0, :])).any(axis=1).mean()
        print(f"  Overlap L↔O range: {overlap_LO*100:.2f}%  (target 0%)")
        print(f"  Overlap L↔C range: {overlap_LC*100:.2f}%")
        print(f"  Overlap O↔H range: {overlap_OH*100:.2f}%")
        print(f"  Overlap C↔H range: {overlap_CH*100:.2f}%")

        # Median ordering: L_q50 < O_q50, L_q50 < C_q50, O_q50 < H_q50, C_q50 < H_q50
        med_LO = (L[:, 1, :] >= O[:, 1, :]).any(axis=1).mean()
        med_LC = (L[:, 1, :] >= C[:, 1, :]).any(axis=1).mean()
        med_OH = (O[:, 1, :] >= H[:, 1, :]).any(axis=1).mean()
        med_CH = (C[:, 1, :] >= H[:, 1, :]).any(axis=1).mean()
        print(f"\n  Median ordering violations (L_q50 ≥ O_q50 и т.д.):")
        print(f"  L≥O med: {med_LO*100:.2f}%   L≥C med: {med_LC*100:.2f}%   "
              f"O≥H med: {med_OH*100:.2f}%   C≥H med: {med_CH*100:.2f}%")

        # Per-channel coverage on actual values
        actuals = {"O": ohlc["O"], "H": ohlc["H"], "L": ohlc["L"], "C": ohlc["C"]}
        print(f"\n  Per-channel coverage (actual в [q_0.10, q_0.90]):")
        for ch in ("O", "H", "L", "C"):
            cov_per_bar = []
            for b in range(FUTURE_BARS):
                a = actuals[ch][:, b]
                in_range = ((a >= parts[ch][:, 0, b]) & (a <= parts[ch][:, 2, b])).mean()
                cov_per_bar.append(in_range)
            print(f"    {ch}:  " + "  ".join(f"{c*100:5.2f}%" for c in cov_per_bar)
                  + f"   avg={np.mean(cov_per_bar)*100:.2f}%")


if __name__ == "__main__":
    main()
