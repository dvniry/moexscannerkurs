"""Sprint 11.3 — side-by-side evaluation: Chronos zero-shot vs custom QuantileOHLCHead.

Сравнивает quantile-предсказания Close-канала:
  - Custom: ensemble_predictions.npz['quantile_pred'][:, 45:60] (C-channel, ATR-norm)
  - Chronos: chronos_close_quantiles.npz['chronos_close_rel_atr'] (ATR-norm)

Метрики (per future bar t+1..t+5):
  - Coverage:  доля actual в [q_0.10, q_0.90]; идеал = 80%
  - Sharpness: mean(q_0.90 − q_0.10); ширина интервала
  - Pinball loss: стандартная quantile loss (ниже = лучше)
  - Median bias: mean(q_0.50 − actual); ~0 = калиброван
  - Score: pinball loss summed across quantiles

Запуск:
    py -m ml.chronos_eval

Требования:
    1. ensemble_predictions.npz существует (с quantile_pred, ohlc_test)
    2. chronos_close_quantiles.npz существует (см. chronos_quantile_pred.py)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

ENS_PATH     = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
CHRONOS_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "chronos_close_quantiles.npz")
QUANTILES    = (0.10, 0.50, 0.90)


def pinball_loss(pred_q: np.ndarray, target: np.ndarray, quantiles=QUANTILES) -> np.ndarray:
    """pred_q: [N, n_q, fb]; target: [N, fb]; returns per-quantile per-bar loss [n_q, fb]."""
    n_q = len(quantiles)
    target_b = target[:, None, :]                     # [N, 1, fb]
    err = target_b - pred_q                            # [N, n_q, fb]
    q = np.array(quantiles).reshape(1, n_q, 1)
    loss = np.where(err >= 0, q * err, (q - 1.0) * err)   # >= 0 always
    return loss.mean(axis=0)                           # [n_q, fb]


def coverage(pred_q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Доля actual в [q10, q90]. pred_q: [N, 3, fb], target: [N, fb] → [fb]."""
    q10 = pred_q[:, 0, :]
    q90 = pred_q[:, 2, :]
    cov = ((target >= q10) & (target <= q90)).mean(axis=0)
    return cov


def sharpness(pred_q: np.ndarray) -> np.ndarray:
    """mean(q90 - q10) per bar. [fb]"""
    return (pred_q[:, 2, :] - pred_q[:, 0, :]).mean(axis=0)


def median_bias(pred_q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """mean(q50 - actual) per bar. [fb]"""
    return (pred_q[:, 1, :] - target).mean(axis=0)


def _format_row(label: str, values: np.ndarray, fmt: str = "+6.3f") -> str:
    cells = " ".join(f"{v:{fmt}}" for v in values)
    return f"  {label:<22} {cells}"


def main():
    if not os.path.exists(ENS_PATH):
        print(f"ERROR: {ENS_PATH} не найден"); sys.exit(1)
    if not os.path.exists(CHRONOS_PATH):
        print(f"ERROR: {CHRONOS_PATH} не найден. Запусти: py -m ml.chronos_quantile_pred")
        sys.exit(2)

    print("Загружаем ensemble_predictions.npz...")
    ens = np.load(ENS_PATH, allow_pickle=True)
    print("Загружаем chronos_close_quantiles.npz...")
    chr_npz = np.load(CHRONOS_PATH, allow_pickle=True)

    valid_idx = chr_npz["valid_idx"]
    N_valid   = len(valid_idx)
    fb        = int(chr_npz["pred_len"])
    print(f"\nN_valid: {N_valid}  fb: {fb}  model: {str(chr_npz['model_name'])}")
    print(f"Уникальных тикеров: {len(set(chr_npz['test_tickers']))}")

    # ── 1. Actuals (C-канал ATR-norm) ───────────────────────────────────
    # ohlc_test layout: [N, fb*4] row-major [t+1_O, t+1_H, t+1_L, t+1_C, t+2_O, ...]
    ohlc_test = ens["ohlc_test"]
    ohlc_3d = ohlc_test.reshape(-1, fb, 4)
    target_C_all = ohlc_3d[:, :, 3]                    # [N_total, fb]
    target_C = target_C_all[valid_idx].astype(np.float64)   # [N_valid, fb]

    # ── 2. Custom quantile_pred (C-канал) ───────────────────────────────
    # quantile_pred layout: [N, 60] = [O||H||L||C] × [q10×fb | q50×fb | q90×fb]
    qp_all = ens["quantile_pred"]
    chunk = qp_all.shape[1] // 4                       # 15
    pred_C_layout = qp_all[:, 3*chunk:4*chunk]         # [N_total, 15]
    n_q = chunk // fb
    pred_C_custom = pred_C_layout.reshape(-1, n_q, fb).astype(np.float64)  # [N_total, 3, fb]
    pred_C_custom = pred_C_custom[valid_idx]            # [N_valid, 3, fb]

    # ── 3. Chronos quantiles (C-канал в ATR-norm) ──────────────────────
    pred_C_chronos = chr_npz["chronos_close_rel_atr"].astype(np.float64)  # [N_valid, 3, fb]

    # ── 4. Метрики ───────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print(f"  Chronos zero-shot vs Custom QuantileOHLCHead — C-channel comparison")
    print(f"  N={N_valid}, fb={fb}")
    print("═" * 78)

    bars_hdr = "  " + " ".join(f"  t+{i+1:>2}" for i in range(fb))
    print(f"\n{'':<24}{bars_hdr.strip()}")

    # COVERAGE
    cov_custom  = coverage(pred_C_custom,  target_C)
    cov_chronos = coverage(pred_C_chronos, target_C)
    print(f"\n── 1. COVERAGE [q10, q90] (ideal 80%) ──")
    print(_format_row("Custom (C-channel):",  cov_custom * 100,  "+6.2f"))
    print(_format_row("Chronos zero-shot:",   cov_chronos * 100, "+6.2f"))
    print(f"  Δ Chronos vs Custom: {(cov_chronos - cov_custom).mean()*100:+.2f}pp avg")

    # SHARPNESS
    sh_custom  = sharpness(pred_C_custom)
    sh_chronos = sharpness(pred_C_chronos)
    print(f"\n── 2. SHARPNESS (mean q90 − q10, ATR-norm units) ──")
    print(_format_row("Custom:",  sh_custom,  "+6.3f"))
    print(_format_row("Chronos:", sh_chronos, "+6.3f"))

    # MEDIAN BIAS
    mb_custom  = median_bias(pred_C_custom,  target_C)
    mb_chronos = median_bias(pred_C_chronos, target_C)
    print(f"\n── 3. MEDIAN BIAS (q50 − actual; ~0 = калиброван) ──")
    print(_format_row("Custom:",  mb_custom,  "+6.3f"))
    print(_format_row("Chronos:", mb_chronos, "+6.3f"))

    # PINBALL LOSS
    pl_custom_all  = pinball_loss(pred_C_custom,  target_C)    # [3, fb]
    pl_chronos_all = pinball_loss(pred_C_chronos, target_C)
    print(f"\n── 4. PINBALL LOSS (per quantile, mean across bars; ниже = лучше) ──")
    print(f"  {'method':<22} q=0.10  q=0.50  q=0.90  | mean")
    pl_c  = pl_custom_all.mean(axis=1)
    pl_ch = pl_chronos_all.mean(axis=1)
    print(f"  {'Custom:':<22}  {pl_c[0]:.4f}  {pl_c[1]:.4f}  {pl_c[2]:.4f}  | {pl_c.mean():.4f}")
    print(f"  {'Chronos:':<22}  {pl_ch[0]:.4f}  {pl_ch[1]:.4f}  {pl_ch[2]:.4f}  | {pl_ch.mean():.4f}")
    diff = (pl_ch.mean() - pl_c.mean())
    print(f"  Δ Chronos vs Custom: {diff:+.4f}  ({'Chronos лучше' if diff < 0 else 'Custom лучше'})")

    # ── 5. По-барный pinball ─────────────────────────────────────────────
    print(f"\n── 5. PINBALL LOSS per bar (sum across quantiles) ──")
    pl_c_per  = pl_custom_all.sum(axis=0)
    pl_ch_per = pl_chronos_all.sum(axis=0)
    print(_format_row("Custom:",  pl_c_per,  "+6.4f"))
    print(_format_row("Chronos:", pl_ch_per, "+6.4f"))

    # ── 6. ВЕРДИКТ ────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print(f"  ВЕРДИКТ")
    print("═" * 78)
    cov_target = 0.80
    custom_dist  = abs(cov_custom.mean()  - cov_target)
    chronos_dist = abs(cov_chronos.mean() - cov_target)
    print(f"  Coverage (cred. interval) — кто ближе к {cov_target*100:.0f}%:")
    print(f"    Custom:  {cov_custom.mean()*100:5.2f}%  (|Δ|={custom_dist*100:.2f}pp)")
    print(f"    Chronos: {cov_chronos.mean()*100:5.2f}%  (|Δ|={chronos_dist*100:.2f}pp)  "
          f"{'⭐ ближе' if chronos_dist < custom_dist else ''}")
    print(f"\n  Pinball mean:")
    print(f"    Custom:  {pl_c.mean():.4f}")
    print(f"    Chronos: {pl_ch.mean():.4f}  "
          f"{'⭐ ниже (лучше)' if pl_ch.mean() < pl_c.mean() else 'выше'}")
    print(f"\n  Median bias |avg|:")
    bi_c  = abs(mb_custom.mean())
    bi_ch = abs(mb_chronos.mean())
    print(f"    Custom:  {bi_c:.3f}")
    print(f"    Chronos: {bi_ch:.3f}  {'⭐ меньше' if bi_ch < bi_c else 'больше'}")

    # Aggregate score: smaller pinball + smaller |bias| + closer to 80% coverage
    score_custom  = pl_c.mean()  + bi_c  + custom_dist
    score_chronos = pl_ch.mean() + bi_ch + chronos_dist
    print(f"\n  Aggregate score (pinball + |bias| + |cov_err|, ниже = лучше):")
    print(f"    Custom:  {score_custom:.4f}")
    print(f"    Chronos: {score_chronos:.4f}  "
          f"{'⭐ ЛУЧШЕ' if score_chronos < score_custom else 'хуже'}")


if __name__ == "__main__":
    main()
