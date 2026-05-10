"""ml/calibrate_platt.py — Sprint 10 B: Per-ticker Platt scaling.

Post-hoc калибровка dir_prob через двухпараметрическую sigmoid:
    P(UP) = sigmoid(a * logit + b)

Отличие от temperature scaling: temperature симметрична (T одинаково
сжимает положительные и отрицательные логиты), Platt с двумя параметрами
корректирует асимметричный bias — например, систематический DOWN-bias,
где predicted > actual во всех bin'ах [0.5..0.8].

Reliability до Platt (2026-05-06 baseline):
  bin [0.50,0.60]: pred=0.55  actual=0.37  bias=-0.17
  bin [0.60,0.70]: pred=0.63  actual=0.43  bias=-0.20
  bin [0.70,0.80]: pred=0.71  actual=0.55  bias=-0.16

Запуск:
    py -m ml.calibrate_platt [--seeds 42 123 7]
    py -m ml.calibrate_platt --apply ml/ensemble/ensemble_predictions.npz
    py -m ml.calibrate_platt --coverage-report

Полезные функции:
    load_ticker_calibrators(path) → {ticker: {"a": float, "b": float}}
    apply_calibration(dir_prob, tickers, calib_map, global_calib) → calibrated
"""
import os, sys, json, argparse
os.environ['GRPC_DNS_RESOLVER'] = 'native'
_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from ml.config import CFG, SCALES
from ml.dataset_v3 import (
    build_full_multiscale_dataset_v3, temporal_split, INDICATOR_COLS,
)
from ml.multiscale_cnn_v3 import MultiScaleHybridV3, _make_loader_v3
from ml.trainer_v3 import _forward_unpack

MIN_SAMPLES_PER_TICKER = 30
A_MIN, A_MAX = 0.1, 5.0      # slope: 0.1 = сильное сжатие, 5.0 = усиление
B_MIN, B_MAX = -3.0, 3.0     # bias: ±3 в logit-пространстве ≈ ±0.95 в prob


# ──────────────────────────────────────────────────────────────────────────────
# Публичные утилиты
# ──────────────────────────────────────────────────────────────────────────────

def load_ticker_calibrators(path: str) -> dict:
    """Загружает {ticker: {"a": float, "b": float}} из JSON."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('per_ticker', {})


def load_global_calibrator(path: str) -> dict:
    """Глобальный fallback {"a": float, "b": float}."""
    if not os.path.exists(path):
        return {"a": 1.0, "b": 0.0}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('global', {"a": 1.0, "b": 0.0})


def apply_calibration(dir_prob: np.ndarray,
                      tickers: np.ndarray,
                      calib_map: dict,
                      global_calib: dict | None = None) -> np.ndarray:
    """Применяет per-ticker Platt scaling к dir_prob.

    Алгоритм:
        logit = log(p / (1-p))
        calibrated_prob = sigmoid(a * logit + b)
    """
    if global_calib is None:
        global_calib = {"a": 1.0, "b": 0.0}

    eps = 1e-6
    p   = np.clip(dir_prob, eps, 1.0 - eps).astype(np.float64)
    logits = np.log(p / (1.0 - p))

    out = np.empty_like(p)
    for i, ticker in enumerate(tickers):
        calib = calib_map.get(str(ticker), global_calib)
        a = float(calib.get('a', 1.0))
        b = float(calib.get('b', 0.0))
        out[i] = 1.0 / (1.0 + np.exp(-(a * logits[i] + b)))

    return out.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration fit
# ──────────────────────────────────────────────────────────────────────────────

def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Calibration ECE в процентах."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        ece += mask.sum() / n * abs(conf - acc)
    return ece * 100.0


def _fit_platt(logits: np.ndarray, labels: np.ndarray,
               max_iter: int = 200) -> tuple[float, float]:
    """Подбирает (a, b) через LBFGS на NLL.

    labels: бинарные (1=UP, 0=DOWN).
    Возвращает (a, b), где a ∈ [A_MIN, A_MAX], b ∈ [B_MIN, B_MAX].
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    a_param = torch.nn.Parameter(torch.ones(1))
    b_param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([a_param, b_param], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        a_clamped = a_param.clamp(A_MIN, A_MAX)
        b_clamped = b_param.clamp(B_MIN, B_MAX)
        scaled = a_clamped * logits_t + b_clamped
        loss = F.binary_cross_entropy_with_logits(scaled, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return (float(a_param.clamp(A_MIN, A_MAX).item()),
            float(b_param.clamp(B_MIN, B_MAX).item()))


def _collect_val_logits(models, val_indices, dataset, device, ctx_dim,
                        use_hourly, batch_size=256):
    """Собирает dir_logits и cls_y для каждого val сэмпла. Идентично
    реализации из calibrate_temperature.py — общий код вынести при
    появлении третьего калибратора (не сейчас, чтобы не тащить рефактор).
    """
    val_ds = Subset(dataset, val_indices.tolist())
    loader = _make_loader_v3(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []
    all_tickers = []

    ticker_for_idx = [dataset.records[int(gi)][0] for gi in val_indices]

    with torch.no_grad():
        global_pos = 0
        for batch in loader:
            imgs_dict, num_dict, cls_y, _, ctx, hourly_data, *_ = batch
            B = cls_y.shape[0]
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            ht    = (hourly_data.to(device)
                     if (use_hourly and hourly_data is not None) else None)
            nums  = ({W: num_dict[W].to(device) for W in SCALES}
                     if num_dict is not None else None)

            batch_logits = []
            for model in models:
                _, _, _, dir_l, *_ = _forward_unpack(model, imgs, nums, ctx_t, ht)
                batch_logits.append(dir_l.cpu().float().numpy())

            avg_logit = np.mean(batch_logits, axis=0)
            all_logits.append(avg_logit)
            all_labels.append(cls_y.numpy())
            all_tickers.extend(ticker_for_idx[global_pos: global_pos + B])
            global_pos += B

    return (np.concatenate(all_logits),
            np.concatenate(all_labels),
            np.array(all_tickers))


def calibrate(seeds=(42, 123, 7),
              out_path='ml/ensemble/platt_per_ticker.json',
              batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Загрузка датасета...')
    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=False, use_hourly=True)
    _, idx_val, _ = temporal_split(ticker_lengths)

    n_ind = len(INDICATOR_COLS)
    models = []
    for seed in seeds:
        path = f'ml/ensemble/model_seed{seed}.pt'
        if not os.path.exists(path):
            print(f'  Пропуск seed {seed}: файл не найден')
            continue
        m = MultiScaleHybridV3(
            ctx_dim=ctx_dim, n_indicator_cols=n_ind,
            future_bars=CFG.future_bars, use_hourly=True,
        ).to(device)
        # Sprint 11.1: strict=False — позволяет загрузить старый checkpoint без quantile_head
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True),
                          strict=False)
        m.eval()
        models.append(m)
        print(f'  Загружен seed {seed}')

    if not models:
        raise RuntimeError('Нет доступных моделей ансамбля в ml/ensemble/')

    print(f'\nСбор val логитов ({len(idx_val)} сэмплов)...')
    logits, labels, tickers = _collect_val_logits(
        models, idx_val, dataset, device, ctx_dim, use_hourly=True,
        batch_size=batch_size)

    mask_ud = labels != 1
    labels_bin = (labels == 0).astype(np.float32)  # 1=UP, 0=DOWN

    a_global, b_global = _fit_platt(logits[mask_ud], labels_bin[mask_ud])
    print(f'Глобальный Platt: a={a_global:.4f}  b={b_global:+.4f}')

    unique_tickers = np.unique(tickers)
    per_ticker: dict = {}
    skipped = 0
    coverage_report: dict[str, dict] = {}

    print(f'\nPer-ticker Platt fit ({len(unique_tickers)} тикеров):')
    for ticker in sorted(unique_tickers):
        mask_t = (tickers == ticker) & mask_ud
        n_t    = int(mask_t.sum())
        if n_t < MIN_SAMPLES_PER_TICKER:
            per_ticker[ticker] = {"a": a_global, "b": b_global}
            skipped += 1
            coverage_report[str(ticker)] = {
                "n_samples":     n_t,
                "a":             round(a_global, 4),
                "b":             round(b_global, 4),
                "used_fallback": True,
                "reason":        f"n<{MIN_SAMPLES_PER_TICKER}",
            }
            continue
        a_t, b_t = _fit_platt(logits[mask_t], labels_bin[mask_t])
        per_ticker[ticker] = {"a": round(a_t, 4), "b": round(b_t, 4)}
        coverage_report[str(ticker)] = {
            "n_samples":     n_t,
            "a":             round(a_t, 4),
            "b":             round(b_t, 4),
            "used_fallback": False,
            "reason":        "fit_ok",
        }

    print(f'  Подобрано per-ticker: {len(unique_tickers) - skipped} / {len(unique_tickers)}')
    print(f'  Глобальный fallback:  {skipped} тикеров (< {MIN_SAMPLES_PER_TICKER} val сэмплов)')

    # ── Диагностика ───────────────────────────────────────────────────────────
    a_vals = [v["a"] for v in per_ticker.values()]
    b_vals = [v["b"] for v in per_ticker.values()]
    print(f'  a range: [{min(a_vals):.3f}, {max(a_vals):.3f}]  '
          f'mean={np.mean(a_vals):.3f}  std={np.std(a_vals):.3f}')
    print(f'  b range: [{min(b_vals):+.3f}, {max(b_vals):+.3f}]  '
          f'mean={np.mean(b_vals):+.3f}  std={np.std(b_vals):.3f}')

    # Тикеры с самым большим bias correction (|b|)
    by_bias = sorted(per_ticker.items(), key=lambda x: abs(x[1]["b"]), reverse=True)[:5]
    print(f'  Top-5 |b| (сильнейшая bias-коррекция):')
    for t, v in by_bias:
        print(f'    {t:8s}  a={v["a"]:.3f}  b={v["b"]:+.3f}')

    # ── ECE-guard: если Platt ухудшает калибровку на val — откатываем к identity ──
    _sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    p_raw_val    = _sigmoid(logits[mask_ud])
    p_platt_val  = _sigmoid(a_global * logits[mask_ud] + b_global)
    ece_raw_val  = _compute_ece(p_raw_val,   labels_bin[mask_ud])
    ece_platt_val= _compute_ece(p_platt_val, labels_bin[mask_ud])
    print(f'\n  Val ECE (global): raw={ece_raw_val:.2f}%  platt={ece_platt_val:.2f}%')
    platt_valid = ece_platt_val < ece_raw_val + 0.5   # Platt должен улучшать хотя бы на 0.5pp
    if not platt_valid:
        print(f'  ⚠️  Platt не улучшил ECE (+{ece_platt_val - ece_raw_val:.2f}pp на val) '
              f'→ сохраняем identity (a=1, b=0) для всех тикеров')
        per_ticker    = {t: {"a": 1.0, "b": 0.0} for t in unique_tickers}
        a_global, b_global = 1.0, 0.0
    else:
        print(f'  ✓  Platt улучшил ECE ({ece_raw_val:.2f}% → {ece_platt_val:.2f}%)')

    # ── Сохранение ────────────────────────────────────────────────────────────
    result = {
        'global':         {"a": round(a_global, 4), "b": round(b_global, 4)},
        'n_val':          int(mask_ud.sum()),
        'seeds':          list(seeds),
        'per_ticker':     per_ticker,
        'coverage_report': coverage_report,
        'min_samples_threshold': MIN_SAMPLES_PER_TICKER,
        'n_fitted':       len(unique_tickers) - skipped,
        'n_fallback':     skipped,
        'platt_valid':    bool(platt_valid),
        'ece_raw_val':    round(ece_raw_val, 4),
        'ece_platt_val':  round(ece_platt_val, 4),
    }
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\nСохранено → {out_path}')

    # Также сразу применяем к ensemble_predictions.npz, чтобы downstream
    # (reliability_report, decision_layer) сразу видели dir_prob_platt
    npz_path = 'ml/ensemble/ensemble_predictions.npz'
    if os.path.exists(npz_path):
        print(f'\nАвтоприменение к {npz_path}...')
        apply_to_npz(npz_path, out_path)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Apply to existing npz
# ──────────────────────────────────────────────────────────────────────────────

def apply_to_npz(npz_path: str, calib_json: str, out_path: str | None = None):
    """Применяет Platt калибровку к npz, добавляет поле `dir_prob_platt`."""
    out_path = out_path or npz_path
    print(f'Загрузка {npz_path}...')
    data = dict(np.load(npz_path, allow_pickle=False))

    if 'dir_prob' not in data:
        print('  Ошибка: dir_prob не найден в npz')
        return
    if 'test_tickers' not in data:
        print('  Ошибка: test_tickers не найден в npz')
        return

    calib_map    = load_ticker_calibrators(calib_json)
    global_calib = load_global_calibrator(calib_json)
    if not calib_map:
        print(f'  Ошибка: {calib_json} не найден или пуст')
        return

    tickers   = data['test_tickers']
    dir_prob  = data['dir_prob']

    calibrated = apply_calibration(dir_prob, tickers, calib_map, global_calib)
    data['dir_prob_platt'] = calibrated

    np.savez(out_path, **data)
    print(f'Сохранено → {out_path}')
    print(f'dir_prob       mean: {dir_prob.mean():.4f} ± {dir_prob.std():.4f}')
    print(f'dir_prob_platt mean: {calibrated.mean():.4f} ± {calibrated.std():.4f}')


# ──────────────────────────────────────────────────────────────────────────────
# Coverage report
# ──────────────────────────────────────────────────────────────────────────────

def show_coverage(path: str) -> None:
    if not os.path.exists(path):
        print(f"Файл не найден: {path}"); return
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    report = data.get('coverage_report')
    if not report:
        print(f"  [INFO] coverage_report отсутствует — устаревший формат "
              f"({path}). Пересчитайте: py -m ml.calibrate_platt")
        return

    n_total    = len(report)
    n_fitted   = sum(1 for v in report.values() if not v["used_fallback"])
    n_fallback = n_total - n_fitted
    g          = data.get('global', {"a": 1.0, "b": 0.0})

    print(f"\n  Platt scaling coverage report ({path})")
    print(f"  ────────────────────────────────────────────")
    print(f"  Тикеров:            {n_total}")
    print(f"  Fit'ы (n≥{data.get('min_samples_threshold', 30)}):       {n_fitted}")
    print(f"  Fallback на global: {n_fallback}  "
          f"(global a={g['a']:.4f} b={g['b']:+.4f})")

    fb = sorted([(t, v["n_samples"]) for t, v in report.items() if v["used_fallback"]],
                key=lambda x: x[1])
    if fb:
        print(f"\n  Fallback тикеры (n_samples < threshold):")
        for t, n in fb:
            print(f"    {t:8s}  n={n:3d}  fallback")

    fitted = sorted(
        [(t, v["a"], v["b"], v["n_samples"]) for t, v in report.items() if not v["used_fallback"]],
        key=lambda x: abs(x[2]), reverse=True)
    if fitted:
        print(f"\n  Top-10 |b| — самые большие bias-коррекции:")
        for t, a, b, n in fitted[:10]:
            print(f"    {t:8s}  a={a:.3f}  b={b:+.3f}  n={n}")


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',      type=int, nargs='+', default=[42, 123, 7])
    parser.add_argument('--out',        default='ml/ensemble/platt_per_ticker.json')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--apply',      default=None,
                        help='Путь к npz: применить калибровку к уже сохранённому файлу')
    parser.add_argument('--calib-json', default='ml/ensemble/platt_per_ticker.json')
    parser.add_argument('--coverage-report', action='store_true')
    args = parser.parse_args()

    if args.coverage_report:
        show_coverage(args.calib_json)
    elif args.apply:
        apply_to_npz(args.apply, args.calib_json)
    else:
        calibrate(
            seeds=tuple(args.seeds),
            out_path=args.out,
            batch_size=args.batch_size,
        )


if __name__ == '__main__':
    main()
