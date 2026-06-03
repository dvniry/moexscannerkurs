"""ml/calibrate_temperature.py — Sprint 7 #7: Per-ticker Temperature Scaling.

Post-hoc калибровка dir_prob по тикеру. Для каждого тикера в val-сете
подбирает температуру T_t: P(UP) = sigmoid(dir_logit / T_t).

Запуск:
    py -m ml.calibrate_temperature [--seeds 42 123 7] [--out ml/ensemble/temperature_per_ticker.json]
    py -m ml.calibrate_temperature --apply ml/ensemble/ensemble_predictions.npz

Полезные функции:
    load_ticker_temperatures(path) → {ticker: T_t}
    apply_calibration(dir_prob, tickers, temp_map) → calibrated_dir_prob
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

MIN_SAMPLES_PER_TICKER = 30   # минимум сэмплов для per-ticker fit; иначе глобальный T
T_MIN = 0.3
T_MAX = 5.0


# ──────────────────────────────────────────────────────────────────────────────
# Публичные утилиты
# ──────────────────────────────────────────────────────────────────────────────

def load_ticker_temperatures(path: str) -> dict:
    """Загружает {ticker: T_t} из JSON. Возвращает пустой dict если файл не найден."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('per_ticker', {})


def apply_calibration(dir_prob: np.ndarray,
                      tickers: np.ndarray,
                      temp_map: dict,
                      global_T: float = 1.0) -> np.ndarray:
    """Применяет per-ticker температурное масштабирование к dir_prob.

    Алгоритм:
        logit = log(p / (1-p))
        calibrated_prob = sigmoid(logit / T_t)
    Если тикер не в temp_map, использует global_T.
    """
    eps = 1e-6
    p   = np.clip(dir_prob, eps, 1.0 - eps).astype(np.float64)
    logits = np.log(p / (1.0 - p))

    calibrated = np.empty_like(p)
    for i, ticker in enumerate(tickers):
        T = temp_map.get(str(ticker), global_T)
        calibrated[i] = 1.0 / (1.0 + np.exp(-logits[i] / T))

    return calibrated.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration fit
# ──────────────────────────────────────────────────────────────────────────────

def _fit_temperature(logits: np.ndarray, labels: np.ndarray,
                     max_iter: int = 200) -> float:
    """Подбирает единственную T через LBFGS на NLL.

    labels: бинарные (1=UP, 0=DOWN).
    Возвращает скалярный float T ∈ [T_MIN, T_MAX].
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    T_param = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T_param], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            logits_t / T_param.clamp(T_MIN, T_MAX), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T_param.clamp(T_MIN, T_MAX).item())


def _collect_val_logits(models, val_indices, dataset, device, ctx_dim,
                        use_hourly, batch_size=256):
    """Собирает dir_logits и cls_y для каждого val сэмпла.

    Возвращает:
        logits [N]  — сырые логиты dir_head
        labels [N]  — cls_y (0=UP, 1=FLAT, 2=DOWN)
        tickers [N] — тикер каждого сэмпла
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
              out_path='ml/ensemble/temperature_per_ticker.json',
              batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Данные ────────────────────────────────────────────────────────────────
    print('Загрузка датасета...')
    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=False, use_hourly=True)
    _, idx_val, _ = temporal_split(ticker_lengths)

    # ── Модели ────────────────────────────────────────────────────────────────
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
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True),
                          strict=False)  # Sprint 11.1: tolerant к quantile_head
        m.eval()
        models.append(m)
        print(f'  Загружен seed {seed}')

    if not models:
        raise RuntimeError('Нет доступных моделей ансамбля в ml/ensemble/')

    # ── Сбор логитов на val-сете ───────────────────────────────────────────
    print(f'\nСбор val логитов ({len(idx_val)} сэмплов)...')
    logits, labels, tickers = _collect_val_logits(
        models, idx_val, dataset, device, ctx_dim, use_hourly=True,
        batch_size=batch_size)

    # Маска UP/DOWN (убираем FLAT для бинарной калибровки)
    mask_ud = labels != 1
    labels_bin = (labels == 0).astype(np.float32)  # 1=UP, 0=DOWN

    # ── Глобальная T ──────────────────────────────────────────────────────────
    T_global = _fit_temperature(logits[mask_ud], labels_bin[mask_ud])
    print(f'Глобальная T = {T_global:.4f}')

    # ── Per-ticker T ──────────────────────────────────────────────────────────
    unique_tickers = np.unique(tickers)
    per_ticker = {}
    skipped = 0

    # B-23 fix: coverage_report — отслеживаем, для каких тикеров pre-fit был fallback
    coverage_report: dict[str, dict] = {}

    print(f'\nPer-ticker калибровка ({len(unique_tickers)} тикеров):')
    for ticker in sorted(unique_tickers):
        mask_t  = (tickers == ticker) & mask_ud
        n_t     = int(mask_t.sum())
        if n_t < MIN_SAMPLES_PER_TICKER:
            per_ticker[ticker] = T_global
            skipped += 1
            coverage_report[str(ticker)] = {
                "n_samples":     n_t,
                "T":             round(T_global, 4),
                "used_fallback": True,
                "reason":        f"n<{MIN_SAMPLES_PER_TICKER}",
            }
            continue
        T_t = _fit_temperature(logits[mask_t], labels_bin[mask_t])
        per_ticker[ticker] = round(T_t, 4)
        coverage_report[str(ticker)] = {
            "n_samples":     n_t,
            "T":             round(T_t, 4),
            "used_fallback": False,
            "reason":        "fit_ok",
        }

    print(f'  Подобрано per-ticker: {len(unique_tickers) - skipped} / {len(unique_tickers)}')
    print(f'  Глобальный fallback:  {skipped} тикеров (< {MIN_SAMPLES_PER_TICKER} val сэмплов)')
    print(f'  Coverage detail: '
          f'{sum(1 for v in coverage_report.values() if not v["used_fallback"])} fitted, '
          f'{sum(1 for v in coverage_report.values() if v["used_fallback"])} fallback')

    # ── Диагностика ───────────────────────────────────────────────────────────
    T_vals = list(per_ticker.values())
    print(f'  T range: [{min(T_vals):.3f}, {max(T_vals):.3f}]  '
          f'mean={np.mean(T_vals):.3f}  std={np.std(T_vals):.3f}')

    top_cold  = sorted(per_ticker.items(), key=lambda x: x[1])[:5]
    top_warm  = sorted(per_ticker.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f'  Самые "холодные" (T<1, уверенная модель): '
          + ', '.join(f'{t}={v:.2f}' for t, v in top_cold))
    print(f'  Самые "тёплые"  (T>1, неуверенная модель): '
          + ', '.join(f'{t}={v:.2f}' for t, v in top_warm))

    # ── Сохранение ────────────────────────────────────────────────────────────
    result = {
        'global_T':       round(T_global, 4),
        'n_val':          int(mask_ud.sum()),
        'seeds':          list(seeds),
        'per_ticker':     per_ticker,
        # B-23 fix: per-ticker метаданные для аудита покрытия
        'coverage_report': coverage_report,
        'min_samples_threshold': MIN_SAMPLES_PER_TICKER,
        'n_fitted':       len(unique_tickers) - skipped,
        'n_fallback':     skipped,
    }
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\nСохранено → {out_path}')
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Apply to existing npz
# ──────────────────────────────────────────────────────────────────────────────

def apply_to_npz(npz_path: str, temp_json: str, out_path: str = None):
    """Применяет температурную калибровку к уже сохранённому npz.

    Добавляет поле `dir_prob_calibrated` в npz.
    """
    out_path = out_path or npz_path
    print(f'Загрузка {npz_path}...')
    data = dict(np.load(npz_path, allow_pickle=False))

    if 'dir_prob' not in data:
        print('  Ошибка: dir_prob не найден в npz')
        return
    if 'test_tickers' not in data:
        print('  Ошибка: test_tickers не найден в npz')
        return

    temp_map  = load_ticker_temperatures(temp_json)
    if not temp_map:
        print(f'  Ошибка: {temp_json} не найден или пуст')
        return

    global_T  = temp_map.get('__global__', 1.0)
    tickers   = data['test_tickers']
    dir_prob  = data['dir_prob']

    calibrated = apply_calibration(dir_prob, tickers, temp_map, global_T)
    data['dir_prob_calibrated'] = calibrated

    np.savez(out_path, **data)
    old_dir_acc = float(((dir_prob > 0.5).astype(int) == (dir_prob > 0.5).astype(int)).mean())
    print(f'Сохранено → {out_path}')
    print(f'dir_prob mean:            {dir_prob.mean():.4f} ± {dir_prob.std():.4f}')
    print(f'dir_prob_calibrated mean: {calibrated.mean():.4f} ± {calibrated.std():.4f}')


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

def show_coverage(path: str) -> None:
    """B-23: показать coverage_report из существующего temperature_per_ticker.json."""
    if not os.path.exists(path):
        print(f"Файл не найден: {path}"); return
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    report = data.get('coverage_report')
    if not report:
        print(f"  [INFO] coverage_report отсутствует — это устаревший формат "
              f"({path}). Пересчитайте: py -m ml.calibrate_temperature")
        per_ticker = data.get('per_ticker', {})
        print(f"  Тикеров с T: {len(per_ticker)} (без n_samples метаданных)")
        return

    n_total    = len(report)
    n_fitted   = sum(1 for v in report.values() if not v["used_fallback"])
    n_fallback = n_total - n_fitted
    global_T   = data.get('global_T', 1.0)

    print(f"\n  Temperature scaling coverage report ({path})")
    print(f"  ────────────────────────────────────────────")
    print(f"  Тикеров:            {n_total}")
    print(f"  Fit'ы (n≥{data.get('min_samples_threshold', 30)}):       {n_fitted}")
    print(f"  Fallback на global: {n_fallback}  (global_T={global_T:.4f})")
    print()

    fb = sorted([(t, v["n_samples"]) for t, v in report.items() if v["used_fallback"]],
                key=lambda x: x[1])
    if fb:
        print(f"  Fallback тикеры (n_samples < threshold):")
        for t, n in fb:
            print(f"    {t:8s}  n={n:3d}  T={global_T:.4f} (fallback)")

    fitted_T = sorted(
        [(t, v["T"], v["n_samples"]) for t, v in report.items() if not v["used_fallback"]],
        key=lambda x: x[1])
    if fitted_T:
        print(f"\n  Самые холодные T (модель уверена) — top 5:")
        for t, T, n in fitted_T[:5]:
            print(f"    {t:8s}  T={T:.3f}  n={n}")
        print(f"  Самые тёплые T (модель неуверенна) — top 5:")
        for t, T, n in fitted_T[-5:]:
            print(f"    {t:8s}  T={T:.3f}  n={n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',      type=int, nargs='+', default=[42, 123, 7])
    parser.add_argument('--out',        default='ml/ensemble/temperature_per_ticker.json')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--apply',      default=None,
                        help='Путь к npz: применить калибровку к уже сохранённому файлу')
    parser.add_argument('--temp-json',  default='ml/ensemble/temperature_per_ticker.json')
    parser.add_argument('--coverage-report', action='store_true',
                        help='B-23: показать coverage_report из существующего --temp-json')
    args = parser.parse_args()

    if args.coverage_report:
        show_coverage(args.temp_json)
    elif args.apply:
        apply_to_npz(args.apply, args.temp_json)
    else:
        calibrate(
            seeds=tuple(args.seeds),
            out_path=args.out,
            batch_size=args.batch_size,
        )


if __name__ == '__main__':
    main()
