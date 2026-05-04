"""ml/feature_importance.py — Sprint 7 #15: Permutation Importance на 37 INDICATOR_COLS.

Запуск:
    py -m ml.feature_importance [--seeds 42 123 7] [--batch-size 256] [--out ml/feature_importance.json]
"""
import os, sys, json, argparse
os.environ['GRPC_DNS_RESOLVER'] = 'native'
_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import Subset

from ml.config import CFG, SCALES
from ml.dataset_v3 import (
    build_full_multiscale_dataset_v3, temporal_split, INDICATOR_COLS,
)
from ml.multiscale_cnn_v3 import MultiScaleHybridV3, _make_loader_v3
from ml.trainer_v3 import _forward_unpack


def _dir_acc(dir_probs: np.ndarray, y: np.ndarray) -> float:
    mask = y != 1  # только UP/DOWN
    if mask.sum() == 0:
        return 0.5
    return float(((dir_probs[mask] > 0.5) == (y[mask] == 0)).mean())


def _run_inference(models, loader, device, ctx_dim, use_hourly,
                   perm_feature_idx: int = -1, rng: np.random.Generator = None):
    """Прогоняет inference через ансамбль.

    perm_feature_idx >= 0: заменяет столбец j в nums[W] случайной перестановкой.
    Возвращает усреднённые dir_probs [N] и y_true [N].
    """
    all_dir = []
    all_y   = []

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, _, ctx, hourly_data, *_ = batch
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            ht    = (hourly_data.to(device)
                     if (use_hourly and hourly_data is not None) else None)

            if num_dict is not None:
                nums = {W: num_dict[W].clone().to(device) for W in SCALES}
                if perm_feature_idx >= 0:
                    B = nums[SCALES[0]].shape[0]
                    perm = rng.permutation(B)
                    for W in SCALES:
                        n = nums[W]
                        if n.ndim == 3:   # [B, T, n_features]
                            n[:, :, perm_feature_idx] = n[perm, :, perm_feature_idx]
                        else:              # [B, n_features]
                            n[:, perm_feature_idx] = n[perm, perm_feature_idx]
            else:
                nums = None

            batch_dir = []
            for model in models:
                _, _, _, dir_l, *_ = _forward_unpack(model, imgs, nums, ctx_t, ht)
                batch_dir.append(torch.sigmoid(dir_l).cpu().numpy())

            all_dir.append(np.mean(batch_dir, axis=0))
            all_y.append(cls_y.numpy())

    return np.concatenate(all_dir), np.concatenate(all_y)


def compute_feature_importance(
    seeds=(42, 123, 7),
    batch_size=256,
    out_path='ml/feature_importance.json',
    n_repeats=3,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Данные ────────────────────────────────────────────────────────────────
    print('Загрузка датасета...')
    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=False, use_hourly=True)
    _, _, idx_test = temporal_split(ticker_lengths)
    y_test = y_all[idx_test]
    te_ds  = Subset(dataset, idx_test.tolist())
    loader = _make_loader_v3(te_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Загрузка моделей ──────────────────────────────────────────────────────
    n_ind = len(INDICATOR_COLS)
    models = []
    for seed in seeds:
        path = f'ml/ensemble/model_seed{seed}.pt'
        if not os.path.exists(path):
            print(f'  Пропуск seed {seed}: файл не найден ({path})')
            continue
        m = MultiScaleHybridV3(
            ctx_dim=ctx_dim, n_indicator_cols=n_ind,
            future_bars=CFG.future_bars, use_hourly=True,
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
        print(f'  Загружен seed {seed}')

    if not models:
        raise RuntimeError('Нет доступных моделей ансамбля в ml/ensemble/')

    # ── Baseline ──────────────────────────────────────────────────────────────
    print('\nBaseline inference...')
    dir_base, y_true = _run_inference(models, loader, device, ctx_dim, use_hourly=True)
    baseline_acc = _dir_acc(dir_base, y_true)
    print(f'Baseline dir_acc: {baseline_acc:.4f}  (N={len(y_true)})')

    # ── Permutation sweep ─────────────────────────────────────────────────────
    print(f'\nPermutation importance ({len(INDICATOR_COLS)} фичей × {n_repeats} повторов)...')
    rng = np.random.default_rng(42)
    importances = {}

    for j, feat in enumerate(INDICATOR_COLS):
        accs = []
        for _ in range(n_repeats):
            dir_p, _ = _run_inference(
                models, loader, device, ctx_dim, use_hourly=True,
                perm_feature_idx=j, rng=rng)
            accs.append(_dir_acc(dir_p, y_true))
        imp = baseline_acc - float(np.mean(accs))
        importances[feat] = round(imp, 6)
        marker = '  ▲' if imp > 0.002 else ('  ▼' if imp < -0.001 else '  ·')
        print(f'  [{j:2d}] {feat:<22}  Δdir_acc = {imp:+.4f}{marker}')

    # ── Сохранение ────────────────────────────────────────────────────────────
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    result = {
        'baseline_dir_acc': baseline_acc,
        'n_test':           int(len(y_true)),
        'n_repeats':        n_repeats,
        'seeds':            list(seeds),
        'importances':      {k: v for k, v in sorted_imp},
    }
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\nСохранено → {out_path}')

    # ── Итоги ─────────────────────────────────────────────────────────────────
    print(f'\n{"═" * 50}')
    print('TOP-10 наиболее важных фичей:')
    for i, (feat, imp) in enumerate(sorted_imp[:10], 1):
        print(f'  {i:2d}. {feat:<22}  {imp:+.4f}')

    negative = [(f, v) for f, v in sorted_imp if v < -0.001]
    if negative:
        print(f'\nФичи с отрицательной важностью (кандидаты на удаление):')
        for feat, imp in negative:
            print(f'  {feat:<22}  {imp:+.4f}')
    else:
        print('\nОтрицательно важных фичей не найдено.')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',      type=int, nargs='+', default=[42, 123, 7])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--out',        default='ml/feature_importance.json')
    parser.add_argument('--repeats',    type=int, default=3,
                        help='Число повторов перестановки (усредняем для стабильности)')
    args = parser.parse_args()

    compute_feature_importance(
        seeds=tuple(args.seeds),
        batch_size=args.batch_size,
        out_path=args.out,
        n_repeats=args.repeats,
    )


if __name__ == '__main__':
    main()
