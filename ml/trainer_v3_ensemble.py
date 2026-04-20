"""Ensemble trainer: обучает N моделей с разными seeds и усредняет.

Запуск:
    python -m ml.trainer_v3_ensemble --n-seeds 3
    python -m ml.trainer_v3_ensemble --n-seeds 5 --epochs 20
"""
import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'
import sys, argparse, random, json, math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset

_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.config import CFG, SCALES
from ml.dataset_v3 import (
    build_full_multiscale_dataset_v3, temporal_split,
    INDICATOR_COLS, class_distribution,
)
from ml.multiscale_cnn_v3 import (
    MultiScaleHybridV3, MultiTaskLossV3, _make_loader_v3,
    evaluate_multiscale_v3,
)
from ml.trainer_v3 import _run_epochs, _init_cls_head


def set_seed(seed: int):
    """Фиксирует все источники случайности."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Для воспроизводимости (небольшая потеря скорости):
    torch.backends.cudnn.deterministic = False  # True дает +20% времени
    torch.backends.cudnn.benchmark = True


def train_one_seed(seed: int, save_path: str, shared_data: dict,
                    epochs: int = 20, max_lr: float = 2e-4,
                    patience: int = 5):
    """Обучает одну модель с фиксированным seed."""
    set_seed(seed)
    device = shared_data['device']

    print(f'\n{"═" * 70}')
    print(f'  🎲 SEED {seed} — обучение модели → {save_path}')
    print(f'{"═" * 70}')

    # Из shared_data берём уже построенный датасет
    tr_loader  = shared_data['tr_loader']
    val_loader = shared_data['val_loader']
    te_ds      = shared_data['te_ds']
    ctx_dim    = shared_data['ctx_dim']
    cls_weights = shared_data['cls_weights']
    n_ind      = shared_data['n_ind']
    y_test     = shared_data['y_test']
    use_hourly = shared_data['use_hourly']

    model = MultiScaleHybridV3(
        ctx_dim=ctx_dim, n_indicator_cols=n_ind,
        future_bars=CFG.future_bars,
        use_hourly=use_hourly).to(device)
    _init_cls_head(model)

    criterion = MultiTaskLossV3(
        cls_weight=cls_weights,
        gamma_per_class=(2.0, 1.0, 2.0),
        label_smoothing=0.05,
        future_bars=CFG.future_bars,
        huber_delta=0.3,
        direction_weight=0.80,
        reg_loss_weight=0.30,
        aux_loss_weight=0.05,
    ).to(device)

    # Раздельные LR
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    hourly_ids   = ({id(p) for p in model.hourly_enc.parameters()}
                    if use_hourly and hasattr(model, 'hourly_enc') else set())
    cls_head_ids = {id(p) for p in model.cls_head.parameters()}
    dir_head_ids = {id(p) for p in model.dir_head.parameters()}
    crit_params  = list(criterion.parameters())

    param_groups = [
        {'params': list(model.backbone.parameters()),
         'lr': max_lr * 0.15, 'name': 'backbone', 'weight_decay': 5e-4},
        {'params': list(model.cls_head.parameters()),
         'lr': max_lr * 0.5,  'name': 'cls_head', 'weight_decay': 1e-4},
        {'params': list(model.dir_head.parameters()),
         'lr': max_lr,        'name': 'dir_head', 'weight_decay': 1e-4},
        {'params': [p for p in model.parameters()
                    if p.requires_grad
                    and id(p) not in backbone_ids
                    and id(p) not in hourly_ids
                    and id(p) not in cls_head_ids
                    and id(p) not in dir_head_ids],
         'lr': max_lr, 'name': 'other', 'weight_decay': 5e-3},
    ]
    if hourly_ids:
        param_groups.insert(3, {
            'params': list(model.hourly_enc.parameters()),
            'lr': max_lr * 0.1, 'name': 'hourly', 'weight_decay': 5e-4})

    optimizer = AdamW(param_groups
                      + [{'params': crit_params, 'lr': max_lr,
                          'name': 'criterion', 'weight_decay': 1e-4}])

    
    scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    _run_epochs(
        model, tr_loader, val_loader, optimizer, scheduler,
        criterion, device,
        n_epochs=epochs,
        patience_limit=patience,
        save_path=save_path,
        phase_name=f'S{seed}',
        ctx_dim=ctx_dim,
        use_hourly=use_hourly,
        accum_steps=2)

    # Вернуть путь + финальные предсказания на test
    print(f'\n  [SEED {seed}] Evaluation on test:')
        # Собираем предсказания для ансамбля
    # ВАЖНО: перезагружаем модель из сохранённого чекпоинта
    model_eval = MultiScaleHybridV3(
        ctx_dim=ctx_dim, n_indicator_cols=n_ind,
        future_bars=CFG.future_bars,
        use_hourly=use_hourly).to(device)
    model_eval.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))
    model_eval.eval()

    from torch.utils.data import DataLoader
    loader = _make_loader_v3(te_ds, batch_size=256, shuffle=False, num_workers=0)
    all_cls_logits, all_dir_prob, all_ohlc_pred = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, hourly_data, *_ = batch
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            ht    = hourly_data.to(device) if use_hourly else None
            nums  = {W: num_dict[W].to(device) for W in SCALES}
            lo, op, _, dir_l = model_eval(imgs, nums, ctx_t, hourly=ht)
            all_cls_logits.append(torch.softmax(lo, dim=1).cpu().numpy())
            all_dir_prob.append(torch.sigmoid(dir_l).cpu().numpy())
            all_ohlc_pred.append(op.cpu().numpy())

    # val dir accuracy
    val_dir_probs, val_trues = [], []
    val_loader_loc = shared_data['val_loader']
    with torch.no_grad():
        for batch in val_loader_loc:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, hourly_data, *_ = batch
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if shared_data['ctx_dim'] > 0 else None
            ht    = hourly_data.to(device) if shared_data['use_hourly'] else None
            nums  = {W: num_dict[W].to(device) for W in SCALES}
            _, _, _, dir_l = model_eval(imgs, nums, ctx_t, hourly=ht)
            val_dir_probs.append(torch.sigmoid(dir_l).cpu().numpy())
            val_trues.append(cls_y.numpy())

    val_dir_probs = np.concatenate(val_dir_probs)
    val_trues_arr = np.concatenate(val_trues)
    mask_ud = val_trues_arr != 1
    if mask_ud.any():
        val_dir_acc = (
            (val_dir_probs[mask_ud] > 0.5).astype(int)
            == (val_trues_arr[mask_ud] == 0).astype(int)
        ).mean()
    else:
        val_dir_acc = 0.5

    print(f'  [SEED {seed}] val_dir_acc = {val_dir_acc:.4f}')

    # Освобождаем память
    del model_eval
    torch.cuda.empty_cache()

    return {
        'seed':       seed,
        'val_dir_acc': float(val_dir_acc),
        'path':       save_path,
        'cls_probs':  np.concatenate(all_cls_logits),
        'dir_prob':   np.concatenate(all_dir_prob),
        'ohlc_pred':  np.concatenate(all_ohlc_pred),
    }


def evaluate_ensemble(results: list, y_test: np.ndarray,
                       ohlc_test: np.ndarray,
                       val_dir_accs: dict = None):
    """v3.18.2: Selective weighted ensemble.
    
    1. Фильтруем модели с val_dir_acc < 0.51 (бесполезные)
    2. Усредняем с весами по val_dir_acc
    """
    from sklearn.metrics import classification_report, f1_score

    print(f'\n{"═" * 70}')
    print(f'  📊 ENSEMBLE EVALUATION')
    print(f'{"═" * 70}\n')

    # ── Фильтрация по качеству ──
    if val_dir_accs is not None:
        alive = [r for r in results
                  if val_dir_accs.get(r['seed'], 0) > 0.51]
        if len(alive) == 0:
            print('  ⚠️  Все модели ниже порога — используем все')
            alive = results
        elif len(alive) < len(results):
            dropped = [r['seed'] for r in results if r not in alive]
            print(f'  🚫 Отброшены seeds с val_dir_acc <= 0.51: {dropped}')
        results = alive

    # ── Расчёт весов (softmax по val_dir_acc - 0.5) ──
    if val_dir_accs is not None:
        edges = np.array([
            max(0, val_dir_accs.get(r['seed'], 0.5) - 0.5)
            for r in results
        ])
        if edges.sum() > 0:
            weights = edges / edges.sum()
        else:
            weights = np.ones(len(results)) / len(results)
    else:
        weights = np.ones(len(results)) / len(results)

    print(f'  Weights: {dict(zip([r["seed"] for r in results], weights.round(3)))}')

    # Взвешенное усреднение
    cls_probs_avg = np.average(
        [r['cls_probs'] for r in results], axis=0, weights=weights)
    dir_prob_avg = np.average(
        [r['dir_prob'] for r in results], axis=0, weights=weights)
    ohlc_pred_avg = np.average(
        [r['ohlc_pred'] for r in results], axis=0, weights=weights)

    preds = cls_probs_avg.argmax(axis=1)
    trues = y_test

    print(classification_report(trues, preds,
                                 target_names=['UP', 'FLAT', 'DOWN'],
                                 digits=4, zero_division=0))

    mask_ud = trues != 1
    dir_acc = 0.5; baseline = 0.5
    if mask_ud.any():
        dir_target = (trues[mask_ud] == 0).astype(int)
        dir_pred   = (dir_prob_avg[mask_ud] > 0.5).astype(int)
        dir_acc = (dir_pred == dir_target).mean()
        baseline = (trues[mask_ud] == 0).mean()
        print(f'\n  📈 Residual dir accuracy (weighted ensemble): {dir_acc:.4f}')
        print(f'  📉 Baseline (always BUY):                      {baseline:.4f}')
        print(f'  💰 Edge:                                        {dir_acc - baseline:+.4f}')

    # Per-model breakdown
    print(f'\n  Per-seed dir accuracy:')
    for r in results:
        dir_p = r['dir_prob']
        if mask_ud.any():
            acc = ((dir_p[mask_ud] > 0.5).astype(int)
                   == (trues[mask_ud] == 0).astype(int)).mean()
            val_a = val_dir_accs.get(r['seed'], 0) if val_dir_accs else 0
            print(f'    seed={r["seed"]:>4}  '
                  f'test_dir={acc:.4f}  val_dir={val_a:.4f}')

    # Agreement
    if len(results) > 1:
        dir_preds_all = [(r['dir_prob'] > 0.5).astype(int) for r in results]
        agreement = np.mean([
            (dir_preds_all[i] == dir_preds_all[j]).mean()
            for i in range(len(results))
            for j in range(i + 1, len(results))
        ])
        print(f'  🤝 Mean pairwise agreement: {agreement:.4f}')

    return cls_probs_avg, dir_prob_avg, ohlc_pred_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Количество моделей в ансамбле')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=None,
                        help='Явный список seeds (если не задан — [42, 123, 7, 2024, 1337])')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--max-lr', type=float, default=3e-4)
    parser.add_argument('--no-hourly', action='store_true')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--save-dir', default='ml/ensemble',
                        help='Куда сохранять модели')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    seeds = args.seeds or [42, 123, 7, 2024, 1337][:args.n_seeds]
    use_hourly = not args.no_hourly

    # ── Подготовка данных (один раз для всех моделей!) ─────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    print(f'  🎲 Обучаем {len(seeds)} моделей с seeds: {seeds}')

    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=args.rebuild, use_hourly=use_hourly)

    idx_tr, idx_val, idx_test = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.15,
        purge_bars=CFG.future_bars)
    y_tr   = y_all[idx_tr]
    y_test = y_all[idx_test]

    # ВАЖНО: ohlc для test — берём из dataset
    # ohlc для test + atr_ratio
    ohlc_test      = []
    atr_ratio_test = []

    for i in idx_test:
        _, _, _, ohlc_y, _, _, aux_y = dataset[int(i)]
        ohlc_test.append(ohlc_y.numpy())
        # aux_y = [vol*100, skew] — модель получает 2 элемента
        # но из кэша читаем напрямую для atr_ratio
        ticker, local_idx = dataset.records[int(i)]
        data = dataset._load(ticker)
        aux_raw = data["aux"][min(local_idx, data["_n"] - 1)]
        # aux_raw[2] = atr_ratio если новый кэш [N,3], иначе fallback
        if len(aux_raw) >= 3:
            atr_v = float(np.clip(aux_raw[2], 0.001, 0.15))
        else:
            atr_v = 0.018   # fallback для старого кэша
        atr_ratio_test.append(atr_v)

    ohlc_test      = np.array(ohlc_test, dtype=np.float32)
    atr_ratio_arr  = np.array(atr_ratio_test, dtype=np.float32)

    print(f'  atr_ratio: mean={atr_ratio_arr.mean():.4f} '
          f'std={atr_ratio_arr.std():.4f} '
          f'min={atr_ratio_arr.min():.4f} '
          f'max={atr_ratio_arr.max():.4f}')
    try:
        from ml.dataset_v3 import _aux_path, _cls_path
        # Собираем atr_ratio из кэшированных файлов по тикерам
        ticker_atr = {}
        for ticker, n_samples in ticker_lengths:
            # atr_ratio хранится как 4-й возврат build_ohlc_labels
            # Мы его не сохраняли отдельно — берём из aux или считаем приближение
            atr_path = f'ml/cache_v3/atr{ticker}.npy'
            if os.path.exists(atr_path):
                ticker_atr[ticker] = np.load(atr_path, mmap_mode='r')

        if ticker_atr:
            # Собираем atr_ratio для test индексов
            # records[i] = (ticker, local_idx)
            records = dataset.records  # список (ticker, local_idx)
            for global_idx in idx_test:
                ticker, local_idx = records[int(global_idx)]
                if ticker in ticker_atr:
                    arr = ticker_atr[ticker]
                    atr_v = float(arr[min(local_idx, len(arr)-1)])
                else:
                    atr_v = 0.018  # fallback: 1.8%
                atr_ratio_test.append(atr_v)
            atr_ratio_arr = np.array(atr_ratio_test, dtype=np.float32)
            print(f'  atr_ratio: mean={atr_ratio_arr.mean():.4f} '
                  f'std={atr_ratio_arr.std():.4f}')
        else:
            raise FileNotFoundError("atr кэш не найден")

    except Exception as e:
        print(f'  ⚠️  atr_ratio не найден ({e}), используем fallback 0.018')
        atr_ratio_arr = np.full(len(idx_test), 0.018, dtype=np.float32)

    tr_subset  = Subset(dataset, idx_tr.tolist())
    val_subset = Subset(dataset, idx_val.tolist())
    tr_loader  = _make_loader_v3(tr_subset, CFG.batch_size,
                                  shuffle=True, sampler=None, num_workers=0)
    val_loader = _make_loader_v3(val_subset, CFG.batch_size,
                                  shuffle=False, num_workers=0)
    te_ds = Subset(dataset, idx_test.tolist())

    counts_dict = Counter(y_tr.tolist()); total_tr = len(y_tr)
    raw_w = [math.sqrt(total_tr / max(counts_dict.get(i, 1), 1)) for i in range(3)]
    max_w = max(raw_w)
    cls_weights = torch.tensor([w / max_w for w in raw_w], dtype=torch.float32).to(device)

    shared_data = {
        'device': device,
        'tr_loader': tr_loader, 'val_loader': val_loader,
        'te_ds': te_ds, 'ctx_dim': ctx_dim,
        'cls_weights': cls_weights, 'n_ind': len(INDICATOR_COLS),
        'y_test': y_test, 'use_hourly': use_hourly,
    }

    # ── Последовательное обучение N моделей ─────────────────────
    results = []
    for seed in seeds:
        save_path = os.path.join(args.save_dir, f'model_seed{seed}.pt')
        result = train_one_seed(
            seed, save_path, shared_data,
            epochs=args.epochs, max_lr=args.max_lr,
            patience=args.patience,
        )
        results.append(result)

    # ── Ансамблевая оценка ──────────────────────────────────────
    val_dir_accs = {r['seed']: r['val_dir_acc'] for r in results}
    evaluate_ensemble(results, y_test, ohlc_test, val_dir_accs=val_dir_accs)

    # Сохраняем ансамблевые предсказания для backtest
    np.savez(
        os.path.join(args.save_dir, 'ensemble_predictions.npz'),
        cls_probs  = np.mean([r['cls_probs'] for r in results], axis=0),
        dir_prob   = np.mean([r['dir_prob']  for r in results], axis=0),
        ohlc_pred  = np.mean([r['ohlc_pred'] for r in results], axis=0),
        y_test     = y_test,
        ohlc_test  = ohlc_test,
        atr_ratio  = atr_ratio_arr,    # ← НОВОЕ
    )
    print(f'\n  📦 → {args.save_dir}/ensemble_predictions.npz')


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()