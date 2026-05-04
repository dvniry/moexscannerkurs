# ml/trainer_v4.py
"""Trainer v4.1 — MultiScaleHybridV4 + DirectionHead (v3.17 changes).

Изменения v4.1:
- Распаковка 4-кортежа forward.
- dir_logit передаётся в criterion.
- val_metric считается по dir_head.
- В Phase-0 заморожено ВСЁ кроме Kronos adapter и голов (не весь backbone).
- gamma_per_class=(2.0, 1.0, 2.0), label_smoothing=0.05.
- WRS отключён (shuffle=True).
- max_lr снижен до 1.5e-4.
"""

import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'
import sys, argparse, math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset

_cert = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from ml.config import CFG, SCALES
from ml.dataset_v3 import class_distribution


def _vram_info(prefix: str = '') -> str:
    if not torch.cuda.is_available():
        return ''
    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f'{prefix}VRAM: {used:.2f}/{total:.2f}GB'


# ────────────────────────────────────────────────────────────
# Phase-0: прогрев ТОЛЬКО адаптера и голов
# ────────────────────────────────────────────────────────────
def _phase0_warmup(
    model, tr_loader, val_loader,
    criterion, device,
    n_epochs: int = 3,
    ctx_dim: int = 0,
    use_hourly: bool = True,
):
    """v4.1: замораживаем ВСЁ кроме kronos.projector/grn + heads.
    Раньше было только kronos.encoder — остальной backbone поверх
    случайного Kronos-выхода коллапсировал.
    """
    from sklearn.metrics import f1_score

    print('\n  [Phase-0] Прогрев адаптера + голов')

    # Замораживаем ВСЁ
    for p in model.parameters():
        p.requires_grad_(False)

    # Размораживаем: kronos projector + grn + все heads + VSN
    unfroze_names = []
    if hasattr(model, 'kronos') and model.use_kronos:
        for n, p in model.kronos.projector.named_parameters():
            p.requires_grad_(True); unfroze_names.append(f'kronos.projector.{n}')
        for n, p in model.kronos.grn.named_parameters():
            p.requires_grad_(True); unfroze_names.append(f'kronos.grn.{n}')
        # _out_proj и _input_proj
        if model.kronos.extractor._out_proj is not None:
            for n, p in model.kronos.extractor._out_proj.named_parameters():
                p.requires_grad_(True)
        if model.kronos.extractor._input_proj is not None:
            for n, p in model.kronos.extractor._input_proj.named_parameters():
                p.requires_grad_(True)
    for n, p in model.vsn.named_parameters():
        p.requires_grad_(True)
    for n, p in model.cls_head.named_parameters():
        p.requires_grad_(True)
    for n, p in model.ohlc_head.named_parameters():
        p.requires_grad_(True)
    for n, p in model.aux_head.named_parameters():
        p.requires_grad_(True)
    for n, p in model.dir_head.named_parameters():
        p.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f'  [Phase-0] Обучаемо: {n_trainable / 1e6:.2f}M параметров')

    optimizer = AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    scaler    = GradScaler('cuda')

    best_f1 = 0.0
    for epoch in range(1, n_epochs + 1):
        model.train(); criterion.train()
        criterion.focal.set_gamma(epoch, warmup_epochs=n_epochs)
        total_loss = 0.0; n_steps = 0

        for batch in tr_loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
            hourly_data = hourly_opt[0] if hourly_opt else None

            imgs   = {W: imgs_dict[W].to(device) for W in SCALES}
            cls_y  = cls_y.to(device)
            ohlc_y = ohlc_y.to(device).clamp(-5., 5.)
            ctx_t  = ctx.to(device) if ctx_dim > 0 else None
            ht     = (hourly_data.to(device)
                     if use_hourly and hourly_data is not None else None)
            nums   = ({W: num_dict[W].to(device) for W in SCALES}
                     if num_dict is not None else None)

            optimizer.zero_grad()
            with autocast('cuda'):
                lo, op, aux, dir_l = model(imgs, nums, ctx_t, hourly=ht)
                lo     = lo.float().clamp(-15., 15.).nan_to_num(0.)
                op     = op.float().nan_to_num(0.)
                dir_l  = dir_l.float().clamp(-15., 15.).nan_to_num(0.)
                loss, _, _, _ = criterion(
                    lo, cls_y, op, ohlc_y.float(), dir_logit=dir_l)

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                n_steps    += 1

        # Быстрая валидация
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
                hourly_data = hourly_opt[0] if hourly_opt else None
                imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
                ctx_t = ctx.to(device) if ctx_dim > 0 else None
                ht    = (hourly_data.to(device)
                        if use_hourly and hourly_data is not None else None)
                nums  = ({W: num_dict[W].to(device) for W in SCALES}
                        if num_dict is not None else None)
                with autocast('cuda'):
                    lo, _, _, _ = model(imgs, nums, ctx_t, hourly=ht)
                val_preds.extend(lo.argmax(1).cpu().numpy())
                val_trues.extend(cls_y.numpy())

        vp = np.array(val_preds); vt = np.array(val_trues)
        macro_f1 = f1_score(vt, vp, average='macro', zero_division=0)
        print(f'  [Phase-0] E{epoch}/{n_epochs} '
              f'loss={total_loss / max(n_steps, 1):.4f} '
              f'val_mF1={macro_f1:.4f} '
              f'{_vram_info()}')
        best_f1 = max(best_f1, macro_f1)

    # Разморозить всё обратно — Phase-1 сам через param_groups разрулит LR
    for p in model.parameters():
        p.requires_grad_(True)

    # Kronos backbone: разморозить только нужные слои
    if hasattr(model, 'kronos') and model.use_kronos:
        enc = model.kronos.extractor.encoder
        if enc is not None:
            # Сначала заморозим весь Kronos encoder
            for p in enc.parameters():
                p.requires_grad_(False)
            # Разморозить последние N слоёв
            n_unfreeze = model.kronos.extractor._n_unfreeze
            if hasattr(enc, 'block'):
                blocks = list(enc.block)
                for block in blocks[-n_unfreeze:]:
                    for p in block.parameters():
                        p.requires_grad_(True)
                if hasattr(enc, 'final_layer_norm'):
                    for p in enc.final_layer_norm.parameters():
                        p.requires_grad_(True)
            elif hasattr(enc, 'layers'):
                layers = list(enc.layers)
                for layer in layers[-n_unfreeze:]:
                    for p in layer.parameters():
                        p.requires_grad_(True)
            print(f'  [Phase-0] Kronos encoder разморожен ({n_unfreeze} layers)')

    print(f'  [Phase-0] Завершён. best_mF1={best_f1:.4f}')
    torch.cuda.empty_cache()


# ────────────────────────────────────────────────────────────
# Phase-1: End-to-end
# ────────────────────────────────────────────────────────────
def _run_epochs_v4(
    model, tr_loader, val_loader,
    optimizer, scheduler, criterion,
    device, n_epochs, patience_limit,
    save_path, phase_name,
    ctx_dim, use_hourly,
    accum_steps: int = 1,
):
    from sklearn.metrics import f1_score, precision_score

    scaler = GradScaler('cuda')
    best_metric = 0.0
    patience    = 0

    seq_params = [
        p for _, m in model.named_modules()
        if isinstance(m, nn.LSTM)
        for p in m.parameters()
    ]
    cls_head_ps = list(model.cls_head.parameters())

    for epoch in range(1, n_epochs + 1):
        criterion.focal.set_gamma(epoch, warmup_epochs=10)

        model.train(); criterion.train()
        total_cls = 0.0; total_reg = 0.0; n_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tr_loader, 1):
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
            hourly_data = hourly_opt[0] if hourly_opt else None

            imgs   = {W: imgs_dict[W].to(device) for W in SCALES}
            cls_y  = cls_y.to(device)
            ohlc_y = ohlc_y.to(device).clamp(-5., 5.)
            ctx_t  = ctx.to(device) if ctx_dim > 0 else None
            ht     = (hourly_data.to(device)
                     if use_hourly and hourly_data is not None else None)
            nums   = ({W: num_dict[W].to(device) for W in SCALES}
                     if num_dict is not None else None)

            with autocast('cuda'):
                lo, op, aux_pred, dir_l = model(imgs, nums, ctx_t, hourly=ht)

                # NaN-чек ДО преобразований
                if (torch.isnan(lo).any() or torch.isnan(op).any()
                    or torch.isnan(dir_l).any()):
                    print(f'  [WARN] e={epoch} s={step} NaN preds — skip')
                    optimizer.zero_grad()
                    continue

                lo    = lo.float().clamp(-15., 15.)
                op    = op.float().clamp(-5., 5.)
                dir_l = dir_l.float().clamp(-15., 15.)

                loss, lcls, lreg, _ = criterion(
                    lo, cls_y, op, ohlc_y.float(), dir_logit=dir_l)

            if not torch.isfinite(loss):
                print(f'  [WARN] e={epoch} s={step} NaN loss — skip')
                optimizer.zero_grad()
                continue

            scaler.scale(loss / accum_steps).backward()

            total_cls += lcls.item()
            total_reg += lreg.item()
            n_steps   += 1

            if step % accum_steps == 0 or step == len(tr_loader):
                scaler.unscale_(optimizer)

                if seq_params:
                    nn.utils.clip_grad_norm_(seq_params, 0.5)
                nn.utils.clip_grad_norm_(cls_head_ps, 0.2)

                if model.use_kronos:
                    kb = model.kronos.get_backbone_params()
                    if kb:
                        nn.utils.clip_grad_norm_(kb, 0.1)

                trainable = [p for g in optimizer.param_groups
                            for p in g['params'] if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable, 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f'  [{phase_name}] e={epoch} s={step} '
                      f'cls={total_cls / max(n_steps, 1):.4f} '
                      f'reg={total_reg / max(n_steps, 1):.5f} '
                      f'{_vram_info()}', end='\r')

        # ── Validation ──
        model.eval(); criterion.eval()
        val_preds, val_trues = [], []
        val_ohlc_pred, val_ohlc_true = [], []
        val_dir_prob = []

        with torch.no_grad():
            for batch in val_loader:
                imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
                hourly_data = hourly_opt[0] if hourly_opt else None
                imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
                ctx_t = ctx.to(device) if ctx_dim > 0 else None
                ht    = (hourly_data.to(device)
                        if use_hourly and hourly_data is not None else None)
                nums  = ({W: num_dict[W].to(device) for W in SCALES}
                        if num_dict is not None else None)
                with autocast('cuda'):
                    lo, op, _, dir_l = model(imgs, nums, ctx_t, hourly=ht)
                val_preds.extend(lo.argmax(1).cpu().numpy())
                val_trues.extend(cls_y.numpy())
                val_ohlc_pred.append(op.cpu().float().numpy())
                val_ohlc_true.append(ohlc_y.float().numpy())
                val_dir_prob.append(torch.sigmoid(dir_l).cpu().numpy())

        vp = np.array(val_preds); vt = np.array(val_trues)
        val_acc   = (vp == vt).mean()
        macro_f1  = f1_score(vt, vp, average='macro', zero_division=0)
        f1pc      = f1_score(vt, vp, average=None, labels=[0, 1, 2], zero_division=0)
        buy_f1, hold_f1, sell_f1 = f1pc[0], f1pc[1], f1pc[2]

        ohlc_p = np.concatenate(val_ohlc_pred, 0)
        ohlc_t = np.concatenate(val_ohlc_true, 0)
        dir_p  = np.concatenate(val_dir_prob, 0)

        dir_acc_head = 0.5; dir_cov = 0.0
        if ohlc_p.shape[1] >= 4:
            delta_c = ohlc_t[:, 3]
            mask = np.abs(delta_c) > 1e-4
            dir_cov = float(mask.mean())
            if mask.any():
                dir_acc_head = float(
                    ((dir_p[mask] > 0.5).astype(int)
                     == (delta_c[mask] > 0).astype(int)).mean())

        prec_pc = precision_score(vt, vp, average=None,
                                  labels=[0, 1, 2], zero_division=0)
        prec_ud = 0.5 * (prec_pc[0] + prec_pc[2])
        val_metric = dir_acc_head * prec_ud

        if ohlc_p.shape[1] >= 4:
            ohlc_mae = float(np.abs(ohlc_p[:, :4] - ohlc_t[:, :4]).mean())
        else:
            ohlc_mae = 999.

        scheduler.step()

        print(f'\n  [{phase_name}] E{epoch:3d}/{n_epochs} '
              f'cls={total_cls / max(n_steps, 1):.4f} '
              f'reg={total_reg / max(n_steps, 1):.5f} | '
              f'val_acc={val_acc:.4f} mF1={macro_f1:.4f} | '
              f'buy={buy_f1:.3f} hold={hold_f1:.3f} sell={sell_f1:.3f} | '
              f'dir_head={dir_acc_head:.4f} (cov={dir_cov:.2f}) | '
              f'prec_ud={prec_ud:.3f} | '
              f'metric={val_metric:.4f} | '
              f'ohlc_mae={ohlc_mae:.4f} | '
              f'lr={optimizer.param_groups[0]["lr"]:.2e} '
              f'{_vram_info()}')

        if val_metric > best_metric:
            best_metric = val_metric
            patience    = 0
            torch.save(model.state_dict(), save_path)
            print(f'  ✓ Saved metric={val_metric:.4f}')
        else:
            patience += 1
            if patience >= patience_limit:
                print(f'  [{phase_name}] Early stop (patience={patience_limit})')
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))
    print(f'  [{phase_name}] Best metric={best_metric:.4f}')


# ────────────────────────────────────────────────────────────
# Main train
# ────────────────────────────────────────────────────────────
def train(
    model_path: str = 'ml/model_multiscale_v4.pt',
    use_hourly: bool = True,
    force_rebuild: bool = False,
    use_kronos: bool = True,
    kronos_model: str = 'amazon/chronos-t5-tiny',
    kronos_n_unfreeze: int = 2,
    phase0_epochs: int = 3,
):
    from ml.dataset_v3 import (
        build_full_multiscale_dataset_v3,
        temporal_split, INDICATOR_COLS,
    )
    from ml.multiscale_cnn_v3 import MultiTaskLossV3, _make_loader_v3
    from ml.multiscale_cnn_v4 import MultiScaleHybridV4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f'  GPU: {p.name} {p.total_memory / 1024**2:.0f} MB VRAM')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=force_rebuild, use_hourly=use_hourly)
    print(f'  Всего сэмплов: {len(y_all)}, ctx_dim={ctx_dim}')

    idx_tr, idx_val, idx_test = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.15,
        purge_bars=CFG.future_bars)
    y_tr  = y_all[idx_tr]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]
    print(f'  Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_test)}')
    print('  Распределение (train):'); class_distribution(y_tr)

    BATCH_SIZE = 256     # v4.1: снижено с 512 для стабильности

    # v4.1: без WRS
    tr_loader = _make_loader_v3(
        Subset(dataset, idx_tr.tolist()),
        BATCH_SIZE, shuffle=True, sampler=None, num_workers=0)
    val_loader = _make_loader_v3(
        Subset(dataset, idx_val.tolist()),
        BATCH_SIZE, shuffle=False, num_workers=0)
    te_ds = Subset(dataset, idx_test.tolist())

    n_ind = len(INDICATOR_COLS)
    print(f'  n_indicator_cols={n_ind}')
    print(f'  use_kronos={use_kronos} model={kronos_model}')

    model = MultiScaleHybridV4(
        ctx_dim=ctx_dim,
        n_indicator_cols=n_ind,
        future_bars=CFG.future_bars,
        use_hourly=use_hourly,
        use_kronos=use_kronos,
        kronos_model=kronos_model,
        kronos_seq_len=64,
        kronos_n_unfreeze=kronos_n_unfreeze,
        kronos_grad_checkpoint=True,
    ).to(device)

    if use_kronos:
        print('  [Kronos] Инициализация backbone...')
        model.init_kronos(device)
        print(f'  {_vram_info("После загрузки Kronos: ")}')

    # Re-init cls_head
    for name, p in model.cls_head.named_parameters():
        if p.ndim >= 2:
            nn.init.xavier_uniform_(p, gain=0.1)
        elif 'bias' in name:
            nn.init.zeros_(p)

    counts_dict = Counter(y_tr.tolist()); total_tr = len(y_tr)
    raw_w = [math.sqrt(total_tr / max(counts_dict.get(i, 1), 1))
             for i in range(3)]
    max_w = max(raw_w)
    cls_weights = torch.tensor(
        [w / max_w for w in raw_w], dtype=torch.float32).to(device)
    print(f'  Class weights: BUY={cls_weights[0]:.3f} '
          f'HOLD={cls_weights[1]:.3f} SELL={cls_weights[2]:.3f}')

    # v4.1: параметры из v3.17
    criterion = MultiTaskLossV3(
        cls_weight=cls_weights,
        gamma_per_class=(2.0, 1.0, 2.0),
        label_smoothing=0.05,
        future_bars=CFG.future_bars,
        huber_delta=0.5,
        direction_weight=0.60,
        reg_loss_weight=0.20,
        aux_loss_weight=0.05,
    ).to(device)

    print(f'  {_vram_info("После инициализации модели: ")}')

    if use_kronos and phase0_epochs > 0:
        _phase0_warmup(
            model, tr_loader, val_loader,
            criterion, device,
            n_epochs=phase0_epochs,
            ctx_dim=ctx_dim,
            use_hourly=use_hourly,
        )

    print('\n  [Phase-1] End-to-end — CAWR T_0=20, раздельные LR')

    max_lr = 1.5e-4          # v4.1: снижено с 3e-4
    _ACCUM    = 1
    _EPOCHS   = 60
    _PATIENCE = 12

    param_groups = model.get_param_groups(max_lr=max_lr)
    crit_params = list(criterion.parameters())
    if crit_params:
        param_groups.append({
            'params': crit_params,
            'lr': max_lr,
            'weight_decay': 1e-4,
            'name': 'criterion',
        })

    optimizer = AdamW(param_groups)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    for g in param_groups:
        print(f'  [LR] {g["name"]:20s}: {g["lr"]:.2e}')

    _run_epochs_v4(
        model, tr_loader, val_loader,
        optimizer, scheduler, criterion,
        device,
        n_epochs=_EPOCHS,
        patience_limit=_PATIENCE,
        save_path=model_path,
        phase_name='V4-E2E',
        ctx_dim=ctx_dim,
        use_hourly=use_hourly,
        accum_steps=_ACCUM,
    )

    print('\n' + '=' * 60 + '\nОценка на test set\n' + '=' * 60)
    from ml.multiscale_cnn_v3 import evaluate_multiscale_v3
    evaluate_multiscale_v3(
        model, te_ds, y_test, ctx_dim,
        use_hourly=use_hourly,
        save_json=model_path.replace('.pt', '_eval.json'),
    )

    return model


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    parser = argparse.ArgumentParser(description='Trainer v4.1 + Kronos')
    parser.add_argument('--model', default='ml/model_multiscale_v4.pt')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--no-hourly', action='store_true')
    parser.add_argument('--no-kronos', action='store_true')
    parser.add_argument('--kronos-model',
                        default='amazon/chronos-t5-tiny',
                        choices=[
                            'amazon/chronos-t5-tiny',
                            'amazon/chronos-t5-mini',
                            'amazon/chronos-t5-small',
                        ])
    parser.add_argument('--kronos-unfreeze', type=int, default=2)
    parser.add_argument('--phase0-epochs', type=int, default=3)
    args = parser.parse_args()

    train(
        model_path=args.model,
        use_hourly=not args.no_hourly,
        force_rebuild=args.rebuild,
        use_kronos=not args.no_kronos,
        kronos_model=args.kronos_model,
        kronos_n_unfreeze=args.kronos_unfreeze,
        phase0_epochs=args.phase0_epochs,
    )