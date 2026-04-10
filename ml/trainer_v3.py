# ml/trainer_v3.py
"""Trainer v3.13 — Улучшение обобщения без утечки будущего.

Изменения v3.13:

FIX 1: Embargo × 3 (purge_bars = future_bars * 3).
  Было: purge_bars = future_bars (5 баров ≈ 1 торговый день).
  Стало: purge_bars = future_bars * 3 (15 баров ≈ 3 дня).
  Причина: на MOEX тикеры одного сектора (нефть, банки) движутся синхронно.
  Без embargo модель видела похожие паттерны в train и val в разных тикерах
  → эффективная утечка через корреляцию.

FIX 2: TemporalBalancedSampler.
  Каждый батч = равные доли из каждого тикера.
  Было: random shuffle → один батч мог содержать 80% одного тикера/периода.
  Стало: модель не может «запомнить» паттерны конкретного эмитента.

FIX 3: OHLCVAugment в цикле обучения.
  amplitude_jitter + volatility_swap + num_jitter — только для истории.
  Метки (cls_y, ohlc_y) не трогаются — они из будущего.
  Применяется с torch.no_grad() чтобы не строить лишний граф.

FIX 4: temporal_cutmix вместо mixup_data.
  Вырезает сегмент из исторического окна одного сэмпла и вставляет в другой.
  Форма вставки сохраняет временной порядок — нет «телепортации» бара из будущего.

FIX 5: Понижен mixup_alpha 0.4 → 0.25.
  Высокий alpha смешивал метки слишком агрессивно → модель видела
  «SELL на 60% + BUY на 40%» как лёгкий пример; снижение усиливает сигнал.

FIX 6: gamma_per_class (1.0,1.0,1.0) → (2.0,1.0,1.5).
  UP-класс получает больший фокус (recall UP был 0.23).
  DOWN немного усиливается; HOLD оставляем без изменений.
"""
import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'
import sys, argparse, math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.config import CFG, SCALES
from ml.dataset_v3 import class_distribution


def _pretrain_mae(model, tr_loader, device, n_epochs=5, mask_ratio=0.3, lr=5e-4):
    from ml.multiscale_cnn_v3 import TRUNK_OUT
    W_short  = min(SCALES)
    decoder  = nn.Sequential(
        nn.Linear(TRUNK_OUT, TRUNK_OUT*2), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(TRUNK_OUT*2, 3*64)).to(device)
    for name, p in model.named_parameters():
        if any(k in name for k in ('cls_head', 'ohlc_head')): p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad] + list(decoder.parameters())
    opt    = AdamW(params, lr=lr, weight_decay=1e-4)
    sched  = CosineAnnealingWarmRestarts(opt, T_0=max(1, n_epochs//2))
    print(f"  [Pretrain] MAE: {n_epochs} эпох, mask_ratio={mask_ratio}, lr={lr}")
    model.train(); decoder.train()
    for epoch in range(1, n_epochs+1):
        total_loss=0.0; n_batches=0
        for batch in tr_loader:
            x     = batch[0][W_short].to(device)
            B,C,T = x.shape
            mask  = (torch.rand(B,1,T,device=device)<mask_ratio).expand(B,C,T)
            feat  = model.backbone(x.masked_fill(mask, 0.0))
            recon = decoder(feat)
            loss  = F.mse_loss(recon[mask.reshape(B,-1)], x.reshape(B,-1)[mask.reshape(B,-1)])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            total_loss += loss.item(); n_batches += 1
        sched.step(epoch)
        print(f"  [Pretrain] Epoch {epoch:2d}/{n_epochs}  loss={total_loss/max(n_batches,1):.6f}")
    for p in model.parameters(): p.requires_grad_(True)
    del decoder
    print("  [Pretrain] Готово. Backbone разморожен.")


def _init_cls_head(model):
    for name, p in model.named_parameters():
        if 'cls_head' in name:
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(p)
            print(f"  [Init] cls_head init: {name}  shape={list(p.shape)}")


def _run_epochs(model, tr_loader, val_loader, optimizer, scheduler,
                criterion, device, n_epochs, patience_limit,
                save_path, phase_name, ctx_dim, use_hourly,
                accum_steps=2, use_mixup=True, mixup_alpha=0.25):
    from ml.multiscale_cnn_v3 import temporal_cutmix, OHLCVAugment
    from sklearn.metrics import f1_score

    best_metric = 0.0; patience = 0
    scaler      = None  # AMP отключён

    DEAD_STREAK_LIMIT = 50
    zero_grad_streak  = 0

    # FIX v3.13: аугментатор истории — создаём один раз, используем в каждом батче
    # Важно: augmenter меняет только imgs/nums (история), не cls_y/ohlc_y (будущее)
    augmenter = OHLCVAugment(
        amplitude_sigma=0.015,
        vol_range=(0.75, 1.25),
        num_jitter_sigma=0.008,
        p=0.45,
    )

    def _reset_cls_head_full(opt):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'cls_head' in name:
                    nn.init.xavier_uniform_(p, gain=0.2) if 'weight' in name else nn.init.zeros_(p)
        for group in opt.param_groups:
            if group.get('name') == 'cls_head':
                for p in group['params']:
                    opt.state.pop(p, None)
        print(f"  [RECOVERY] Dead gradient → reset cls_head + optimizer state")

    bilstm_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LSTM):
            bilstm_params.extend(list(module.parameters()))

    for epoch in range(1, n_epochs+1):
        model.train(); criterion.train()
        total_cls=0.0; total_reg=0.0; n_steps=0
        optimizer.zero_grad()
        _last_lo=None; _last_cls_y=None

        for step, batch in enumerate(tr_loader, 1):
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
            hourly_data = hourly_opt[0] if hourly_opt else None
            imgs     = {W: imgs_dict[W].to(device) for W in SCALES}
            cls_y    = cls_y.to(device)
            ohlc_y   = ohlc_y.to(device).clamp(-5.0, 5.0)
            ctx_t    = ctx.to(device) if ctx_dim > 0 else None
            hourly_t = (hourly_data.to(device) if (use_hourly and hourly_data is not None) else None)
            nums     = ({W: num_dict[W].to(device) for W in SCALES} if num_dict is not None else None)

            # ── DBG первый батч ───────────────────────────────────────────
            if epoch==1 and step==1:
                print("\n  [DBG] ── Диагностика первого батча (v3.13) ──")
                for W,t in imgs.items():
                    print(f"  [DBG] imgs[{W}]: shape={t.shape} nan={torch.isnan(t).any().item()} "
                          f"min={t.min():.3f} max={t.max():.3f}")
                if nums:
                    for W,t in nums.items():
                        print(f"  [DBG] nums[{W}]: nan={torch.isnan(t).any().item()}")
                print(f"  [DBG] cls_y:  unique={cls_y.unique().tolist()}")
                print(f"  [DBG] ohlc_y: nan={torch.isnan(ohlc_y).any().item()} "
                      f"min={ohlc_y.min():.3f} max={ohlc_y.max():.3f}")
                if ctx_t is not None: print(f"  [DBG] ctx:    shape={ctx_t.shape}")
                if hourly_t is not None: print(f"  [DBG] hourly: shape={hourly_t.shape}")
                with torch.no_grad():
                    lo_d, op_d = model(imgs, nums, ctx_t, hourly=hourly_t)
                    lo_d = lo_d.float().clamp(-15., 15.).nan_to_num(0.)
                print(f"  [DBG] logits:    nan={torch.isnan(lo_d).any().item()} "
                      f"min={lo_d.min():.3f} max={lo_d.max():.3f}")
                print(f"  [DBG] AMP: ОТКЛЮЧЁН (float32)")
                print(f"  [DBG] Augmenter: p={augmenter.p}  "
                      f"vol_range={augmenter.vol_range}")
                print("  [DBG] ──────────────────────────────────────────\n")

            # ── FIX v3.13: аугментация истории (torch.no_grad — не строим граф) ──
            # ГАРАНТИЯ: cls_y и ohlc_y не трогаются — они из будущего
            with torch.no_grad():
                imgs, nums = augmenter(imgs, nums)

            effective_alpha = mixup_alpha if use_mixup else 0.0

            if use_mixup and effective_alpha > 0:
                # FIX v3.13: temporal_cutmix вместо mixup_data
                # Срез [cut_start:cut_start+cut_len] — только история, не будущее
                m_imgs,m_nums,cls_a,cls_b,m_ohlc,m_ctx,lam,m_hourly = temporal_cutmix(
                    imgs,nums,cls_y,ohlc_y.float(),ctx_t,effective_alpha,hourly=hourly_t)
                lo,op = model(m_imgs,m_nums,m_ctx,hourly=m_hourly)
                lo = lo.float().clamp(-15.,15.).nan_to_num(nan=0.,posinf=15.,neginf=-15.)
                op = op.float().nan_to_num(nan=0.,posinf=10.,neginf=-10.)
                la,lca,lra = criterion(lo,cls_a,op,m_ohlc)
                lb,lcb,lrb = criterion(lo,cls_b,op,m_ohlc)
                loss=lam*la+(1-lam)*lb; lcls=lam*lca+(1-lam)*lcb; lreg=lam*lra+(1-lam)*lrb
            else:
                lo,op = model(imgs,nums,ctx_t,hourly=hourly_t)
                lo = lo.float().clamp(-15.,15.).nan_to_num(nan=0.,posinf=15.,neginf=-15.)
                op = op.float().nan_to_num(nan=0.,posinf=10.,neginf=-10.)
                loss,lcls,lreg = criterion(lo,cls_y,op,ohlc_y.float())

            if not torch.isfinite(loss):
                print(f"  [WARN] e={epoch} s={step} loss=nan — пропускаем батч")
                optimizer.zero_grad(); continue

            (loss/accum_steps).backward()
            _last_lo=lo.detach(); _last_cls_y=cls_y.detach()
            total_cls += lcls.item(); total_reg += lreg.item(); n_steps += 1

            if step % accum_steps == 0 or step == len(tr_loader):
                trainable   = [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
                cls_head_ps = list(model.cls_head.parameters())

                if bilstm_params:
                    nn.utils.clip_grad_norm_(bilstm_params, max_norm=1.0)
                nn.utils.clip_grad_norm_(cls_head_ps, max_norm=0.2)
                grad_norm = nn.utils.clip_grad_norm_(trainable, 2.0)

                if step % 100 == 0:
                    cls_gn = sum(p.grad.norm().item()**2 for p in cls_head_ps
                                 if p.grad is not None and torch.isfinite(p.grad).all())**0.5
                    print(f"    [GRAD] e={epoch} s={step} grad_norm={grad_norm:.3f} "
                          f"cls_head_gn={cls_gn:.4f}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                gn = float(grad_norm)
                if gn < 1e-7:
                    zero_grad_streak += 1
                    if zero_grad_streak >= DEAD_STREAK_LIMIT:
                        _reset_cls_head_full(optimizer)
                        zero_grad_streak = -100
                else:
                    if zero_grad_streak < 0: zero_grad_streak += 1
                    else: zero_grad_streak = 0

            if step % 200 == 0 and _last_lo is not None:
                with torch.no_grad():
                    _probs    = torch.softmax(_last_lo.float(), dim=1)
                    _pred_cls = _probs.argmax(dim=1)
                    _pt   = [_probs[_last_cls_y==c,c].mean().item() if (_last_cls_y==c).any() else -1. for c in range(3)]
                    _dist = [(_pred_cls==c).float().mean().item() for c in range(3)]
                    _tr_acc = (_pred_cls==_last_cls_y).float().mean().item()
                    cls_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name')=='cls_head'), 0.0)
                print(f"  [FOCAL] e={epoch} s={step} "
                      f"pt: BUY={_pt[0]:.3f} HOLD={_pt[1]:.3f} SELL={_pt[2]:.3f} | "
                      f"pred_dist: BUY={_dist[0]:.3f} HOLD={_dist[1]:.3f} SELL={_dist[2]:.3f} | "
                      f"tr_acc={_tr_acc:.3f} | cls_lr={cls_lr:.2e}")

        model.eval(); criterion.eval(); val_preds=[]; val_trues=[]
        with torch.no_grad():
            for batch in val_loader:
                imgs_dict,num_dict,cls_y,ohlc_y,ctx,*hourly_opt = batch
                hourly_data = hourly_opt[0] if hourly_opt else None
                imgs     = {W: imgs_dict[W].to(device) for W in SCALES}
                cls_y    = cls_y.to(device)
                ctx_t    = ctx.to(device) if ctx_dim > 0 else None
                hourly_t = (hourly_data.to(device) if (use_hourly and hourly_data is not None) else None)
                nums     = ({W: num_dict[W].to(device) for W in SCALES} if num_dict is not None else None)
                preds    = model(imgs,nums,ctx_t,hourly=hourly_t)[0].argmax(1)
                val_preds.extend(preds.cpu().numpy()); val_trues.extend(cls_y.cpu().numpy())

        vp=np.array(val_preds); vt=np.array(val_trues)
        val_acc  = (vp==vt).mean()
        macro_f1 = f1_score(vt, vp, average='macro', zero_division=0)
        f1pc     = f1_score(vt, vp, average=None, labels=[0,1,2], zero_division=0)
        buy_f1, hold_f1, sell_f1 = f1pc[0], f1pc[1], f1pc[2]
        val_metric = 0.3*val_acc + 0.5*macro_f1 + 0.2*min(buy_f1,hold_f1,sell_f1)
        reg_w = getattr(criterion, 'reg_loss_weight', 0.4)
        print(f"  [{phase_name}] E{epoch:3d}/{n_epochs} "
              f"cls={total_cls/max(n_steps,1):.4f} reg={total_reg/max(n_steps,1):.5f} | "
              f"val_acc={val_acc:.4f} mF1={macro_f1:.4f} | "
              f"buy={buy_f1:.3f} hold={hold_f1:.3f} sell={sell_f1:.3f} | "
              f"reg_w={reg_w:.2f} | metric={val_metric:.4f}")
        if val_metric > best_metric:
            best_metric=val_metric; patience=0
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ saved  metric={val_metric:.4f}")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  [{phase_name}] Early stop (patience={patience_limit})"); break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"  [{phase_name}] Best metric={best_metric:.4f}")


def train(model_path='ml/model_multiscale_v3.pt', use_hourly=True,
          force_rebuild=False, do_pretrain=True, pretrain_epochs=5):
    from ml.dataset_v3 import build_full_multiscale_dataset_v3, temporal_split
    from ml.multiscale_cnn_v3 import (MultiScaleHybridV3, MultiTaskLossV3,
                                       _make_loader_v3, TemporalBalancedSampler)
    from torch.utils.data import Subset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type=='cuda':
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}  {getattr(p,'total_memory',0)/1024**2:.0f} MB VRAM")

    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=force_rebuild, use_hourly=use_hourly)
    print(f"  Всего сэмплов: {len(y_all)}, ctx_dim={ctx_dim}")

    # FIX v3.13: embargo × 3 — защита от утечки через секторальную корреляцию MOEX
    # future_bars баров ≈ 1 день → × 3 ≈ 3 торговых дня зазор между train/val
    embargo = CFG.future_bars * 3
    print(f"  [Embargo] purge_bars={embargo} (= future_bars×3 = {CFG.future_bars}×3)")
    idx_tr, idx_val, idx_test = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.15, purge_bars=embargo)
    y_tr=y_all[idx_tr]; y_val=y_all[idx_val]; y_test=y_all[idx_test]
    print(f"  Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_test)}")
    print("  Распределение (все):"); class_distribution(y_all)
    print("  Распределение (train):"); class_distribution(y_tr)

    # FIX v3.13: TemporalBalancedSampler
    # Строим маппинг ticker → LOCAL индексы в tr_subset
    # LOCAL = позиция в Subset (0..len(idx_tr)-1), не глобальный индекс датасета
    tr_loader = None
    try:
        idx_tr_list  = idx_tr.tolist()
        idx_to_local = {gi: li for li, gi in enumerate(idx_tr_list)}
        ticker_to_local_idx = {}
        offset = 0
        for t_idx, entry in enumerate(ticker_lengths):
            # ticker_lengths может быть list[int] или list[(name, int)]
            n = entry if isinstance(entry, int) else int(entry[1])
            name = f'ticker_{t_idx}' if isinstance(entry, int) else str(entry[0])
            local_inds = [idx_to_local[gi]
                          for gi in range(offset, offset + n)
                          if gi in idx_to_local]
            min_samples = max(4, CFG.batch_size // 8)
            if len(local_inds) >= min_samples:
                ticker_to_local_idx[name] = local_inds
            offset += n

        if len(ticker_to_local_idx) > 1:
            sampler = TemporalBalancedSampler(
                ticker_to_local_idx, CFG.batch_size, drop_last=True)
            tr_loader = _make_loader_v3(
                Subset(dataset, idx_tr_list), CFG.batch_size,
                shuffle=False, sampler=sampler)
            print(f"  [Sampler] TemporalBalancedSampler: "
                  f"{len(ticker_to_local_idx)} тикеров, "
                  f"per_group≈{sampler.per_group}")
        else:
            print("  [Sampler] Мало тикеров → стандартный shuffle")
    except Exception as e:
        print(f"  [Sampler] Ошибка при построении: {e} → стандартный shuffle")

    if tr_loader is None:
        tr_loader = _make_loader_v3(
            Subset(dataset, idx_tr.tolist()), CFG.batch_size, shuffle=True)

    val_loader = _make_loader_v3(
        Subset(dataset, idx_val.tolist()), CFG.batch_size, shuffle=False)
    te_ds = Subset(dataset, idx_test.tolist())

    model = MultiScaleHybridV3(ctx_dim=ctx_dim, n_indicator_cols=30,
                                future_bars=CFG.future_bars, use_hourly=use_hourly).to(device)
    _init_cls_head(model)

    counts_dict = Counter(y_tr.tolist()); total_tr = len(y_tr)
    raw_w       = [total_tr/(counts_dict.get(i,1)*3) for i in range(3)]
    max_w       = max(raw_w)
    cls_weights = torch.tensor([w/max_w*3. for w in raw_w], dtype=torch.float32).to(device)
    print(f"  Class weights: BUY={cls_weights[0]:.3f}  HOLD={cls_weights[1]:.3f}  SELL={cls_weights[2]:.3f}")

    # FIX v3.13: gamma_per_class повышен для UP (recall был 0.23) и немного для DOWN
    criterion = MultiTaskLossV3(
        cls_weight=cls_weights,
        gamma_per_class=(2.0, 1.0, 1.5),   # было (1.0,1.0,1.0) в v3.12
        label_smoothing=0.12,               # немного снижено: 0.15→0.12
        huber_delta=0.5,
        direction_weight=0.3,
        reg_loss_weight=0.10).to(device)

    if do_pretrain:
        _pretrain_mae(model, tr_loader, device, n_epochs=pretrain_epochs,
                      mask_ratio=0.30, lr=3e-4)

    print("\n  [Phase-1] Full training, layer-wise LR decay")
    max_lr = 3e-4

    backbone_ids = {id(p) for p in model.backbone.parameters()}
    hourly_ids   = ({id(p) for p in model.hourly_enc.parameters()}
                    if use_hourly and hasattr(model,'hourly_enc') else set())
    cls_head_ids = {id(p) for p in model.cls_head.parameters()}
    crit_params  = list(criterion.parameters())

    param_groups = [
        {'params': list(model.backbone.parameters()),
         'lr': max_lr*0.15, 'weight_decay': 5e-4, 'name': 'backbone'},
        {'params': list(model.cls_head.parameters()),
         'lr': max_lr*0.5,  'weight_decay': 1e-4, 'name': 'cls_head'},
        {'params': [p for p in model.parameters() if p.requires_grad
                    and id(p) not in backbone_ids and id(p) not in hourly_ids
                    and id(p) not in cls_head_ids],
         'lr': max_lr, 'weight_decay': 5e-3, 'name': 'other'},
    ]
    if hourly_ids:
        param_groups.insert(2, {
            'params': list(model.hourly_enc.parameters()),
            'lr': max_lr*0.1, 'weight_decay': 5e-4, 'name': 'hourly'})

    optimizer = AdamW(
        param_groups + [{'params': crit_params, 'lr': max_lr,
                         'weight_decay': 1e-4, 'name': 'criterion'}])

    _ACCUM=2; _EPOCHS=50
    n_steps = math.ceil(len(tr_loader)/_ACCUM)*_EPOCHS
    print(f"  [Sched] batches={len(tr_loader)} accum={_ACCUM} epochs={_EPOCHS} → total_steps={n_steps}")
    print(f"  [LR] backbone={max_lr*0.15:.2e}  cls_head={max_lr*0.5:.2e}  "
          f"other={max_lr:.2e}  hourly={max_lr*0.1:.2e}")

    scheduler = OneCycleLR(optimizer, max_lr=[g['lr'] for g in optimizer.param_groups],
                            total_steps=n_steps, pct_start=0.15,
                            div_factor=20, final_div_factor=500, anneal_strategy='cos')

    _run_epochs(model, tr_loader, val_loader, optimizer, scheduler, criterion, device,
                n_epochs=50, patience_limit=10, save_path=model_path, phase_name='F1-full',
                ctx_dim=ctx_dim, use_hourly=use_hourly, accum_steps=2,
                use_mixup=True, mixup_alpha=0.25)  # FIX v3.13: alpha 0.4→0.25

    print("\n" + "="*60 + "\nОценка на test set\n" + "="*60)
    from ml.multiscale_cnn_v3 import evaluate_multiscale_v3
    evaluate_multiscale_v3(model, te_ds, y_test, ctx_dim, use_hourly=use_hourly,
                            save_json=model_path.replace('.pt', '_eval.json'))
    try:
        from ml.visualize_predictions import predict_and_plot
        predict_and_plot(model_path, te_ds, y_test, ctx_dim, use_hourly=use_hourly, n_examples=8)
    except ImportError:
        print("  [WARN] visualize_predictions не найден — пропускаем")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer v3.13 — MultiScale CNN для MOEX')
    parser.add_argument('--model',           default='ml/model_multiscale_v3.pt')
    parser.add_argument('--rebuild',         action='store_true')
    parser.add_argument('--no-hourly',       action='store_true')
    parser.add_argument('--no-pretrain',     action='store_true')
    parser.add_argument('--pretrain-epochs', type=int, default=5)
    args = parser.parse_args()
    train(model_path=args.model, use_hourly=not args.no_hourly,
          force_rebuild=args.rebuild, do_pretrain=not args.no_pretrain,
          pretrain_epochs=args.pretrain_epochs)
