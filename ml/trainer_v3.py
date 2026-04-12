# ml/trainer_v3.py
"""Trainer v3.13

Изменения v3.13:
- [2.3] WeightedRandomSampler заменяет TemporalBalancedSampler
  Веса = 1/class_count для каждого сэмпла train set → равномерный батч
- [2.4] Aux labels: 7-кортеж из датасета, aux_loss в criterion
  aux_loss_weight=0.05 — не мешает cls, добавляет регуляризацию фич
- [3.x] n_indicator_cols=37 (было 30)
- gamma_per_class=(1.5, 3.5, 1.5) — FLAT получает сильный фокус
- reg_loss_weight=0.30 (было 0.10) — регрессия полноправная задача
- patience_limit=12 (было 8)
- pct_start=0.15, div_factor=20
- [FOCAL] лог показывает aux_loss
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
from torch.utils.data import WeightedRandomSampler

_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.config import CFG, SCALES
from ml.dataset_v3 import (build_full_multiscale_dataset_v3,
                            temporal_split, class_distribution, INDICATOR_COLS)


# ══════════════════════════════════════════════════════════════════
# [2.3] WeightedRandomSampler
# ══════════════════════════════════════════════════════════════════

def make_weighted_sampler(y_tr: np.ndarray) -> WeightedRandomSampler:
    """1/class_count weight для каждого сэмпла. Нет утечки — работает
    только внутри уже изолированного train set."""
    counts = Counter(y_tr.tolist())
    weights = np.array([1.0 / counts[int(y)] for y in y_tr], dtype=np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )


# ══════════════════════════════════════════════════════════════════
# Pretrain MAE
# ══════════════════════════════════════════════════════════════════

def _pretrain_mae(model, tr_loader, device, n_epochs=5,
                  mask_ratio=0.3, lr=5e-4):
    from ml.multiscale_cnn_v3 import TRUNK_OUT
    W_short = min(SCALES)
    decoder = nn.Sequential(
        nn.Linear(TRUNK_OUT, TRUNK_OUT * 2), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(TRUNK_OUT * 2, 3 * 64)).to(device)

    for name, p in model.named_parameters():
        if any(k in name for k in ('cls_head', 'ohlc_head', 'aux_head')):
            p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad] + list(decoder.parameters())
    opt    = AdamW(params, lr=lr, weight_decay=1e-4)
    sched  = CosineAnnealingWarmRestarts(opt, T_0=max(1, n_epochs // 2))

    print(f"  [Pretrain] MAE: {n_epochs} эпох, mask={mask_ratio}, lr={lr}")
    model.train(); decoder.train()

    for epoch in range(1, n_epochs + 1):
        total_loss = 0.; n_batches = 0
        for batch in tr_loader:
            # v3.3: 7-кортеж
            x = batch[0][W_short].to(device)
            B, C, T = x.shape
            mask = (torch.rand(B, 1, T, device=device) < mask_ratio).expand(B, C, T)
            feat  = model.backbone(x.masked_fill(mask, 0.))
            recon = decoder(feat)
            loss  = F.mse_loss(
                recon[mask.reshape(B, -1)],
                x.reshape(B, -1)[mask.reshape(B, -1)])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            total_loss += loss.item(); n_batches += 1
        sched.step(epoch)
        print(f"  [Pretrain] E{epoch:2d}/{n_epochs} loss={total_loss/max(n_batches,1):.6f}")

    for p in model.parameters(): p.requires_grad_(True)
    del decoder
    print("  [Pretrain] Готово.")


# ══════════════════════════════════════════════════════════════════
# Инициализация cls_head
# ══════════════════════════════════════════════════════════════════

def _init_cls_head(model):
    for name, p in model.named_parameters():
        if 'cls_head' in name:
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(p)
            print(f"  [Init] {name}  shape={list(p.shape)}")


# ══════════════════════════════════════════════════════════════════
# Основной цикл обучения
# ══════════════════════════════════════════════════════════════════

def _run_epochs(model, tr_loader, val_loader, optimizer, scheduler,
                criterion, device, n_epochs, patience_limit,
                save_path, phase_name, ctx_dim, use_hourly,
                accum_steps=2, use_mixup=True, mixup_alpha=0.4):
    from ml.multiscale_cnn_v3 import mixup_data
    from sklearn.metrics import f1_score

    best_metric = 0.; patience = 0
    scaler = None   # AMP отключён

    DEAD_STREAK_LIMIT = 50
    zero_grad_streak  = 0

    def _reset_cls_head(opt):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'cls_head' in name:
                    nn.init.xavier_uniform_(p, gain=0.2) if p.dim() >= 2 \
                        else nn.init.zeros_(p)
        for group in opt.param_groups:
            if group.get('name') == 'cls_head':
                for p in group['params']: opt.state.pop(p, None)
        print("  [RECOVERY] cls_head reset")

    bilstm_params = [p for n, m in model.named_modules()
                     if isinstance(m, torch.nn.LSTM)
                     for p in m.parameters()]

    for epoch in range(1, n_epochs + 1):
        model.train(); criterion.train()
        total_cls = 0.; total_reg = 0.; total_aux = 0.; n_steps = 0
        optimizer.zero_grad()
        _last_lo = None; _last_cls_y = None

        for step, batch in enumerate(tr_loader, 1):
            # v3.3: (imgs, nums, cls_y, ohlc_y, ctx, hourly, aux_y)
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, hourly_data, aux_y = batch

            imgs     = {W: imgs_dict[W].to(device) for W in SCALES}
            cls_y    = cls_y.to(device)
            ohlc_y   = ohlc_y.to(device).clamp(-5., 5.)
            aux_y    = aux_y.to(device)
            ctx_t    = ctx.to(device) if ctx_dim > 0 else None
            hourly_t = (hourly_data.to(device)
                        if use_hourly and hourly_data is not None else None)
            nums     = ({W: num_dict[W].to(device) for W in SCALES}
                        if num_dict is not None else None)

            # ── DBG первый батч ────────────────────────────────────
            if epoch == 1 and step == 1:
                print("\n  [DBG] ── Первый батч v3.13 ──")
                for W, t in imgs.items():
                    print(f"  [DBG] imgs[{W}]: {t.shape} "
                          f"nan={torch.isnan(t).any().item()} "
                          f"min={t.min():.2f} max={t.max():.2f}")
                if nums:
                    for W, t in nums.items():
                        n_cols = t.shape[-1]
                        print(f"  [DBG] nums[{W}]: {t.shape} "
                              f"nan={torch.isnan(t).any().item()} "
                              f"n_cols={n_cols}")
                        if n_cols != 37:
                            print(f"  [WARN] Ожидали 37 cols, получили {n_cols}!")
                print(f"  [DBG] aux_y: {aux_y.shape} "
                      f"vol={aux_y[:,0].mean():.3f} skew={aux_y[:,1].mean():.3f}")
                print(f"  [DBG] cls_y unique: {cls_y.unique().tolist()}")
                with torch.no_grad():
                    lo_d, op_d, au_d = model(imgs, nums, ctx_t, hourly=hourly_t)
                    lo_d = lo_d.float().clamp(-15., 15.).nan_to_num(0.)
                print(f"  [DBG] logits: min={lo_d.min():.2f} max={lo_d.max():.2f}")
                print(f"  [DBG] aux_pred: {au_d.detach().cpu().numpy()[:3]}")
                print("  [DBG] ──────────────────────────────\n")

            # ── Mixup ──────────────────────────────────────────────
            if use_mixup and mixup_alpha > 0:
                (m_imgs, m_nums, cls_a, cls_b,
                 m_ohlc, m_ctx, lam,
                 m_hourly, m_aux) = mixup_data(
                    imgs, nums, cls_y, ohlc_y.float(),
                    ctx_t, mixup_alpha,
                    hourly=hourly_t, aux_y=aux_y)

                lo, op, au = model(m_imgs, m_nums, m_ctx, hourly=m_hourly)
                lo = lo.float().clamp(-15.,15.).nan_to_num(nan=0.,posinf=15.,neginf=-15.)
                op = op.float().nan_to_num(nan=0.,posinf=10.,neginf=-10.)

                la, lca, lra, laa = criterion(lo, cls_a, op, m_ohlc, au, m_aux)
                lb, lcb, lrb, lab = criterion(lo, cls_b, op, m_ohlc, au, m_aux)
                loss = lam*la + (1-lam)*lb
                lcls = lam*lca + (1-lam)*lcb
                lreg = lam*lra + (1-lam)*lrb
                laux = lam*laa + (1-lam)*lab
            else:
                lo, op, au = model(imgs, nums, ctx_t, hourly=hourly_t)
                lo = lo.float().clamp(-15.,15.).nan_to_num(nan=0.,posinf=15.,neginf=-15.)
                op = op.float().nan_to_num(nan=0.,posinf=10.,neginf=-10.)
                loss, lcls, lreg, laux = criterion(
                    lo, cls_y, op, ohlc_y.float(), au, aux_y)

            if not torch.isfinite(loss):
                print(f"  [WARN] e={epoch} s={step} loss=nan — пропуск")
                optimizer.zero_grad(); continue

            (loss / accum_steps).backward()
            _last_lo = lo.detach(); _last_cls_y = cls_y.detach()
            total_cls += lcls.item()
            total_reg += lreg.item()
            total_aux += laux.item()
            n_steps += 1

            if step % accum_steps == 0 or step == len(tr_loader):
                trainable   = [p for g in optimizer.param_groups
                               for p in g['params'] if p.requires_grad]
                cls_head_ps = list(model.cls_head.parameters())

                if bilstm_params:
                    nn.utils.clip_grad_norm_(bilstm_params, max_norm=1.0)
                nn.utils.clip_grad_norm_(cls_head_ps, max_norm=0.5)
                grad_norm = nn.utils.clip_grad_norm_(trainable, 2.0)

                if step % 100 == 0:
                    cls_gn = sum(
                        p.grad.norm().item() ** 2 for p in cls_head_ps
                        if p.grad is not None and torch.isfinite(p.grad).all()
                    ) ** 0.5
                    print(f"    [GRAD] e={epoch} s={step} "
                          f"gn={grad_norm:.3f} cls_gn={cls_gn:.4f}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                gn = float(grad_norm)
                if gn < 1e-7:
                    zero_grad_streak += 1
                    if zero_grad_streak >= DEAD_STREAK_LIMIT:
                        _reset_cls_head(optimizer)
                        zero_grad_streak = -100
                else:
                    zero_grad_streak = max(0, zero_grad_streak + 1) \
                        if zero_grad_streak < 0 else 0

            if step % 200 == 0 and _last_lo is not None:
                with torch.no_grad():
                    probs    = torch.softmax(_last_lo.float(), dim=1)
                    pred_cls = probs.argmax(dim=1)
                    pt   = [probs[_last_cls_y==c, c].mean().item()
                            if (_last_cls_y==c).any() else -1. for c in range(3)]
                    dist = [(pred_cls==c).float().mean().item() for c in range(3)]
                    tr_acc = (pred_cls == _last_cls_y).float().mean().item()
                    cls_lr = next((g['lr'] for g in optimizer.param_groups
                                   if g.get('name') == 'cls_head'), 0.)
                print(f"  [FOCAL] e={epoch} s={step} "
                      f"pt: UP={pt[0]:.3f} FL={pt[1]:.3f} DN={pt[2]:.3f} | "
                      f"dist: UP={dist[0]:.3f} FL={dist[1]:.3f} DN={dist[2]:.3f} | "
                      f"tr_acc={tr_acc:.3f} aux={total_aux/max(n_steps,1):.4f} "
                      f"cls_lr={cls_lr:.2e}")

        # ── Validation ────────────────────────────────────────────
        model.eval(); criterion.eval()
        val_preds = []; val_trues = []
        with torch.no_grad():
            for batch in val_loader:
                imgs_dict, num_dict, cls_y, ohlc_y, ctx, hourly_data, *_ = batch
                imgs     = {W: imgs_dict[W].to(device) for W in SCALES}
                cls_y    = cls_y.to(device)
                ctx_t    = ctx.to(device) if ctx_dim > 0 else None
                hourly_t = (hourly_data.to(device)
                            if use_hourly and hourly_data is not None else None)
                nums     = ({W: num_dict[W].to(device) for W in SCALES}
                            if num_dict is not None else None)
                preds    = model(imgs, nums, ctx_t, hourly=hourly_t)[0].argmax(1)
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(cls_y.cpu().numpy())

        vp = np.array(val_preds); vt = np.array(val_trues)
        val_acc  = (vp == vt).mean()
        macro_f1 = f1_score(vt, vp, average='macro', zero_division=0)
        f1pc     = f1_score(vt, vp, average=None, labels=[0,1,2], zero_division=0)
        buy_f1, hold_f1, sell_f1 = f1pc

        # Composite metric — штрафуем за низкий min-class F1
        val_metric = (0.25 * val_acc
                      + 0.50 * macro_f1
                      + 0.25 * min(buy_f1, hold_f1, sell_f1))

        print(f"  [{phase_name}] E{epoch:3d}/{n_epochs} "
              f"cls={total_cls/max(n_steps,1):.4f} "
              f"reg={total_reg/max(n_steps,1):.4f} "
              f"aux={total_aux/max(n_steps,1):.4f} | "
              f"val_acc={val_acc:.4f} mF1={macro_f1:.4f} | "
              f"UP={buy_f1:.3f} FL={hold_f1:.3f} DN={sell_f1:.3f} | "
              f"metric={val_metric:.4f}")

        if val_metric > best_metric:
            best_metric = val_metric; patience = 0
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ saved  metric={val_metric:.4f}")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  [{phase_name}] Early stop (patience={patience_limit})")
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))
    print(f"  [{phase_name}] Best metric={best_metric:.4f}")


# ══════════════════════════════════════════════════════════════════
# train()
# ══════════════════════════════════════════════════════════════════

def train(model_path='ml/model_multiscale_v3.pt', use_hourly=True,
          force_rebuild=False, do_pretrain=True, pretrain_epochs=5):
    from ml.dataset_v3 import (build_full_multiscale_dataset_v3,
                                temporal_split)
    from ml.multiscale_cnn_v3 import (MultiScaleHybridV3, MultiTaskLossV3,
                                       _make_loader_v3)
    from torch.utils.data import Subset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}  {p.total_memory/1024**2:.0f} MB")

    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=force_rebuild, use_hourly=use_hourly)
    print(f"  Сэмплов: {len(y_all)}, ctx_dim={ctx_dim}")

    idx_tr, idx_val, idx_test = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.15,
        purge_bars=CFG.future_bars)

    y_tr = y_all[idx_tr]; y_val = y_all[idx_val]; y_test = y_all[idx_test]
    print(f"  Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_test)}")
    print("  Дистрибуция (train):"); class_distribution(y_tr)

    # [2.3] WeightedRandomSampler для train
    wrs = make_weighted_sampler(y_tr)
    print(f"  [WRS] WeightedRandomSampler: {len(wrs)} сэмплов, replacement=True")

    tr_loader  = _make_loader_v3(Subset(dataset, idx_tr.tolist()),
                                  CFG.batch_size, sampler=wrs)
    val_loader = _make_loader_v3(Subset(dataset, idx_val.tolist()),
                                  CFG.batch_size, shuffle=False)
    te_ds = Subset(dataset, idx_test.tolist())

    # ── Модель ───────────────────────────────────────────────────
    model = MultiScaleHybridV3(
        ctx_dim=ctx_dim,
        n_indicator_cols=len(INDICATOR_COLS),  # было: 30
        future_bars=CFG.future_bars,
        use_hourly=use_hourly,
    ).to(device)
    _init_cls_head(model)

    # ── Веса классов ─────────────────────────────────────────────
    counts_dict = Counter(y_tr.tolist()); total_tr = len(y_tr)
    raw_w       = [total_tr / (counts_dict.get(i, 1) * 3) for i in range(3)]
    max_w       = max(raw_w)
    cls_weights = torch.tensor(
        [w / max_w * 3. for w in raw_w], dtype=torch.float32).to(device)
    print(f"  Class weights: UP={cls_weights[0]:.3f}  "
          f"FL={cls_weights[1]:.3f}  DN={cls_weights[2]:.3f}")

    # ── Criterion ─────────────────────────────────────────────────
    criterion = MultiTaskLossV3(
        cls_weight=cls_weights,
        gamma_per_class=(1.5, 3.5, 1.5),
        label_smoothing=0.05,        # ← было 0.12 — слишком сглаживает, UP F1=0.28
        future_bars=CFG.future_bars,
        direction_weight=0.40,       # ← было 0.2 — direction_acc=49.7% хуже монетки
        reg_loss_weight=0.30,        # ок, не трогаем
        aux_loss_weight=0.05,        # ок
    ).to(device)

    # ── Pretrain MAE ─────────────────────────────────────────────
    if do_pretrain:
        _pretrain_mae(model, tr_loader, device,
                      n_epochs=pretrain_epochs, mask_ratio=0.30, lr=3e-4)

    # ── Param groups с per-group weight_decay ─────────────────────
    print("\n  [Phase-1] Full training, layer-wise LR + per-group WD")
    max_lr = 3e-4

    backbone_ids  = {id(p) for p in model.backbone.parameters()}
    hourly_ids    = ({id(p) for p in model.hourly_enc.parameters()}
                     if use_hourly and hasattr(model, 'hourly_enc') else set())
    cls_head_ids  = {id(p) for p in model.cls_head.parameters()}
    aux_head_ids  = {id(p) for p in model.aux_head.parameters()}
    crit_params   = list(criterion.parameters())

    param_groups = [
        {'params': list(model.backbone.parameters()),
         'lr': max_lr * 0.15, 'weight_decay': 5e-4, 'name': 'backbone'},
        {'params': list(model.cls_head.parameters()),
         'lr': max_lr * 0.5,  'weight_decay': 1e-4, 'name': 'cls_head'},
        {'params': list(model.aux_head.parameters()),
         'lr': max_lr * 0.5,  'weight_decay': 1e-4, 'name': 'aux_head'},
        {'params': [p for p in model.parameters()
                    if p.requires_grad
                    and id(p) not in backbone_ids
                    and id(p) not in hourly_ids
                    and id(p) not in cls_head_ids
                    and id(p) not in aux_head_ids],
         'lr': max_lr, 'weight_decay': 5e-3, 'name': 'other'},
    ]
    if hourly_ids:
        param_groups.insert(3, {
            'params': list(model.hourly_enc.parameters()),
            'lr': max_lr * 0.1, 'weight_decay': 1e-3, 'name': 'hourly'})

    optimizer = AdamW(
        param_groups + [{'params': crit_params,
                         'lr': max_lr, 'weight_decay': 1e-4,
                         'name': 'criterion'}])

    _ACCUM = 2; _EPOCHS = 50
    n_steps = math.ceil(len(tr_loader) / _ACCUM) * _EPOCHS
    print(f"  [Sched] batches={len(tr_loader)} accum={_ACCUM} "
          f"epochs={_EPOCHS} → steps={n_steps}")
    print(f"  [LR] backbone={max_lr*0.15:.2e}  cls_head={max_lr*0.5:.2e}  "
          f"other={max_lr:.2e}")

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[g['lr'] for g in optimizer.param_groups],
        total_steps=n_steps,
        pct_start=0.15,         # 0.20 → 0.15
        div_factor=20,          # 10 → 20
        final_div_factor=500,
        anneal_strategy='cos')

    _run_epochs(
        model, tr_loader, val_loader, optimizer, scheduler,
        criterion, device,
        n_epochs=50, patience_limit=12,    # 8 → 12
        save_path=model_path,
        phase_name='F1-full',
        ctx_dim=ctx_dim, use_hourly=use_hourly,
        accum_steps=_ACCUM,
        use_mixup=True, mixup_alpha=0.4)

    print("\n" + "=" * 60 + "\nОценка на test set\n" + "=" * 60)
    from ml.multiscale_cnn_v3 import evaluate_multiscale_v3
    evaluate_multiscale_v3(
        model, te_ds, y_test, ctx_dim,
        use_hourly=use_hourly,
        save_json=model_path.replace('.pt', '_eval.json'))

    try:
        from ml.visualize_predictions import predict_and_plot
        predict_and_plot(model_path, te_ds, y_test, ctx_dim,
                         use_hourly=use_hourly, n_examples=8)
    except ImportError:
        print("  [WARN] visualize_predictions не найден")

    return model


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trainer v3.13 — MultiScale CNN для MOEX')
    parser.add_argument('--model',           default='ml/model_multiscale_v3.pt')
    parser.add_argument('--rebuild',         action='store_true',
                        help='Принудительный rebuild кэша (нужен при v3.3!)')
    parser.add_argument('--no-hourly',       action='store_true')
    parser.add_argument('--no-pretrain',     action='store_true')
    parser.add_argument('--pretrain-epochs', type=int, default=5)
    args = parser.parse_args()
    train(
        model_path=args.model,
        use_hourly=not args.no_hourly,
        force_rebuild=args.rebuild,
        do_pretrain=not args.no_pretrain,
        pretrain_epochs=args.pretrain_epochs,
    )