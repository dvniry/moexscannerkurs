"""Точка входа v3: python -m ml.trainer_v3 [--rebuild] [--no-hourly] [--model PATH]"""
import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from ml.config import CFG, SCALES
from ml.dataset_v3 import class_distribution


# ══════════════════════════════════════════════════════════════════
# Pretrain: MAE-style masked reconstruction на backbone
# ══════════════════════════════════════════════════════════════════

def _pretrain_mae(model, tr_loader, device, n_epochs=5, mask_ratio=0.3, lr=5e-5):
    from ml.multiscale_cnn_v3 import TRUNK_OUT
    in_dim = 3 * 64
    decoder = nn.Linear(TRUNK_OUT, in_dim).to(device)
    params = list(model.backbone.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    print(f"  [Pretrain] {n_epochs} эпох, mask_ratio={mask_ratio}, lr={lr}")
    model.train(); decoder.train()
    W_short = min(SCALES)
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0; n_batches = 0
        for batch in tr_loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
            x = imgs_dict[W_short].to(device)
            B, C, T = x.shape
            mask = torch.rand(B, 1, T, device=device) < mask_ratio
            x_masked = x.masked_fill(mask.expand_as(x), 0.0)
            feat = model.backbone(x_masked)
            recon = decoder(feat)
            target = x.reshape(B, -1)
            mask_flat = mask.expand(B, C, T).reshape(B, -1)
            loss = F.mse_loss(recon[mask_flat], target[mask_flat])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        print(f"  [Pretrain] Epoch {epoch:2d}/{n_epochs} loss={total_loss/max(n_batches,1):.6f}")
    print("  [Pretrain] Готово. Backbone разморожен.")
    del decoder


# ══════════════════════════════════════════════════════════════════
# Одна фаза обучения
# ══════════════════════════════════════════════════════════════════

def _run_epochs(
    model, tr_loader, val_loader, optimizer, scheduler,
    criterion, device, n_epochs, patience_limit,
    save_path, phase_name, ctx_dim, use_hourly,
    accum_steps=1, use_mixup=True, mixup_alpha=0.2, 
):
    from ml.multiscale_cnn_v3 import mixup_data
    from sklearn.metrics import f1_score

    best_metric = 0.0; patience = 0
    n_val = sum(len(b[2]) for b in val_loader)

    for epoch in range(1, n_epochs + 1):
        model.train(); criterion.train()
        total_cls = 0.0; total_reg = 0.0; n_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tr_loader, 1):
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
            hourly_data = hourly_opt[0] if hourly_opt else None

            imgs     = {W: imgs_dict[W].to(device) for W in SCALES}
            cls_y    = cls_y.to(device)
            ohlc_y   = ohlc_y.to(device)
            ctx_t    = ctx.to(device) if ctx_dim > 0 else None
            hourly_t = hourly_data.to(device) if (use_hourly and hourly_data is not None) else None
            nums     = ({W: num_dict[W].to(device) for W in SCALES}
                        if num_dict is not None else None)

            if use_mixup and mixup_alpha > 0:
                m_imgs, m_nums, cls_a, cls_b, m_ohlc, m_ctx, lam, m_hourly = mixup_data(
                    imgs, nums, cls_y, ohlc_y, ctx_t, mixup_alpha, hourly=hourly_t)
                cls_logits, ohlc_pred = model(m_imgs, m_nums, m_ctx, hourly=m_hourly)
                loss_a, lcls_a, lreg = criterion(cls_logits, cls_a, ohlc_pred, m_ohlc)
                loss_b, lcls_b, _    = criterion(cls_logits, cls_b, ohlc_pred, m_ohlc)
                loss = lam * loss_a + (1 - lam) * loss_b
                lcls = lam * lcls_a + (1 - lam) * lcls_b
            else:
                cls_logits, ohlc_pred = model(imgs, nums, ctx_t, hourly=hourly_t)
                loss, lcls, lreg = criterion(cls_logits, cls_y, ohlc_pred, ohlc_y)

            (loss / accum_steps).backward()
            total_cls += lcls.item(); total_reg += lreg.item(); n_steps += 1

            if step % accum_steps == 0 or step == len(tr_loader):
                trainable = [p for g in optimizer.param_groups for p in g['params']
                             if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

        # ── Val ──────────────────────────────────────────────────
        model.eval(); criterion.eval()
        val_correct = 0; val_preds = []; val_trues = []
        with torch.no_grad():
            for batch in val_loader:
                imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
                hourly_data = hourly_opt[0] if hourly_opt else None
                imgs     = {W: imgs_dict[W].to(device) for W in SCALES}
                cls_y    = cls_y.to(device)
                ctx_t    = ctx.to(device) if ctx_dim > 0 else None
                hourly_t = hourly_data.to(device) if (use_hourly and hourly_data is not None) else None
                nums     = ({W: num_dict[W].to(device) for W in SCALES}
                            if num_dict is not None else None)
                cls_logits, ohlc_pred = model(imgs, nums, ctx_t, hourly=hourly_t)
                preds = cls_logits.argmax(1)
                val_correct += (preds == cls_y).sum().item()
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(cls_y.cpu().numpy())

        val_acc      = val_correct / max(n_val, 1)
        val_preds_np = np.array(val_preds)
        val_trues_np = np.array(val_trues)
        macro_f1     = f1_score(val_trues_np, val_preds_np, average='macro', zero_division=0)
        # FIX: мультиклассовый f1 вместо binary
        f1_per_cls   = f1_score(val_trues_np, val_preds_np, average=None,
                                labels=[0, 1, 2], zero_division=0)
        buy_f1, hold_f1, sell_f1 = f1_per_cls[0], f1_per_cls[1], f1_per_cls[2]
        min_cls      = min(buy_f1, hold_f1, sell_f1)
        val_metric   = 0.3 * val_acc + 0.5 * macro_f1 + 0.2 * min_cls

        sigma_cls = torch.exp(criterion.log_sigma_cls).item()
        sigma_reg = torch.exp(criterion.log_sigma_reg).item()
        print(f"  [{phase_name}] Epoch {epoch:3d}/{n_epochs} "
              f"cls={total_cls/n_steps:.4f} reg={total_reg/n_steps:.6f} | "
              f"val_acc={val_acc:.4f} macro_f1={macro_f1:.4f} | "
              f"up={buy_f1:.3f} flat={hold_f1:.3f} down={sell_f1:.3f} | "
              f"σ_cls={sigma_cls:.3f} σ_reg={sigma_reg:.3f} | metric={val_metric:.4f}")

        if val_metric > best_metric:
            best_metric = val_metric; patience = 0
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ saved metric={val_metric:.4f}")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  [{phase_name}] Early stop (patience={patience_limit})")
                break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"  [{phase_name}] Best metric={best_metric:.4f}")


# ══════════════════════════════════════════════════════════════════
# Точка входа тренировки
# ══════════════════════════════════════════════════════════════════

def train(model_path: str = 'ml/model_multiscale_v3.pt',
          use_hourly: bool = True,
          force_rebuild: bool = False):
    from ml.dataset_v3 import build_full_multiscale_dataset_v3, temporal_split
    from ml.multiscale_cnn_v3 import MultiScaleHybridV3, MultiTaskLossV3, _make_loader_v3
    from torch.utils.data import Subset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        mem = getattr(p, 'total_mem', None) or getattr(p, 'total_memory', 0)
        print(f"  GPU: {p.name}  {mem/1024**2:.0f} MB VRAM")

    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=force_rebuild, use_hourly=use_hourly)
    print(f"  Всего сэмплов: {len(y_all)}, ctx_dim={ctx_dim}")

    idx_tr, idx_val, idx_test = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.15, purge_bars=CFG.future_bars)
    y_tr   = y_all[idx_tr];  y_val = y_all[idx_val];  y_test = y_all[idx_test]
    print(f"  Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_test)}")
    class_distribution(y_all)

    tr_ds  = Subset(dataset, idx_tr.tolist())
    val_ds = Subset(dataset, idx_val.tolist())
    te_ds  = Subset(dataset, idx_test.tolist())
    tr_loader  = _make_loader_v3(tr_ds,  CFG.batch_size, shuffle=True)
    val_loader = _make_loader_v3(val_ds, CFG.batch_size, shuffle=False)

    model = MultiScaleHybridV3(
        ctx_dim=ctx_dim, n_indicator_cols=30,
        future_bars=CFG.future_bars, use_hourly=use_hourly,
    ).to(device)

    counts = np.bincount(y_tr, minlength=3).astype(float)
    weights = torch.ones(3, dtype=torch.float).to(device)
    print(f"  UP={weights[0]:.2f}  FLAT={weights[1]:.2f}  DOWN={weights[2]:.2f}")

    criterion = MultiTaskLossV3(
        cls_weight=weights,
        gamma_per_class=(2.0, 3.0, 2.0),
        label_smoothing=0.15,
        huber_delta=0.02,
        direction_weight=0.1,
    ).to(device)

    # Фаза 0: MAE pretrain (backbone заморожен)
    for p in model.backbone.parameters(): p.requires_grad = False
    _pretrain_mae(model, tr_loader, device, n_epochs=5, mask_ratio=0.3, lr=5e-5)
    for p in model.backbone.parameters(): p.requires_grad = True

    # Фаза 1: Сразу полная разморозка, backbone с очень малым LR
    # (frozen фаза убрана — при замороженном backbone головы не могут разделить классы)
    print("\n  [Finetune] Full unlock от старта")
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    hourly_ids   = {id(p) for p in model.hourly_enc.parameters()} if use_hourly else set()
    crit_params  = list(criterion.parameters())
    max_lr_ft = 5e-5

    param_groups = [
        {'params': list(model.backbone.parameters()),    'lr': max_lr_ft * 0.02},
        {'params': [p for p in model.parameters()
                    if p.requires_grad
                    and id(p) not in backbone_ids
                    and id(p) not in hourly_ids],         'lr': max_lr_ft},
    ]
    if use_hourly and hourly_ids:
        param_groups.insert(1, {
            'params': list(model.hourly_enc.parameters()),
            'lr': max_lr_ft * 0.1,
        })

    opt1 = torch.optim.AdamW(
        param_groups + [{'params': crit_params, 'lr': max_lr_ft}],
        weight_decay=1e-3)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(
        opt1, max_lr=[g['lr'] for g in opt1.param_groups],
        total_steps=math.ceil(len(tr_loader) * 40),
        pct_start=0.3, div_factor=5, final_div_factor=200)
    _run_epochs(model, tr_loader, val_loader, opt1, sched1, criterion, device,
                n_epochs=40, patience_limit=10, save_path=model_path,
                phase_name='F1-full', ctx_dim=ctx_dim, use_hourly=use_hourly,
                use_mixup=True, mixup_alpha=0.2, accum_steps=2)

        # Финальная оценка
    print("\n" + "="*60 + "\nОценка на test set\n" + "="*60)
    from ml.multiscale_cnn_v3 import evaluate_multiscale_v3
    evaluate_multiscale_v3(model, te_ds, y_test, ctx_dim,
                           use_hourly=use_hourly,
                           save_json=model_path.replace('.pt', '_eval.json'))
    print("\n  Визуализация предсказаний...")
    from ml.visualize_predictions import predict_and_plot
    predict_and_plot(model_path, te_ds, y_test, ctx_dim,
                    use_hourly=use_hourly, n_examples=8)
    return model




# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     default='ml/model_multiscale_v3.pt')
    parser.add_argument('--rebuild',   action='store_true')
    parser.add_argument('--no-hourly', action='store_true')
    args = parser.parse_args()
    train(model_path=args.model, use_hourly=not args.no_hourly,
          force_rebuild=args.rebuild)
