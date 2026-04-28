# ml/trainer_v3.py
"""Trainer v3.20.0

Изменения v3.20.0:
- [SPRINT 1.5] добавлен masked intraday loss поверх hourly-ветки
- [SPRINT 1.5] trainer поддерживает batch:
  imgs, nums, cls_y, ohlc_y, ctx, hourly, aux_y, intraday_y, intraday_mask
- [SPRINT 1.5] trainer поддерживает model(...)->5 выходов:
  logits, ohlc, aux, dir_logit, intraday_pred
- Все фиксы v3.19.2 сохранены
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset, WeightedRandomSampler

_cert = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from ml.config import CFG, SCALES
from ml.dataset_v3 import class_distribution


INTRADAY_LOSS_WEIGHT = 0.15  # можно и 0.2
ECON_LOSS_WEIGHT     = 0.50  # Sprint 2: 0.30→0.50 — edge_head не сходился при 0.30
                              # 0.5 → 0.3 после sanity: econ доминировал и
                              # выкручивал edge_pred к границам clamp.


def _make_weighted_sampler(y_tr: np.ndarray) -> WeightedRandomSampler:
    counts = Counter(y_tr.tolist())
    n_total = len(y_tr)
    class_weight = {cls: n_total / max(cnt, 1) for cls, cnt in counts.items()}
    sample_weights = np.array(
        [class_weight[int(y)] for y in y_tr], dtype=np.float32)
    print(f'  [WRS] counts={dict(counts)}')
    print('  [WRS] weights='
          + '  '.join(f'{c}:{class_weight[c]:.1f}'
                      for c in sorted(counts)))
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True)


def _pretrain_tfc(model, tr_loader, device, n_epochs=5, lr=3e-4):
    from ml.multiscale_cnn_v3 import TRUNK_OUT
    W_short = min(SCALES)
    projector = nn.Sequential(
        nn.Linear(TRUNK_OUT, 128), nn.ReLU(), nn.Linear(128, 64)
    ).to(device)

    for name, p in model.named_parameters():
        if any(k in name for k in ('cls_head', 'ohlc_head', 'aux_head', 'dir_head')):
            p.requires_grad_(False)

    params = ([p for p in model.parameters() if p.requires_grad]
              + list(projector.parameters()))
    opt = AdamW(params, lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingWarmRestarts(opt, T_0=max(1, n_epochs // 2))

    def _aug_time(x):
        scale = 1.0 + torch.randn(
            x.shape[0], x.shape[1], 1, device=x.device) * 0.05
        return (x + torch.randn_like(x) * 0.02) * scale

    def _aug_freq(x):
        X = torch.fft.rfft(x, dim=-1)
        X[:, :, :3] = 0
        X = X + torch.randn_like(X) * 0.01
        return torch.fft.irfft(X, n=x.shape[-1], dim=-1)

    def _nt_xent(z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / temperature
        sim.fill_diagonal_(-1e9)
        labels = torch.cat([
            torch.arange(B, 2 * B), torch.arange(B)]).to(device)
        return F.cross_entropy(sim, labels)

    print(f'  [TF-C] {n_epochs} эпох, W={W_short}')
    model.train()
    projector.train()

    for epoch in range(1, n_epochs + 1):
        total = 0.0
        nb = 0
        for batch in tr_loader:
            x = batch[0][W_short].to(device).float()
            feat_t = model.backbones[str(W_short)](_aug_time(x))
            feat_f = model.backbones[str(W_short)](_aug_freq(x))
            loss = _nt_xent(projector(feat_t), projector(feat_f))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total += loss.item()
            nb += 1
        print(f'  [TF-C] E{epoch}/{n_epochs} loss={total / max(nb, 1):.4f}')

    for p in model.parameters():
        p.requires_grad_(True)
    del projector
    print('  [TF-C] done.')


def _init_cls_head(model):
    for name, p in model.cls_head.named_parameters():
        if p.ndim >= 2:
            nn.init.xavier_uniform_(p, gain=0.1)
        elif 'bias' in name:
            nn.init.zeros_(p)
    print('  [Init] cls_head re-initialized')


def _masked_intraday_loss(pred, target, mask):
    if pred is None or target is None or mask is None:
        dev = (pred.device if pred is not None else
               target.device if target is not None else 'cpu')
        return torch.tensor(0.0, device=dev)

    pred   = torch.sigmoid(pred.float()).nan_to_num(nan=0.5)  # ← [0,1]
    target = target.float().nan_to_num(nan=0.)
    mask   = mask.float().nan_to_num(nan=0.)

    # shape guard вместо raise
    h = min(pred.shape[-1], target.shape[-1], mask.shape[-1])
    pred, target, mask = pred[..., :h], target[..., :h], mask[..., :h]

    valid = mask > 0.5
    if valid.sum().item() == 0:
        return torch.tensor(0.0, device=pred.device)

    return F.binary_cross_entropy(pred[valid], target[valid])


def _forward_unpack(model, imgs, nums, ctx_t, hourly_t):
    """Универсальная распаковка forward модели.

    Поддерживает (для back-compat):
      4-tuple: logits, ohlc, aux, dir_logit                  ← v3.19, v4 Kronos
      5-tuple: + intraday_pred                               ← v3.20.0
      6-tuple: + econ (dict)                                 ← v3.21 Sprint 2
    """
    out = model(imgs, nums, ctx_t, hourly=hourly_t)
    if not isinstance(out, (tuple, list)):
        raise ValueError(f"Forward returned {type(out)}, expected tuple/list")
    n = len(out)
    if n == 6:
        lo, op, aux, dir_l, intraday_p, econ = out
    elif n == 5:
        lo, op, aux, dir_l, intraday_p = out
        econ = None
    elif n == 4:
        lo, op, aux, dir_l = out
        intraday_p = None
        econ = None
    else:
        raise ValueError(f"Forward returned {n} values, expected 4/5/6")
    return lo, op, aux, dir_l, intraday_p, econ


def _run_epochs(model, tr_loader, val_loader, optimizer, scheduler,
                criterion, device, n_epochs, patience_limit,
                save_path, phase_name, ctx_dim, use_hourly,
                accum_steps=2, use_mixup=False, mixup_alpha=0.0,
                econ_criterion=None, decision_layer=None):
    from ml.multiscale_cnn_v3 import mixup_data
    from sklearn.metrics import f1_score, precision_score, recall_score

    best_metric = 0.0
    patience = 0

    seq_params = []
    for _, module in model.named_modules():
        if isinstance(module, nn.LSTM):
            seq_params.extend(list(module.parameters()))

    for epoch in range(1, n_epochs + 1):
        criterion.focal.set_gamma(epoch, warmup_epochs=5)

        model.train()
        criterion.train()
        total_cls = 0.0
        total_reg = 0.0
        total_aux = 0.0
        total_intra = 0.0
        total_econ = 0.0
        n_steps = 0

        optimizer.zero_grad()
        _last_lo = None
        _last_cls_y = None

        for step, batch in enumerate(tr_loader, 1):
            (imgs_dict, num_dict, cls_y, ohlc_y, ctx,
             hourly_data, aux_y, intraday_y, intraday_mask, econ_y) = batch

            imgs = {W: imgs_dict[W].to(device) for W in SCALES}
            cls_y = cls_y.to(device)
            ohlc_y = ohlc_y.to(device).clamp(-5.0, 5.0)
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            hourly_t = (hourly_data.to(device)
                        if (use_hourly and hourly_data is not None) else None)
            aux_t = (aux_y.to(device) if aux_y is not None else None)
            intraday_t = (intraday_y.to(device) if intraday_y is not None else None)
            intraday_m = (intraday_mask.to(device) if intraday_mask is not None else None)
            econ_t = (econ_y.to(device) if econ_y is not None else None)
            nums = ({W: num_dict[W].to(device) for W in SCALES}
                    if num_dict is not None else None)

            if epoch == 1 and step == 1:
                print('\n  [DBG] ── первый батч ──')
                for W, t in imgs.items():
                    print(f'  [DBG] imgs[{W}]: {t.shape} nan={torch.isnan(t).any().item()}')
                bc = Counter(cls_y.cpu().numpy().tolist())
                print(f'  [DBG] batch dist: {dict(bc)}')

                if aux_t is not None:
                    print(f'  [DBG] aux_y vol:  mean={aux_t[:,0].mean():.5f} std={aux_t[:,0].std():.5f}')
                    print(f'  [DBG] aux_y skew: mean={aux_t[:,1].mean():.4f} std={aux_t[:,1].std():.4f}')
                else:
                    print('  [DBG] aux_y = None ← проверь датасет!')

                if intraday_m is not None:
                    cov = float((intraday_m > 0.5).float().mean().item())
                    print(f'  [DBG] intraday mask cov={cov:.3f}')

                if econ_t is not None:
                    print(f'  [DBG] econ_y future_ret={econ_t[:,0].mean():+.5f} '
                          f'mfe_l={econ_t[:,1].mean():.4f} mae_l={econ_t[:,2].mean():.4f} '
                          f'fill_l={econ_t[:,7].mean():.3f} '
                          f'edge_l={econ_t[:,9].mean():+.5f}')

                with torch.no_grad():
                    lo_d, op_d, aux_d, dir_d, intra_d, econ_d = _forward_unpack(
                        model, imgs, nums, ctx_t, hourly_t
                    )
                    lo_d = lo_d.float().clamp(-15., 15.).nan_to_num(0.)
                    l, lc, lr2, la = criterion(
                        lo_d, cls_y,
                        op_d.float(), ohlc_y.float(),
                        dir_logit=dir_d,
                        aux_pred=aux_d.float(),
                        aux_true=aux_t,
                    )
                    li = _masked_intraday_loss(intra_d, intraday_t, intraday_m)
                    le_dbg = torch.tensor(0., device=device)
                    if econ_criterion is not None and econ_d is not None and econ_t is not None:
                        le_dbg, _ = econ_criterion(econ_d, econ_t.float())
                print(f'  [DBG] loss={l.item():.4f}  cls={lc.item():.4f}  reg={lr2.item():.4f}  aux={la.item():.4f}  intra={li.item():.4f}  econ={le_dbg.item():.5f}')
                if aux_d is not None:
                    print(f'  [DBG] aux_pred[:3]: {aux_d[:3].detach().cpu().numpy()}')
                if intra_d is not None:
                    print(f'  [DBG] intraday_pred.shape={tuple(intra_d.shape)}')
                if econ_d is not None:
                    print(f'  [DBG] econ_pred mfe_mae[:2]: {econ_d["mfe_mae"][:2].detach().cpu().numpy()}')
                    print(f'  [DBG] econ_pred edge[:2]:    {econ_d["edge_pred"][:2].detach().cpu().numpy()}')
                print('  [DBG] ──────────────────\n')

            lo, op, aux, dir_l, intraday_p, econ_p = _forward_unpack(
                model, imgs, nums, ctx_t, hourly_t
            )

            lo = lo.float().clamp(-15., 15.).nan_to_num(nan=0.)
            op = op.float().nan_to_num(nan=0.)
            dir_l = dir_l.float().clamp(-15., 15.).nan_to_num(nan=0.)
            aux = aux.float().nan_to_num(nan=0.)

            base_loss, lcls, lreg, laux = criterion(
                lo, cls_y,
                op, ohlc_y.float(),
                dir_logit=dir_l,
                aux_pred=aux,
                aux_true=aux_t,
            )

            lintra = _masked_intraday_loss(intraday_p, intraday_t, intraday_m)

            # Sprint 2: cost-aware EconomicLoss
            lecon = torch.tensor(0., device=device)
            if econ_criterion is not None and econ_p is not None and econ_t is not None:
                lecon, _ = econ_criterion(econ_p, econ_t.float())

            loss = (base_loss
                    + INTRADAY_LOSS_WEIGHT * lintra
                    + ECON_LOSS_WEIGHT     * lecon)

            if not torch.isfinite(loss):
                print(f'  [WARN] e={epoch} s={step} loss=nan — skip')
                optimizer.zero_grad()
                continue

            (loss / accum_steps).backward()

            _last_lo = lo.detach()
            _last_cls_y = cls_y.detach()
            total_cls += lcls.item()
            total_reg += lreg.item()
            total_aux += laux.item()
            total_intra += lintra.item()
            total_econ += lecon.item()
            n_steps += 1

            if step % accum_steps == 0 or step == len(tr_loader):
                trainable = [p for g in optimizer.param_groups
                             for p in g['params'] if p.requires_grad]
                cls_head_ps = list(model.cls_head.parameters())
                aux_head_ps = list(model.aux_head.parameters())
                cls_head_ps_ids = {id(p) for p in cls_head_ps}
                aux_head_ps_ids = {id(p) for p in aux_head_ps}

                trunk_ps = [p for p in trainable
                            if id(p) not in cls_head_ps_ids
                            and id(p) not in aux_head_ps_ids]

                if seq_params:
                    nn.utils.clip_grad_norm_(seq_params, max_norm=0.5)
                if trunk_ps:
                    nn.utils.clip_grad_norm_(trunk_ps, max_norm=1.0)
                if cls_head_ps:
                    nn.utils.clip_grad_norm_(cls_head_ps, max_norm=0.5)
                if aux_head_ps:
                    nn.utils.clip_grad_norm_(aux_head_ps, max_norm=0.3)

                grad_norm = nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                if step % 200 == 0:
                    print(f'  [GRAD] e={epoch} s={step} norm={grad_norm:.3f}')

                optimizer.step()
                optimizer.zero_grad()

            if step % 200 == 0 and _last_lo is not None:
                with torch.no_grad():
                    probs = torch.softmax(_last_lo.float(), dim=1)
                    pred_cls = probs.argmax(dim=1)
                    dist = [(pred_cls == c).float().mean().item() for c in range(3)]
                    tr_acc = (pred_cls == _last_cls_y).float().mean().item()
                    print(f'  [DIST] e={epoch} s={step} BUY={dist[0]:.3f} HOLD={dist[1]:.3f} SELL={dist[2]:.3f} tr_acc={tr_acc:.3f}')

        model.eval()
        criterion.eval()
        val_preds = []
        val_trues = []
        val_dir_prob = []
        val_intra_total = 0.0
        val_intra_steps = 0
        val_decision_signals = []         # Sprint 2: BUY/HOLD/SELL coverage
        val_decision_correct = []         # hit на не-HOLD

        from ml.decision_layer import DecisionLayer, costs_from_config, SIG_BUY, SIG_SELL, SIG_HOLD
        _dl_runtime = decision_layer or DecisionLayer(costs_from_config())

        with torch.no_grad():
            for batch in val_loader:
                (imgs_dict, num_dict, cls_y, ohlc_y, ctx,
                 hourly_data, aux_y, intraday_y, intraday_mask, econ_y) = batch

                imgs = {W: imgs_dict[W].to(device) for W in SCALES}
                ctx_t = ctx.to(device) if ctx_dim > 0 else None
                hourly_t = (hourly_data.to(device)
                            if (use_hourly and hourly_data is not None) else None)
                intraday_t = (intraday_y.to(device) if intraday_y is not None else None)
                intraday_m = (intraday_mask.to(device) if intraday_mask is not None else None)
                nums = ({W: num_dict[W].to(device) for W in SCALES}
                        if num_dict is not None else None)

                lo, op, _, dir_l, intraday_p, econ_p = _forward_unpack(
                    model, imgs, nums, ctx_t, hourly_t
                )

                val_preds.extend(lo.argmax(1).cpu().numpy())
                val_trues.extend(cls_y.numpy())
                val_dir_prob.append(torch.sigmoid(dir_l).cpu().numpy())

                li = _masked_intraday_loss(intraday_p, intraday_t, intraday_m)
                val_intra_total += li.item()
                val_intra_steps += 1

                # Sprint 2: decision-aware метрика
                if econ_p is not None:
                    dec = _dl_runtime.decide(dir_l, econ_p)
                    sig = dec["signal"].cpu().numpy()
                    val_decision_signals.append(sig)
                    # hit: BUY → cls_y == 0 (UP); SELL → cls_y == 2 (DOWN)
                    cls_np = cls_y.numpy()
                    is_buy  = sig == SIG_BUY
                    is_sell = sig == SIG_SELL
                    correct = np.zeros_like(sig, dtype=bool)
                    correct[is_buy]  = (cls_np[is_buy]  == 0)
                    correct[is_sell] = (cls_np[is_sell] == 2)
                    not_hold = sig != SIG_HOLD
                    if not_hold.any():
                        val_decision_correct.append(correct[not_hold])

        vp = np.array(val_preds)
        vt = np.array(val_trues)
        val_acc = (vp == vt).mean()
        macro_f1 = f1_score(vt, vp, average='macro', zero_division=0)
        f1pc = f1_score(vt, vp, average=None, labels=[0, 1, 2], zero_division=0)
        rec_pc = recall_score(vt, vp, average=None, labels=[0, 1, 2], zero_division=0)
        prec_pc = precision_score(vt, vp, average=None, labels=[0, 1, 2], zero_division=0)

        buy_f1, hold_f1, sell_f1 = f1pc
        buy_rec, hold_rec, sell_rec = rec_pc
        prec_ud = 0.5 * (float(prec_pc[0]) + float(prec_pc[2]))

        dir_p_np = np.concatenate(val_dir_prob, axis=0)
        mask_ud = vt != 1
        dir_cov = float(mask_ud.mean())
        dir_acc_head = 0.5
        if mask_ud.any():
            dir_target = (vt[mask_ud] == 0).astype(int)
            dir_pred = (dir_p_np[mask_ud] > 0.5).astype(int)
            dir_acc_head = float((dir_pred == dir_target).mean())

        # Sprint 2: decision-aware coverage и hit rate
        dec_buy_pct = dec_hold_pct = dec_sell_pct = 0.0
        dec_hit = 0.5
        if val_decision_signals:
            sig_all = np.concatenate(val_decision_signals)
            n_sig = max(len(sig_all), 1)
            dec_buy_pct  = float((sig_all == SIG_BUY ).sum() / n_sig)
            dec_hold_pct = float((sig_all == SIG_HOLD).sum() / n_sig)
            dec_sell_pct = float((sig_all == SIG_SELL).sum() / n_sig)
            if val_decision_correct:
                hit_arr = np.concatenate(val_decision_correct)
                if len(hit_arr) > 0:
                    dec_hit = float(hit_arr.mean())

        # Sprint 2: метрика = dir_acc (главный сигнал, стабилен) +
        # бонус за dec_hit > 0.5 (сигналы полезнее случайных).
        # prec_ud убран — слишком шумный (0.18→0.61 между эпохами),
        # доминировал и приводил к выбору случайных чекпоинтов.
        val_metric = dir_acc_head + 0.3 * max(dec_hit - 0.5, 0.0)
        scheduler.step()

        print(f'  [{phase_name}] E{epoch:3d}/{n_epochs} '
              f'cls={total_cls/max(n_steps,1):.4f} '
              f'reg={total_reg/max(n_steps,1):.5f} '
              f'aux={total_aux/max(n_steps,1):.5f} '
              f'intra={total_intra/max(n_steps,1):.5f} '
              f'econ={total_econ/max(n_steps,1):.5f} | '
              f'acc={val_acc:.4f} mF1={macro_f1:.4f} | '
              f'F1: up={buy_f1:.3f} fl={hold_f1:.3f} dn={sell_f1:.3f} | '
              f'REC: up={buy_rec:.3f} fl={hold_rec:.3f} dn={sell_rec:.3f} | '
              f'dir={dir_acc_head:.4f}(cov={dir_cov:.2f}) '
              f'prec_ud={prec_ud:.3f} '
              f'val_intra={val_intra_total/max(val_intra_steps,1):.5f} | '
              f'DEC[B={dec_buy_pct:.2f} H={dec_hold_pct:.2f} S={dec_sell_pct:.2f} hit={dec_hit:.3f}] | '
              f'metric={val_metric:.4f} | '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}')

        if val_metric > best_metric:
            best_metric = val_metric
            patience = 0
            torch.save(model.state_dict(), save_path)
            print(f'  ✓ saved metric={val_metric:.4f}')
        else:
            patience += 1
            if patience >= patience_limit:
                print(f'  [{phase_name}] Early stop (patience={patience_limit})')
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))
    print(f'  [{phase_name}] Best={best_metric:.4f}')


def train(model_path='ml/model_multiscale_v3.pt', use_hourly=True,
          force_rebuild=False, do_pretrain=True, pretrain_epochs=5):
    from ml.dataset_v3 import (build_full_multiscale_dataset_v3,
                               temporal_split, INDICATOR_COLS)
    from ml.multiscale_cnn_v3 import (MultiScaleHybridV3, MultiTaskLossV3,
                                      EconomicLoss, _make_loader_v3)
    from ml.decision_layer import DecisionLayer, costs_from_config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f'  GPU: {p.name} {getattr(p, "total_memory", 0) / 1024**2:.0f} MB')

    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=force_rebuild, use_hourly=use_hourly)
    print(f'  Сэмплов: {len(y_all)}, ctx_dim={ctx_dim}')

    idx_tr, idx_val, idx_test = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.15,
        purge_bars=CFG.future_bars)
    y_tr = y_all[idx_tr]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]
    print(f'  Train={len(y_tr)} Val={len(y_val)} Test={len(y_test)}')
    class_distribution(y_tr)

    tr_subset = Subset(dataset, idx_tr.tolist())
    val_subset = Subset(dataset, idx_val.tolist())

    wrs = _make_weighted_sampler(y_tr)
    tr_loader = _make_loader_v3(
        tr_subset, CFG.batch_size, shuffle=False, sampler=wrs, num_workers=2)
    val_loader = _make_loader_v3(
        val_subset, CFG.batch_size, shuffle=False, num_workers=2)
    te_ds = Subset(dataset, idx_test.tolist())
    n_ind = len(INDICATOR_COLS)

    model = MultiScaleHybridV3(
        ctx_dim=ctx_dim,
        n_indicator_cols=n_ind,
        future_bars=CFG.future_bars,
        use_hourly=use_hourly,
    ).to(device)
    _init_cls_head(model)

    counts_dict = Counter(y_tr.tolist())
    total_tr = len(y_tr)
    raw_w = [math.sqrt(total_tr / max(counts_dict.get(i, 1), 1)) for i in range(3)]
    max_w = max(raw_w)
    cls_weights = torch.tensor(
        [w / max_w for w in raw_w], dtype=torch.float32).to(device)
    print(f'  cls_w: UP={cls_weights[0]:.3f} FLAT={cls_weights[1]:.3f} DOWN={cls_weights[2]:.3f}')

    criterion = MultiTaskLossV3(
        cls_weight=cls_weights,
        gamma_per_class=(3.0, 1.0, 3.0),
        label_smoothing=0.03,
        future_bars=CFG.future_bars,
        huber_delta=0.3,
        direction_weight=0.40,
        reg_loss_weight=0.20,
        aux_loss_weight=0.10,
    ).to(device)

    # Sprint 2: cost-aware loss + DecisionLayer
    _costs = costs_from_config()
    econ_criterion = EconomicLoss(cost_roundtrip=_costs.roundtrip).to(device)
    decision_layer = DecisionLayer(_costs)

    if do_pretrain:
        _pretrain_tfc(model, tr_loader, device,
                      n_epochs=pretrain_epochs, lr=3e-4)

    max_lr = 2e-4

    backbone_ids = {id(p) for p in model.backbone.parameters()}
    hourly_ids = ({id(p) for p in model.hourly_enc.parameters()}
                  if use_hourly and hasattr(model, 'hourly_enc') else set())
    cls_head_ids = {id(p) for p in model.cls_head.parameters()}
    dir_head_ids = {id(p) for p in model.dir_head.parameters()}
    econ_head_ids = ({id(p) for p in model.econ_heads.parameters()}
                     if hasattr(model, 'econ_heads') else set())
    crit_params = list(criterion.parameters())

    param_groups = [
        {'params': list(model.backbone.parameters()),
         'lr': max_lr * 0.15, 'name': 'backbone', 'weight_decay': 5e-4},
        {'params': list(model.cls_head.parameters()),
         'lr': max_lr * 0.5, 'name': 'cls_head', 'weight_decay': 1e-4},
        {'params': list(model.dir_head.parameters()),
         'lr': max_lr, 'name': 'dir_head', 'weight_decay': 1e-4},
        {'params': list(model.econ_heads.parameters()) if econ_head_ids else [],
         'lr': max_lr * 1.5, 'name': 'econ_heads', 'weight_decay': 1e-4},
        {'params': [p for p in model.parameters()
                    if (p.requires_grad
                        and id(p) not in backbone_ids
                        and id(p) not in hourly_ids
                        and id(p) not in cls_head_ids
                        and id(p) not in dir_head_ids
                        and id(p) not in econ_head_ids)],
         'lr': max_lr, 'name': 'other', 'weight_decay': 5e-3},
    ]

    if hourly_ids:
        param_groups.insert(3, {
            'params': list(model.hourly_enc.parameters()),
            'lr': max_lr * 0.1, 'name': 'hourly', 'weight_decay': 5e-4
        })

    optimizer = AdamW(
        param_groups + [{
            'params': crit_params, 'lr': max_lr,
            'name': 'criterion', 'weight_decay': 1e-4
        }]
    )

    _ACCUM = 2
    _EPOCHS = 20
    _PATIENCE = 5

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=_EPOCHS, T_mult=1, eta_min=5e-6)

    _run_epochs(
        model, tr_loader, val_loader, optimizer, scheduler,
        criterion, device,
        n_epochs=_EPOCHS, patience_limit=_PATIENCE,
        save_path=model_path, phase_name='F1',
        ctx_dim=ctx_dim, use_hourly=use_hourly,
        accum_steps=_ACCUM,
        econ_criterion=econ_criterion,
        decision_layer=decision_layer)

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

    parser = argparse.ArgumentParser(description='Trainer v3.20.0')
    parser.add_argument('--model', default='ml/model_multiscale_v3.pt')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--no-hourly', action='store_true')
    parser.add_argument('--no-pretrain', action='store_true')
    parser.add_argument('--pretrain-epochs', type=int, default=5)
    args = parser.parse_args()

    train(
        model_path=args.model,
        use_hourly=not args.no_hourly,
        force_rebuild=args.rebuild,
        do_pretrain=not args.no_pretrain,
        pretrain_epochs=args.pretrain_epochs,
    )