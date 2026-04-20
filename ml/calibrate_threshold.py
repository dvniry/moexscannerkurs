"""Поиск оптимального threshold для dir_head, чтобы торговать только в уверенных сигналах."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import Subset

from ml.config import CFG, SCALES
from ml.dataset_v3 import build_full_multiscale_dataset_v3, temporal_split, INDICATOR_COLS
from ml.multiscale_cnn_v3 import MultiScaleHybridV3, _make_loader_v3


def main(model_path='ml/model_multiscale_v3.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=False, use_hourly=True)
    _, idx_val, idx_test = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.15,
        purge_bars=CFG.future_bars)

    model = MultiScaleHybridV3(
        ctx_dim=ctx_dim, n_indicator_cols=len(INDICATOR_COLS),
        future_bars=CFG.future_bars, use_hourly=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()

    # Собираем предсказания на val (для калибровки) и test (для финальной оценки)
    def collect(idx):
        ds = Subset(dataset, idx.tolist())
        loader = _make_loader_v3(ds, batch_size=256, shuffle=False, num_workers=0)
        all_dir_prob, all_ohlc_pred, all_cls_y, all_ohlc_y = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                imgs_dict, num_dict, cls_y, ohlc_y, ctx, hourly_data, *_ = batch
                imgs = {W: imgs_dict[W].to(device) for W in SCALES}
                nums = {W: num_dict[W].to(device) for W in SCALES}
                ctx_t = ctx.to(device)
                ht = hourly_data.to(device)
                lo, op, _, dir_l = model(imgs, nums, ctx_t, hourly=ht)
                all_dir_prob.append(torch.sigmoid(dir_l).cpu().numpy())
                all_ohlc_pred.append(op.cpu().numpy())
                all_cls_y.append(cls_y.numpy())
                all_ohlc_y.append(ohlc_y.numpy())
        return (np.concatenate(all_dir_prob),
                np.concatenate(all_ohlc_pred),
                np.concatenate(all_cls_y),
                np.concatenate(all_ohlc_y))

    print('Collecting val predictions...')
    p_val, op_val, cls_val, ohlc_val = collect(idx_val)
    print('Collecting test predictions...')
    p_te, op_te, cls_te, ohlc_te = collect(idx_test)

    # Анализ по test (на val — только для выбора threshold)
    print('\n═══ ANALYSIS: dir_head performance by confidence threshold ═══')
    print(f'{"thr":>6}  {"coverage":>8}  {"n":>6}  {"hit_rate":>10}  {"edge_vs_base":>14}')

    # В test: считаем ТОЛЬКО бинарные сэмплы (cls != 1 = HOLD)
    mask_ud = cls_te != 1
    p_bin = p_te[mask_ud]
    true_bin = (cls_te[mask_ud] == 0).astype(int)   # 1 = UP (BUY), 0 = DOWN

    base_rate = true_bin.mean()      # доля UP в non-HOLD
    print(f'  Base rate (always BUY in non-HOLD): {base_rate:.4f}')
    print(f'  Total non-HOLD samples: {len(true_bin)}')
    print()

    for thr in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75]:
        # LONG сигналы: p > thr
        long_mask = p_bin > thr
        # SHORT сигналы: p < (1 - thr)
        short_mask = p_bin < (1 - thr)
        # Берём оба направления:
        signal_mask = long_mask | short_mask
        n = signal_mask.sum()
        if n < 20: continue

        # Предсказание для каждого сигнала
        pred = np.where(p_bin > 0.5, 1, 0)  # 1=LONG, 0=SHORT
        hits = (pred[signal_mask] == true_bin[signal_mask]).mean()
        coverage = signal_mask.mean()
        edge = hits - 0.5
        print(f'  {thr:>5.2f}  {coverage:>7.2%}  {n:>6}  {hits:>9.4f}  '
              f'{edge:>+13.4f}')

    # ── Стратегия buy-low/sell-high ──
    print('\n═══ STRATEGY: limit-buy on predicted low, TP at predicted high ═══')
    # op_te shape: [N, 4] for future_bars=1, в формате [ΔO, ΔH, ΔL, ΔC]
    # ohlc_te имеет тот же формат

    for dir_thr in [0.55, 0.60, 0.65]:
        for entry_depth in [0.5, 0.7]:   # покупаем на % от pred_low
            for tp_depth in [0.6, 0.8]:  # TP на % от pred_high
                # LONG trades
                long_mask = (p_te > dir_thr) & (cls_te != 1)
                if long_mask.sum() < 50: continue

                pred_L = op_te[long_mask, 2]   # ΔLow predicted (negative)
                pred_H = op_te[long_mask, 1]   # ΔHigh predicted (positive)
                real_L = ohlc_te[long_mask, 2]
                real_H = ohlc_te[long_mask, 1]
                real_C = ohlc_te[long_mask, 3]

                entry_price = pred_L * entry_depth    # например -0.5 * ΔLow
                tp_price    = pred_H * tp_depth        # например 0.8 * ΔHigh

                # Fill rate: реальный low достиг entry?
                filled = real_L <= entry_price
                if filled.sum() < 20: continue

                # Для заполненных — TP сработал или вышли по close?
                tp_hit = (real_H >= tp_price) & filled
                close_exit = filled & ~tp_hit

                returns = np.zeros(filled.sum())
                idx_filled = np.where(filled)[0]

                for i, k in enumerate(idx_filled):
                    fee = 0.001 * 2   # в обе стороны
                    if tp_hit[k]:
                        returns[i] = tp_price[k] - entry_price[k] - fee
                    else:
                        returns[i] = real_C[k] - entry_price[k] - fee

                n_trades = filled.sum()
                mean_ret = returns.mean()
                std_ret  = returns.std() + 1e-9
                sharpe = mean_ret / std_ret * np.sqrt(252)
                win_rate = (returns > 0).mean()
                fill_rate = filled.mean()

                print(f'  thr={dir_thr:.2f} entry={entry_depth} tp={tp_depth}: '
                      f'n={n_trades:>4} fill={fill_rate:.2%} '
                      f'win={win_rate:.2%} '
                      f'ret={mean_ret * 100:+.3f}% '
                      f'sharpe={sharpe:.2f}')


if __name__ == '__main__':
    main()