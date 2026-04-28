# ml/export_predictions_csv.py
"""Экспортирует предсказания модели в CSV для MT5 тестера."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from ml.config import CFG, SCALES
from ml.dataset_v3 import (
    build_full_multiscale_dataset_v3,
    temporal_split, INDICATOR_COLS,
    _load_daily_candles_chunked,
)
from ml.multiscale_cnn_v3 import MultiScaleHybridV3, _make_loader_v3
from ml.hourly_encoder import N_HOURLY_CHANNELS, N_HOURS_PER_DAY, N_INTRADAY_DAYS
from ml.trainer_v3 import _forward_unpack
from ml.decision_layer import DecisionLayer, costs_from_config, SIG_BUY, SIG_HOLD, SIG_SELL


def export_predictions(
    save_dir:   str   = 'ml/ensemble',
    seeds:      list  = [42, 123, 7],
    ctx_dim:    int   = 21,
    output_dir: str   = 'ml/mt5_signals',
    use_all:    bool  = True,   # True = весь датасет, False = только test
):
    """
    Генерирует CSV файлы с предсказаниями для каждого тикера.
    
    Формат CSV:
    Date, Open, High, Low, Close, Volume,
    dir_prob, signal, cls_buy, cls_hold, cls_sell,
    pred_dOpen_pct, pred_dHigh_pct, pred_dLow_pct, pred_dClose_pct,
    atr_ratio
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    # ── Загружаем модели ─────────────────────────────────────
    models = []
    for seed in seeds:
        path = os.path.join(save_dir, f'model_seed{seed}.pt')
        if not os.path.exists(path):
            print(f'  [WARN] {path} не найден')
            continue

        ckpt = torch.load(path, map_location=device, weights_only=True)
        has_hourly = any('hourly_enc' in k for k in ckpt.keys())

        model = MultiScaleHybridV3(
            ctx_dim          = ctx_dim,
            n_indicator_cols = len(INDICATOR_COLS),
            future_bars      = CFG.future_bars,
            use_hourly       = has_hourly,
        ).to(device)
        model.load_state_dict(ckpt)
        model.eval()
        models.append({'model': model, 'use_hourly': has_hourly})
        print(f'  Загружена seed={seed}')

    if not models:
        raise RuntimeError("Нет загруженных моделей!")

    # ── Загружаем датасет ────────────────────────────────────
    dataset, y_all, ctx_dim_loaded, ticker_lengths = \
        build_full_multiscale_dataset_v3(use_hourly=True)

    idx_tr, idx_val, idx_test = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.15,
        purge_bars=CFG.future_bars)

    if use_all:
        idx_use = np.concatenate([idx_tr, idx_val, idx_test])
    else:
        idx_use = idx_test

    # ── Инференс по тикерам ──────────────────────────────────
    # Группируем индексы по тикерам
    ticker_indices = {}
    for global_idx in idx_use:
        ticker, local_idx = dataset.records[int(global_idx)]
        if ticker not in ticker_indices:
            ticker_indices[ticker] = []
        ticker_indices[ticker].append((global_idx, local_idx))

    # ── Загружаем реальные свечи для дат ────────────────────
    try:
        from api.routes.candles import get_client
        client = get_client()
        can_load_candles = True
    except Exception:
        can_load_candles = False
        print('  [WARN] API недоступен — даты будут синтетическими')

    all_results = []

    for ticker, idx_list in ticker_indices.items():
        print(f'\n  Обрабатываем {ticker}: {len(idx_list)} баров...')

        # Загружаем свечи для получения реальных дат
        dates = None
        if can_load_candles:
            try:
                figi = client.find_figi(ticker)
                df_candles = _load_daily_candles_chunked(client, figi)
                dates = df_candles.index
            except Exception as e:
                print(f'  [WARN] Свечи {ticker}: {e}')

        # Инференс
        subset   = Subset(dataset, [gi for gi, _ in idx_list])
        loader   = _make_loader_v3(subset, batch_size=256,
                                   shuffle=False, num_workers=0)

        all_cls, all_dir, all_ohlc = [], [], []
        all_mfe, all_fill, all_edge = [], [], []   # Sprint 2

        with torch.no_grad():
            for batch in loader:
                imgs_d, num_d, cls_y, ohlc_y, ctx, hourly_d, *_ = batch
                imgs = {W: imgs_d[W].to(device) for W in SCALES}
                nums = {W: num_d[W].to(device)  for W in SCALES}
                ctx_t = ctx.to(device) if ctx_dim > 0 else None

                batch_cls, batch_dir, batch_ohlc = [], [], []
                batch_mfe, batch_fill, batch_edge = [], [], []

                for entry in models:
                    m = entry['model']
                    if entry['use_hourly'] and hourly_d is not None:
                        ht = hourly_d.to(device)
                    else:
                        ht = torch.zeros(
                            imgs[min(SCALES)].shape[0],
                            N_INTRADAY_DAYS,
                            N_HOURLY_CHANNELS,
                            N_HOURS_PER_DAY,
                            device=device)

                    lo, op, _, dir_l, _, econ_p = _forward_unpack(
                        m, imgs, nums, ctx_t, ht)
                    batch_cls.append(
                        torch.softmax(lo, dim=1).cpu().numpy())
                    batch_dir.append(
                        torch.sigmoid(dir_l).cpu().numpy())
                    batch_ohlc.append(op.cpu().numpy())

                    if econ_p is not None:
                        batch_mfe.append(econ_p["mfe_mae"].cpu().numpy())
                        batch_fill.append(torch.sigmoid(econ_p["fill_logit"]).cpu().numpy())
                        batch_edge.append(econ_p["edge_pred"].cpu().numpy())

                all_cls.append(np.mean(batch_cls, axis=0))
                all_dir.append(np.mean(batch_dir, axis=0))
                all_ohlc.append(np.mean(batch_ohlc, axis=0))
                if batch_mfe:
                    all_mfe.append(np.mean(batch_mfe, axis=0))
                    all_fill.append(np.mean(batch_fill, axis=0))
                    all_edge.append(np.mean(batch_edge, axis=0))

        cls_probs = np.concatenate(all_cls)   # [N, 3]
        dir_probs = np.concatenate(all_dir)   # [N]
        ohlc_pred = np.concatenate(all_ohlc)  # [N, 4]

        has_econ_export = bool(all_mfe)
        if has_econ_export:
            mfe_mae   = np.concatenate(all_mfe)    # [N, 4]
            fill_prob = np.concatenate(all_fill)   # [N, 2]
            edge_pred = np.concatenate(all_edge)   # [N, 2]
            dl = DecisionLayer(costs_from_config())
            dec = dl.decide_numpy(
                dir_prob=dir_probs,
                mfe_mae=mfe_mae,
                fill_prob=fill_prob,
                edge_pred=edge_pred,
            )
            decision_signal = dec['signal']        # [N] 0/1/2
            decision_conf   = dec['confidence']    # [N]
        else:
            mfe_mae = fill_prob = edge_pred = None
            decision_signal = decision_conf = None

        # Денормализация OHLC через atr_ratio
        atr_ratios = []
        for gi, li in idx_list:
            t, loc = dataset.records[int(gi)]
            d = dataset._load(t)
            aux = d['aux']
            if aux is not None and loc < len(aux):
                arr = aux[loc]
                atr_v = float(arr[2]) if len(arr) >= 3 else 0.018
            else:
                atr_v = 0.018
            atr_ratios.append(atr_v)

        atr_arr   = np.array(atr_ratios)
        norm_fact = atr_arr * np.sqrt(CFG.future_bars)

        ohlc_pct  = ohlc_pred * norm_fact[:, np.newaxis]  # в долях

        # Composite signal
        p_buy  = cls_probs[:, 0]
        p_sell = cls_probs[:, 2]
        dc     = np.clip(ohlc_pct[:, 3], -0.1, 0.1)
        signal = np.clip(
            0.5 * (dir_probs - 0.5) * 2
            + 0.3 * (p_buy - p_sell)
            + 0.2 * np.sign(dc) * np.abs(dc) / 0.1,
            -1., 1.)

        # Формируем строки
        for j, (gi, li) in enumerate(idx_list):
            # Дата
            if dates is not None and li < len(dates):
                date_str = str(dates[li].date())
            else:
                date_str = f"BAR_{li}"

            # Свечные данные
            d = dataset._load(ticker)
            ohlc_true = d['ohlc'][li] if li < d['_n'] else [0]*4

            row = {
                'Ticker':         ticker,
                'Date':           date_str,
                'BarIndex':       li,
                'dir_prob':       round(float(dir_probs[j]), 5),
                'signal':         round(float(signal[j]), 5),
                'cls_buy':        round(float(cls_probs[j, 0]), 5),
                'cls_hold':       round(float(cls_probs[j, 1]), 5),
                'cls_sell':       round(float(cls_probs[j, 2]), 5),
                'pred_dHigh_pct': round(float(ohlc_pct[j, 1]) * 100, 4),
                'pred_dLow_pct':  round(float(ohlc_pct[j, 2]) * 100, 4),
                'pred_dClose_pct':round(float(ohlc_pct[j, 3]) * 100, 4),
                'atr_ratio':      round(float(atr_arr[j]), 6),
                'direction':      ['BUY','HOLD','SELL'][
                                   int(cls_probs[j].argmax())],
            }
            # Sprint 2: decision-aware колонки
            if has_econ_export:
                sig_int = int(decision_signal[j])
                row['decision']           = ['BUY', 'HOLD', 'SELL'][sig_int]
                row['decision_conf']      = round(float(decision_conf[j]), 5)
                row['pred_mfe_long_pct']  = round(float(mfe_mae[j, 0]) * 100, 4)
                row['pred_mae_long_pct']  = round(float(mfe_mae[j, 1]) * 100, 4)
                row['pred_mfe_short_pct'] = round(float(mfe_mae[j, 2]) * 100, 4)
                row['pred_mae_short_pct'] = round(float(mfe_mae[j, 3]) * 100, 4)
                row['fill_long']          = round(float(fill_prob[j, 0]), 5)
                row['fill_short']         = round(float(fill_prob[j, 1]), 5)
                row['edge_long_pct']      = round(float(edge_pred[j, 0]) * 100, 4)
                row['edge_short_pct']     = round(float(edge_pred[j, 1]) * 100, 4)
            all_results.append(row)

        # Сохраняем CSV для тикера
        df_out = pd.DataFrame(
            [r for r in all_results if r['Ticker'] == ticker])
        csv_path = os.path.join(output_dir, f'{ticker}_signals.csv')
        df_out.to_csv(csv_path, index=False)
        print(f'  ✓ {csv_path} ({len(df_out)} строк)')

    # Общий CSV
    df_all  = pd.DataFrame(all_results)
    all_csv = os.path.join(output_dir, 'all_signals.csv')
    df_all.to_csv(all_csv, index=False)
    print(f'\n  ✓ Общий файл: {all_csv}')
    print(f'  Всего строк: {len(df_all)}')
    print(f'\n  Скопируй CSV в папку MT5:')
    print(f'  C:\\Users\\{os.environ.get("USERNAME", "user")}\\'
          f'AppData\\Roaming\\MetaQuotes\\Terminal\\'
          f'<ID брокера>\\MQL5\\Files\\')
    return df_all


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--save-dir',    default='ml/ensemble')
    p.add_argument('--seeds',       type=int, nargs='+', default=[42, 123, 7])
    p.add_argument('--ctx-dim',     type=int, default=21)
    p.add_argument('--output-dir',  default='ml/mt5_signals')
    p.add_argument('--test-only',   action='store_true')
    args = p.parse_args()

    export_predictions(
        save_dir   = args.save_dir,
        seeds      = args.seeds,
        ctx_dim    = args.ctx_dim,
        output_dir = args.output_dir,
        use_all    = not args.test_only,
    )