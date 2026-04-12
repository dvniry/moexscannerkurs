# ml/visualize_predictions.py
"""Визуализация предсказаний модели v3.11+

Поддерживает:
- QuantileOHLCHead (65 outputs = 13 fields × 5 bars) — основной режим
- OHLCHead (20 outputs = 4 × 5) — обратная совместимость

BUG FIX: ohlc_pred.reshape(fb, 4) → автоопределение формата по размеру output
BUG FIX: close_hist из img_c[2] (пиксели) → из num_dict['close'] (реальные данные)
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


try:
    from ml.config import CFG, SCALES
except ImportError:
    from config import CFG, SCALES


# Определяем индекс колонки 'close' в INDICATOR_COLS один раз при импорте
try:
    from ml.dataset_v3 import INDICATOR_COLS as _IND_COLS
    _IND_LIST = list(_IND_COLS)
    _CLOSE_IDX = _IND_LIST.index('close') if 'close' in _IND_LIST else None
except Exception:
    _IND_LIST  = []
    _CLOSE_IDX = None


# Константы квантильного формата
QOHLC_N_FIELDS = 13
_Q10 = [0, 3, 6,  9]
_Q50 = [1, 4, 7, 10]
_Q90 = [2, 5, 8, 11]


def _parse_ohlc_output(ohlc_raw: np.ndarray, future_bars: int):
    total = ohlc_raw.shape[0]

    if total == future_bars * QOHLC_N_FIELDS:
        arr = ohlc_raw.reshape(future_bars, QOHLC_N_FIELDS)
        return arr[:, _Q50], arr[:, _Q10], arr[:, _Q90], 'quantile'

    if total == future_bars * 4:
        return ohlc_raw.reshape(future_bars, 4), None, None, 'point'

    if total % future_bars == 0:
        arr = ohlc_raw.reshape(future_bars, total // future_bars)
        return arr[:, :4], None, None, 'point'

    raise ValueError(
        f"Не удалось разобрать ohlc_head output: size={total}, future_bars={future_bars}. "
        f"Ожидалось {future_bars * QOHLC_N_FIELDS} (quantile) или {future_bars * 4} (point)."
    )


def _load_model(model_path, ctx_dim, use_hourly, n_indicator_cols=None):
    from ml.multiscale_cnn_v3 import MultiScaleHybridV3
    if n_indicator_cols is None:
        n_indicator_cols = len(_IND_LIST) if _IND_LIST else 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MultiScaleHybridV3(
        ctx_dim=ctx_dim,
        n_indicator_cols=n_indicator_cols,
        future_bars=CFG.future_bars,
        use_hourly=use_hourly,
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device


def _extract_close_hist(nums_arr: np.ndarray) -> np.ndarray:
    """
    Извлекает историю close из nums [W, n_cols] или [W, n_cols] тензора.
    Возвращает 1D массив длиной W.
    """
    if nums_arr.ndim == 2:
        # nums_arr: [window, n_cols]
        if _CLOSE_IDX is not None and nums_arr.shape[1] > _CLOSE_IDX:
            return nums_arr[:, _CLOSE_IDX]
        # fallback: первая колонка
        return nums_arr[:, 0]
    # fallback если что-то неожиданное
    return nums_arr.mean(axis=0) if nums_arr.ndim == 2 else nums_arr


def predict_and_plot(model_path, te_ds, y_test, ctx_dim,
                     use_hourly=True, n_examples=8,
                     output_dir='ml/plots',
                     n_indicator_cols=None):
    model, device = _load_model(model_path, ctx_dim, use_hourly, n_indicator_cols)
    loader = DataLoader(te_ds, batch_size=n_examples * 2, shuffle=False,
                        num_workers=0, pin_memory=False)

    os.makedirs(output_dir, exist_ok=True)
    fb = CFG.future_bars
    label_map = {0: 'UP ▲', 1: 'FLAT ─', 2: 'DOWN ▼'}
    color_map  = {0: '#4ec9b0', 1: '#dcdcaa', 2: '#f48771'}

    collected = {
        'nums_short': [],          # ← ДОБАВЛЕНО: [W, n_cols] для отрисовки истории
        'cls_pred':   [], 'cls_true': [],
        'q50': [], 'q10': [], 'q90': [],
        'ohlc_true':  [], 'mode': None,
    }

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
            hourly_data = hourly_opt[0] if hourly_opt else None
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            ht    = (hourly_data.to(device)
                     if (use_hourly and hourly_data is not None) else None)
            nums  = ({W: num_dict[W].to(device) for W in SCALES}
                     if num_dict is not None else None)

            lo, op, *_ = model(imgs, nums, ctx_t, hourly=ht)
            cls_pred    = lo.argmax(1).cpu().numpy()
            ohlc_raw    = op.cpu().numpy()
            cls_true    = cls_y.numpy()
            ohlc_true_np = ohlc_y.numpy()

            W_short = min(SCALES)
            B = cls_pred.shape[0]
            for i in range(B):
                if len(collected['cls_pred']) >= n_examples:
                    break
                q50, q10, q90, mode = _parse_ohlc_output(ohlc_raw[i], fb)
                collected['cls_pred'].append(cls_pred[i])
                collected['cls_true'].append(cls_true[i])
                collected['q50'].append(q50)
                collected['q10'].append(q10)
                collected['q90'].append(q90)
                collected['ohlc_true'].append(ohlc_true_np[i, :fb*4].reshape(fb, 4))
                collected['mode'] = mode
                # ← ИСПРАВЛЕНО: берём nums вместо пикселей img
                collected['nums_short'].append(
                    num_dict[W_short][i].cpu().numpy()  # [W, n_cols]
                )

            if len(collected['cls_pred']) >= n_examples:
                break

    n    = len(collected['cls_pred'])
    mode = collected['mode'] or 'point'
    print(f"  [VIZ] {n} примеров, режим={mode}, future_bars={fb}, "
          f"close_idx={_CLOSE_IDX}")

    if not HAS_MPL:
        _print_text_summary(collected, n, label_map)
        return

    _plot_examples(collected, n, fb, label_map, color_map, output_dir, mode)


def _print_text_summary(collected, n, label_map):
    correct = sum(p == t for p, t in zip(collected['cls_pred'], collected['cls_true']))
    print(f"  [VIZ] Accuracy на {n} примерах: {correct/n:.3f}")
    for i in range(n):
        p, t = collected['cls_pred'][i], collected['cls_true'][i]
        mark = '✓' if p == t else '✗'
        print(f"    [{i+1}] pred={label_map[p]}  true={label_map[t]}  {mark}")
        print(f"         ΔClose pred={collected['q50'][i][0,3]:+.3f}  "
              f"true={collected['ohlc_true'][i][0,3]:+.3f}")


def _plot_examples(collected, n, fb, label_map, color_map, output_dir, mode):
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.5, nrows * 3.5),
                              facecolor='#1a1a2e')
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        ax.set_facecolor('#0d1117')

        # ── История close из nums ──────────────────────────────────────────────
        close_hist = _extract_close_hist(collected['nums_short'][i])
        T      = len(close_hist)
        t_hist = np.arange(-T, 0)
        ax.plot(t_hist, close_hist, color='#569cd6', lw=1.2,
                alpha=0.8, label='close')

        # ── Предсказание OHLC ─────────────────────────────────────────────────
        q50       = collected['q50'][i]
        ohlc_true = collected['ohlc_true'][i]
        last      = close_hist[-1]

        pred_close = last + np.cumsum(q50[:, 3])
        true_close = last + np.cumsum(ohlc_true[:, 3])
        t_future   = np.arange(1, fb + 1)

        ax.plot(t_future, pred_close, color='#dcdcaa', lw=1.5,
                marker='o', markersize=3, label='pred q50')
        ax.plot(t_future, true_close, color='#ce9178', lw=1.5,
                ls='--', marker='x', markersize=3, label='true')

        if mode == 'quantile' and collected['q10'][i] is not None:
            q10 = collected['q10'][i]
            q90 = collected['q90'][i]
            ax.fill_between(t_future,
                            last + np.cumsum(q10[:, 3]),
                            last + np.cumsum(q90[:, 3]),
                            color='#dcdcaa', alpha=0.15, label='q10-q90')

        ax.axvline(0, color='#555', lw=0.8, ls=':')

        # ── Заголовок ─────────────────────────────────────────────────────────
        p, t = collected['cls_pred'][i], collected['cls_true'][i]
        mark  = '✓' if p == t else '✗'
        ax.set_title(f"{mark} pred: {label_map[p]}  true: {label_map[t]}",
                     color=color_map[p], fontsize=8, pad=4)

        ax.tick_params(colors='#888', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#333')

        if i == 0:
            ax.legend(fontsize=6, facecolor='#1a1a2e',
                      labelcolor='white', loc='upper left')

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    correct = sum(p == t for p, t in zip(collected['cls_pred'], collected['cls_true']))
    fig.suptitle(
        f"Predictions ({n} samples) — acc={correct/n:.2f} | mode={mode}",
        color='white', fontsize=11, y=1.01,
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'predictions_sample.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [VIZ] Saved → {out_path}")


if __name__ == '__main__':
    print("visualize_predictions.py — запускается только из trainer_v3.py")