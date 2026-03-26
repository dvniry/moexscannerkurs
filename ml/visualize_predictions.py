# visualize_predictions.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ml.config import CFG, SCALES
from ml.multiscale_cnn_v3 import MultiScaleHybridV3
from ml.dataset_v3 import build_full_multiscale_dataset_v3

LABEL_NAMES = ['UP', 'FLAT', 'DOWN']
COLORS      = {'UP': '#26a69a', 'FLAT': '#888', 'DOWN': '#ef5350'}

def render_candle_chart(ax, ohlc_abs, title, signal=None, signal_color='gray'):
    """ohlc_abs: (N, 4) — Open, High, Low, Close абсолютные цены"""
    for i, (o, h, l, c) in enumerate(ohlc_abs):
        color = '#26a69a' if c >= o else '#ef5350'
        ax.plot([i, i], [l, h], color=color, lw=1.2)           # фитиль
        ax.bar(i, abs(c - o), bottom=min(o, c),
               color=color, width=0.6, alpha=0.85)              # тело
    if signal:
        ax.set_title(f"{title}  →  {signal}", color=signal_color, fontsize=11)
    else:
        ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.8, len(ohlc_abs) - 0.2)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=8)

def predict_and_plot(model_path, te_ds, y_test, ctx_dim,
                     use_hourly=True, n_examples=6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MultiScaleHybridV3(ctx_dim=ctx_dim,
                                use_hourly=use_hourly).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Берём n_examples случайных сэмплов
    indices = np.random.choice(len(te_ds), n_examples, replace=False)

    fig, axes = plt.subplots(n_examples, 2,
                             figsize=(14, n_examples * 3.5),
                             gridspec_kw={'width_ratios': [2, 3]})
    fig.suptitle('Предсказания модели: история + прогноз 5 свечей',
                 fontsize=13, fontweight='bold')

    for row, idx in enumerate(indices):
        sample = te_ds[idx]
        # sample: (imgs_by_scale, num_by_scale, cls_label, ohlc_target, ctx, hourly)
        imgs_dict, num_dict, cls_true, ohlc_true, ctx, *hourly_opt = sample
        hourly = hourly_opt[0] if hourly_opt else None

        # Батч из 1 элемента
        imgs   = {W: imgs_dict[W].unsqueeze(0).to(device) for W in SCALES}
        nums   = ({W: num_dict[W].unsqueeze(0).to(device) for W in SCALES}
                  if num_dict is not None else None)
        ctx_t  = ctx.unsqueeze(0).to(device) if ctx_dim > 0 else None
        hrly_t = (hourly.unsqueeze(0).to(device)
                  if (use_hourly and hourly is not None) else None)

        with torch.no_grad():
            cls_logits, ohlc_pred = model(imgs, nums, ctx_t, hourly=hrly_t)

        probs      = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
        pred_label = int(probs.argmax())
        true_label = int(cls_true)

        # ohlc_pred → (5, 4) дельты → абсолютные цены
        fb          = CFG.future_bars
        pred_deltas = ohlc_pred[0].cpu().numpy().reshape(fb, 4)
        true_deltas = ohlc_true.numpy().reshape(fb, 4)

        # Восстанавливаем цены (начало = 100 для нормализации)
        def deltas_to_abs(deltas, start=100.0):
            prices = np.zeros_like(deltas)
            cur = start
            for i, (do, dh, dl, dc) in enumerate(deltas):
                o = cur * (1 + do)
                c = cur * (1 + dc)
                h = cur * (1 + dh)
                l = cur * (1 + dl)
                prices[i] = [o, h, l, c]
                cur = c
            return prices

        pred_abs = deltas_to_abs(pred_deltas)
        true_abs = deltas_to_abs(true_deltas)

        # --- Левый график: факт (зелёный/красный) ---
        ax_true = axes[row, 0]
        render_candle_chart(ax_true, true_abs,
                            title=f"Факт [{LABEL_NAMES[true_label]}]",
                            signal=LABEL_NAMES[true_label],
                            signal_color=COLORS[LABEL_NAMES[true_label]])

        # --- Правый график: прогноз ---
        ax_pred = axes[row, 1]
        render_candle_chart(ax_pred, pred_abs,
                            title=(f"Прогноз [{LABEL_NAMES[pred_label]}]  "
                                   f"P={probs[pred_label]:.2f}  "
                                   f"{'✓' if pred_label==true_label else '✗'}"),
                            signal=LABEL_NAMES[pred_label],
                            signal_color=COLORS[LABEL_NAMES[pred_label]])

        # Подписи вероятностей
        prob_str = '  '.join(
            f"{LABEL_NAMES[i]}:{probs[i]:.2f}" for i in range(3))
        ax_pred.set_xlabel(prob_str, fontsize=8, color='#555')

    plt.tight_layout()
    plt.savefig('ml/predictions_sample.png', dpi=130, bbox_inches='tight')
    plt.show()
    print("  ✓ Сохранено: ml/predictions_sample.png")
