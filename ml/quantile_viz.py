"""ml/quantile_viz.py — Sprint 11.1 matplotlib визуализация predictions.

Генерирует PNG для каждого примера: actual OHLC candle (left) + predicted
quantile ranges (right) на одной y-шкале.

Запуск:
    py -m ml.quantile_viz                              # 5 случайных в ml/viz/
    py -m ml.quantile_viz --examples 8 --out ml/viz/   # 8 примеров
    py -m ml.quantile_viz --indices 100 1500 8000      # конкретные индексы
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NPZ_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
QUANTILES = (0.10, 0.50, 0.90)
FUTURE_BARS = 5

CANDLE_BODY_W = 0.18
PRED_BAND_W   = 0.18
GROUP_GAP     = 0.18    # промежуток между actual и pred внутри bar


def split_quantiles_ohlc(qpred: np.ndarray) -> dict:
    """[N, 60] → {O,H,L,C} каждый shape [N, 3 quantiles, fb].

    Layout: [O × 3q × fb || H × 3q × fb || L × 3q × fb || C × 3q × fb]
    Backward-compat: [N, 30] (старый формат low+high) → возвращает только
    {L, H} с заглушками O=L, C=H.
    """
    N = qpred.shape[0]
    n_q = len(QUANTILES)
    chunk = n_q * FUTURE_BARS
    if qpred.shape[1] >= 4 * chunk:
        # Новый формат — 4 канала
        return {
            "O": qpred[:, 0*chunk:1*chunk].reshape(N, n_q, FUTURE_BARS),
            "H": qpred[:, 1*chunk:2*chunk].reshape(N, n_q, FUTURE_BARS),
            "L": qpred[:, 2*chunk:3*chunk].reshape(N, n_q, FUTURE_BARS),
            "C": qpred[:, 3*chunk:4*chunk].reshape(N, n_q, FUTURE_BARS),
        }
    # Старый формат — low+high only
    half = qpred.shape[1] // 2
    L = qpred[:, :half].reshape(N, n_q, FUTURE_BARS)
    H = qpred[:, half:].reshape(N, n_q, FUTURE_BARS)
    return {"O": L.copy(), "H": H, "L": L, "C": H.copy()}


def draw_candle(ax, x: float, O: float, H: float, L: float, C: float,
                width: float = CANDLE_BODY_W):
    """Классическая свеча: wick L→H + тело O→C (зелёная если C>O, красная иначе)."""
    is_bull = C >= O
    body_color = "#26a69a" if is_bull else "#ef5350"   # green/red
    edge_color = "#000"
    # Wick
    ax.plot([x, x], [L, H], color=edge_color, linewidth=1.0, zorder=2)
    # Body
    body_h = abs(C - O)
    body_b = min(O, C)
    if body_h < 1e-6:
        # Doji — тонкая линия
        ax.plot([x - width / 2, x + width / 2], [O, O], color=edge_color,
                linewidth=1.5, zorder=3)
    else:
        rect = mpatches.Rectangle(
            (x - width / 2, body_b), width, body_h,
            facecolor=body_color, edgecolor=edge_color, linewidth=0.7, zorder=3)
        ax.add_patch(rect)


def draw_quantile_band(ax, x: float, q10: float, q50: float, q90: float,
                       color: str, label: str | None = None,
                       width: float = PRED_BAND_W,
                       alpha: float = 0.30):
    """Горизонтальная плашка [q10, q90] со средней линией q50."""
    rect = mpatches.Rectangle(
        (x - width / 2, q10), width, max(q90 - q10, 1e-9),
        facecolor=color, edgecolor="#333", linewidth=0.5, alpha=alpha,
        label=label, zorder=2)
    ax.add_patch(rect)
    ax.plot([x - width / 2, x + width / 2], [q50, q50],
            color=color, linewidth=2.0, zorder=4)
    ax.plot([x - width / 2, x + width / 2], [q10, q10],
            color=color, linewidth=0.7, linestyle=":", zorder=4)
    ax.plot([x - width / 2, x + width / 2], [q90, q90],
            color=color, linewidth=0.7, linestyle=":", zorder=4)


def _channel_colors():
    return {"O": "#9467bd", "H": "#d62728", "L": "#1f77b4", "C": "#2ca02c"}


def plot_example(idx: int, data: dict, qparts: dict,
                 out_path: str | None = None):
    """qparts = {O,H,L,C} → каждый [N, 3, fb]."""
    ohlc = data["ohlc_test"][idx].reshape(FUTURE_BARS, 4)
    ticker = str(data["test_tickers"][idx])
    date = str(data["test_dates"][idx])
    y = int(data["y_test"][idx])
    p_up = float(data.get("dir_prob_platt", data["dir_prob"])[idx])
    atr_r = float(data["atr_ratio"][idx])
    cls_name = {0: "UP", 1: "FLAT", 2: "DOWN"}.get(y, "?")
    colors = _channel_colors()

    has_full_ohlc = not np.allclose(qparts["O"][idx], qparts["L"][idx])

    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    bars_x = np.arange(FUTURE_BARS)
    for b in range(FUTURE_BARS):
        x_actual = b - GROUP_GAP * 1.2
        O_a, H_a, L_a, C_a = ohlc[b]

        # ── Actual candle (left) ──
        draw_candle(ax, x_actual, O_a, H_a, L_a, C_a)

        # ── Predicted quantile bands (right): 4 columns spaced 0.06 apart ──
        if has_full_ohlc:
            ch_order = ["L", "O", "C", "H"]    # bottom-up по логике L<O,C<H
            spacing = 0.07
            base_x  = b + GROUP_GAP * 0.5
            for ci, ch in enumerate(ch_order):
                xc = base_x + ci * spacing
                q10, q50, q90 = qparts[ch][idx, :, b]
                draw_quantile_band(
                    ax, xc, q10, q50, q90, color=colors[ch], width=0.06,
                    label=f"q_{ch} [0.1, 0.9]" if b == 0 else None,
                    alpha=0.35,
                )
        else:
            # Backward-compat (старый формат — только L и H)
            x_pred = b + GROUP_GAP
            for ch in ("L", "H"):
                q10, q50, q90 = qparts[ch][idx, :, b]
                draw_quantile_band(
                    ax, x_pred, q10, q50, q90, color=colors[ch],
                    label=f"q_{ch} [0.1, 0.9]" if b == 0 else None,
                )

        # Markers actual H/L on top of candle
        in_range = lambda v, c: qparts[c][idx, 0, b] <= v <= qparts[c][idx, 2, b]
        for actual_v, ch_name, marker in [
            (H_a, "H", "^"),
            (L_a, "L", "v"),
            (O_a, "O", "o"),
            (C_a, "C", "s"),
        ]:
            ok = in_range(actual_v, ch_name) if has_full_ohlc else (
                qparts["L"][idx, 0, b] <= actual_v <= qparts["H"][idx, 2, b]
            )
            edge = "#888" if ok else colors[ch_name]
            ax.plot([x_actual + 0.12], [actual_v], marker=marker, color=edge,
                    markersize=6, markeredgecolor="#000", markeredgewidth=0.4,
                    alpha=0.85, zorder=5)

    for b in range(FUTURE_BARS):
        ax.axvline(b, color="#aaa", linewidth=0.3, linestyle=":", zorder=1)
    ax.axhline(0, color="#666", linewidth=0.6, linestyle="-", zorder=1)

    ax.set_xlim(-0.55, FUTURE_BARS - 0.30)
    ax.set_xticks(bars_x)
    ax.set_xticklabels([f"t+{i+1}" for i in range(FUTURE_BARS)])
    ax.set_xlabel("Future bar (left = actual candle | right = predicted L/O/C/H quantile bands)")
    ax.set_ylabel("ATR-normalized price")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_title(f"{ticker}  {date}  y_true={cls_name}  "
                 f"dir_prob_platt={p_up:.3f}  atr_ratio={atr_r:.4f}",
                 fontsize=11)

    handles = [
        mpatches.Patch(facecolor="#26a69a", edgecolor="#000", label="actual bull candle"),
        mpatches.Patch(facecolor="#ef5350", edgecolor="#000", label="actual bear candle"),
    ]
    if has_full_ohlc:
        handles += [
            mpatches.Patch(facecolor=colors["L"], alpha=0.35, label="q_L [0.1, 0.9]"),
            mpatches.Patch(facecolor=colors["O"], alpha=0.35, label="q_O [0.1, 0.9]"),
            mpatches.Patch(facecolor=colors["C"], alpha=0.35, label="q_C [0.1, 0.9]"),
            mpatches.Patch(facecolor=colors["H"], alpha=0.35, label="q_H [0.1, 0.9]"),
        ]
    else:
        handles += [
            mpatches.Patch(facecolor=colors["L"], alpha=0.30, label="q_L [0.1, 0.9]"),
            mpatches.Patch(facecolor=colors["H"], alpha=0.30, label="q_H [0.1, 0.9]"),
        ]
    handles += [
        plt.Line2D([0], [0], color="#888", marker="^", markeredgecolor="#000",
                   markersize=7, linestyle="None", label="actual H"),
        plt.Line2D([0], [0], color="#888", marker="v", markeredgecolor="#000",
                   markersize=7, linestyle="None", label="actual L"),
        plt.Line2D([0], [0], color="#888", marker="o", markeredgecolor="#000",
                   markersize=6, linestyle="None", label="actual O"),
        plt.Line2D([0], [0], color="#888", marker="s", markeredgecolor="#000",
                   markersize=6, linestyle="None", label="actual C"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.9, ncol=2)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path
    plt.show()
    return None


def plot_grid(indices: list[int], data: dict, qparts: dict, out_path: str):
    """Один большой PNG с N примерами в grid (упрощённый: только L/H bands)."""
    n = len(indices)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 5 * rows), squeeze=False)
    colors = _channel_colors()
    has_full = not np.allclose(qparts["O"], qparts["L"])

    for plot_i, idx in enumerate(indices):
        ax = axes[plot_i // cols][plot_i % cols]
        ohlc = data["ohlc_test"][idx].reshape(FUTURE_BARS, 4)
        ticker = str(data["test_tickers"][idx])
        date = str(data["test_dates"][idx])
        y = int(data["y_test"][idx])
        p_up = float(data.get("dir_prob_platt", data["dir_prob"])[idx])
        cls_name = {0: "UP", 1: "FLAT", 2: "DOWN"}.get(y, "?")

        for b in range(FUTURE_BARS):
            x_actual = b - GROUP_GAP
            O, H, L, C = ohlc[b]
            draw_candle(ax, x_actual, O, H, L, C)
            if has_full:
                spacing = 0.07
                base_x  = b + GROUP_GAP * 0.5
                for ci, ch in enumerate(["L", "O", "C", "H"]):
                    xc = base_x + ci * spacing
                    q10, q50, q90 = qparts[ch][idx, :, b]
                    draw_quantile_band(ax, xc, q10, q50, q90, colors[ch],
                                       width=0.06, alpha=0.35)
            else:
                x_pred = b + GROUP_GAP
                for ch in ("L", "H"):
                    q10, q50, q90 = qparts[ch][idx, :, b]
                    draw_quantile_band(ax, x_pred, q10, q50, q90, colors[ch])

        ax.axhline(0, color="#666", linewidth=0.5)
        ax.set_xlim(-0.55, FUTURE_BARS - 0.30)
        ax.set_xticks(np.arange(FUTURE_BARS))
        ax.set_xticklabels([f"t+{i+1}" for i in range(FUTURE_BARS)])
        ax.set_title(f"{ticker} {date}  y={cls_name}  p_up={p_up:.2f}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    for plot_i in range(n, rows * cols):
        axes[plot_i // cols][plot_i % cols].axis("off")

    title = ("OHLC quantile predictions" if has_full
             else "L/H quantile predictions (старый формат)")
    fig.suptitle(f"{title} ({n} примеров)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="ml/viz")
    parser.add_argument("--indices", type=int, nargs="*",
                        help="Конкретные индексы вместо random sample")
    parser.add_argument("--grid", action="store_true",
                        help="Один PNG с grid-раскладкой вместо отдельных файлов")
    args = parser.parse_args()

    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден"); sys.exit(1)

    npz = np.load(NPZ_PATH, allow_pickle=False)
    if "quantile_pred" not in npz.files:
        print(f"ERROR: quantile_pred отсутствует. Ребилд V3: py -m ml.retrain_all --rebuild ensemble")
        sys.exit(2)

    data = {k: npz[k] for k in npz.files}
    qpred = data["quantile_pred"]
    qparts = split_quantiles_ohlc(qpred)
    has_full = qpred.shape[1] >= 4 * len(QUANTILES) * FUTURE_BARS
    print(f"  quantile_pred shape: {qpred.shape}  "
          f"({'OHLC 4-channel' if has_full else 'L/H 2-channel legacy'})")

    if args.indices:
        indices = list(args.indices)
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(qpred), size=args.examples, replace=False).tolist()

    os.makedirs(args.out, exist_ok=True)

    if args.grid:
        out = os.path.join(args.out, f"quantile_grid_{len(indices)}.png")
        plot_grid(indices, data, qparts, out)
        print(f"  → {out}")
    else:
        for i, idx in enumerate(indices):
            ticker = str(data["test_tickers"][idx])
            date = str(data["test_dates"][idx])
            fname = f"q_{i+1:02d}_{ticker}_{date}_idx{idx}.png"
            out = os.path.join(args.out, fname)
            plot_example(idx, data, qparts, out)
            print(f"  → {out}")


if __name__ == "__main__":
    main()
