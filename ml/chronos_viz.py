"""Sprint 11.3 — matplotlib визуализация: custom OHLC quantiles + Chronos close band.

Расширение `quantile_viz`: к стандартному отрисовыванию predicted L/O/C/H bands
добавляет Chronos close-band (узкая колонка справа от custom-bands). Видно где
Chronos шире/уже custom C-канала, видно покрывает ли actual close.

Запуск:
    py -m ml.chronos_viz                              # 5 random, in ml/viz/
    py -m ml.chronos_viz --examples 8 --out ml/viz/
    py -m ml.chronos_viz --indices 100 1500 8000      # конкретные сэмплы
    py -m ml.chronos_viz --only-with-chronos           # пропускать has_chronos=False
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

from ml.quantile_viz import (
    NPZ_PATH, QUANTILES, FUTURE_BARS, CANDLE_BODY_W, GROUP_GAP,
    split_quantiles_ohlc, draw_candle, draw_quantile_band, _channel_colors,
)

CHRONOS_COLOR = "#ff9800"   # orange — chronos is different "voice"


def plot_with_chronos(idx: int, data: dict, qparts: dict,
                      out_path: str | None = None):
    """OHLC candles + custom quantile bands + chronos close band на одном графике."""
    ohlc = data["ohlc_test"][idx].reshape(FUTURE_BARS, 4)
    ticker = str(data["test_tickers"][idx])
    date = str(data["test_dates"][idx])
    y = int(data["y_test"][idx])
    p_up_raw  = float(data["dir_prob"][idx])
    p_up_meta = (float(data["meta_dir_prob"][idx])
                 if "meta_dir_prob" in data and np.isfinite(data["meta_dir_prob"][idx])
                 else None)
    atr_r = float(data["atr_ratio"][idx])
    cls_name = {0: "UP", 1: "FLAT", 2: "DOWN"}.get(y, "?")
    colors = _channel_colors()

    # Chronos data (может отсутствовать)
    has_chronos = bool(data["has_chronos"][idx]) if "has_chronos" in data else False
    chr_q10 = data["chronos_close_q10"][idx] if has_chronos else None
    chr_q50 = data["chronos_close_q50"][idx] if has_chronos else None
    chr_q90 = data["chronos_close_q90"][idx] if has_chronos else None
    chr_width_avg = (float(np.nanmean(chr_q90 - chr_q10))
                     if has_chronos else None)

    has_full_ohlc = not np.allclose(qparts["O"][idx], qparts["L"][idx])

    fig, ax = plt.subplots(1, 1, figsize=(14, 6.5))
    bars_x = np.arange(FUTURE_BARS)

    for b in range(FUTURE_BARS):
        x_actual = b - GROUP_GAP * 1.3
        O_a, H_a, L_a, C_a = ohlc[b]

        # ── Actual candle (left) ──
        draw_candle(ax, x_actual, O_a, H_a, L_a, C_a)

        # ── Custom quantile bands (center): 4 columns L/O/C/H ──
        if has_full_ohlc:
            ch_order = ["L", "O", "C", "H"]
            spacing  = 0.07
            base_x   = b + GROUP_GAP * 0.3
            for ci, ch in enumerate(ch_order):
                xc = base_x + ci * spacing
                q10, q50, q90 = qparts[ch][idx, :, b]
                draw_quantile_band(
                    ax, xc, q10, q50, q90, color=colors[ch], width=0.06,
                    label=f"q_{ch}_custom" if b == 0 else None,
                    alpha=0.35,
                )

        # ── Chronos close band (right of custom bands) ──
        if has_chronos:
            x_chr = b + GROUP_GAP * 0.3 + 4 * 0.07 + 0.05    # справа от H column
            cq10 = float(chr_q10[b])
            cq50 = float(chr_q50[b])
            cq90 = float(chr_q90[b])
            draw_quantile_band(
                ax, x_chr, cq10, cq50, cq90, color=CHRONOS_COLOR, width=0.10,
                label="q_C_chronos" if b == 0 else None,
                alpha=0.40,
            )
            # отметка: попадает ли actual_C в chronos band?
            chr_hit = cq10 <= C_a <= cq90
            edge = "#888" if chr_hit else CHRONOS_COLOR
            ax.plot([x_chr], [C_a], marker="*", color=edge, markersize=10,
                    markeredgecolor="#000", markeredgewidth=0.5,
                    alpha=0.95, zorder=6,
                    label="actual C (chronos hit)" if b == 0 and chr_hit else (
                          "actual C (chronos MISS)" if b == 0 and not chr_hit else None))

        # ── Markers actual O/H/L/C поверх actual candle ──
        in_range = lambda v, c: qparts[c][idx, 0, b] <= v <= qparts[c][idx, 2, b]
        for actual_v, ch_name, marker in [
            (H_a, "H", "^"),
            (L_a, "L", "v"),
            (O_a, "O", "o"),
            (C_a, "C", "s"),
        ]:
            ok = in_range(actual_v, ch_name) if has_full_ohlc else True
            edge = "#888" if ok else colors[ch_name]
            ax.plot([x_actual + 0.12], [actual_v], marker=marker, color=edge,
                    markersize=6, markeredgecolor="#000", markeredgewidth=0.4,
                    alpha=0.85, zorder=5)

    for b in range(FUTURE_BARS):
        ax.axvline(b, color="#aaa", linewidth=0.3, linestyle=":", zorder=1)
    ax.axhline(0, color="#666", linewidth=0.6, linestyle="-", zorder=1)

    ax.set_xlim(-0.65, FUTURE_BARS - 0.20)
    ax.set_xticks(bars_x)
    ax.set_xticklabels([f"t+{i+1}" for i in range(FUTURE_BARS)])
    ax.set_xlabel("Future bar  | left: actual candle · center: custom L/O/C/H quantiles · right (orange): Chronos close")
    ax.set_ylabel("ATR-normalized price")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    # Title с meta_dir_prob если есть
    title = f"{ticker}  {date}  y_true={cls_name}  raw_p_up={p_up_raw:.3f}"
    if p_up_meta is not None:
        title += f"  meta_p_up={p_up_meta:.3f}"
    title += f"  atr={atr_r:.4f}"
    if has_chronos:
        title += f"\nChronos: width_avg={chr_width_avg:.2f} ATR (uncertainty)"
    else:
        title += "  ⚠ NO CHRONOS (fallback)"
    ax.set_title(title, fontsize=10)

    # Legend
    handles = [
        mpatches.Patch(facecolor="#26a69a", edgecolor="#000", label="actual bull"),
        mpatches.Patch(facecolor="#ef5350", edgecolor="#000", label="actual bear"),
    ]
    if has_full_ohlc:
        for ch, col in colors.items():
            handles.append(mpatches.Patch(facecolor=col, alpha=0.35,
                                          label=f"q_{ch}_custom"))
    if has_chronos:
        handles.append(mpatches.Patch(facecolor=CHRONOS_COLOR, alpha=0.40,
                                      label="q_C_chronos (zero-shot/LoRA)"))
        handles.append(plt.Line2D([0], [0], marker="*", color=CHRONOS_COLOR,
                                  markeredgecolor="#000", markersize=10,
                                  linestyle="None", label="actual C vs chronos"))
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


def plot_grid_with_chronos(indices: list[int], data: dict, qparts: dict,
                           out_path: str):
    """Grid PNG — упрощённый: только custom C + chronos C-band, без O/H/L."""
    n = len(indices)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 5 * rows), squeeze=False)
    colors = _channel_colors()

    for plot_i, idx in enumerate(indices):
        ax = axes[plot_i // cols][plot_i % cols]
        ohlc = data["ohlc_test"][idx].reshape(FUTURE_BARS, 4)
        ticker = str(data["test_tickers"][idx])
        date = str(data["test_dates"][idx])
        y = int(data["y_test"][idx])
        cls_name = {0: "UP", 1: "FLAT", 2: "DOWN"}.get(y, "?")
        has_chronos = bool(data["has_chronos"][idx]) if "has_chronos" in data else False

        for b in range(FUTURE_BARS):
            x_actual = b - GROUP_GAP
            O, H, L, C = ohlc[b]
            draw_candle(ax, x_actual, O, H, L, C)

            # Только custom C-канал (center)
            x_cc = b + GROUP_GAP * 0.4
            q10, q50, q90 = qparts["C"][idx, :, b]
            draw_quantile_band(ax, x_cc, q10, q50, q90, colors["C"],
                               width=0.10, alpha=0.40)

            # Chronos close (right)
            if has_chronos:
                x_chr = b + GROUP_GAP * 0.4 + 0.15
                cq10 = float(data["chronos_close_q10"][idx, b])
                cq50 = float(data["chronos_close_q50"][idx, b])
                cq90 = float(data["chronos_close_q90"][idx, b])
                draw_quantile_band(ax, x_chr, cq10, cq50, cq90, CHRONOS_COLOR,
                                   width=0.10, alpha=0.45)

        ax.axhline(0, color="#666", linewidth=0.5)
        ax.set_xlim(-0.55, FUTURE_BARS - 0.10)
        ax.set_xticks(np.arange(FUTURE_BARS))
        ax.set_xticklabels([f"t+{i+1}" for i in range(FUTURE_BARS)])
        chr_tag = "✓chr" if has_chronos else "✗chr"
        ax.set_title(f"{ticker} {date}  y={cls_name}  {chr_tag}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    for plot_i in range(n, rows * cols):
        axes[plot_i // cols][plot_i % cols].axis("off")

    fig.suptitle(f"Custom C (green) vs Chronos close (orange)  —  {n} примеров",
                 fontsize=13)

    # Общий legend
    handles = [
        mpatches.Patch(facecolor="#26a69a", edgecolor="#000", label="actual bull"),
        mpatches.Patch(facecolor="#ef5350", edgecolor="#000", label="actual bear"),
        mpatches.Patch(facecolor=colors["C"], alpha=0.40, label="q_C_custom [0.1, 0.9]"),
        mpatches.Patch(facecolor=CHRONOS_COLOR, alpha=0.45,
                       label="q_C_chronos [0.1, 0.9]"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
                        help="Один PNG grid вместо отдельных файлов")
    parser.add_argument("--only-with-chronos", action="store_true",
                        help="Пропустить сэмплы без has_chronos (~21 percent)")
    args = parser.parse_args()

    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден"); sys.exit(1)

    npz = np.load(NPZ_PATH, allow_pickle=False)
    if "quantile_pred" not in npz.files:
        print("ERROR: quantile_pred отсутствует. Ребилд V3"); sys.exit(2)
    if "chronos_close_q50" not in npz.files:
        print("WARN: chronos_close_q* отсутствует. Сначала:")
        print("  py -m ml.chronos_quantile_pred --max-samples 0")
        print("  py -m ml.merge_chronos_into_npz")
        # Не падаем — просто рисуем без chronos overlay
    data = {k: npz[k] for k in npz.files}
    qpred = data["quantile_pred"]
    qparts = split_quantiles_ohlc(qpred)
    has_full = qpred.shape[1] >= 4 * len(QUANTILES) * FUTURE_BARS
    n_total = len(qpred)
    print(f"  quantile_pred: {qpred.shape} ({'OHLC 4-ch' if has_full else 'L/H legacy'})")
    if "has_chronos" in data:
        n_chr = int(data["has_chronos"].sum())
        print(f"  chronos coverage: {n_chr}/{n_total} ({n_chr/n_total*100:.1f}%)")

    # Sample selection
    if args.indices:
        indices = list(args.indices)
    else:
        rng = np.random.default_rng(args.seed)
        if args.only_with_chronos and "has_chronos" in data:
            pool = np.where(data["has_chronos"])[0]
        else:
            pool = np.arange(n_total)
        if len(pool) < args.examples:
            print(f"WARN: pool size {len(pool)} < requested {args.examples}")
            args.examples = len(pool)
        indices = rng.choice(pool, size=args.examples, replace=False).tolist()

    os.makedirs(args.out, exist_ok=True)

    if args.grid:
        out = os.path.join(args.out, f"chronos_grid_{len(indices)}.png")
        plot_grid_with_chronos(indices, data, qparts, out)
        print(f"  → {out}")
    else:
        for i, idx in enumerate(indices):
            ticker = str(data["test_tickers"][idx])
            date = str(data["test_dates"][idx])
            fname = f"chr_{i+1:02d}_{ticker}_{date}_idx{idx}.png"
            out = os.path.join(args.out, fname)
            plot_with_chronos(idx, data, qparts, out)
            print(f"  → {out}")


if __name__ == "__main__":
    main()
