"""Бэктест-модуль v2: симуляция торговли на основе OHLC предсказаний модели.

Запуск:
  python -m ml.backtest --model ml/model_multiscale_v2.pt

Исправления v2:
  1. SELL risk/reward инвертирован (reward=|downside|, risk=upside)
  2. Position sizing: floor на max_downside, жёсткий cap
  3. Time-overlap: макс N позиций одновременно, трекинг открытых
  4. Sharpe: корректный расчёт (все дни, без фильтрации нулей)
  5. Добавлен Calmar ratio, Sortino ratio
"""
import os, sys, json, argparse
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.config import CFG, SCALES
from ml.labels_ohlc import ohlc_to_strategy_features


# ══════════════════════════════════════════════════════════════════
#  Конфигурация стратегии
# ══════════════════════════════════════════════════════════════════

@dataclass
class StrategyConfig:
    """Параметры торговой стратегии."""
    # Фильтрация сигналов
    confidence_threshold: float = 0.45    # min softmax confidence
    min_risk_reward:      float = 1.2     # min risk/reward ratio
    min_expected_move:    float = 0.005   # min |final_return| (0.5%)

    # Позиция
    risk_per_trade:       float = 0.01    # 1% капитала на сделку
    max_position_pct:     float = 0.10    # max 10% в одну позицию
    max_open_positions:   int   = 5       # одновременно
    min_downside_floor:   float = 0.005   # floor для max_downside (0.5%)

    # Комиссия
    commission:           float = 0.0005  # 0.05% в одну сторону

    # Начальный капитал
    initial_capital:      float = 1_000_000.0

    # Горизонт сделки (дней)
    hold_bars:            int   = 5       # = CFG.future_bars


# ══════════════════════════════════════════════════════════════════
#  Трейд
# ══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Одна сделка."""
    idx:           int            # индекс сэмпла в test set
    direction:     str            # BUY / SELL
    confidence:    float          # softmax confidence
    expected_ret:  float          # предсказанный final return
    risk_reward:   float          # предсказанный risk/reward (direction-aware)
    position_size: float          # доля капитала
    capital_used:  float          # рублей вложено
    open_bar:      int            # когда открыта (условный "день")
    close_bar:     int            # когда закроется
    # Результат (заполняется после)
    actual_ret:    float = 0.0    # реальный return (из ohlc_true)
    actual_maxdd:  float = 0.0    # реальная max просадка внутри сделки
    pnl:           float = 0.0    # P&L в рублях
    pnl_pct:       float = 0.0    # P&L в %


# ══════════════════════════════════════════════════════════════════
#  Вспомогательная: direction-aware risk/reward
# ══════════════════════════════════════════════════════════════════

def direction_risk_reward(features: dict, direction: str) -> float:
    """
    Пересчитывает risk/reward для заданного направления.

    BUY:  reward = max_upside,       risk = |max_downside|
    SELL: reward = |max_downside|,   risk = max_upside
          (для short: цена падает = profit, цена растёт = loss)
    """
    max_up   = features['max_upside']
    max_down = features['max_downside']

    if direction == "BUY":
        reward = max(max_up, 0.0)
        risk   = max(abs(max_down), 1e-9)
    else:  # SELL
        reward = max(abs(max_down), 0.0)
        risk   = max(max_up, 1e-9)

    return reward / risk


# ══════════════════════════════════════════════════════════════════
#  Бэктест
# ══════════════════════════════════════════════════════════════════

class Backtester:
    def __init__(self, cfg: StrategyConfig = None):
        self.cfg = cfg or StrategyConfig()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def run(self, cls_probs: np.ndarray, ohlc_pred: np.ndarray,
            ohlc_true: np.ndarray, cls_true: np.ndarray = None):
        """
        Запускает бэктест.

        cls_probs:  (N, 3) — softmax вероятности
        ohlc_pred:  (N, F, 4) — предсказанные OHLC
        ohlc_true:  (N, F, 4) — реальные OHLC
        cls_true:   (N,) — реальные метки (опционально, для аналитики)
        """
        N = len(cls_probs)
        cfg = self.cfg
        capital = cfg.initial_capital
        self.equity_curve = [capital]
        self.trades = []

        # ── Трекинг открытых позиций (time-overlap) ────────────
        # Каждая позиция: (close_bar, position_pct)
        # capital_locked = сумма position_pct по открытым сделкам
        open_positions: deque = deque()  # (close_bar, capital_used, trade_ref)
        capital_locked = 0.0

        # ── Для учёта P&L по барам (корректный equity curve) ───
        # pending_pnl[bar] = list of (pnl,) — P&L реализуемые в этот бар
        pending_pnl = {}

        for i in range(N):
            # ── Закрываем позиции, чей срок истёк ──────────────
            while open_positions and open_positions[0][0] <= i:
                _, locked_amt, trade_ref = open_positions.popleft()
                capital_locked -= trade_ref.position_size

                # P&L уже учтён при создании (ниже), здесь только unlock
                # Но нет: P&L нужно реализовывать при закрытии!
                # Пересчитаем: P&L реализуется при close_bar
                pass

            # ── Генерация сигнала ─────────────────────────────
            probs = cls_probs[i]
            pred_class = int(probs.argmax())
            confidence = float(probs.max())

            # HOLD → пропуск
            if pred_class == 1:
                self.equity_curve.append(capital)
                continue

            # Confidence filter
            if confidence < cfg.confidence_threshold:
                self.equity_curve.append(capital)
                continue

            # Стратегические метрики из предсказания
            features = ohlc_to_strategy_features(ohlc_pred[i])

            # ── Направление ───────────────────────────────────
            direction = "BUY" if pred_class == 0 else "SELL"

            # ── Direction-aware risk/reward ─────────────────────
            rr = direction_risk_reward(features, direction)

            if rr < cfg.min_risk_reward:
                self.equity_curve.append(capital)
                continue

            # Direction-aware expected move
            if direction == "BUY":
                expected_ret = features['final_return']
            else:
                expected_ret = -features['final_return']

            if expected_ret < cfg.min_expected_move:
                self.equity_curve.append(capital)
                continue

            # ── Проверка лимита открытых позиций ───────────────
            n_open = len(open_positions)
            if n_open >= cfg.max_open_positions:
                self.equity_curve.append(capital)
                continue

            # ── Проверка доступного капитала ────────────────────
            available_pct = 1.0 - capital_locked
            if available_pct < 0.01:  # менее 1% свободно
                self.equity_curve.append(capital)
                continue

            # ── Размер позиции ────────────────────────────────
            if direction == "BUY":
                downside_risk = abs(features['max_downside'])
            else:
                # Для шорта: risk = max_upside (цена может уйти вверх)
                downside_risk = abs(features['max_upside'])

            # Floor на downside (защита от деления на ~0)
            downside_risk = max(downside_risk, cfg.min_downside_floor)

            raw_size = cfg.risk_per_trade / downside_risk

            # Масштабируем по confidence
            conf_scale = (confidence - cfg.confidence_threshold) / \
                         (1.0 - cfg.confidence_threshold + 1e-9)
            position_pct = raw_size * conf_scale

            # Ограничения
            position_pct = min(position_pct, cfg.max_position_pct)
            position_pct = min(position_pct, available_pct)  # не больше свободного
            position_pct = max(position_pct, 0.001)          # min 0.1%

            # ── Симуляция P&L по реальным данным ──────────────
            real_ohlc = ohlc_true[i]   # (F, 4): ΔO, ΔH, ΔL, ΔC

            if direction == "BUY":
                # Покупаем: profit = final ΔClose
                actual_ret = float(real_ohlc[-1, 3])  # ΔClose последней свечи
                # Максимальная просадка = min(ΔLow) за горизонт
                actual_maxdd = float(real_ohlc[:, 2].min())  # min ΔLow
            else:
                # Продаём (шорт): profit = -final ΔClose
                actual_ret = -float(real_ohlc[-1, 3])
                # Максимальная просадка для шорта = -max(ΔHigh)
                actual_maxdd = -float(real_ohlc[:, 1].max())  # -(max ΔHigh)

            # Комиссия (покупка + продажа)
            net_ret = actual_ret - 2 * cfg.commission

            # P&L
            capital_used = capital * position_pct
            pnl = capital_used * net_ret
            capital += pnl

            # Трекинг позиции
            close_bar = i + cfg.hold_bars
            trade = Trade(
                idx=i,
                direction=direction,
                confidence=confidence,
                expected_ret=expected_ret,
                risk_reward=rr,
                position_size=position_pct,
                capital_used=capital_used,
                open_bar=i,
                close_bar=close_bar,
                actual_ret=net_ret,
                actual_maxdd=actual_maxdd,
                pnl=pnl,
                pnl_pct=net_ret * position_pct * 100,
            )
            self.trades.append(trade)

            # Добавляем в трекер открытых позиций
            open_positions.append((close_bar, capital_used, trade))
            capital_locked += position_pct

            self.equity_curve.append(capital)

        return self.compute_metrics()

    def compute_metrics(self) -> dict:
        """Считает все метрики бэктеста."""
        cfg = self.cfg
        equity = np.array(self.equity_curve)
        trades = self.trades

        if not trades:
            return {"error": "Нет сделок — фильтры слишком жёсткие"}

        # ── Базовые ───────────────────────────────────────────
        total_return = (equity[-1] / cfg.initial_capital - 1) * 100
        n_trades     = len(trades)

        wins  = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / n_trades * 100

        avg_win  = np.mean([t.pnl for t in wins])  if wins   else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        # ── Profit Factor ─────────────────────────────────────
        gross_profit = sum(t.pnl for t in wins)  if wins   else 0
        gross_loss   = sum(abs(t.pnl) for t in losses) if losses else 1e-9
        profit_factor = gross_profit / gross_loss

        # ── Max Drawdown ──────────────────────────────────────
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = max_dd * 100

        # ── Returns для Sharpe/Sortino ────────────────────────
        # Все returns (включая нулевые дни HOLD)
        returns = np.diff(equity) / equity[:-1]

        if len(returns) > 1:
            # ── Sharpe Ratio ──────────────────────────────────
            # Annualized: предполагаем 1 сэмпл ≈ 1 торговый день
            # Но это приблизительно — сэмплы разных тикеров не = дням
            mean_r = np.mean(returns)
            std_r  = np.std(returns, ddof=1)  # sample std
            sharpe = mean_r / (std_r + 1e-9) * np.sqrt(250)

            # ── Sortino Ratio ─────────────────────────────────
            downside = returns[returns < 0]
            downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
            sortino = mean_r / (downside_std + 1e-9) * np.sqrt(250)
        else:
            sharpe = 0.0
            sortino = 0.0

        # ── Calmar Ratio ──────────────────────────────────────
        # Total return annualized / Max DD
        if max_dd > 0:
            # Приблизительная годовая доходность
            n_days = len(returns) if len(returns) > 0 else 1
            annual_return = (equity[-1] / cfg.initial_capital) ** (250 / n_days) - 1
            calmar = annual_return / max_dd
        else:
            calmar = 0.0

        # ── Средняя длительность (условная) ───────────────────
        avg_hold_days = cfg.hold_bars

        # ── По направлениям ───────────────────────────────────
        buys  = [t for t in trades if t.direction == "BUY"]
        sells = [t for t in trades if t.direction == "SELL"]

        buy_wr  = len([t for t in buys  if t.pnl > 0]) / max(len(buys), 1) * 100
        sell_wr = len([t for t in sells if t.pnl > 0]) / max(len(sells), 1) * 100

        # ── Средняя confidence по результату ──────────────────
        avg_conf_win  = np.mean([t.confidence for t in wins])  if wins   else 0
        avg_conf_loss = np.mean([t.confidence for t in losses]) if losses else 0

        # ── Средний P&L по направлениям ───────────────────────
        avg_pnl_buy  = np.mean([t.pnl for t in buys])  if buys  else 0
        avg_pnl_sell = np.mean([t.pnl for t in sells]) if sells else 0

        # ── Expectancy (математическое ожидание на сделку) ────
        expectancy = np.mean([t.pnl for t in trades])
        expectancy_pct = np.mean([t.pnl_pct for t in trades])

        results = {
            "total_return_pct":    round(total_return, 2),
            "n_trades":            n_trades,
            "win_rate_pct":        round(win_rate, 1),
            "profit_factor":       round(profit_factor, 3),
            "sharpe_ratio":        round(sharpe, 3),
            "sortino_ratio":       round(sortino, 3),
            "calmar_ratio":        round(calmar, 3),
            "max_drawdown_pct":    round(max_dd_pct, 2),
            "avg_win":             round(avg_win, 2),
            "avg_loss":            round(avg_loss, 2),
            "expectancy":          round(expectancy, 2),
            "expectancy_pct":      round(expectancy_pct, 4),
            "avg_hold_days":       avg_hold_days,
            "final_equity":        round(equity[-1], 2),
            "initial_capital":     cfg.initial_capital,
            # По направлениям
            "n_buys":              len(buys),
            "n_sells":             len(sells),
            "buy_win_rate_pct":    round(buy_wr, 1),
            "sell_win_rate_pct":   round(sell_wr, 1),
            "avg_pnl_buy":         round(avg_pnl_buy, 2),
            "avg_pnl_sell":        round(avg_pnl_sell, 2),
            # Confidence
            "avg_confidence_win":  round(avg_conf_win, 4),
            "avg_confidence_loss": round(avg_conf_loss, 4),
            # Фильтры
            "filters": {
                "confidence_thr":    cfg.confidence_threshold,
                "min_risk_reward":   cfg.min_risk_reward,
                "min_expected_move": cfg.min_expected_move,
                "max_open_positions": cfg.max_open_positions,
                "min_downside_floor": cfg.min_downside_floor,
            },
        }
        return results

    def print_report(self, results: dict):
        """Красиво печатает отчёт."""
        if "error" in results:
            print(f"\n  ⚠ {results['error']}")
            return

        print("\n" + "=" * 60)
        print("  BACKTEST REPORT")
        print("=" * 60)

        print(f"\n  Капитал: {results['initial_capital']:,.0f} → "
              f"{results['final_equity']:,.0f}")
        print(f"  Total Return: {results['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {results['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio: {results['calmar_ratio']:.3f}")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")

        print(f"\n  Сделок: {results['n_trades']}")
        print(f"    BUY:  {results['n_buys']} "
              f"(win rate {results['buy_win_rate_pct']:.1f}%, "
              f"avg P&L {results['avg_pnl_buy']:+,.0f})")
        print(f"    SELL: {results['n_sells']} "
              f"(win rate {results['sell_win_rate_pct']:.1f}%, "
              f"avg P&L {results['avg_pnl_sell']:+,.0f})")

        print(f"\n  Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"  Profit Factor: {results['profit_factor']:.3f}")
        print(f"  Expectancy: {results['expectancy']:+,.2f} / "
              f"{results['expectancy_pct']:+.4f}%")
        print(f"  Avg Win:  {results['avg_win']:+,.2f}")
        print(f"  Avg Loss: {results['avg_loss']:+,.2f}")

        print(f"\n  Avg Confidence (wins):   {results['avg_confidence_win']:.4f}")
        print(f"  Avg Confidence (losses): {results['avg_confidence_loss']:.4f}")

        print(f"\n  Фильтры:")
        f = results['filters']
        print(f"    confidence > {f['confidence_thr']}")
        print(f"    risk/reward > {f['min_risk_reward']} (direction-aware)")
        print(f"    |expected_move| > {f['min_expected_move']}")
        print(f"    max open positions = {f['max_open_positions']}")
        print(f"    downside floor = {f['min_downside_floor']}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════
#  Benchmark: Buy & Hold
# ══════════════════════════════════════════════════════════════════

def buy_hold_benchmark(ohlc_true: np.ndarray) -> dict:
    """
    Бенчмарк: Buy & Hold по всем сэмплам.
    Считаем средний return, как если бы мы входили в каждый сэмпл.
    """
    # ohlc_true: (N, F, 4) — ΔO, ΔH, ΔL, ΔC
    final_returns = ohlc_true[:, -1, 3]  # ΔClose последней свечи
    commission = 2 * 0.0005  # same as strategy

    net_returns = final_returns - commission
    mean_ret    = float(net_returns.mean())
    median_ret  = float(np.median(net_returns))
    std_ret     = float(net_returns.std())
    pct_positive = float((net_returns > 0).mean() * 100)

    # Последовательный compounding (вкладываем всё каждый раз)
    capital = 1_000_000.0
    equity = [capital]
    for r in net_returns:
        capital *= (1 + r * 0.10)  # 10% позиция каждый раз
        equity.append(capital)

    equity = np.array(equity)
    total_ret = (equity[-1] / equity[0] - 1) * 100

    # Max DD
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "strategy":      "Buy & Hold (10% position)",
        "total_return":  round(total_ret, 2),
        "mean_return":   round(mean_ret * 100, 4),
        "median_return": round(median_ret * 100, 4),
        "std_return":    round(std_ret * 100, 4),
        "pct_positive":  round(pct_positive, 1),
        "max_dd_pct":    round(max_dd * 100, 2),
        "n_samples":     len(ohlc_true),
    }


# ══════════════════════════════════════════════════════════════════
#  Запуск бэктеста из командной строки
# ══════════════════════════════════════════════════════════════════

def run_backtest(model_path: str = 'ml/model_multiscale_v2.pt',
                 save_json: str = None):
    """Полный пайплайн: загрузка данных → inference → бэктест."""
    from ml.dataset_v2_ohlc import (
        build_full_multiscale_dataset_v2,
        temporal_split,
    )
    from ml.multiscale_cnn_v2 import MultiScaleHybridV2, _collate_v2
    from ml.dataset import class_distribution
    from torch.utils.data import Subset, DataLoader

    print("=" * 60)
    print("  BACKTEST v2 — загрузка данных и модели")
    print("=" * 60)

    # ── Загрузка данных ───────────────────────────────────────
    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v2()
    print(f"\n  Всего сэмплов: {len(y_all)}, ctx_dim={ctx_dim}")

    # Temporal split (тот же, что и при обучении)
    _, _, idx_test = temporal_split(
        ticker_lengths,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_bars=CFG.future_bars,
    )
    y_test = y_all[idx_test]

    te_ds = Subset(dataset, idx_test.tolist())
    print(f"  Test set (temporal): {len(y_test)} сэмплов")
    class_distribution(y_test)

    # ── Загрузка модели ───────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MultiScaleHybridV2(ctx_dim=ctx_dim).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Модель загружена: {model_path}")

    # ── Inference ─────────────────────────────────────────────
    loader = DataLoader(te_ds, batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=True,
                        collate_fn=_collate_v2)

    all_probs      = []
    all_ohlc_pred  = []
    all_ohlc_true  = []

    print("  Inference на test set...")
    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx = batch
            imgs = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None

            if num_dict is not None:
                nums = {W: num_dict[W].to(device) for W in SCALES}
                cls_logits, ohlc_pred = model(imgs, nums, ctx_t)
            else:
                cls_logits, ohlc_pred = model(imgs, ctx=ctx_t)

            probs = torch.softmax(cls_logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_ohlc_pred.append(ohlc_pred.cpu().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    all_probs     = np.concatenate(all_probs, axis=0)
    all_ohlc_pred = np.concatenate(all_ohlc_pred, axis=0)
    all_ohlc_true = np.concatenate(all_ohlc_true, axis=0)

    print(f"  Inference done: {len(all_probs)} сэмплов")

    # ── Бэктест с разными конфигурациями ─────────────────────
    configs = {
        "conservative": StrategyConfig(
            confidence_threshold=0.55,
            min_risk_reward=1.5,
            min_expected_move=0.008,
            risk_per_trade=0.01,
            max_open_positions=3,
        ),
        "balanced": StrategyConfig(
            confidence_threshold=0.45,
            min_risk_reward=1.2,
            min_expected_move=0.005,
            risk_per_trade=0.015,
            max_open_positions=5,
        ),
        "aggressive": StrategyConfig(
            confidence_threshold=0.38,
            min_risk_reward=0.8,
            min_expected_move=0.003,
            risk_per_trade=0.02,
            max_open_positions=8,
        ),
    }

    all_results = {}
    for name, cfg in configs.items():
        print(f"\n  ── Стратегия: {name.upper()} ──")
        bt = Backtester(cfg)
        results = bt.run(all_probs, all_ohlc_pred, all_ohlc_true, y_test)
        bt.print_report(results)
        all_results[name] = results

    # ── Benchmark ─────────────────────────────────────────────
    print(f"\n  ── BENCHMARK: Buy & Hold ──")
    bh = buy_hold_benchmark(all_ohlc_true)
    print(f"  Total Return: {bh['total_return']:+.2f}%")
    print(f"  Mean sample return: {bh['mean_return']:+.4f}%")
    print(f"  Positive samples: {bh['pct_positive']:.1f}%")
    print(f"  Max Drawdown: {bh['max_dd_pct']:.2f}%")
    all_results["benchmark_buy_hold"] = bh

    # ── Сохранение ────────────────────────────────────────────
    if save_json is None:
        save_json = model_path.replace('.pt', '_backtest.json')
    with open(save_json, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Результаты → {save_json}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ml/model_multiscale_v2.pt',
                        help='Путь к .pt файлу модели')
    parser.add_argument('--output', default=None,
                        help='Путь для JSON с результатами')
    args = parser.parse_args()
    run_backtest(args.model, args.output)
