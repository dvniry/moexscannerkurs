"""Sprint 2: DecisionLayer — детерминированный cost-aware фильтр над предсказаниями.

Не обучается. Принимает выходы модели (dir_logit + EconomicHeads) и решает:
  signal[B] ∈ {0=BUY, 1=HOLD, 2=SELL}
  confidence[B] ∈ [0, +inf)

HOLD = "ожидаемый edge не покрывает костов после комиссий, спреда и проскальзывания".
Это не третий класс модели — это вывод критериального фильтра, поэтому модели не нужно
учиться предсказывать FLAT (что ранее давало class imbalance и UP recall=3%).

Векторно через torch boolean masks — без python for-loop по batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


SIG_BUY  = 0
SIG_HOLD = 1
SIG_SELL = 2


@dataclass
class TradingCosts:
    """Стоимости в долях цены (одна сторона). Дефолты согласованы с CFG."""
    commission: float = 0.0005
    slippage:   float = 0.0003
    spread:     float = 0.0002

    @property
    def roundtrip(self) -> float:
        """Полный round-trip cost: вход + выход."""
        return 2.0 * (self.commission + self.slippage + self.spread)


def costs_from_config() -> TradingCosts:
    """Берёт costs из ml.config.CFG (econ_commission/slippage/spread)."""
    from ml.config import CFG
    return TradingCosts(
        commission=CFG.econ_commission,
        slippage=CFG.econ_slippage,
        spread=CFG.econ_spread,
    )


class DecisionLayer:
    """Cost-aware фильтр поверх EconomicHeads.

    Условие BUY:
      P(UP)            >= min_dir_prob       (направление уверенно)
      fill_long        >= min_fill_prob      (лимитка скорее исполнится)
      mfe_long/mae_long >= min_rr            (асимметрия в нашу пользу)
      net_edge_long    >= min_edge_ratio * cost_roundtrip   (покроет костов)

    Симметрично для SELL. Если оба ok — выбираем по dir_prob.

    Sprint 8.1/8.2 (extremes): если передан extremes [N, 2|3] — первые две колонки
    (pred_high, pred_low) в ATR-units. Третья колонка (high_first_logit, S8-2)
    игнорируется здесь и используется только в `simulate_path_aware_strategy`.
    range_pred = high - low; при range_pred > min_range_threshold добавляется
    бонус к edge_pred: edge_adj = edge + extremes_weight * clip(range - threshold, 0, 0.1).
    """
    def __init__(
        self,
        costs: Optional[TradingCosts] = None,
        # B-15 (sweep 2026-05-01): оптимальные пороги по expectancy.
        # До sweep: edge_r=4.0, dir=0.70, sell=0.85 → exp=-1.84%/trade (убыточно).
        # После sweep:                                  exp=+1.43%/trade (прибыльно).
        min_edge_ratio:      float = 5.0,    # 4.0 → 5.0 (edge_pred ~0, нужен жёстче cut)
        min_dir_prob:        float = 0.75,   # 0.70 → 0.75 (более уверенный BUY)
        min_sell_dir_prob:   float = 0.55,   # 0.85 → 0.55: 0.85 отбрасывал 76% SELL.
                                             # 0.55 → dir_prob ≤ 0.45 (вместо ≤ 0.15)
        min_fill_prob:       float = 0.40,
        min_rr:              float = 1.2,
        # Sprint 8.1: extremes head support
        extremes_weight:     float = 0.15,   # вес edge-бонуса от predicted range
        min_range_threshold: float = 0.005,  # мин. диапазон (доля) для активации бонуса
    ):
        self.costs = costs or TradingCosts()
        self.min_edge_ratio    = float(min_edge_ratio)
        self.min_dir_prob      = float(min_dir_prob)
        self.min_sell_dir_prob = float(min_sell_dir_prob)
        self.min_fill_prob     = float(min_fill_prob)
        self.min_rr            = float(min_rr)
        self.extremes_weight     = float(extremes_weight)
        self.min_range_threshold = float(min_range_threshold)

    @torch.no_grad()
    def decide(self, dir_logit: torch.Tensor, econ: dict,
               extremes: Optional[torch.Tensor] = None) -> dict:
        """
        dir_logit: [B] — logit из dir_head (перед sigmoid)
        econ: dict из EconomicHeads.forward с ключами:
            mfe_mae    [B, 4]  (mfe_l, mae_l, mfe_s, mae_s)
            fill_logit [B, 2]  (fill_l, fill_s) — logits
            edge_pred  [B, 2]  (edge_l, edge_s) — net edge prediction
        extremes: [B, 2|3] optional — (pred_high, pred_low[, high_first_logit])
            в ATR-нормированных единицах. Sprint 8.1: range_pred = high - low
            используется как бонус к edge. Sprint 8.2: 3-я колонка high_first_logit
            читается только в `simulate_path_aware_strategy`, здесь игнорируется.

        Returns dict:
            signal     [B] long  — 0/1/2
            confidence [B] float — для sizing
            dir_prob, rr_long, rr_short, fill_long, fill_short, edge_ratio_*
        """
        dir_prob = torch.sigmoid(dir_logit.float())            # [B]

        mfe_mae  = econ["mfe_mae"].float()                      # [B, 4]
        fill_lg  = econ["fill_logit"].float()                   # [B, 2]
        edge_pr  = econ["edge_pred"].float()                    # [B, 2]

        mfe_l, mae_l, mfe_s, mae_s = mfe_mae.unbind(-1)
        fill_l, fill_s             = torch.sigmoid(fill_lg).unbind(-1)
        edge_l, edge_s             = edge_pr.unbind(-1)

        # Sprint 8.1: extremes бонус к edge_pred
        if extremes is not None and self.extremes_weight > 0:
            ext = extremes.float()
            range_pred  = (ext[:, 0] - ext[:, 1]).clamp(min=0.0)   # [B] high-low
            bonus = (self.extremes_weight
                     * (range_pred - self.min_range_threshold).clamp(0.0, 0.10))
            edge_l = edge_l + bonus
            edge_s = edge_s + bonus

        rr_l = mfe_l / mae_l.clamp_min(1e-6)
        rr_s = mfe_s / mae_s.clamp_min(1e-6)

        cost = self.costs.roundtrip
        er_l = edge_l / max(cost, 1e-9)
        er_s = edge_s / max(cost, 1e-9)

        long_ok = (
            (dir_prob >= self.min_dir_prob)
            & (fill_l  >= self.min_fill_prob)
            & (rr_l    >= self.min_rr)
            & (er_l    >= self.min_edge_ratio)
        )
        short_ok = (
            ((1.0 - dir_prob) >= self.min_sell_dir_prob)
            & (fill_s         >= self.min_fill_prob)
            & (rr_s           >= self.min_rr)
            & (er_s           >= self.min_edge_ratio)
        )

        # При текущих B-15 порогах (min_dir_prob=0.75, min_sell_dir_prob=0.55)
        # для коллизии нужно (1-dir_prob)≥0.55 И dir_prob≥0.75, что невозможно.
        # Оставляем приоритет long над short как safeguard на случай tuned порогов
        # (regime-aware sweep может занизить min_dir_prob → коллизия станет возможна).
        is_buy  = long_ok
        is_sell = short_ok & ~long_ok

        sig_long  = torch.full_like(dir_prob, SIG_BUY,  dtype=torch.long)
        sig_hold  = torch.full_like(dir_prob, SIG_HOLD, dtype=torch.long)
        sig_short = torch.full_like(dir_prob, SIG_SELL, dtype=torch.long)
        signal = torch.where(is_buy, sig_long,
                 torch.where(is_sell, sig_short, sig_hold))

        confidence = torch.where(
            is_buy,  dir_prob * rr_l.clamp(0., 5.),
            torch.where(is_sell, (1.0 - dir_prob) * rr_s.clamp(0., 5.),
                        torch.zeros_like(dir_prob))
        )

        return {
            "signal":     signal,
            "confidence": confidence,
            "dir_prob":   dir_prob,
            "rr_long":    rr_l,
            "rr_short":   rr_s,
            "fill_long":  fill_l,
            "fill_short": fill_s,
            "edge_ratio_long":  er_l,
            "edge_ratio_short": er_s,
            "long_ok":    long_ok,
            "short_ok":   short_ok,
        }

    def decide_numpy(
        self,
        dir_prob:   np.ndarray,              # [N] sigmoid вероятность UP (уже без logit)
        mfe_mae:    np.ndarray,              # [N, 4]
        fill_prob:  np.ndarray,              # [N, 2] уже sigmoid
        edge_pred:  np.ndarray,              # [N, 2]
        extremes:   Optional[np.ndarray] = None,  # [N, 2] (pred_high, pred_low) — Sprint 8.1
    ) -> dict:
        """Numpy-вариант для оффлайн-инференса (бэктест, export_csv)."""
        dir_logit = torch.from_numpy(np.log(np.clip(dir_prob, 1e-6, 1.0 - 1e-6)
                                            / np.clip(1 - dir_prob, 1e-6, 1.0 - 1e-6)))
        # fill_prob уже sigmoid → переводим обратно в logit для совместимости
        fill_logit = np.log(np.clip(fill_prob, 1e-6, 1 - 1e-6)
                            / np.clip(1 - fill_prob, 1e-6, 1 - 1e-6))
        econ = {
            "mfe_mae":    torch.from_numpy(mfe_mae.astype(np.float32)),
            "fill_logit": torch.from_numpy(fill_logit.astype(np.float32)),
            "edge_pred":  torch.from_numpy(edge_pred.astype(np.float32)),
        }
        ext_t = (torch.from_numpy(extremes.astype(np.float32))
                 if extremes is not None else None)
        out = self.decide(dir_logit, econ, extremes=ext_t)
        return {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}


class RegimeAwareDecisionLayer:
    """Sprint 5 / Idea #3: per-regime thresholds. Обновлено Sprint 9.5 (2026-05-04).

    ─────────────────────────────────────────────────────────────────────
    ИСТОРИЯ DEFAULT'ов
    ─────────────────────────────────────────────────────────────────────
    Sprint 5 (2026-05-01, in-sample sweep):
        bear (8%):  +0.375%/trade  edge=7.0/dir=0.80/sell=0.50   ✅
        side (44%): +0.006%/trade  edge=6.5/dir=0.55/sell=0.50   ✅
        bull (48%): −0.190%/trade  → отключаем (`bull=OFF`)
        Backtest: Sharpe +1.19, Total +1.61% — лучший результат проекта.

    Sprint 6 walk-forward (2026-05-02): Sprint 5 пороги дали coverage 0% OOS
    на 5 фолдах — пороги были оптимизированы под in-sample distribution.

    Sprint 9.5 (2026-05-04, новый sweep на 19067 сэмплов):
        bear (5%):  −0.284%/trade  edge=2.0/dir=0.80/sell=0.50   ⚠️
        side (42%): −0.164%/trade  edge=4.0/dir=0.80/sell=0.50   ⚠️
        bull (52%): −0.151%/trade  edge=5.0/dir=0.75/sell=0.50   ⚠️ enabled

    Все три режима показали отрицательный exp% на полной test-выборке —
    это согласуется с Reliability anal. (модель overconfident на 12-46%).
    Однако на fresh-window (последние 60 дней, 71.6% bull):
        bull: +0.021%/trade  edge=2.0/dir=0.80/sell=0.50         ✅
        side: +0.323%/trade  edge=5.0/dir=0.55/sell=0.50         ✅
    → значит distribution drift'ит, и `bull=OFF` (Sprint 5) точно устарел.

    ─────────────────────────────────────────────────────────────────────
    ТЕКУЩАЯ СТРАТЕГИЯ DEFAULT (Sprint 9.5)
    ─────────────────────────────────────────────────────────────────────
    "Least bad" пороги из полного sweep — НЕ best in-sample, а наиболее
    устойчивые на полных 19k сэмплов. Bull больше не отключён, т.к.:
      1. На full sweep bull least bad из трёх (-0.151 vs side -0.164, bear -0.284)
      2. На fresh данных bull прибыльный
      3. `bull=OFF` дало 0% coverage в walk-forward → не торгуем вообще

    ⚠️ Для production используйте walk_forward `--adaptive-thresholds` (B-22)
    вместо этих DEFAULT — quantile-based thresholds стабильнее под drift.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Sprint 10 (2026-05-06): per-ticker blacklist — модель плохо калибрована
    # для этих тикеров (ECE > 18% даже после Platt scaling).
    # Обновлено после ребилда с quantile heads (2026-05-06):
    #   AFLT 30.96% (-), OZON 27.71% (-), HEAD 23.47% (новый),
    #   ENPG 19.25% (новый), LENT 18.33% (новый)
    # TATN, VKCO, CBOM выпали из blacklist (их ECE улучшился после ребилда).
    # Сделки на blacklist'е форсятся в HOLD — лучше пропустить чем торговать
    # на шумных вероятностях. Список собирается из reliability_report и
    # должен периодически пересматриваться (после каждого ребилда V3).
    # ─────────────────────────────────────────────────────────────────────
    TICKER_BLACKLIST: frozenset[str] = frozenset({
        "AFLT", "OZON", "HEAD", "ENPG", "LENT",
    })

    # Регимы: 0=bear, 1=side, 2=bull
    DEFAULT_REGIME_THRESHOLDS = {
        0: {  # bear: рынок падает — модель нашла лучшее на full sweep
            "min_edge_ratio":      2.0,   # Sprint 5: 7.0 → 2.0 (низкий cost-bar нужен в bear)
            "min_dir_prob":        0.80,
            "min_sell_dir_prob":   0.50,
            "min_fill_prob":       0.40,
            "min_rr":              1.2,
        },
        1: {  # side: боковик — умеренный edge cutoff
            "min_edge_ratio":      4.0,   # Sprint 5: 6.5 → 4.0
            "min_dir_prob":        0.80,  # Sprint 5: 0.55 → 0.80 (требуем уверенности)
            "min_sell_dir_prob":   0.50,
            "min_fill_prob":       0.40,
            "min_rr":              1.2,
        },
        2: {  # bull: enabled (Sprint 5 OFF устарел, см. docstring)
            "min_edge_ratio":      5.0,   # Sprint 5: 99.0 (OFF) → 5.0 (enabled)
            "min_dir_prob":        0.75,
            "min_sell_dir_prob":   0.50,
            "min_fill_prob":       0.40,
            "min_rr":              1.2,
        },
        -1: {  # unknown regime: fallback на B-15 default
            "min_edge_ratio":      5.0,
            "min_dir_prob":        0.75,
            "min_sell_dir_prob":   0.55,
            "min_fill_prob":       0.40,
            "min_rr":              1.2,
        },
    }
    REGIME_NAMES = {0: "bear", 1: "side", 2: "bull", -1: "unknown"}

    def __init__(
        self,
        costs: Optional[TradingCosts] = None,
        regime_thresholds: Optional[dict[int, dict]] = None,
        ticker_blacklist: Optional[frozenset[str]] = None,
    ):
        self.costs = costs or TradingCosts()
        self.regime_thresholds = regime_thresholds or self.DEFAULT_REGIME_THRESHOLDS
        # Sprint 10: ticker_blacklist=None → используем class default; пустой
        # set отключит blacklist (например, для walk-forward на чистых данных).
        self.ticker_blacklist = (self.TICKER_BLACKLIST if ticker_blacklist is None
                                 else frozenset(ticker_blacklist))
        # Pre-build per-regime DecisionLayer instances
        self._layers: dict[int, DecisionLayer] = {}
        for rid, params in self.regime_thresholds.items():
            self._layers[rid] = DecisionLayer(costs=self.costs, **params)

    def decide_numpy(
        self,
        dir_prob:   np.ndarray,
        mfe_mae:    np.ndarray,
        fill_prob:  np.ndarray,
        edge_pred:  np.ndarray,
        regime:     np.ndarray,                   # [N] int8: 0=bear, 1=side, 2=bull, -1=unknown
        extremes:   Optional[np.ndarray] = None,  # [N, 2] — Sprint 8.1
        tickers:    Optional[np.ndarray] = None,  # Sprint 10: для blacklist-фильтра
    ) -> dict:
        """Применяет per-regime пороги. Сэмплы группируются по regime ID,
        для каждой группы вызывается соответствующий DecisionLayer.

        Sprint 10: если tickers переданы и self.ticker_blacklist непуст,
        blacklist'нутые тикеры force-HOLD'ятся (модель шумит на них).
        """
        n = len(dir_prob)
        signal     = np.full(n, SIG_HOLD, dtype=np.int64)
        confidence = np.zeros(n, dtype=np.float32)

        # Sprint 10: blacklist mask. NB: применяется ПОСЛЕ per-regime decision,
        # чтобы статистика coverage_per_regime отражала фактический blacklist effect.
        blacklist_mask = None
        if tickers is not None and self.ticker_blacklist:
            blacklist_mask = np.array([str(t) in self.ticker_blacklist
                                       for t in tickers], dtype=bool)

        unique_regimes = np.unique(regime)
        for rid in unique_regimes:
            rid_int = int(rid)
            mask = (regime == rid)
            if not mask.any():
                continue
            layer = self._layers.get(rid_int, self._layers.get(-1))
            if layer is None:
                continue
            out = layer.decide_numpy(
                dir_prob  = dir_prob[mask],
                mfe_mae   = mfe_mae[mask],
                fill_prob = fill_prob[mask],
                edge_pred = edge_pred[mask],
                extremes  = extremes[mask] if extremes is not None else None,
            )
            signal[mask]     = out["signal"]
            confidence[mask] = out["confidence"]

        if blacklist_mask is not None and blacklist_mask.any():
            n_filtered = int((signal[blacklist_mask] != SIG_HOLD).sum())
            signal[blacklist_mask]     = SIG_HOLD
            confidence[blacklist_mask] = 0.0
            if n_filtered > 0:
                # Молчаливая статистика — пишем в результат для downstream-логирования
                pass

        return {
            "signal":     signal,
            "confidence": confidence,
            "blacklist_filtered": (int(blacklist_mask.sum()) if blacklist_mask is not None else 0),
        }

    def coverage_per_regime(self, signal: np.ndarray, regime: np.ndarray) -> dict:
        """Coverage по регимам — для диагностики."""
        out = {}
        for rid in (0, 1, 2, -1):
            mask = (regime == rid)
            n = int(mask.sum())
            if n == 0:
                continue
            sig_r = signal[mask]
            n_buy  = int((sig_r == SIG_BUY ).sum())
            n_sell = int((sig_r == SIG_SELL).sum())
            out[self.REGIME_NAMES[rid]] = {
                "n":        n,
                "buy":      n_buy,
                "sell":     n_sell,
                "coverage": (n_buy + n_sell) / n,
            }
        return out


def coverage_report(signal: torch.Tensor) -> dict:
    """Доля BUY/HOLD/SELL для логирования."""
    n = signal.numel()
    if n == 0:
        return {"buy": 0., "hold": 0., "sell": 0.}
    return {
        "buy":  float((signal == SIG_BUY ).float().mean().item()),
        "hold": float((signal == SIG_HOLD).float().mean().item()),
        "sell": float((signal == SIG_SELL).float().mean().item()),
    }
