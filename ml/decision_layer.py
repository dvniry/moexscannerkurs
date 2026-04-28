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
    """
    def __init__(
        self,
        costs: Optional[TradingCosts] = None,
        min_edge_ratio: float = 2.0,   # edge >= 2× cost — реальный буфер над костами
        min_dir_prob:   float = 0.62,  # топ-10% уверенных предсказаний dir_head
        min_fill_prob:  float = 0.40,
        min_rr:         float = 1.2,   # mfe > mae — асимметрия в нашу пользу
    ):
        self.costs = costs or TradingCosts()
        self.min_edge_ratio = float(min_edge_ratio)
        self.min_dir_prob   = float(min_dir_prob)
        self.min_fill_prob  = float(min_fill_prob)
        self.min_rr         = float(min_rr)

    @torch.no_grad()
    def decide(self, dir_logit: torch.Tensor, econ: dict) -> dict:
        """
        dir_logit: [B] — sigmoid выход dir_head (logit)
        econ: dict из EconomicHeads.forward с ключами:
            mfe_mae    [B, 4]  (mfe_l, mae_l, mfe_s, mae_s)
            fill_logit [B, 2]  (fill_l, fill_s) — logits
            edge_pred  [B, 2]  (edge_l, edge_s) — net edge prediction

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
            ((1.0 - dir_prob) >= self.min_dir_prob)
            & (fill_s         >= self.min_fill_prob)
            & (rr_s           >= self.min_rr)
            & (er_s           >= self.min_edge_ratio)
        )

        # Если оба условия выполнены — направление по dir_prob;
        # SELL только если long_ok=False или dir_prob < 0.5.
        is_buy  = long_ok  & (~short_ok | (dir_prob >= 0.5))
        is_sell = short_ok & ~is_buy

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
        dir_prob:   np.ndarray,    # [N] sigmoid вероятность UP (уже без logit)
        mfe_mae:    np.ndarray,    # [N, 4]
        fill_prob:  np.ndarray,    # [N, 2] уже sigmoid
        edge_pred:  np.ndarray,    # [N, 2]
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
        out = self.decide(dir_logit, econ)
        return {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}


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
