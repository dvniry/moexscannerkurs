"""Sprint 9.5 (Sprint 8 polish): unit-тесты для simulate_path_aware_strategy.

Покрывает 9 сценариев:
  1. HOLD сигнал → no trade
  2. low_first BUY с TP hit (entry + TP оба сработали)
  3. low_first BUY без entry (limit не достигнут)
  4. low_first BUY entry hit, TP miss (exit at close)
  5. low_first SELL с TP hit
  6. high_first BUY с TP hit (entry at open, TP at pred_high)
  7. high_first BUY без TP (exit at close)
  8. uncertain hf_p (между порогами) → skip
  9. Аномальный PnL (|pnl_capital| > 0.05) → trade dropped

Запуск:
    py -m ml.tests.test_path_aware
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from ml.backtest_strategy import simulate_path_aware_strategy
from ml.decision_layer import SIG_BUY, SIG_HOLD, SIG_SELL

FEE = 0.001  # default in simulator


def _build_inputs(rows: list[dict]) -> dict:
    """Удобный билдер: каждая row = {sig, conf, hf_p, pred_H, pred_L, real_H, real_L, real_C, y}."""
    n = len(rows)
    sig  = np.zeros(n, dtype=np.int64)
    conf = np.zeros(n, dtype=np.float32)
    hf   = np.zeros(n, dtype=np.float32)
    pred = np.zeros((n, 4), dtype=np.float32)  # ΔO ΔH ΔL ΔC
    true = np.zeros((n, 4), dtype=np.float32)
    y    = np.zeros(n, dtype=np.int8)
    for i, r in enumerate(rows):
        sig[i]  = r["sig"]
        conf[i] = r.get("conf", 0.5)
        hf[i]   = r["hf_p"]
        pred[i, 1] = r.get("pred_H", 0.02)
        pred[i, 2] = r.get("pred_L", -0.02)
        true[i, 1] = r.get("real_H", 0.0)
        true[i, 2] = r.get("real_L", 0.0)
        true[i, 3] = r.get("real_C", 0.0)
        y[i] = r.get("y", 1)  # FLAT default
    return dict(decision_signal=sig, decision_conf=conf,
                ohlc_pred_pct=pred, ohlc_true_pct=true,
                y_true=y, high_first_prob=hf)


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def _run(label: str, rows: list[dict], expected_n: int,
         **kwargs) -> tuple[list[dict], dict]:
    """Запускает симулятор + базовые ассерты."""
    inputs = _build_inputs(rows)
    trades, diag = simulate_path_aware_strategy(**inputs, **kwargs)
    assert len(trades) == expected_n, \
        f"[{label}] Ожидалось {expected_n} сделок, получено {len(trades)}"
    return trades, diag


# ──────────────────────────────────────────────────────────────────────
# Сценарии
# ──────────────────────────────────────────────────────────────────────

def test_hold_no_trade():
    """1. HOLD сигнал → trade не создаётся."""
    rows = [{"sig": SIG_HOLD, "hf_p": 0.30, "real_H": 0.05, "real_L": -0.05}]
    trades, diag = _run("hold", rows, expected_n=0)
    assert diag["n_hold"] == 1
    print("  ✓ test_hold_no_trade")


def test_low_first_buy_tp_hit():
    """2. low_first BUY: entry at pred_low, TP at pred_high. Оба сработали."""
    rows = [{
        "sig": SIG_BUY, "conf": 0.8,
        "hf_p": 0.20,                # < 0.45 → low_first
        "pred_H": 0.020, "pred_L": -0.015,
        "real_H": 0.025,             # >= pred_H → TP hit
        "real_L": -0.020,            # <= pred_L → entry hit
        "real_C": 0.010,
    }]
    trades, _ = _run("low_first BUY TP", rows, expected_n=1)
    t = trades[0]
    assert t["direction"] == "LONG"
    assert t["mode"] == "low_first"
    assert t["exit"] == "TP"
    expected_gross = 0.020 - (-0.015)   # tp_price - entry_delta
    assert _approx(t["gross_pnl"], expected_gross), f"gross={t['gross_pnl']} vs {expected_gross}"
    assert _approx(t["net_pnl_pct"], expected_gross - 2 * FEE)
    print(f"  ✓ test_low_first_buy_tp_hit  gross={t['gross_pnl']:.4f}")


def test_low_first_buy_entry_miss():
    """3. low_first BUY: real_L > entry → лимитка не сработала, trade skipped."""
    rows = [{
        "sig": SIG_BUY, "hf_p": 0.20,
        "pred_H": 0.030, "pred_L": -0.025,
        "real_H": 0.040,
        "real_L": -0.010,            # > pred_L (-0.025) → entry NOT hit
        "real_C": 0.015,
    }]
    trades, _ = _run("low_first BUY no entry", rows, expected_n=0)
    print("  ✓ test_low_first_buy_entry_miss")


def test_low_first_buy_tp_miss():
    """4. low_first BUY: entry hit, TP miss → exit at close."""
    rows = [{
        "sig": SIG_BUY, "hf_p": 0.20,
        "pred_H": 0.030, "pred_L": -0.020,
        "real_H": 0.020,             # < pred_H → TP miss
        "real_L": -0.025,            # <= pred_L → entry hit
        "real_C": 0.005,
    }]
    trades, _ = _run("low_first BUY TP miss", rows, expected_n=1)
    t = trades[0]
    assert t["exit"] == "close"
    expected_gross = 0.005 - (-0.020)  # real_C - entry_delta
    assert _approx(t["gross_pnl"], expected_gross), f"gross={t['gross_pnl']} vs {expected_gross}"
    print(f"  ✓ test_low_first_buy_tp_miss  exit=close gross={t['gross_pnl']:.4f}")


def test_low_first_sell_tp_hit():
    """5. low_first SELL: entry at pred_high, TP at pred_low. Оба сработали."""
    rows = [{
        "sig": SIG_SELL, "hf_p": 0.30,
        "pred_H": 0.020, "pred_L": -0.018,
        "real_H": 0.025,             # >= pred_H → SELL entry hit
        "real_L": -0.020,            # <= pred_L → TP hit
        "real_C": -0.010,
    }]
    trades, _ = _run("low_first SELL TP", rows, expected_n=1)
    t = trades[0]
    assert t["direction"] == "SHORT"
    assert t["exit"] == "TP"
    expected_gross = 0.020 - (-0.018)  # entry - tp_price for short
    assert _approx(t["gross_pnl"], expected_gross), f"gross={t['gross_pnl']} vs {expected_gross}"
    print(f"  ✓ test_low_first_sell_tp_hit  gross={t['gross_pnl']:.4f}")


def test_high_first_buy_tp_hit():
    """6. high_first BUY: entry at open (delta=0), TP at pred_high → TP hit."""
    rows = [{
        "sig": SIG_BUY, "hf_p": 0.70,    # > 0.55 → high_first
        "pred_H": 0.025, "pred_L": -0.020,
        "real_H": 0.030,                  # >= pred_H → TP hit
        "real_L": 0.005,
        "real_C": 0.020,
    }]
    trades, _ = _run("high_first BUY TP", rows, expected_n=1)
    t = trades[0]
    assert t["mode"] == "high_first"
    assert t["entry_delta"] == 0.0
    assert t["exit"] == "TP"
    assert _approx(t["gross_pnl"], 0.025), f"gross={t['gross_pnl']}"
    print(f"  ✓ test_high_first_buy_tp_hit  gross={t['gross_pnl']:.4f}")


def test_high_first_buy_no_tp():
    """7. high_first BUY без TP: real_H < pred_H → exit at close."""
    rows = [{
        "sig": SIG_BUY, "hf_p": 0.70,
        "pred_H": 0.030, "pred_L": -0.020,
        "real_H": 0.015,             # < pred_H → no TP
        "real_L": -0.010,
        "real_C": 0.012,
    }]
    trades, _ = _run("high_first BUY close", rows, expected_n=1)
    t = trades[0]
    assert t["exit"] == "close"
    assert _approx(t["gross_pnl"], 0.012), f"gross={t['gross_pnl']}"
    print(f"  ✓ test_high_first_buy_no_tp  gross={t['gross_pnl']:.4f}")


def test_uncertain_hf_skip():
    """8. hf_p строго в зоне неопределённости (low_thr, high_thr) → trade skipped.

    Реализация: low_first if hf_p < 0.45 (строго); high_first if hf_p > 0.55 (строго).
    Граничные 0.45/0.55 НЕ попадают в uncertain — это by design, чтобы избежать
    дрейфа сигналов на боковых hf_p.
    """
    rows = [
        {"sig": SIG_BUY, "hf_p": 0.50, "pred_H": 0.02, "pred_L": -0.02},  # центр зоны
        {"sig": SIG_BUY, "hf_p": 0.46, "pred_H": 0.02, "pred_L": -0.02},  # внутри
        {"sig": SIG_BUY, "hf_p": 0.54, "pred_H": 0.02, "pred_L": -0.02},  # внутри
    ]
    trades, diag = _run("uncertain", rows, expected_n=0)
    assert diag["n_uncertain"] == 3, \
        f"Ожидалось 3 uncertain, получено {diag['n_uncertain']}"
    print(f"  ✓ test_uncertain_hf_skip  uncertain={diag['n_uncertain']}")


def test_anomalous_pnl_dropped():
    """9. |pnl_capital| > 0.05 → trade сбрасывается как аномальный."""
    # max_position_pct=0.5, conf=1.0 → size=0.5
    # gross должен дать |gross| × 0.5 > 0.05 → |gross| > 0.10
    rows = [{
        "sig": SIG_BUY, "conf": 1.0,
        "hf_p": 0.70,                # high_first
        "pred_H": 0.20, "pred_L": -0.05,
        "real_H": 0.25, "real_L": 0.0,
        "real_C": 0.20,
    }]
    trades, _ = _run("anomalous pnl dropped", rows, expected_n=0,
                     max_position_pct=0.5)
    print("  ✓ test_anomalous_pnl_dropped")


def test_position_size_scales_with_conf():
    """Бонус: position_size = max_position_pct × clip(conf, 0, 1)."""
    rows = [{
        "sig": SIG_BUY, "conf": 0.4,
        "hf_p": 0.20, "pred_H": 0.020, "pred_L": -0.015,
        "real_H": 0.025, "real_L": -0.020, "real_C": 0.010,
    }]
    trades, _ = _run("size scales", rows, expected_n=1, max_position_pct=0.02)
    t = trades[0]
    assert _approx(t["size"], 0.02 * 0.4), f"size={t['size']} vs 0.008"
    print(f"  ✓ test_position_size_scales_with_conf  size={t['size']:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_hold_no_trade,
        test_low_first_buy_tp_hit,
        test_low_first_buy_entry_miss,
        test_low_first_buy_tp_miss,
        test_low_first_sell_tp_hit,
        test_high_first_buy_tp_hit,
        test_high_first_buy_no_tp,
        test_uncertain_hf_skip,
        test_anomalous_pnl_dropped,
        test_position_size_scales_with_conf,
    ]
    print(f"\n  Sprint 8 polish — path-aware unit tests")
    print(f"  ───────────────────────────────────────")
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ {fn.__name__}: FAIL — {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: ERROR — {type(e).__name__}: {e}")
            failed += 1
    print(f"\n  Итого: {len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
