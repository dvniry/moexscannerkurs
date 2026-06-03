"""Sprint 12.A — Data Foundation: расширение тикеров 55 → 100 с liquidity gate.

Тянет полный список TQBR-акций через T-Bank API, считает per-ticker liquidity
metrics на дневных свечах за `--days` дней (default 60), применяет фильтры и
ранжирует по обороту. Core-55 из CFG.tickers всегда включён (force_include) —
liquidity-фильтр применяется только к кандидатам в расширение.

Запуск:
    py -m ml.ticker_universe --refresh                # tian universe + write json
    py -m ml.ticker_universe --inspect SBER           # показать метрики
    py -m ml.ticker_universe --top-n 100              # отобрать 100 лучших
    py -m ml.ticker_universe --refresh --days 90      # окно метрик 90 дней

Выход:
    ml/ensemble/ticker_universe.json
      {version, generated_at, criteria, tickers: [...],
       cost_overrides: {ticker: float}, dropped: [{ticker, reasons}],
       metrics: {ticker: {turnover_rub_med, range_pct_med, n_days, last_dt}}}

Используется в config.py: при наличии файла CFG.tickers подменяется на universe.tickers.
Cost overrides — для backtest_strategy и decision_layer (per-ticker cost вместо global 0.2%).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from typing import Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

UNIVERSE_PATH = os.path.join(os.path.dirname(__file__), "ensemble", "ticker_universe.json")
UNIVERSE_VERSION = 1

# Default thresholds (эшелон-1 MOEX)
MIN_TURNOVER_RUB = 100_000_000.0   # 100M ₽/день медиана
MIN_TURNOVER_RUB_ECHELON2 = 50_000_000.0   # 50M ₽ — для эшелон-2 с cost-warning
MAX_RANGE_PCT = 0.05               # (H-L)/C ≤ 5% медиана — отсев слишком волатильных
MIN_DAYS = 250                     # минимум 250 дневных свечей в истории
DEFAULT_COMMISSION = 0.001         # 0.1% per leg (round-trip = 0.2% + slippage)
COST_CAP = 0.005                   # round-trip cap = 0.5%


def _bootstrap_env() -> None:
    os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
    cert = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "russian_ca.cer"))
    if os.path.exists(cert):
        os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", cert)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return
    except ImportError:
        pass
    # Минимальный fallback-парсер .env: KEY=VALUE построчно
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def fetch_full_universe(client) -> dict[str, str]:
    """Возвращает {ticker: figi} для всех TQBR-shares из кэша shares_table.

    Использует существующий _ensure_shares_table в TinkoffDataClient (TTL 30д).
    """
    client._ensure_shares_table()
    figi_map: dict[str, str] = {}
    for t, figi in client._figi_cache.items():
        if figi:
            figi_map[t] = figi
    return figi_map


def compute_liquidity_metrics(
    client,
    ticker: str,
    figi: str,
    days_back: int = 60,
) -> dict | None:
    """Считает метрики ликвидности по дневным свечам.

    Returns dict с {turnover_rub_med, range_pct_med, n_days, last_dt} или None
    если данных недостаточно/ошибка API.
    """
    try:
        df = client.get_candles(figi=figi, interval="1d", days_back=days_back, use_cache=True)
    except Exception as exc:
        return {"_error": str(exc)[:120]}

    if df is None or len(df) == 0:
        return None

    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    # Turnover ≈ close × volume (volume в штуках для акций TQBR)
    turnover = close * volume
    # Range pct — (H-L)/C, прокси волатильности и спреда
    with np.errstate(divide="ignore", invalid="ignore"):
        range_pct = np.where(close > 0, (high - low) / close, np.nan)

    valid = np.isfinite(turnover) & (turnover > 0) & np.isfinite(range_pct)
    n_valid = int(valid.sum())
    if n_valid < 5:
        return None

    return {
        "turnover_rub_med": float(np.median(turnover[valid])),
        "range_pct_med":    float(np.median(range_pct[valid])),
        "n_days":           int(len(df)),
        "n_valid":          n_valid,
        "last_dt":          str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
    }


def liquidity_gate(
    metrics: dict,
    *,
    min_turnover: float = MIN_TURNOVER_RUB,
    max_range_pct: float = MAX_RANGE_PCT,
    min_days: int = MIN_DAYS,
) -> tuple[bool, list[str]]:
    """Возвращает (passed, reasons_if_failed). reasons пуст если passed=True."""
    reasons: list[str] = []
    if not metrics or metrics.get("_error"):
        return False, ["no_data" if not metrics else f"api_error:{metrics['_error']}"]
    if metrics["turnover_rub_med"] < min_turnover:
        reasons.append(
            f"low_turnover ({metrics['turnover_rub_med']/1e6:.1f}M < {min_turnover/1e6:.0f}M)"
        )
    if metrics["range_pct_med"] > max_range_pct:
        reasons.append(
            f"high_range ({metrics['range_pct_med']*100:.2f}% > {max_range_pct*100:.1f}%)"
        )
    if metrics["n_days"] < min_days:
        reasons.append(f"short_history ({metrics['n_days']} < {min_days}d)")
    return (len(reasons) == 0), reasons


def estimate_cost(range_pct_med: float, commission: float = DEFAULT_COMMISSION) -> float:
    """Round-trip cost оценка: 2× комиссия + 0.5× range (slippage proxy).

    Range_pct — daily range, не bid-ask. Половина диапазона ≈ оптимистичная
    оценка slippage при market-order в течение дня. Cap'ируется COST_CAP.
    """
    cost = 2.0 * commission + 0.5 * range_pct_med
    return float(min(cost, COST_CAP))


def select_top_n(
    metrics_map: dict[str, dict],
    *,
    top_n: int = 100,
    force_include: Iterable[str] = (),
    min_turnover: float = MIN_TURNOVER_RUB,
    max_range_pct: float = MAX_RANGE_PCT,
    min_days: int = MIN_DAYS,
) -> dict:
    """Применяет gate, ранжирует по turnover, возвращает structured result.

    force_include — тикеры, которые включаются всегда (core-55), gate
    применяется только к расширению. Если force-тикер не проходит gate —
    он остаётся в выборке, но логируется в `force_overrides`.
    """
    force_set = {t.upper() for t in force_include}
    passed: list[tuple[str, dict]] = []
    dropped: list[dict] = []
    force_overrides: list[dict] = []

    for ticker, m in metrics_map.items():
        is_forced = ticker in force_set
        ok, reasons = liquidity_gate(
            m, min_turnover=min_turnover, max_range_pct=max_range_pct, min_days=min_days,
        )
        if ok:
            passed.append((ticker, m))
        elif is_forced:
            force_overrides.append({"ticker": ticker, "reasons": reasons})
            passed.append((ticker, m))
        else:
            dropped.append({"ticker": ticker, "reasons": reasons})

    # Сортируем по turnover убыванию; force-тикеры всегда первыми
    def sort_key(item):
        t, m = item
        is_force = t in force_set
        turnover = m.get("turnover_rub_med", 0.0) if m else 0.0
        return (0 if is_force else 1, -turnover)

    passed.sort(key=sort_key)

    # Ограничиваем top_n но force-тикеры всегда в выборке
    forced_in = [(t, m) for t, m in passed if t in force_set]
    non_forced = [(t, m) for t, m in passed if t not in force_set]
    budget = max(0, top_n - len(forced_in))
    final = forced_in + non_forced[:budget]

    # cost_overrides: для тикеров где estimated cost > 2 × default (0.002)
    cost_overrides: dict[str, float] = {}
    for t, m in final:
        if not m or m.get("_error"):
            continue
        est = estimate_cost(m["range_pct_med"])
        if est > 2.0 * DEFAULT_COMMISSION:
            cost_overrides[t] = round(est, 5)

    return {
        "version":     UNIVERSE_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "criteria": {
            "min_turnover_rub": min_turnover,
            "max_range_pct":    max_range_pct,
            "min_days":         min_days,
            "top_n":            top_n,
        },
        "tickers":         [t for t, _ in final],
        "cost_overrides":  cost_overrides,
        "force_overrides": force_overrides,
        "dropped":         dropped,
        "metrics": {
            t: {k: v for k, v in m.items() if not k.startswith("_")}
            for t, m in metrics_map.items() if m
        },
    }


def build_universe(
    client,
    *,
    days_back: int = 60,
    top_n: int = 100,
    force_include: Iterable[str] = (),
    min_turnover: float = MIN_TURNOVER_RUB,
    max_range_pct: float = MAX_RANGE_PCT,
    min_days: int = MIN_DAYS,
    candidates: Iterable[str] | None = None,
    log: bool = True,
) -> dict:
    """Полный пайплайн: tian universe → метрики → gate → ranking.

    Если `candidates` передан — ограничиваем выборку этими тикерами (для отладки).
    """
    figi_map = fetch_full_universe(client)
    if candidates:
        cand_set = {t.upper() for t in candidates}
        figi_map = {t: f for t, f in figi_map.items() if t in cand_set}

    if log:
        print(f"[universe] кандидатов из TQBR: {len(figi_map)}")
        print(f"[universe] окно метрик: {days_back}d, top_n: {top_n}, "
              f"force_include: {len(set(force_include))}")

    metrics_map: dict[str, dict] = {}
    t0 = time.time()
    for i, (ticker, figi) in enumerate(sorted(figi_map.items()), 1):
        m = compute_liquidity_metrics(client, ticker, figi, days_back=days_back)
        metrics_map[ticker] = m or {}
        if log and (i % 20 == 0 or i == len(figi_map)):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(figi_map)}] {ticker} — {elapsed:.0f}s")

    result = select_top_n(
        metrics_map,
        top_n=top_n,
        force_include=force_include,
        min_turnover=min_turnover,
        max_range_pct=max_range_pct,
        min_days=min_days,
    )
    if log:
        print(f"\n[universe] отобрано: {len(result['tickers'])} | "
              f"dropped: {len(result['dropped'])} | "
              f"force_overrides: {len(result['force_overrides'])} | "
              f"cost_overrides: {len(result['cost_overrides'])}")
    return result


def load_universe(path: str = UNIVERSE_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def save_universe(result: dict, path: str = UNIVERSE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


def _print_summary(result: dict) -> None:
    print(f"\nUniverse v{result['version']} @ {result['generated_at']}")
    print(f"Criteria: {result['criteria']}")
    print(f"Selected:  {len(result['tickers'])} тикеров")
    print(f"Dropped:   {len(result['dropped'])}")
    print(f"Force-overrides: {len(result['force_overrides'])}")
    print(f"Cost-overrides:  {len(result['cost_overrides'])}")
    print(f"Saved: {UNIVERSE_PATH}")

    if result["force_overrides"]:
        print("\n⚠️  Force-included тикеры с провалами gate:")
        for fo in result["force_overrides"][:10]:
            print(f"  {fo['ticker']:6s} — {', '.join(fo['reasons'])}")
        if len(result["force_overrides"]) > 10:
            print(f"  ... +{len(result['force_overrides'])-10} more")

    if result["cost_overrides"]:
        print(f"\n💸 Cost-overrides (top by cost):")
        items = sorted(result["cost_overrides"].items(), key=lambda x: -x[1])[:10]
        for t, c in items:
            print(f"  {t:6s} — round-trip {c*100:.2f}%")


def main():
    p = argparse.ArgumentParser(description="Sprint 12.A — ticker universe builder")
    p.add_argument("--refresh", action="store_true",
                   help="Пересчитать universe + записать json")
    p.add_argument("--inspect", metavar="TICKER",
                   help="Показать метрики одного тикера")
    p.add_argument("--days", type=int, default=60,
                   help="Окно для метрик (daily candles)")
    p.add_argument("--top-n", type=int, default=100, help="Максимум тикеров в universe")
    p.add_argument("--min-turnover-mln", type=float, default=MIN_TURNOVER_RUB / 1e6,
                   help="Минимальный медианный оборот, млн ₽")
    p.add_argument("--max-range-pct", type=float, default=MAX_RANGE_PCT,
                   help="Максимальный медианный (H-L)/C")
    p.add_argument("--min-days", type=int, default=MIN_DAYS,
                   help="Минимум дневных свечей в истории")
    p.add_argument("--candidates", nargs="*", default=None,
                   help="Ограничить выборку этими тикерами (для дебага)")
    args = p.parse_args()

    _bootstrap_env()
    from data.tinkoff_client import TinkoffDataClient
    from ml.config import CFG

    token = os.getenv("TINKOFF_TOKEN", "")
    if not token:
        print("TINKOFF_TOKEN не задан"); sys.exit(1)
    client = TinkoffDataClient(token)

    if args.inspect:
        t = args.inspect.upper()
        figi_map = fetch_full_universe(client)
        figi = figi_map.get(t)
        if not figi:
            print(f"❌ {t}: figi не найден в TQBR shares"); sys.exit(2)
        m = compute_liquidity_metrics(client, t, figi, days_back=args.days)
        if not m:
            print(f"❌ {t}: нет данных"); sys.exit(3)
        ok, reasons = liquidity_gate(
            m,
            min_turnover=args.min_turnover_mln * 1e6,
            max_range_pct=args.max_range_pct,
            min_days=args.min_days,
        )
        cost = estimate_cost(m["range_pct_med"])
        print(f"\n{t}")
        print(f"  turnover_med:  {m['turnover_rub_med']/1e6:.1f}M ₽/день")
        print(f"  range_pct_med: {m['range_pct_med']*100:.3f}%")
        print(f"  n_days:        {m['n_days']} (valid: {m['n_valid']})")
        print(f"  last_dt:       {m['last_dt']}")
        print(f"  est_cost (RT): {cost*100:.3f}%")
        print(f"  gate: {'✅ PASS' if ok else '❌ FAIL — ' + ', '.join(reasons)}")
        return

    if args.refresh:
        result = build_universe(
            client,
            days_back=args.days,
            top_n=args.top_n,
            force_include=list(CFG.tickers),
            min_turnover=args.min_turnover_mln * 1e6,
            max_range_pct=args.max_range_pct,
            min_days=args.min_days,
            candidates=args.candidates,
            log=True,
        )
        save_universe(result)
        _print_summary(result)
        return

    # default: показать сохранённый universe
    saved = load_universe()
    if not saved:
        print("ticker_universe.json отсутствует. Запусти: py -m ml.ticker_universe --refresh")
        sys.exit(0)
    _print_summary(saved)


if __name__ == "__main__":
    main()
