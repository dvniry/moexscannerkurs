"""Sprint 9 (B-21): дивидендные показатели → 5 фич для MetaLearner v3.

Источник: T-Bank API метод getDividends (через TinkoffDataClient.get_dividends).
Для тикеров типа SBER, GAZP ex-dividend gap создаёт искусственный DOWN-сигнал
на T0 (record_date+1) — раньше модель этого не знала.

5 фич на (ticker, date):
  0. days_to_next_record_date / 60       (clip 0..1, 1.0 если в течение 60 дней)
  1. is_ex_div_today                     (binary: T0 == record_date+1)
  2. gap_pct_expected                    (dividend_net / close на T-1, в долях)
  3. dy_ttm                              (суммарный yield за 12мес, в долях)
  4. coupon_density_30d                  (кол-во record_date в окне ±30 дней / 5)

Запуск:
    py -m ml.dividends_loader --refresh-cache
    py -m ml.dividends_loader --inspect SBER 2026-05-03
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

N_DIV_FEATURES = 5


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # ISO с tz / без tz
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return None


def featurize_dividends(
    dividends: list[dict] | None,
    target_date: datetime,
    *,
    last_close: float | None = None,
) -> np.ndarray:
    """Строит 5-мерный вектор фич для (ticker, target_date).

    target_date — naive datetime (день для которого делается прогноз).
    last_close — close на T-1 для нормализации gap_pct (если нет, gap_pct=0).
    """
    out = np.zeros(N_DIV_FEATURES, dtype=np.float32)
    if not dividends:
        return out

    # Распаковываем record_date'ы единожды
    parsed = []
    for d in dividends:
        rd = _parse_date(d.get("record_date"))
        if rd is None:
            continue
        parsed.append((rd, d))
    if not parsed:
        return out

    # 1. days_to_next_record_date
    future = [(rd, d) for rd, d in parsed if rd >= target_date]
    if future:
        future.sort(key=lambda x: x[0])
        nxt_rd, nxt_d = future[0]
        days = (nxt_rd - target_date).days
        out[0] = float(np.clip(days, 0, 60) / 60.0)

        # 2. is_ex_div_today: target_date == record_date + 1 рабочий день
        # Упрощённо: разница 1-3 дня (учёт выходных)
        recent_past = [(rd, d) for rd, d in parsed
                       if 0 < (target_date - rd).days <= 3]
        out[1] = 1.0 if recent_past else 0.0

        # 3. gap_pct_expected: размер ближайшего предстоящего дивиденда / цена
        if last_close and last_close > 0:
            net = float(nxt_d.get("dividend_net", 0.0) or 0.0)
            out[2] = float(np.clip(net / last_close, 0.0, 0.30))
    else:
        # все дивы в прошлом — окно с прошлого record_date
        last_rd, last_d = max(parsed, key=lambda x: x[0])
        if 0 < (target_date - last_rd).days <= 3:
            out[1] = 1.0

    # 4. dy_ttm: сумма yield_value за последние 365 дней
    cutoff = target_date - timedelta(days=365)
    ttm_yield = sum(
        float(d.get("yield_value", 0.0) or 0.0)
        for rd, d in parsed if rd >= cutoff and rd <= target_date
    )
    out[3] = float(np.clip(ttm_yield / 100.0, 0.0, 0.30))  # в долях

    # 5. coupon_density_30d: кол-во record_date в окне ±30 дней / 5
    win_lo = target_date - timedelta(days=30)
    win_hi = target_date + timedelta(days=30)
    n_in_win = sum(1 for rd, _ in parsed if win_lo <= rd <= win_hi)
    out[4] = float(min(n_in_win / 5.0, 1.0))

    return out


def build_dividends_map(
    client,
    tickers: list[str],
    *,
    log: bool = False,
) -> dict[str, list[dict]]:
    """Тянет дивиденды для всех тикеров (использует диск-кэш TinkoffDataClient)."""
    out: dict[str, list[dict]] = {}
    for i, t in enumerate(tickers):
        if log:
            print(f"  [{i+1:2d}/{len(tickers)}] {t} ", end="", flush=True)
        try:
            divs = client.get_dividends(t, ttl_days=1)
        except Exception as e:
            if log: print(f"  ошибка: {e}")
            divs = []
        out[t] = divs or []
        if log:
            print(f"  {len(out[t])} записей")
    return out


def featurize_for_dates(
    dividends: list[dict] | None,
    dates: list[str],
    closes: list[float] | None = None,
) -> np.ndarray:
    """Возвращает [N_dates, 5] для списка дат (YYYY-MM-DD)."""
    closes = closes or [None] * len(dates)
    rows = []
    for d_str, c in zip(dates, closes):
        target = _parse_date(d_str)
        if target is None:
            rows.append(np.zeros(N_DIV_FEATURES, dtype=np.float32))
            continue
        rows.append(featurize_dividends(dividends, target, last_close=c))
    return np.stack(rows, axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _bootstrap_env() -> None:
    os.environ.setdefault('GRPC_DNS_RESOLVER', 'native')
    cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
    if os.path.exists(cert):
        os.environ.setdefault('GRPC_DEFAULT_SSL_ROOTS_FILE_PATH', cert)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--inspect", nargs=2, metavar=("TICKER", "DATE"),
                   help="Показать 5 фич для тикера на дату YYYY-MM-DD")
    p.add_argument("--save-map", metavar="PATH",
                   default=os.path.join(os.path.dirname(__file__), "ensemble",
                                        "dividends_map.json"))
    args = p.parse_args()

    _bootstrap_env()
    from data.tinkoff_client import TinkoffDataClient
    from ml.config import CFG

    token = os.getenv("TINKOFF_TOKEN", "")
    if not token:
        print("TINKOFF_TOKEN не задан"); sys.exit(1)

    client = TinkoffDataClient(token)

    if args.refresh_cache:
        dmap = build_dividends_map(client, list(CFG.tickers), log=True)
        os.makedirs(os.path.dirname(args.save_map), exist_ok=True)
        with open(args.save_map, "w", encoding="utf-8") as fh:
            json.dump(dmap, fh, ensure_ascii=False, indent=2)
        n_with_divs = sum(1 for v in dmap.values() if v)
        total = sum(len(v) for v in dmap.values())
        print(f"\n  ✓ Сохранено: {args.save_map}")
        print(f"    {n_with_divs}/{len(dmap)} тикеров с дивидендами, всего записей: {total}")
        return

    if args.inspect:
        ticker, date_s = args.inspect
        ticker = ticker.upper()
        target = _parse_date(date_s)
        if target is None:
            print(f"Невалидная дата: {date_s}"); sys.exit(1)
        divs = client.get_dividends(ticker)
        feats = featurize_dividends(divs, target, last_close=100.0)
        print(f"\n{ticker} on {date_s}")
        print(f"  Дивидендов в истории: {len(divs or [])}")
        names = [
            "days_to_next_record_date / 60",
            "is_ex_div_today",
            "gap_pct_expected (assuming close=100)",
            "dy_ttm",
            "coupon_density_30d / 5",
        ]
        for n, v in zip(names, feats):
            print(f"    {n:42s} = {v:+.4f}")
        return

    p.print_help()


if __name__ == "__main__":
    main()
