"""Sprint 9 (B-21): фундаментальные показатели компании → 12 фич для MetaLearner v3.

Использует T-Bank API метод getAssetFundamentals (через TinkoffDataClient.get_fundamentals).
Применяет sector z-score: для каждого тикера берём raw поля, считаем z-score
относительно компаний того же сектора. По спецификации T-Bank: значение 0
эквивалентно "нет данных" → заменяем на NaN перед z-score, потом fillna(0).

Запуск:
    py -m ml.fundamentals_loader --refresh-cache    # тянет API для всех тикеров
    py -m ml.fundamentals_loader --inspect SBER     # показывает 12 фич
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# 12 фундаментальных показателей, доступных стабильно для российских акций.
# Порядок фиксирован — определяет колонки в `build_fundamentals_matrix`.
FUNDAMENTAL_FIELDS: list[str] = [
    "pe_ratio_ttm",                           # 1. цена / прибыль
    "price_to_sales_ttm",                     # 2. цена / выручка
    "price_to_book_ttm",                      # 3. цена / балансовая стоимость
    "ev_to_ebitda_mrq",                       # 4. EV / EBITDA
    "total_debt_to_equity_mrq",               # 5. долг / капитал
    "current_ratio_mrq",                      # 6. ликвидность
    "roe",                                    # 7. рентабельность капитала
    "roa",                                    # 8. рентабельность активов
    "net_margin_mrq",                         # 9. чистая маржа
    "dividend_yield_daily_ttm",               # 10. дивидендная доходность
    "one_year_annual_revenue_growth_rate",    # 11. рост выручки YoY
    "free_float",                             # 12. free float
]
N_FUND_FEATURES = len(FUNDAMENTAL_FIELDS)


def _load_raw(client, ticker: str) -> dict | None:
    """Возвращает raw dict из get_fundamentals — с диск-кэшем (TTL 7 дней)."""
    return client.get_fundamentals(ticker, ttl_days=7)


def build_sector_stats(
    client,
    tickers: list[str],
    *,
    log: bool = False,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Возвращает {sector: {field: (median, mad_or_std)}} для z-score нормализации.

    Используется robust статистика: median + MAD (median absolute deviation),
    устойчивая к outliers (PLZL P/E может быть 200, испортит mean/std).
    Fallback на mean/std если в секторе < 3 валидных значений.
    """
    raw_by_ticker: dict[str, dict] = {}
    sector_by_ticker: dict[str, str] = {}

    for i, t in enumerate(tickers):
        if log:
            print(f"  [{i+1:2d}/{len(tickers)}] {t} ", end="", flush=True)
        sec = client.get_sector(t) or "unknown"
        sector_by_ticker[t] = sec
        try:
            raw = _load_raw(client, t)
        except Exception as e:
            if log: print(f"  fundamentals error: {e}")
            continue
        if raw is None:
            if log: print("  пусто")
            continue
        raw_by_ticker[t] = raw
        if log:
            print(f"  sector={sec}  pe={raw.get('pe_ratio_ttm', 0):.2f}  "
                  f"roe={raw.get('roe', 0):.2f}")
        time.sleep(0.15)  # rate-limit friendliness

    # Группируем по секторам
    sector_values: dict[str, dict[str, list[float]]] = {}
    for t, raw in raw_by_ticker.items():
        sec = sector_by_ticker[t]
        bucket = sector_values.setdefault(sec, {f: [] for f in FUNDAMENTAL_FIELDS})
        for f in FUNDAMENTAL_FIELDS:
            v = raw.get(f, 0.0)
            # 0.0 = "нет данных" по спецификации T-Bank
            if v != 0.0 and v is not None:
                bucket[f].append(float(v))

    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for sec, fields in sector_values.items():
        per_field: dict[str, tuple[float, float]] = {}
        for f, vals in fields.items():
            if len(vals) >= 3:
                arr = np.asarray(vals, dtype=np.float64)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med))) * 1.4826  # ≈ std для нормального распр.
                if mad < 1e-9:
                    mad = float(arr.std() + 1e-9)
                per_field[f] = (med, mad)
            elif len(vals) > 0:
                arr = np.asarray(vals, dtype=np.float64)
                per_field[f] = (float(arr.mean()), float(arr.std() + 1.0))
            else:
                per_field[f] = (0.0, 1.0)  # нейтральный fallback
        stats[sec] = per_field

    return stats


def featurize_ticker(
    raw: dict | None,
    sector: str,
    sector_stats: dict[str, dict[str, tuple[float, float]]],
) -> np.ndarray:
    """Превращает raw fundamentals в вектор из 12 z-scored фич [-3, 3]."""
    out = np.zeros(N_FUND_FEATURES, dtype=np.float32)
    if raw is None:
        return out
    sec_stats = sector_stats.get(sector) or {}
    for i, field in enumerate(FUNDAMENTAL_FIELDS):
        v = raw.get(field, 0.0)
        if v == 0.0 or v is None:
            continue  # NaN-эквивалент → оставляем 0 (нейтральный после z-score)
        med, mad = sec_stats.get(field, (0.0, 1.0))
        z = (float(v) - med) / mad
        out[i] = float(np.clip(z, -3.0, 3.0))
    return out


def build_fundamentals_map(
    client,
    tickers: list[str],
    *,
    log: bool = False,
) -> dict[str, np.ndarray]:
    """Полный pipeline: API → sector stats → 12 фич на тикер.

    Возвращает {ticker: np.float32[12]}.
    """
    sector_stats = build_sector_stats(client, tickers, log=log)
    out: dict[str, np.ndarray] = {}
    for t in tickers:
        sec = client.get_sector(t) or "unknown"
        try:
            raw = client.get_fundamentals(t, ttl_days=7)
        except Exception:
            raw = None
        out[t] = featurize_ticker(raw, sec, sector_stats)
    return out


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
    p.add_argument("--refresh-cache", action="store_true",
                   help="Запросить fundamentals для всех CFG.tickers и закэшировать")
    p.add_argument("--inspect", metavar="TICKER",
                   help="Показать 12 фич для тикера")
    p.add_argument("--save-map", metavar="PATH",
                   default=os.path.join(os.path.dirname(__file__), "ensemble",
                                        "fundamentals_map.json"),
                   help="Куда сохранить итоговую таблицу features")
    args = p.parse_args()

    _bootstrap_env()
    from data.tinkoff_client import TinkoffDataClient
    from ml.config import CFG

    token = os.getenv("TINKOFF_TOKEN", "")
    if not token:
        print("TINKOFF_TOKEN не задан"); sys.exit(1)

    client = TinkoffDataClient(token)

    if args.refresh_cache:
        fmap = build_fundamentals_map(client, list(CFG.tickers), log=True)
        os.makedirs(os.path.dirname(args.save_map), exist_ok=True)
        # Сохраняем как json {ticker: [12 floats]}
        with open(args.save_map, "w", encoding="utf-8") as fh:
            json.dump({k: v.tolist() for k, v in fmap.items()}, fh,
                      ensure_ascii=False, indent=2)
        print(f"\n  ✓ Сохранено: {args.save_map} ({len(fmap)} тикеров × {N_FUND_FEATURES} фич)")
        return

    if args.inspect:
        t = args.inspect.upper()
        sec = client.get_sector(t)
        raw = client.get_fundamentals(t)
        sector_stats = build_sector_stats(client, [t] + [x for x in CFG.tickers if x != t][:20])
        feats = featurize_ticker(raw, sec or "unknown", sector_stats)
        print(f"\n{t} (sector={sec})")
        print(f"  raw: {len(raw or {})} полей")
        print(f"  z-score features:")
        for name, v in zip(FUNDAMENTAL_FIELDS, feats):
            raw_v = (raw or {}).get(name, 0.0)
            print(f"    {name:42s}  raw={raw_v:>14.4f}  z={v:+.3f}")
        return

    p.print_help()


if __name__ == "__main__":
    main()
