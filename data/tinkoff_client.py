# data/tinkoff_client.py
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from t_tech.invest import Client, CandleInterval, Quotation
from t_tech.invest.exceptions import RequestError
from t_tech.invest.schemas import GetAssetFundamentalsRequest
from t_tech.invest.utils import now

logger = logging.getLogger(__name__)

TARGET = "invest-public-api.tbank.ru:443"

INTERVALS = {
    "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
}


def _is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in s
        or "request limit exceeded" in s
        or "ratelimit_remaining=0" in s
    )


def _extract_retry_seconds(exc: Exception, default: int = 20) -> int:
    try:
        for arg in getattr(exc, "args", ()):
            reset = getattr(arg, "ratelimit_reset", None)
            if reset is not None:
                return max(int(reset), 1)
    except Exception:
        pass
    return default


_CACHE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))


def _ensure_cache_dir() -> str:
    os.makedirs(_CACHE_ROOT, exist_ok=True)
    return _CACHE_ROOT


def _is_fresh_cache(path: str, ttl_seconds: int) -> bool:
    """True если файл существует и моложе ttl_seconds."""
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < ttl_seconds


class TinkoffDataClient:
    def __init__(self, token: str):
        self.token = token
        self._figi_cache: dict[str, str] = {}
        self._uid_cache: dict[str, str] = {}
        self._asset_uid_cache: dict[str, str] = {}
        self._indicative_uid_cache: dict[str, str] = {}
        self._candle_cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _q(q: Quotation) -> float:
        return q.units + q.nano / 1e9

    def find_figi(self, ticker: str) -> str | None:
        if ticker in self._figi_cache:
            return self._figi_cache[ticker]

        with Client(self.token, target=TARGET) as api:
            response = api.instruments.find_instrument(query=ticker)

            # Приоритет 1: BBG FIGI + TQBR class_code
            for inst in response.instruments:
                if (getattr(inst, "ticker", None) == ticker
                        and getattr(inst, "figi", "").startswith("BBG")
                        and getattr(inst, "class_code", None) == "TQBR"):
                    self._figi_cache[ticker] = inst.figi
                    if getattr(inst, "uid", None):
                        self._uid_cache[ticker] = inst.uid
                    return inst.figi

            # Приоритет 2: любой BBG FIGI
            for inst in response.instruments:
                if (getattr(inst, "ticker", None) == ticker
                        and getattr(inst, "figi", "").startswith("BBG")):
                    self._figi_cache[ticker] = inst.figi
                    if getattr(inst, "uid", None):
                        self._uid_cache[ticker] = inst.uid
                    return inst.figi

            # Fallback: первый попавшийся (старое поведение)
            for inst in response.instruments:
                if getattr(inst, "ticker", None) == ticker:
                    figi = getattr(inst, "figi", None)
                    if figi:
                        self._figi_cache[ticker] = figi
                        return figi

        logger.warning("FIGI not found for ticker=%s", ticker)
        return None

    def find_uid(self, ticker: str) -> str | None:
        if ticker in self._uid_cache:
            return self._uid_cache[ticker]

        with Client(self.token, target=TARGET) as api:
            response = api.instruments.find_instrument(query=ticker)
            for inst in response.instruments:
                if getattr(inst, "ticker", None) == ticker and getattr(inst, "uid", None):
                    self._uid_cache[ticker] = inst.uid
                    if getattr(inst, "figi", None):
                        self._figi_cache[ticker] = inst.figi
                    return inst.uid

            for inst in response.instruments:
                if getattr(inst, "ticker", None) == ticker:
                    uid = getattr(inst, "uid", None)
                    if uid:
                        self._uid_cache[ticker] = uid
                        if getattr(inst, "figi", None):
                            self._figi_cache[ticker] = inst.figi
                        return uid

        logger.warning("UID not found for ticker=%s", ticker)
        return None

    def find_indicative_uid(self, ticker: str) -> str | None:
        if ticker in self._indicative_uid_cache:
            return self._indicative_uid_cache[ticker]

        with Client(self.token, target=TARGET) as api:
            response = api.instruments.find_instrument(query=ticker)
            for inst in response.instruments:
                if getattr(inst, "ticker", None) == ticker:
                    uid = getattr(inst, "uid", None)
                    if uid:
                        self._indicative_uid_cache[ticker] = uid
                        return uid

        logger.warning("Indicative UID not found for ticker=%s", ticker)
        return None

    def _load_candles_chunked(
        self,
        *,
        interval: str,
        days_back: int,
        figi: str | None = None,
        uid: str | None = None,
    ) -> pd.DataFrame:
        if interval not in INTERVALS:
            raise ValueError(f"Неизвестный интервал '{interval}'. Доступны: {list(INTERVALS)}")
        if not figi and not uid:
            raise ValueError("Нужно передать figi или uid")

        chunk_days = 365 if interval == "1d" else 85 if interval == "1h" else days_back
        all_frames = []
        end = now()
        remaining = int(days_back)
        ident = figi or uid or "unknown"

        with Client(self.token, target=TARGET) as api:
            while remaining > 0:
                chunk = min(remaining, chunk_days)
                start = end - timedelta(days=chunk)

                kwargs = {
                    "from_": start,
                    "to": end,
                    "interval": INTERVALS[interval],
                }
                if figi is not None:
                    kwargs["figi"] = figi
                else:
                    kwargs["instrument_id"] = uid

                candles = None
                last_exc = None

                for attempt in range(6):
                    try:
                        candles = api.market_data.get_candles(**kwargs).candles
                        last_exc = None
                        break
                    except RequestError as e:
                        last_exc = e
                        if _is_rate_limit_error(e):
                            reset_s = _extract_retry_seconds(e, default=20)
                            sleep_s = min(max(reset_s + random.uniform(0.3, 1.5), 1.0), 65.0)
                            logger.warning(
                                "Rate limit on get_candles(%s, %s) %s→%s, attempt %d/6, sleep %.1fs",
                                ident, interval, start.date(), end.date(), attempt + 1, sleep_s
                            )
                            time.sleep(sleep_s)
                            continue
                        raise
                    except Exception as e:
                        last_exc = e
                        if "RESOURCE_EXHAUSTED" in str(e):
                            sleep_s = 20.0 + random.uniform(0.3, 1.5)
                            logger.warning(
                                "RESOURCE_EXHAUSTED on get_candles(%s, %s) %s→%s, attempt %d/6, sleep %.1fs",
                                ident, interval, start.date(), end.date(), attempt + 1, sleep_s
                            )
                            time.sleep(sleep_s)
                            continue
                        raise

                if last_exc is not None and candles is None:
                    raise last_exc

                if candles:
                    df_chunk = pd.DataFrame({
                        "time":   [c.time for c in candles],
                        "open":   [self._q(c.open) for c in candles],
                        "high":   [self._q(c.high) for c in candles],
                        "low":    [self._q(c.low) for c in candles],
                        "close":  [self._q(c.close) for c in candles],
                        "volume": [c.volume for c in candles],
                    }).set_index("time")
                    all_frames.append(df_chunk)

                end = start
                remaining -= chunk

                time.sleep(0.08 if interval == "1d" else 0.12)

        if not all_frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        result = pd.concat(all_frames).sort_index()
        return result[~result.index.duplicated(keep="first")]

    def get_candles(
        self,
        figi: str,
        interval: str = "1h",
        days_back: int = 100,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_key = f"{figi}_{interval}_{days_back}"
        if use_cache and cache_key in self._candle_cache:
            return self._candle_cache[cache_key].copy()

        df = self._load_candles_chunked(
            figi=figi,
            interval=interval,
            days_back=days_back,
        )

        if use_cache:
            self._candle_cache[cache_key] = df

        logger.info("Loaded %d candles: %s %s %dd", len(df), figi, interval, days_back)
        return df.copy()

    def get_candles_by_uid(
        self,
        uid: str,
        interval: str = "1d",
        days_back: int = 730,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_key = f"uid_{uid}_{interval}_{days_back}"
        if use_cache and cache_key in self._candle_cache:
            return self._candle_cache[cache_key].copy()

        df = self._load_candles_chunked(
            uid=uid,
            interval=interval,
            days_back=days_back,
        )

        if use_cache:
            self._candle_cache[cache_key] = df

        logger.info("Loaded %d candles by uid: %s %s %dd", len(df), uid, interval, days_back)
        return df.copy()

    # ──────────────────────────────────────────────────────────────────
    # Sprint 9: Fundamentals + Dividends (B-21)
    # ──────────────────────────────────────────────────────────────────

    def _ensure_shares_table(self) -> None:
        """Подгружает таблицу всех акций (asset_uid + sector) и кэширует один раз.

        find_instrument не возвращает asset_uid — нужен полный shares() запрос.
        Кэшируется на диске, TTL = 30 дней (asset_uid стабилен).
        """
        if self._asset_uid_cache and getattr(self, "_sector_cache", None):
            return

        cache_dir = _ensure_cache_dir()
        cache_path = os.path.join(cache_dir, "shares_table.json")

        if _is_fresh_cache(cache_path, 30 * 86400):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    table = json.load(fh)
                self._asset_uid_cache.update(table.get("asset_uid", {}))
                self._sector_cache = dict(table.get("sector", {}))
                return
            except Exception:
                pass

        # Полный запрос к API
        with Client(self.token, target=TARGET) as api:
            for attempt in range(5):
                try:
                    resp = api.instruments.shares()
                    break
                except RequestError as e:
                    if _is_rate_limit_error(e):
                        time.sleep(_extract_retry_seconds(e, default=10))
                        continue
                    raise
                except Exception as e:
                    if "UNAVAILABLE" in str(e) and attempt < 4:
                        time.sleep(2.0); continue
                    raise
            else:
                logger.warning("_ensure_shares_table: ретраи исчерпаны")
                return

        # Приоритет TQBR при дубликатах ticker'а
        by_ticker_tqbr: dict[str, object] = {}
        by_ticker_other: dict[str, object] = {}
        for s in resp.instruments:
            t = s.ticker
            if not t:
                continue
            if s.class_code == "TQBR":
                by_ticker_tqbr.setdefault(t, s)
            else:
                by_ticker_other.setdefault(t, s)

        sector_map: dict[str, str] = {}
        for t in set(by_ticker_tqbr) | set(by_ticker_other):
            s = by_ticker_tqbr.get(t) or by_ticker_other.get(t)
            if s.asset_uid:
                self._asset_uid_cache[t] = s.asset_uid
            if s.figi:
                self._figi_cache.setdefault(t, s.figi)
            if s.uid:
                self._uid_cache.setdefault(t, s.uid)
            if s.sector:
                sector_map[t] = s.sector

        self._sector_cache = sector_map

        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "asset_uid": dict(self._asset_uid_cache),
                    "sector":    sector_map,
                }, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Не удалось записать кэш shares_table: %s", e)

    def find_asset_uid(self, ticker: str) -> str | None:
        """asset_uid отличается от instrument uid — нужен для get_asset_fundamentals."""
        if ticker in self._asset_uid_cache:
            return self._asset_uid_cache[ticker]
        self._ensure_shares_table()
        aid = self._asset_uid_cache.get(ticker)
        if not aid:
            logger.warning("asset_uid not found for ticker=%s", ticker)
        return aid

    def get_sector(self, ticker: str) -> str | None:
        """Сектор тикера от T-Bank API (financial/oil_and_gas/metals/...)."""
        self._ensure_shares_table()
        return getattr(self, "_sector_cache", {}).get(ticker)

    def get_fundamentals(
        self,
        ticker: str,
        *,
        asset_uid: str | None = None,
        ttl_days: int = 7,
        use_cache: bool = True,
    ) -> dict | None:
        """Возвращает dict из StatisticResponse как plain primitives.

        По спецификации T-Bank: значение 0.0 эквивалентно "нет данных". Loader
        (`ml.fundamentals_loader`) применит fallback / sector z-score.

        Кэш: data/cache/fundamentals_{TICKER}.json, TTL = 7 дней.
        """
        cache_dir = _ensure_cache_dir()
        cache_path = os.path.join(cache_dir, f"fundamentals_{ticker}.json")

        if use_cache and _is_fresh_cache(cache_path, ttl_days * 86400):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass  # битый кэш — перезапросим

        if asset_uid is None:
            asset_uid = self.find_asset_uid(ticker)
        if not asset_uid:
            logger.warning("get_fundamentals: no asset_uid for %s", ticker)
            return None

        with Client(self.token, target=TARGET) as api:
            for attempt in range(5):
                try:
                    req = GetAssetFundamentalsRequest(assets=[asset_uid])
                    resp = api.instruments.get_asset_fundamentals(request=req)
                    break
                except RequestError as e:
                    if _is_rate_limit_error(e):
                        sleep_s = _extract_retry_seconds(e, default=20)
                        logger.warning(
                            "Rate limit on get_fundamentals(%s) attempt %d/5, sleep %.1fs",
                            ticker, attempt + 1, sleep_s)
                        time.sleep(sleep_s)
                        continue
                    raise
            else:
                logger.warning("get_fundamentals: %s — все ретраи исчерпаны", ticker)
                return None

        if not resp.fundamentals:
            return None

        stat = resp.fundamentals[0]
        # Сериализуем все public поля как plain (datetime → ISO, остальное float/str/bool)
        out: dict = {}
        for name in dir(stat):
            if name.startswith("_"):
                continue
            try:
                val = getattr(stat, name)
            except Exception:
                continue
            if callable(val):
                continue
            if isinstance(val, datetime):
                out[name] = val.isoformat()
            elif isinstance(val, (int, float, str, bool)) or val is None:
                out[name] = val
            # пропускаем сложные структуры

        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Не удалось записать кэш фундаменталки %s: %s", ticker, e)

        return out

    def get_dividends(
        self,
        ticker: str,
        *,
        figi: str | None = None,
        years_back: int = 3,
        ttl_days: int = 1,
        use_cache: bool = True,
    ) -> list[dict] | None:
        """Возвращает список дивидендов с record_date в [now - years_back, now + 1y].

        Поля каждого dict:
          record_date, payment_date, declared_date, last_buy_date,
          dividend_net (float, рубли), yield_value (float, % годовых), close_price (float)

        Кэш: data/cache/dividends_{TICKER}.json, TTL = 1 день.
        """
        cache_dir = _ensure_cache_dir()
        cache_path = os.path.join(cache_dir, f"dividends_{ticker}.json")

        if use_cache and _is_fresh_cache(cache_path, ttl_days * 86400):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass

        if figi is None:
            figi = self.find_figi(ticker)
        if not figi:
            logger.warning("get_dividends: no figi for %s", ticker)
            return None

        from_ = datetime.now(timezone.utc) - timedelta(days=365 * years_back)
        to_   = datetime.now(timezone.utc) + timedelta(days=365)

        with Client(self.token, target=TARGET) as api:
            for attempt in range(5):
                try:
                    resp = api.instruments.get_dividends(
                        figi=figi, from_=from_, to=to_)
                    break
                except RequestError as e:
                    if _is_rate_limit_error(e):
                        sleep_s = _extract_retry_seconds(e, default=20)
                        logger.warning(
                            "Rate limit on get_dividends(%s) attempt %d/5, sleep %.1fs",
                            ticker, attempt + 1, sleep_s)
                        time.sleep(sleep_s)
                        continue
                    raise
            else:
                logger.warning("get_dividends: %s — все ретраи исчерпаны", ticker)
                return None

        out: list[dict] = []
        for d in (resp.dividends or []):
            out.append({
                "record_date":   d.record_date.isoformat()   if d.record_date else None,
                "payment_date":  d.payment_date.isoformat()  if d.payment_date else None,
                "declared_date": d.declared_date.isoformat() if d.declared_date else None,
                "last_buy_date": d.last_buy_date.isoformat() if d.last_buy_date else None,
                "dividend_net":  self._q(d.dividend_net) if d.dividend_net else 0.0,
                "yield_value":   self._q(d.yield_value)  if d.yield_value  else 0.0,
                "close_price":   self._q(d.close_price)  if d.close_price  else 0.0,
                "regularity":    d.regularity or "",
                "dividend_type": d.dividend_type or "",
            })

        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Не удалось записать кэш дивидендов %s: %s", ticker, e)

        return out