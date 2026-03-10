"""Валидация входящих данных для всех эндпоинтов."""
import re
from litestar.exceptions import HTTPException

_TICKER_RE = re.compile(r'^[A-Z]{1,10}$')

_MAX_FORMULA_LEN = 500
_MAX_NAME_LEN    = 64


def validate_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if not _TICKER_RE.match(t):
        raise HTTPException(
            status_code=422,
            detail=f"Неверный тикер '{ticker}'. Только A-Z, максимум 10 символов."
        )
    return t


def validate_formula(formula: str | None, field: str) -> str | None:
    if formula is None:
        return None
    if len(formula) > _MAX_FORMULA_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"Поле '{field}' слишком длинное (максимум {_MAX_FORMULA_LEN} символов)."
        )
    return formula.strip()


def validate_days(days: int | None, default: int = 365) -> int:
    d = days if days is not None else default
    if not (1 <= d <= 365):
        raise HTTPException(
            status_code=422,
            detail=f"days должно быть от 1 до 365, получено: {days}."
        )
    return d


def validate_capital(capital: float) -> float:
    if capital <= 0:
        raise HTTPException(
            status_code=422,
            detail=f"capital должен быть больше 0, получено: {capital}."
        )
    if capital > 1_000_000_000:
        raise HTTPException(
            status_code=422,
            detail="capital не может превышать 1 000 000 000."
        )
    return capital


def validate_lots(lots: int) -> int:
    if not (1 <= lots <= 1000):
        raise HTTPException(
            status_code=422,
            detail=f"lots должно быть от 1 до 1000, получено: {lots}."
        )
    return lots


def validate_direction(direction: str) -> str:
    if direction not in ("BUY", "SELL"):
        raise HTTPException(
            status_code=422,
            detail=f"direction должен быть BUY или SELL, получено: '{direction}'."
        )
    return direction


def validate_name(name: str) -> str:
    n = name.strip()
    if not n:
        return "Strategy"
    return n[:_MAX_NAME_LEN]

_VALID_INTERVALS = {"1m", "5m", "15m", "1h", "1d"}

def validate_interval(interval: str) -> str:
    if interval not in _VALID_INTERVALS:
        raise HTTPException(
            status_code=422,
            detail=f"Неверный интервал '{interval}'. Допустимые: {sorted(_VALID_INTERVALS)}"
        )
    return interval


def validate_size(size: float) -> float:
    if not (0.1 <= size <= 100.0):
        raise HTTPException(
            status_code=422,
            detail=f"size должен быть от 0.1 до 100, получено: {size}"
        )
    return size
