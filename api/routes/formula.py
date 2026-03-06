"""POST /api/formula — расчёт пользовательской формулы."""
import os
import sys
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from litestar import post
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from indicators.formula import Formula
from api.routes.candles import get_client, _to_unix, _executor


@dataclass
class FormulaRequest:
    ticker:   str
    formula:  str
    name:     str            = "Custom"
    interval: str            = "1h"
    days:     int            = 30
    params:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormulaPoint:
    time:  int
    value: float


@dataclass
class FormulaResponse:
    name:   str
    points: List[FormulaPoint]
    last:   float
    error:  Optional[str] = None


def _fetch_and_calculate(req: FormulaRequest) -> tuple:
    """Синхронно: загружаем данные + считаем формулу."""
    client = get_client()
    figi   = client.find_figi(req.ticker.upper())
    df     = client.get_candles(figi=figi, interval=req.interval, days_back=req.days)
    
    ind    = Formula(name=req.name, formula=req.formula, params=req.params)
    result = ind(df)
    return df, result


@post("/formula")
async def calculate_formula(data: FormulaRequest) -> FormulaResponse:
    """Рассчитать формулу и вернуть точки для графика."""
    try:
        loop     = asyncio.get_event_loop()
        df, result = await loop.run_in_executor(
            _executor,
            lambda: _fetch_and_calculate(data)
        )
    except (SyntaxError, ValueError, RuntimeError) as e:
        return FormulaResponse(name=data.name, points=[], last=0.0, error=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    unix = _to_unix(df)

    points = [
        FormulaPoint(time=t, value=round(float(v), 6))
        for t, v in zip(unix, result)
        if pd.notna(v)
    ]

    last_val = float(result.dropna().iloc[-1]) if not result.dropna().empty else 0.0

    return FormulaResponse(
        name=data.name,
        points=points,
        last=round(last_val, 6),
        error=None,
    )
