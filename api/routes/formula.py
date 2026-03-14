"""POST /api/formula — расчёт пользовательской формулы."""
import os, sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from litestar import post
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from indicators.formula    import Formula
from api.routes._common    import MAX_DAYS, run_in_thread
from api.routes.candles    import get_client, _to_unix


# ── Датаклассы ────────────────────────────────────────────

@dataclass
class FormulaRequest:
    ticker:   str
    formula:  str
    name:     str            = "Custom"
    interval: str            = "1h"
    days:     Optional[int]  = None
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


# ── Логика ────────────────────────────────────────────────

def _fetch_and_calculate(req: FormulaRequest) -> tuple:
    client = get_client()
    figi   = client.find_figi(req.ticker.upper())
    days   = req.days or MAX_DAYS.get(req.interval, 7)
    df     = client.get_candles(figi=figi, interval=req.interval, days_back=days)
    result = Formula(name=req.name, formula=req.formula, params=req.params)(df)
    return df, result


# ── Endpoint ──────────────────────────────────────────────

@post("/formula")
async def calculate_formula(data: FormulaRequest) -> FormulaResponse:
    try:
        df, result = await run_in_thread(lambda: _fetch_and_calculate(data))
    except (SyntaxError, ValueError, RuntimeError) as e:
        return FormulaResponse(name=data.name, points=[], last=0.0, error=str(e))

    unix     = _to_unix(df)
    points   = [
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
