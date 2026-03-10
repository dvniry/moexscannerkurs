import re
import ast
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
import logging

from .base import Indicator
from .library import SMA, EMA, RSI, MACD, BollingerBands, ATR

logger = logging.getLogger(__name__)


# ─── Запрещённые паттерны (текстовый фильтр) ─────────────

_FORBIDDEN_PATTERNS = [
    r'\bimport\b',
    r'\b__\w+__\b',       # __import__, __builtins__, __class__ и т.д.
    r'\bopen\b',
    r'\bexec\b',
    r'\beval\b',
    r'\bcompile\b',
    r'\bgetattr\b',
    r'\bsetattr\b',
    r'\bdelattr\b',
    r'\bhasattr\b',
    r'\bglobals\b',
    r'\blocals\b',
    r'\bvars\b',
    r'\bdir\b',
    r'\bsubprocess\b',
    r'\bos\b',
    r'\bsys\b',
    r'\bshutil\b',
    r'\bpickle\b',
    r'\bsocket\b',
]

_FORBIDDEN_RE = re.compile('|'.join(_FORBIDDEN_PATTERNS), re.IGNORECASE)


# ─── AST whitelist разрешённых узлов ─────────────────────

_ALLOWED_AST_NODES = {
    # Структура
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign,
    # Литералы
    ast.Constant,
    # Операции
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,  # Mul → Mult
    ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    # Имена и вызовы
    ast.Name, ast.Load, ast.Store,
    ast.Call, ast.Attribute,
    # Коллекции
    ast.Tuple, ast.List,
    # Условия
    ast.IfExp,
    # Python 3.12+
    ast.arguments, ast.arg,
}


def _check_ast(code: str, formula_name: str) -> None:
    """Проверяем AST — только разрешённые узлы."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"Синтаксическая ошибка в формуле '{formula_name}': {e}")

    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in _ALLOWED_AST_NODES:
            raise SecurityError(
                f"Запрещённая конструкция в формуле '{formula_name}': "
                f"{node_type.__name__}. "
                f"Разрешены только математические выражения."
            )


class SecurityError(ValueError):
    """Формула содержит запрещённые конструкции."""
    pass


# ─── Встроенные функции формул ────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False).mean()

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=int(period)).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=int(period), adjust=False).mean()
    avg_loss = loss.ewm(span=int(period), adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _macd_line(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    return series.ewm(span=int(fast), adjust=False).mean() \
         - series.ewm(span=int(slow), adjust=False).mean()

def _bb_upper(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    m = series.rolling(int(period)).mean()
    s = series.rolling(int(period)).std()
    return m + s * float(std)

def _bb_lower(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    m = series.rolling(int(period)).mean()
    s = series.rolling(int(period)).std()
    return m - s * float(std)

def _bb_middle(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(int(period)).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=int(period), adjust=False).mean()

def _rolling_max(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period)).max()

def _rolling_min(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period)).min()

def _std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period)).std()

def _prev(series: pd.Series) -> pd.Series:
    return series.shift(1)

def _cross_up(a, b) -> pd.Series:
    if isinstance(a, pd.Series):
        idx = a.index
    elif isinstance(b, pd.Series):
        idx = b.index
    else:
        raise TypeError("CROSS_UP: хотя бы один аргумент должен быть Series")
    if not isinstance(a, pd.Series):
        a = pd.Series(float(a), index=idx)
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=idx)
    return ((a.shift(1) <= b.shift(1)) & (a > b)).astype(float)

def _cross_down(a, b) -> pd.Series:
    if isinstance(a, pd.Series):
        idx = a.index
    elif isinstance(b, pd.Series):
        idx = b.index
    else:
        raise TypeError("CROSS_DOWN: хотя бы один аргумент должен быть Series")
    if not isinstance(a, pd.Series):
        a = pd.Series(float(a), index=idx)
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=idx)
    return ((a.shift(1) >= b.shift(1)) & (a < b)).astype(float)

def _if(cond, true_val, false_val) -> pd.Series:
    if isinstance(cond, pd.Series):
        return pd.Series(np.where(cond, true_val, false_val), index=cond.index)
    return true_val if cond else false_val


# ─── Безопасный namespace (без builtins) ─────────────────

def _make_namespace(df: pd.DataFrame, params_obj) -> dict:
    return {
        '__builtins__': {},          # ← полностью отключаем builtins
        # Функции индикаторов
        '_ema': _ema, '_sma': _sma, '_rsi': _rsi,
        '_macd_line': _macd_line,   '_atr': _atr,
        '_bb_upper': _bb_upper,     '_bb_lower': _bb_lower,
        '_bb_middle': _bb_middle,
        '_rolling_max': _rolling_max, '_rolling_min': _rolling_min,
        '_std': _std,               '_prev': _prev,
        '_cross_up': _cross_up,     '_cross_down': _cross_down,
        '_if': _if,
        # OHLCV
        'CLOSE':  df['close'],
        'OPEN':   df['open'],
        'HIGH':   df['high'],
        'LOW':    df['low'],
        'VOLUME': df['volume'],
        '_df':    df,
        # Параметры
        '_params': params_obj,
        # Минимум numpy без опасных функций
        'np': np,
        'pd': pd,
    }


# ─── Класс Formula ────────────────────────────────────────

_MAX_FORMULA_LENGTH = 500   # символов


class Formula(Indicator):

    def __init__(
        self,
        name: str,
        formula: str,
        params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, params=params or {})
        self.formula = formula.strip()
        self._params_obj = type('P', (), params or {})()

        # Проверка длины
        if len(self.formula) > _MAX_FORMULA_LENGTH:
            raise ValueError(
                f"Формула '{name}' слишком длинная "
                f"(максимум {_MAX_FORMULA_LENGTH} символов)"
            )

        # Уровень 1: текстовый фильтр
        match = _FORBIDDEN_RE.search(self.formula)
        if match:
            raise SecurityError(
                f"Запрещённое слово в формуле '{name}': '{match.group()}'"
            )

        # Перевод в Python
        self._python_code = self._translate(self.formula)

        # Уровень 2: AST проверка
        _check_ast(self._python_code, name)

        # Компиляция
        try:
            self._compiled = compile(
                self._python_code,
                f'<formula:{name}>',
                'exec'
            )
        except SyntaxError as e:
            raise SyntaxError(
                f"Ошибка компиляции формулы '{name}':\n"
                f"  Формула: {formula}\n"
                f"  Код:     {self._python_code}\n"
                f"  Ошибка:  {e}"
            )

    def _translate(self, formula: str) -> str:
        code = formula

        # EMA(9) → _ema(CLOSE, 9)
        code = re.sub(r'\bEMA\((\d+)\)',    r'_ema(CLOSE, \1)',  code)
        code = re.sub(r'\bSMA\((\d+)\)',    r'_sma(CLOSE, \1)',  code)
        code = re.sub(r'\bRSI\((\d+)\)',    r'_rsi(CLOSE, \1)',  code)
        code = re.sub(r'\bATR\((\d+)\)',    r'_atr(_df, \1)',    code)

        # С явной series
        code = re.sub(r'\bEMA\(',           '_ema(',             code)
        code = re.sub(r'\bSMA\(',           '_sma(',             code)
        code = re.sub(r'\bRSI\(',           '_rsi(',             code)
        code = re.sub(r'\bMACD\(',          '_macd_line(CLOSE, ',code)
        code = re.sub(r'\bBB_UPPER\(',      '_bb_upper(CLOSE, ', code)
        code = re.sub(r'\bBB_LOWER\(',      '_bb_lower(CLOSE, ', code)
        code = re.sub(r'\bBB_MIDDLE\(',     '_bb_middle(CLOSE, ',code)
        code = re.sub(r'\bMAX\(',           '_rolling_max(',     code)
        code = re.sub(r'\bMIN\(',           '_rolling_min(',     code)
        code = re.sub(r'\bSTD\(',           '_std(',             code)
        code = re.sub(r'\bPREV\(',          '_prev(',            code)
        code = re.sub(r'\bCROSS_UP\(',      '_cross_up(',        code)
        code = re.sub(r'\bCROSS_DOWN\(',    '_cross_down(',      code)
        code = re.sub(r'\bIF\(',            '_if(',              code)
        code = re.sub(r'\bparams\.',        '_params.',          code)

        return code

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        # Уровень 3: изолированный namespace
        namespace = _make_namespace(df, self._params_obj)

        try:
            exec(self._compiled, namespace)
        except SecurityError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Ошибка в формуле '{self.name}': "
                f"{type(e).__name__}: {e}\n"
                f"Формула: {self.formula}"
            )

        if 'RESULT' not in namespace:
            raise ValueError(
                f"Формула '{self.name}': переменная RESULT не найдена. "
                f"Добавьте: RESULT = ..."
            )

        result = namespace['RESULT']
        if isinstance(result, (pd.Series, pd.DataFrame)):
            result.index = df.index

        return result

    def __repr__(self) -> str:
        return f"Formula(name={self.name!r}, formula={self.formula!r})"
