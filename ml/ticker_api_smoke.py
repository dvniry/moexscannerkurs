# ml/ticker_api_smoke.py
"""
Smoke-test доступности тикеров через T-Bank Invest API.

Для каждого тикера из CFG.tickers проверяет:
  1. find_figi  — инструмент найден (TQBR / BBG FIGI)
  2. candles    — фактически возвращаются свечи за последние 30 дней (1d)
  3. history    — минимальный размер истории (≥ 500 дневных свечей ≈ 2 года)

Запуск:
    python -m ml.ticker_api_smoke                  # все тикеры
    python -m ml.ticker_api_smoke --new-only        # только новые тикеры
    python -m ml.ticker_api_smoke --ticker SIBN TATNP POSI  # конкретные
"""
import os, sys, time, argparse
os.environ['GRPC_DNS_RESOLVER'] = 'native'

_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ml.config import CFG

GREEN  = '\033[92m'; RED    = '\033[91m'; YELLOW = '\033[93m'
CYAN   = '\033[96m'; RESET  = '\033[0m';  BOLD   = '\033[1m'
DIM    = '\033[2m'

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg): print(f"  {CYAN}·{RESET} {msg}")

# Тикеры которые добавили в этой сессии (для подсветки)
NEW_TICKERS = {
    "SBERP", "TATNP", "SIBN", "ENPG", "MTLR", "RASP", "TRMK",
    "POSI", "FIXP", "LSRG", "NMTP", "FESH", "OGKB", "NKNC", "KAZT",
}

MIN_CANDLES_RECENT = 5     # за 30 дней (торговых ~21, но берём с запасом)
MIN_CANDLES_HISTORY = 500  # ≈ 2 года дневных свечей


def check_ticker(client, ticker: str) -> dict:
    result = {
        "ticker":   ticker,
        "figi":     None,
        "uid":      None,
        "recent":   0,    # свечей за 30 дней
        "history":  0,    # свечей за 3 года
        "first_dt": None,
        "last_dt":  None,
        "status":   "FAIL",
        "note":     "",
    }

    # ── Шаг 1: поиск FIGI ──────────────────────────────────────────
    try:
        figi = client.find_figi(ticker)
    except Exception as e:
        result["note"] = f"find_figi error: {e}"
        return result

    if figi is None:
        result["note"] = "FIGI не найден (нет инструмента TQBR/BBG)"
        return result

    result["figi"] = figi

    # Также сохраним UID если закешировался
    result["uid"] = client._uid_cache.get(ticker)

    # ── Шаг 2: свежие свечи (30 дней) ────────────────────────────
    try:
        df_recent = client._load_candles_chunked(
            figi=figi, interval="1d", days_back=30
        )
        result["recent"] = len(df_recent)
        if len(df_recent) > 0:
            result["last_dt"] = df_recent.index[-1].strftime("%Y-%m-%d")
    except Exception as e:
        result["note"] = f"candles(30d) error: {e}"
        return result

    if result["recent"] < MIN_CANDLES_RECENT:
        result["note"] = f"только {result['recent']} свечей за 30 дней"
        result["status"] = "WARN"
        return result

    # ── Шаг 3: глубина истории (3 года) ──────────────────────────
    try:
        df_hist = client._load_candles_chunked(
            figi=figi, interval="1d", days_back=1095  # 3 года
        )
        result["history"] = len(df_hist)
        if len(df_hist) > 0:
            result["first_dt"] = df_hist.index[0].strftime("%Y-%m-%d")
    except Exception as e:
        result["note"] = f"history error: {e}"
        result["status"] = "WARN"
        return result

    if result["history"] < MIN_CANDLES_HISTORY:
        result["status"] = "WARN"
        result["note"] = f"история только {result['history']} свечей (< {MIN_CANDLES_HISTORY})"
    else:
        result["status"] = "OK"

    return result


def run(tickers: list[str]):
    token = os.getenv("TINKOFF_TOKEN", "")
    if not token:
        print(f"{RED}TINKOFF_TOKEN не задан в .env{RESET}")
        sys.exit(1)

    from data.tinkoff_client import TinkoffDataClient
    client = TinkoffDataClient(token)

    results = []
    total = len(tickers)

    print(f"\n{BOLD}{CYAN}{'═'*65}{RESET}")
    print(f"{BOLD}  Ticker API Smoke Test — {total} тикеров{RESET}")
    print(f"{BOLD}{CYAN}{'═'*65}{RESET}\n")

    for i, ticker in enumerate(tickers, 1):
        new_mark = f" {YELLOW}[NEW]{RESET}" if ticker in NEW_TICKERS else ""
        print(f"[{i:2d}/{total}] {BOLD}{ticker}{RESET}{new_mark} ", end="", flush=True)

        t0 = time.time()
        r  = check_ticker(client, ticker)
        elapsed = time.time() - t0

        results.append(r)

        status = r["status"]
        if status == "OK":
            color = GREEN
            detail = f"FIGI={r['figi']}  hist={r['history']}d  {r['first_dt']}→{r['last_dt']}"
        elif status == "WARN":
            color = YELLOW
            detail = r["note"]
        else:
            color = RED
            detail = r["note"]

        print(f"[{color}{status}{RESET}]  {DIM}{detail}  ({elapsed:.1f}s){RESET}")

        # Пауза чтобы не перегружать API
        time.sleep(0.3)

    # ── Сводная таблица ───────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*65}{RESET}")
    print(f"{BOLD}  ИТОГО{RESET}")
    print(f"{BOLD}{CYAN}{'═'*65}{RESET}")

    ok_list   = [r["ticker"] for r in results if r["status"] == "OK"]
    warn_list = [r["ticker"] for r in results if r["status"] == "WARN"]
    fail_list = [r["ticker"] for r in results if r["status"] == "FAIL"]

    print(f"\n  {GREEN}✓ OK   ({len(ok_list)}){RESET}:   {', '.join(ok_list)}")
    if warn_list:
        print(f"  {YELLOW}⚠ WARN ({len(warn_list)}){RESET}:   {', '.join(warn_list)}")
    if fail_list:
        print(f"  {RED}✗ FAIL ({len(fail_list)}){RESET}:   {', '.join(fail_list)}")

    if fail_list or warn_list:
        print(f"\n  {BOLD}Рекомендация:{RESET} удали из ml/config.py:")
        remove = fail_list + [r["ticker"] for r in results
                              if r["status"] == "WARN" and r["recent"] == 0]
        for t in remove:
            r = next(x for x in results if x["ticker"] == t)
            print(f"    {RED}✗ {t:8s}{RESET}  {DIM}{r['note']}{RESET}")

    print(f"\n{BOLD}{CYAN}{'═'*65}{RESET}\n")

    return fail_list, warn_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка тикеров через T-Bank API")
    parser.add_argument("--new-only",  action="store_true",
                        help="Проверить только новые тикеры")
    parser.add_argument("--ticker",    nargs="+", metavar="TICKER",
                        help="Конкретные тикеры (по умолчанию — все из CFG)")
    args = parser.parse_args()

    if args.ticker:
        tickers = [t.upper() for t in args.ticker]
    elif args.new_only:
        tickers = [t for t in CFG.tickers if t in NEW_TICKERS]
    else:
        tickers = list(CFG.tickers)

    fail_list, warn_list = run(tickers)
    sys.exit(1 if fail_list else 0)
