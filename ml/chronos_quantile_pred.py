"""Sprint 11.3 — POC zero-shot Chronos quantile prediction для MOEX close prices.

Гипотеза: Chronos (Amazon T5-based foundation model для time-series) обучен на 84B
наблюдениях и нативно выдаёт probabilistic quantile forecasts. Можем сравнить его
zero-shot quantiles с нашим custom QuantileOHLCHead (C-канал) и понять — стоит ли
заменять/дополнять собственную голову Chronos-предсказаниями.

Архитектура (гибрид по выбору пользователя):
1. Берём test_dates/test_tickers из ensemble_predictions.npz
2. Для каждого (ticker, date) грузим N последних close-цен через TinkoffDataClient (TTL-кеш)
3. Прогоняем через ChronosPipeline.predict(context, prediction_length=5, num_samples=20)
4. Извлекаем q10/q50/q90 из sample distribution для каждого бара t+1..t+5
5. Конвертируем в ATR-norm units (соответствует layout quantile_pred[:, 45:60] C-channel)
6. Сохраняем chronos_close_quantiles.npz; quantile_eval.py подхватит для сравнения

Запуск:
    pip install chronos-forecasting
    py -m ml.chronos_quantile_pred --variant tiny --max-samples 200
    py -m ml.chronos_quantile_pred --variant tiny --max-samples 0   # 0 = все 19076

Варианты:
    tiny  — amazon/chronos-t5-tiny  (8M, fast)
    mini  — amazon/chronos-t5-mini  (20M)
    small — amazon/chronos-t5-small (46M)
    bolt  — amazon/chronos-bolt-tiny (быстрый bolt)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

NPZ_PATH   = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
OUT_PATH   = os.path.join(os.path.dirname(__file__), "ensemble", "chronos_close_quantiles.npz")
CTX_LEN    = 64        # длина контекстного окна для Chronos (~3 мес daily)
PRED_LEN   = 5         # future_bars
N_SAMPLES  = 20        # сколько траекторий sample'ить для quantile extraction
QUANTILES  = (0.10, 0.50, 0.90)

VARIANT_MAP = {
    "tiny":  "amazon/chronos-t5-tiny",
    "mini":  "amazon/chronos-t5-mini",
    "small": "amazon/chronos-t5-small",
    "bolt":  "amazon/chronos-bolt-tiny",
}


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
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip(); v = v.strip().strip("'").strip('"')
                if k and k not in os.environ:
                    os.environ[k] = v


def load_close_series_by_ticker(client, tickers_set: set[str]) -> dict[str, "pd.DataFrame"]:
    """Загружает daily candles для всех тикеров (TTL-кеш Tinkoff client)."""
    import pandas as pd
    out: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(sorted(tickers_set), 1):
        figi = client.find_figi(ticker)
        if not figi:
            print(f"  [WARN] {ticker}: figi not found"); continue
        df = client.get_candles(figi=figi, interval="1d", days_back=400, use_cache=True)
        if df is None or len(df) < CTX_LEN + PRED_LEN:
            print(f"  [WARN] {ticker}: only {0 if df is None else len(df)} candles, skip")
            continue
        out[ticker] = df
        if i % 10 == 0 or i == len(tickers_set):
            print(f"  [{i}/{len(tickers_set)}] {ticker}: {len(df)} candles")
    return out


def build_contexts(
    test_dates: np.ndarray,
    test_tickers: np.ndarray,
    close_by_ticker: dict,
    ctx_len: int = CTX_LEN,
) -> tuple[list[np.ndarray], list[int], list[float], list[float]]:
    """Для каждого (date, ticker) собирает (ctx_len)-окно close до date-1.

    Returns:
      contexts:   list of [ctx_len] numpy arrays (close prices)
      valid_idx:  индексы в test_dates, для которых контекст найден
      last_close: список цен закрытия в день D (для денормировки)
      atr_proxy:  ATR-прокси (mean abs daily return * sqrt(prediction_length))
    """
    import pandas as pd
    contexts: list[np.ndarray] = []
    valid_idx: list[int] = []
    last_close: list[float] = []
    atr_proxy: list[float] = []

    for i, (d_str, t) in enumerate(zip(test_dates, test_tickers)):
        d = str(d_str)[:10]
        df = close_by_ticker.get(t)
        if df is None:
            continue
        try:
            dt = pd.to_datetime(d)
            # tz fix: df.index — UTC; убираем tz для сравнения
            idx = df.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            # last bar строго ДО (или ==) date — будем брать ctx_len окно ровно до неё
            mask = idx <= dt
            pos = int(mask.sum()) - 1
            if pos < ctx_len:
                continue
            window = df["close"].to_numpy()[pos - ctx_len + 1 : pos + 1].astype(np.float32)
            if np.any(~np.isfinite(window)) or window[-1] <= 0:
                continue
            # ATR-прокси: среднее abs относительного изменения за последние 14 баров × sqrt(fb)
            last14 = window[-15:]
            rets = np.diff(last14) / last14[:-1].clip(min=1e-9)
            atr = float(np.abs(rets).mean() * np.sqrt(PRED_LEN))
            contexts.append(window)
            valid_idx.append(i)
            last_close.append(float(window[-1]))
            atr_proxy.append(atr)
        except Exception as exc:
            print(f"  [WARN] {t} {d}: {exc}")
            continue

    return contexts, valid_idx, last_close, atr_proxy


def predict_quantiles_batched(
    pipeline,
    contexts: list[np.ndarray],
    batch_size: int = 16,
    pred_len: int = PRED_LEN,
    n_samples: int = N_SAMPLES,
    quantiles: tuple[float, ...] = QUANTILES,
) -> np.ndarray:
    """Прогоняет Chronos батчами. Возвращает [N, len(quantiles), pred_len]."""
    n_q = len(quantiles)
    N = len(contexts)
    out = np.zeros((N, n_q, pred_len), dtype=np.float32)

    t0 = time.time()
    # Универсальный вызов: новые версии chronos-forecasting используют positional/inputs;
    # старые — context=. Пытаемся в порядке предпочтения; кэшируем результат на 1-м батче.
    call_mode: str | None = None

    def _call(batch):
        nonlocal call_mode
        if call_mode is None:
            # Поочерёдно пробуем варианты
            for mode in ("positional", "inputs", "context"):
                try:
                    if mode == "positional":
                        s = pipeline.predict(batch, prediction_length=pred_len,
                                             num_samples=n_samples)
                    elif mode == "inputs":
                        s = pipeline.predict(inputs=batch, prediction_length=pred_len,
                                             num_samples=n_samples)
                    else:
                        s = pipeline.predict(context=batch, prediction_length=pred_len,
                                             num_samples=n_samples)
                    call_mode = mode
                    print(f"  [Chronos API] call_mode={mode}")
                    return s
                except TypeError:
                    continue
            raise RuntimeError(
                "Ни один вариант ChronosPipeline.predict() не сработал. "
                "Версия chronos-forecasting несовместима."
            )
        if call_mode == "positional":
            return pipeline.predict(batch, prediction_length=pred_len,
                                    num_samples=n_samples)
        if call_mode == "inputs":
            return pipeline.predict(inputs=batch, prediction_length=pred_len,
                                    num_samples=n_samples)
        return pipeline.predict(context=batch, prediction_length=pred_len,
                                num_samples=n_samples)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[start:end]]
        samples = _call(batch)
        # samples: torch.Tensor [B, num_samples, pred_len] (chronos-bolt может возвращать
        # [B, 9, pred_len] фиксированных квантилей — обрабатываем оба случая)
        s_np = samples.detach().cpu().numpy().astype(np.float32) if hasattr(samples, "detach") else np.asarray(samples, dtype=np.float32)
        if s_np.shape[1] == n_samples:
            # T5: sampling-based → берём quantile across samples
            q = np.quantile(s_np, quantiles, axis=1)   # [n_q, B, pred_len]
            out[start:end] = q.transpose(1, 0, 2)
        else:
            # Bolt: возвращает фиксированные квантили [0.1..0.9 step 0.1] — индексы 0/4/8
            # для (0.1, 0.5, 0.9). Если другая раскладка — придётся скорректировать.
            n_fixed = s_np.shape[1]
            # Equispaced quantile levels
            levels = np.linspace(1, n_fixed, n_fixed) / (n_fixed + 1)
            for qi, q_target in enumerate(quantiles):
                idx = int(np.argmin(np.abs(levels - q_target)))
                out[start:end, qi] = s_np[:, idx, :]

        if (start // batch_size) % 5 == 0:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1e-6)
            eta = (N - end) / max(rate, 1e-6)
            print(f"  [{end}/{N}] {rate:.1f} samp/s · ETA {eta:.0f}s")

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANT_MAP.keys()), default="tiny",
                   help="Chronos вариант: tiny (8M, default) / mini / small / bolt")
    p.add_argument("--max-samples", type=int, default=200,
                   help="Лимит test-сэмплов для POC (0 = все ~19K). Default 200 для быстрого POC.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default=OUT_PATH)
    p.add_argument("--adapter", default=None,
                   help="Sprint 11.3: путь к LoRA-адаптеру (см. chronos_finetune.py)")
    args = p.parse_args()

    _bootstrap_env()

    # ── 1. Импортируем chronos pipeline ─────────────────────────────────
    try:
        from chronos import ChronosPipeline
    except ImportError:
        print("ERROR: chronos package не установлен. Запусти: pip install chronos-forecasting")
        sys.exit(1)

    model_name = VARIANT_MAP[args.variant]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Chronos] Загрузка {model_name} ({device})...")
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )
    print(f"[Chronos] Загружен")

    # Sprint 11.3: подключаем LoRA-адаптер если задан
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError:
            print("ERROR: --adapter указан, но peft не установлен. pip install peft")
            sys.exit(2)
        if not os.path.isdir(args.adapter):
            print(f"ERROR: адаптер не найден: {args.adapter}")
            sys.exit(3)
        print(f"[Chronos] Загрузка LoRA-адаптера: {args.adapter}")
        pipeline.model.model = PeftModel.from_pretrained(
            pipeline.model.model, args.adapter,
        )
        pipeline.model.model.eval()
        print(f"[Chronos] LoRA адаптер активирован")

    # ── 2. Загружаем test_dates/test_tickers ────────────────────────────
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} не найден"); sys.exit(2)
    d = np.load(NPZ_PATH, allow_pickle=True)
    test_dates   = np.array([str(x)[:10] for x in d["test_dates"]])
    test_tickers = np.array([str(x) for x in d["test_tickers"]])
    N_total = len(test_dates)
    print(f"\nN_total в ensemble_predictions.npz: {N_total}")
    if args.max_samples > 0 and args.max_samples < N_total:
        # Random-sample детерминированно (seed=42), чтобы охватить разные тикеры/даты.
        rng = np.random.RandomState(42)
        idx_pool = rng.choice(N_total, size=args.max_samples, replace=False)
        idx_pool.sort()
        test_dates   = test_dates[idx_pool]
        test_tickers = test_tickers[idx_pool]
        n_uniq_t = len(set(test_tickers))
        print(f"POC: random-sample {len(test_dates)} сэмплов (seed=42), "
              f"{n_uniq_t} уникальных тикеров")

    # ── 3. Грузим daily candles для всех тикеров через Tinkoff (TTL-кеш) ─
    from data.tinkoff_factory import get_client
    client = get_client()
    print(f"\nЗагрузка daily candles ({len(set(test_tickers))} тикеров)...")
    close_by_ticker = load_close_series_by_ticker(client, set(test_tickers))

    # ── 4. Строим контексты ─────────────────────────────────────────────
    print(f"\nСтроим контексты (ctx_len={CTX_LEN})...")
    contexts, valid_idx, last_close, atr_proxy = build_contexts(
        test_dates, test_tickers, close_by_ticker, ctx_len=CTX_LEN,
    )
    print(f"  Валидных сэмплов: {len(contexts)}/{len(test_dates)}")
    if not contexts:
        print("ERROR: ни одного валидного контекста"); sys.exit(3)

    # ── 5. Predict ─────────────────────────────────────────────────────
    print(f"\nChronos predict (batch={args.batch_size}, n_samples={N_SAMPLES})...")
    q_pred_raw = predict_quantiles_batched(
        pipeline, contexts,
        batch_size=args.batch_size, pred_len=PRED_LEN,
        n_samples=N_SAMPLES, quantiles=QUANTILES,
    )
    # q_pred_raw: [N_valid, 3, 5] — в RAW close prices

    # ── 6. Конверсия в ATR-norm relative C-deltas ──────────────────────
    # Наш quantile_pred[:, 45:60] = C-канал в ATR-norm: (ΔC_pct / atr_ratio / sqrt(fb))
    # Chronos выдаёт ABSOLUTE prices → конвертируем: ΔC = (chronos - last_close) / last_close
    # Затем normalize → ATR-units (через atr_proxy чтобы не зависеть от npz["atr_ratio"])
    last_close_arr = np.array(last_close, dtype=np.float32)[:, None, None]      # [N, 1, 1]
    atr_arr        = np.array(atr_proxy, dtype=np.float32)[:, None, None]       # [N, 1, 1]
    rel_pct  = (q_pred_raw - last_close_arr) / last_close_arr.clip(min=1e-9)    # ΔC relative
    rel_atr  = rel_pct / atr_arr.clip(min=1e-9)                                  # ATR-normalized

    # ── 7. Сохраняем ────────────────────────────────────────────────────
    valid_idx_arr = np.array(valid_idx, dtype=np.int64)
    out = {
        "chronos_close_q":        q_pred_raw,         # raw price quantiles [N_valid, 3, 5]
        "chronos_close_rel_pct":  rel_pct.astype(np.float32),    # ΔC/C as fraction
        "chronos_close_rel_atr":  rel_atr.astype(np.float32),    # ATR-normalized (compare to our C-channel)
        "valid_idx":              valid_idx_arr,      # индексы в test_dates/test_tickers
        "test_dates":             test_dates[valid_idx_arr],
        "test_tickers":           test_tickers[valid_idx_arr],
        "last_close":             np.array(last_close, dtype=np.float32),
        "atr_proxy":              np.array(atr_proxy, dtype=np.float32),
        "quantiles":              np.array(QUANTILES, dtype=np.float32),
        "model_name":             model_name,
        "ctx_len":                CTX_LEN,
        "pred_len":               PRED_LEN,
        "n_samples":              N_SAMPLES,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print(f"\n✅ Сохранено: {args.out}")
    print(f"   shape chronos_close_q: {q_pred_raw.shape}")
    print(f"   валидных: {len(valid_idx)}")
    print(f"\nДля сравнения с QuantileOHLCHead: см. quantile_eval с флагом --chronos (TODO).")


if __name__ == "__main__":
    main()
