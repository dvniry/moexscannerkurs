"""Sprint 11.3 — LoRA fine-tune chronos-t5-tiny на MOEX close-series.

Цель: исправить anti-signal directional output Chronos zero-shot (win 40-44%
ниже coin-flip) через адаптацию на MOEX-данных. LoRA добавляет r×8 rank матрицы
к attention слоям T5, не трогая 8M базовых весов.

Архитектура:
  1. Dataset: sliding window (ctx_len=64 → pred_len=5) по close-series тикеров
  2. Train/val split: chronological (80/20) ДО min(test_dates) из ensemble_predictions.npz
     — чтобы LoRA НЕ видел test-период (avoid leakage)
  3. ChronosPipeline.tokenizer.context_input_transform(context) → token_ids, scale
     ChronosPipeline.tokenizer.label_input_transform(label, scale) → label tokens
  4. T5 forward(input_ids, attention_mask, labels=label_ids).loss — native cross-entropy
     на quantile tokens (то, на чём Chronos pretrained)
  5. peft LoRA: target_modules=["q","v"], rank=8 — стандарт для T5 attention
  6. Save adapter to ml/ensemble/chronos_lora_adapter/

Запуск:
    pip install peft accelerate
    py -m ml.chronos_finetune                                # POC: 1 epoch, default
    py -m ml.chronos_finetune --epochs 3 --batch-size 8     # дольше
    py -m ml.chronos_finetune --rank 16                     # больше LoRA capacity
    py -m ml.chronos_finetune --eval-only                   # только val pinball без обучения

После обучения для использования адаптера в chronos_quantile_pred:
    py -m ml.chronos_quantile_pred --variant tiny --adapter ml/ensemble/chronos_lora_adapter --max-samples 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NPZ_PATH    = os.path.join(os.path.dirname(__file__), "ensemble", "ensemble_predictions.npz")
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "ensemble", "chronos_lora_adapter")
CTX_LEN     = 64
PRED_LEN    = 5

VARIANT_MAP = {
    "tiny":  "amazon/chronos-t5-tiny",
    "mini":  "amazon/chronos-t5-mini",
    "small": "amazon/chronos-t5-small",
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


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────

class ChronosCloseDataset(Dataset):
    """Sliding window over close-series. Возвращает (context[ctx_len], label[pred_len])."""

    def __init__(self, close_series: list[np.ndarray], ctx_len: int, pred_len: int, stride: int = 1):
        self.ctx_len  = ctx_len
        self.pred_len = pred_len
        self.windows: list[tuple[int, int]] = []   # (series_idx, start_pos)
        self.series  = close_series
        for si, s in enumerate(close_series):
            N = len(s)
            max_start = N - ctx_len - pred_len
            if max_start <= 0:
                continue
            for pos in range(0, max_start + 1, stride):
                self.windows.append((si, pos))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        si, pos = self.windows[idx]
        s = self.series[si]
        ctx   = s[pos : pos + self.ctx_len]
        label = s[pos + self.ctx_len : pos + self.ctx_len + self.pred_len]
        return (torch.from_numpy(ctx.astype(np.float32)),
                torch.from_numpy(label.astype(np.float32)))


def build_close_series(tickers: list[str], cutoff_date: str | None,
                       min_len: int = CTX_LEN + PRED_LEN + 50) -> list[np.ndarray]:
    """Загружает close-серии для каждого тикера через TinkoffDataClient TTL-кеш.

    cutoff_date: если задано, серии обрезаются ДО этой даты (исключаем test-период из train).
    """
    import pandas as pd
    from data.tinkoff_factory import get_client
    client = get_client()
    out: list[np.ndarray] = []
    cutoff = pd.to_datetime(cutoff_date) if cutoff_date else None

    for i, ticker in enumerate(sorted(tickers), 1):
        figi = client.find_figi(ticker)
        if not figi:
            print(f"  [WARN] {ticker}: figi не найден"); continue
        df = client.get_candles(figi=figi, interval="1d", days_back=2000, use_cache=True)
        if df is None or len(df) < min_len:
            print(f"  [WARN] {ticker}: only {0 if df is None else len(df)} candles, skip")
            continue
        idx = df.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        df.index = idx
        if cutoff is not None:
            df = df[df.index < cutoff]
        close = df["close"].to_numpy(dtype=np.float32)
        if len(close) < min_len:
            continue
        if np.any(~np.isfinite(close)) or np.any(close <= 0):
            print(f"  [WARN] {ticker}: invalid close values, skip"); continue
        out.append(close)
        if i % 10 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] {ticker}: {len(close)} bars (cutoff {cutoff_date})")
    return out


# ────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────

def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    contexts = torch.stack([b[0] for b in batch], dim=0)   # [B, ctx_len]
    labels   = torch.stack([b[1] for b in batch], dim=0)   # [B, pred_len]
    return contexts, labels


def _label_tokens_no_assert(tokenizer, label, scale):
    """Токенизация label без assert prediction_length==model.config.prediction_length.

    Chronos pretrained c prediction_length=64, но fine-tune на 5-bar horizon —
    обходим стандартную `label_input_transform` и вызываем `_input_transform`
    напрямую, затем добавляем EOS токен если конфиг требует.
    """
    token_ids, attention_mask, _ = tokenizer._input_transform(context=label, scale=scale)
    if getattr(tokenizer.config, "use_eos_token", False):
        token_ids, attention_mask = tokenizer._append_eos_token(token_ids, attention_mask)
    return token_ids, attention_mask


def step_train(model, tokenizer, batch, device):
    """Один forward/backward."""
    ctx, lab = batch
    ctx_tokens, ctx_mask, scale = tokenizer.context_input_transform(ctx)
    lab_tokens, _               = _label_tokens_no_assert(tokenizer, lab, scale)

    out = model(
        input_ids=ctx_tokens.to(device),
        attention_mask=ctx_mask.to(device),
        labels=lab_tokens.to(device),
    )
    return out.loss


@torch.no_grad()
def step_eval(pipeline, model, batch, device, num_samples: int = 20):
    """Predict quantile samples и считает pinball loss vs actual label."""
    ctx, lab = batch
    # Используем pipeline.predict (no_grad), но через нашу model (которая может быть LoRA-wrapped)
    pipeline.model.model = model   # для совместимости
    samples = pipeline.predict(
        list(ctx),
        prediction_length=lab.shape[1],
        num_samples=num_samples,
    )
    s_np = samples.detach().cpu().numpy().astype(np.float32)
    # Quantiles
    q10, q50, q90 = np.quantile(s_np, [0.1, 0.5, 0.9], axis=1)
    lab_np = lab.cpu().numpy()
    # Pinball per quantile
    def pb(p, q):
        e = lab_np - p
        return np.where(e >= 0, q * e, (q - 1) * e).mean()
    return {
        "pb_q10": float(pb(q10, 0.1)),
        "pb_q50": float(pb(q50, 0.5)),
        "pb_q90": float(pb(q90, 0.9)),
        "mse_q50": float(((q50 - lab_np) ** 2).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANT_MAP.keys()), default="tiny")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--alpha", type=int, default=16, help="LoRA alpha (обычно 2×rank)")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--stride", type=int, default=2,
                   help="Stride для sliding window (1=макс overlap, 5=disjoint windows)")
    p.add_argument("--max-val-batches", type=int, default=50,
                   help="Сколько val батчей для оценки (limit для скорости)")
    p.add_argument("--adapter-dir", default=ADAPTER_DIR)
    p.add_argument("--eval-only", action="store_true",
                   help="Только оценить текущий (base или с адаптером) — без training")
    args = p.parse_args()

    _bootstrap_env()

    # ── 1. Зависимости ───────────────────────────────────────
    try:
        from chronos import ChronosPipeline
    except ImportError:
        print("ERROR: chronos package. pip install chronos-forecasting"); sys.exit(1)
    try:
        from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    except ImportError:
        print("ERROR: peft не установлен. pip install peft"); sys.exit(2)

    # ── 2. Load Chronos ───────────────────────────────────────
    model_name = VARIANT_MAP[args.variant]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Chronos] Загрузка {model_name} ({device})...")
    pipeline = ChronosPipeline.from_pretrained(
        model_name, device_map=device, torch_dtype=torch.float32,
    )
    tokenizer = pipeline.tokenizer
    t5_base = pipeline.model.model
    print(f"[Chronos] Base model params: "
          f"{sum(p.numel() for p in t5_base.parameters()):,}")

    # ── 3. Берём cutoff = min(test_dates), чтобы train не пересекался ─
    cutoff_date = None
    tickers: list[str]
    if os.path.exists(NPZ_PATH):
        d = np.load(NPZ_PATH, allow_pickle=True)
        td = [str(x)[:10] for x in d["test_dates"]]
        # NumPy 2.x: .min() на string-массиве не работает → используем Python min на list
        cutoff_date = min(td)
        tickers = sorted(set(str(t) for t in d["test_tickers"]))
        print(f"\n[data] Cutoff (min test_date): {cutoff_date}")
        print(f"[data] Tickers из ensemble: {len(tickers)}")
    else:
        from ml.config import CFG
        tickers = list(CFG.tickers)
        print(f"[data] ensemble_predictions.npz не найден; берём CFG.tickers ({len(tickers)})")

    # ── 4. Build close-series + dataset ───────────────────────
    print(f"\n[data] Загрузка close-series...")
    series = build_close_series(tickers, cutoff_date=cutoff_date)
    if not series:
        print("ERROR: ни одной серии не загружено"); sys.exit(3)
    print(f"[data] Загружено {len(series)} серий, "
          f"total bars: {sum(len(s) for s in series):,}")

    # Train/val split: разделяем КАЖДУЮ серию хронологически (80/20)
    train_series = [s[:int(len(s) * 0.8)] for s in series]
    val_series   = [s[int(len(s) * 0.8):] for s in series]

    train_ds = ChronosCloseDataset(train_series, CTX_LEN, PRED_LEN, stride=args.stride)
    val_ds   = ChronosCloseDataset(val_series,   CTX_LEN, PRED_LEN, stride=PRED_LEN)
    print(f"[data] Train windows: {len(train_ds):,}  Val windows: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_batch, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_batch, num_workers=0)

    # ── 5. Eval-only path (sanity check before training) ─────
    if args.eval_only:
        print(f"\n[eval-only] Pinball loss на val ({min(args.max_val_batches, len(val_loader))} батчей)...")
        metrics_all = []
        for i, b in enumerate(val_loader):
            if i >= args.max_val_batches: break
            metrics_all.append(step_eval(pipeline, t5_base, b, device))
        agg = {k: np.mean([m[k] for m in metrics_all]) for k in metrics_all[0]}
        print(f"  pb_q10={agg['pb_q10']:.4f}  pb_q50={agg['pb_q50']:.4f}  "
              f"pb_q90={agg['pb_q90']:.4f}  mean={(agg['pb_q10']+agg['pb_q50']+agg['pb_q90'])/3:.4f}")
        print(f"  mse_q50={agg['mse_q50']:.4f}")
        return

    # ── 6. Apply LoRA ─────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=["q", "v"],
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    t5_lora = get_peft_model(t5_base, lora_config)
    n_trainable = sum(p.numel() for p in t5_lora.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in t5_lora.parameters())
    print(f"\n[LoRA] rank={args.rank}, alpha={args.alpha}, target=q,v")
    print(f"[LoRA] Trainable: {n_trainable:,} / {n_total:,} ({n_trainable/n_total*100:.2f}%)")

    # Wire back into pipeline.model.model for eval calls
    pipeline.model.model = t5_lora

    # ── 7. Optimizer + scheduler ──────────────────────────────
    optim = torch.optim.AdamW(
        [p for p in t5_lora.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )

    # ── 8. Training loop ──────────────────────────────────────
    print(f"\n[train] {args.epochs} epoch × {len(train_loader)} batch = {total_steps} steps")
    print(f"[train] batch_size={args.batch_size}, lr={args.lr}, stride={args.stride}")
    t0 = time.time()
    best_val_pb = float("inf")

    for ep in range(args.epochs):
        t5_lora.train()
        running_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader):
            optim.zero_grad()
            loss = step_train(t5_lora, tokenizer, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(t5_lora.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1
            if step % 50 == 0:
                elapsed = time.time() - t0
                done = ep * len(train_loader) + step + 1
                rate = done / max(elapsed, 1e-6)
                eta = (total_steps - done) / max(rate, 1e-6)
                print(f"  [E{ep+1}/{args.epochs} S{step}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}  lr={optim.param_groups[0]['lr']:.2e}  "
                      f"ETA {eta:.0f}s")

        train_loss = running_loss / max(n_batches, 1)

        # Validation
        t5_lora.eval()
        val_metrics = []
        for i, b in enumerate(val_loader):
            if i >= args.max_val_batches: break
            val_metrics.append(step_eval(pipeline, t5_lora, b, device))
        if val_metrics:
            agg = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
            val_pb = (agg["pb_q10"] + agg["pb_q50"] + agg["pb_q90"]) / 3
            print(f"\n  [E{ep+1}] train_loss={train_loss:.4f}  "
                  f"val pb_q10={agg['pb_q10']:.4f} pb_q50={agg['pb_q50']:.4f} "
                  f"pb_q90={agg['pb_q90']:.4f}  mean={val_pb:.4f}  "
                  f"mse_q50={agg['mse_q50']:.4f}")
            if val_pb < best_val_pb:
                best_val_pb = val_pb
                Path(args.adapter_dir).mkdir(parents=True, exist_ok=True)
                t5_lora.save_pretrained(args.adapter_dir)
                print(f"  ⭐ best val_pb={val_pb:.4f} → saved {args.adapter_dir}")

    print(f"\n[done] Best val pinball mean: {best_val_pb:.4f}")
    print(f"Adapter saved: {args.adapter_dir}")
    print(f"\nДля использования в predict:")
    print(f"  py -m ml.chronos_quantile_pred --variant {args.variant} --adapter {args.adapter_dir}")


if __name__ == "__main__":
    main()
