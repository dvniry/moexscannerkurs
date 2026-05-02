"""Sprint 4: Trainer для HourlySpecialist.

Обучает HourlySpecialist на часовых барах MOEX.
Сохраняет model.pt + val_predictions.npz для MetaLearner.

Запуск:
    python -m ml.trainer_hourly                    # полный прогон
    python -m ml.trainer_hourly --epochs 5 --smoke # 3 тикера, быстрый тест
    python -m ml.trainer_hourly --rebuild           # пересборка кэша
"""
from __future__ import annotations

import argparse
import os
import random
import time

os.environ["GRPC_DNS_RESOLVER"] = "native"
_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "russian_ca.cer"))
if os.path.exists(_cert):
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = _cert

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset

from ml.hourly_only_dataset import HourlyDataset, temporal_split, HOURLY_WINDOW, N_HOURLY_FEAT
from ml.hourly_specialist import build_hourly_specialist
from ml.dataset_v3 import INDICATOR_COLS

# B-3: индекс канала, используемого как realized-vol proxy.
# range_norm = (high - low) / close — уже нормализован, доступен в признаках
# (не требует rebuild кэша).
VOL_TARGET_IDX = INDICATOR_COLS.index("range_norm")
VOL_LOSS_WEIGHT = 0.1

ENSEMBLE_DIR    = os.path.join(os.path.dirname(__file__), "ensemble")
MODEL_PATH      = os.path.join(ENSEMBLE_DIR, "hourly_specialist.pt")
# B-9: раздельные npz для val и test (внутренний отчёт HourlySpecialist).
PRED_PATH       = os.path.join(ENSEMBLE_DIR, "hourly_val_predictions.npz")
PRED_TEST_PATH  = os.path.join(ENSEMBLE_DIR, "hourly_test_predictions.npz")
# B-12: полный набор предсказаний (train+val+test) с метками split — для MetaLearner.
# Hourly и V3 имеют разные temporal-split → их test-наборы почти не пересекаются.
# Чтобы MetaLearner получил достаточно сэмплов, прогоняем HourlySpecialist на ВСЕХ датах
# и используем split-флаг для контроля лика. Hourly target (next-day) ортогонален
# V3 target (3-class 5d), поэтому даже train-предсказания дают полезный сигнал.
PRED_ALL_PATH   = os.path.join(ENSEMBLE_DIR, "hourly_all_predictions.npz")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate(batch):
    """DataLoader collate: игнорирует date- и ticker-строки."""
    xs, ys, dates, tickers = zip(*batch)
    return (
        torch.stack(xs),
        torch.tensor(ys, dtype=torch.long),
        list(dates),
        list(tickers),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_dir_logits, all_ys, all_vol_pred, all_vol_tgt = [], [], [], []
    for x, y, _dates, _tickers in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        all_dir_logits.append(out["dir_logit"])
        all_ys.append(y)
        all_vol_pred.append(out["vol_pred"])
        all_vol_tgt.append(x[:, :, VOL_TARGET_IDX].mean(dim=1))
    all_dir_logits = torch.cat(all_dir_logits)
    all_ys         = torch.cat(all_ys)
    all_vol_pred   = torch.cat(all_vol_pred)
    all_vol_tgt    = torch.cat(all_vol_tgt)

    dir_prob = torch.sigmoid(all_dir_logits)
    pred     = (dir_prob >= 0.5).long()
    acc      = float((pred == all_ys).float().mean())
    loss     = float(F.binary_cross_entropy_with_logits(
        all_dir_logits, all_ys.float()))
    vol_mse  = float(F.mse_loss(all_vol_pred, all_vol_tgt))

    return {"dir_acc": acc, "loss": loss, "vol_mse": vol_mse}


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Собирает dir_prob, vol_pred, y_true, dates, tickers (B-13) для сохранения."""
    model.eval()
    all_prob, all_vol, all_y, all_dates, all_tickers = [], [], [], [], []
    for x, y, dates, tickers in loader:
        x = x.to(device)
        out = model(x)
        all_prob.append(torch.sigmoid(out["dir_logit"]).cpu().numpy())
        all_vol.append(out["vol_pred"].cpu().numpy())
        all_y.append(y.numpy())
        all_dates.extend(dates)
        all_tickers.extend(tickers)
    return {
        "dir_prob": np.concatenate(all_prob),
        "vol_pred": np.concatenate(all_vol),
        "y_true":   np.concatenate(all_y),
        "dates":    np.array(all_dates, dtype="U10"),
        "tickers":  np.array(all_tickers, dtype="U16"),
    }


def train(
    epochs:     int   = 30,             # откат v2→v1: углубление не дало прироста
    lr:         float = 3e-4,
    batch_size: int   = 256,
    seed:       int   = 42,
    rebuild:    bool  = False,
    tickers:    list | None = None,
    patience:   int   = 10,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" HourlySpecialist trainer  device={device}  seed={seed}")
    print(f"{'='*60}\n")

    # ── Dataset ──────────────────────────────────────────────────
    ds = HourlyDataset(tickers=tickers, rebuild=rebuild)
    if len(ds) < 100:
        raise RuntimeError(f"Слишком мало сэмплов: {len(ds)}")

    tr_idx, va_idx, te_idx = temporal_split(ds)
    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=batch_size,
                           shuffle=True,  num_workers=0, collate_fn=_collate)
    va_loader = DataLoader(Subset(ds, va_idx), batch_size=batch_size,
                           shuffle=False, num_workers=0, collate_fn=_collate)
    te_loader = DataLoader(Subset(ds, te_idx), batch_size=batch_size,
                           shuffle=False, num_workers=0, collate_fn=_collate)

    # Метки train для pos_weight
    y_tr = np.array([ds[i][1] for i in tr_idx])
    n_up  = int((y_tr == 1).sum())
    n_dn  = int((y_tr == 0).sum())
    pos_weight = torch.tensor(n_dn / max(n_up, 1), dtype=torch.float32).to(device)
    pos_weight = pos_weight.clamp(0.5, 3.0)
    print(f"  train UP={n_up} DOWN={n_dn}  pos_weight={pos_weight.item():.2f}")

    # ── Model ─────────────────────────────────────────────────────
    model = build_hourly_specialist().to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = OneCycleLR(
        optimizer,
        max_lr     = lr,
        steps_per_epoch = len(tr_loader),
        epochs     = epochs,
        pct_start  = 0.15,
        anneal_strategy = "cos",
    )

    # ── Training loop ─────────────────────────────────────────────
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    best_val_acc = 0.0
    patience_cnt = 0
    history = []

    always_up_acc = float((y_tr == 1).mean())
    print(f"  Baseline always-UP: {always_up_acc:.4f}")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total   = 0
        t0 = time.time()

        for x, y, _dates, _tickers in tr_loader:
            x, y = x.to(device), y.float().to(device)
            out  = model(x)

            dir_loss = F.binary_cross_entropy_with_logits(
                out["dir_logit"], y, pos_weight=pos_weight)
            # B-3: vol_target = средний range_norm по окну (proxy для realized vol).
            vol_target = x[:, :, VOL_TARGET_IDX].mean(dim=1).detach()
            vol_loss   = F.mse_loss(out["vol_pred"], vol_target)
            loss       = dir_loss + VOL_LOSS_WEIGHT * vol_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss    += loss.item() * len(y)
            ep_correct += int(((out["dir_logit"] >= 0) == y.bool()).sum())
            ep_total   += len(y)

        ep_loss /= max(ep_total, 1)
        tr_acc   = ep_correct / max(ep_total, 1)
        va_met   = evaluate(model, va_loader, device)
        val_acc  = va_met["dir_acc"]

        elapsed = time.time() - t0
        print(f"  E{epoch:02d}/{epochs}  "
              f"tr_loss={ep_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"val_acc={val_acc:.4f}  vol_mse={va_met['vol_mse']:.5f}  "
              f"({elapsed:.1f}s)")

        history.append({"epoch": epoch, "tr_acc": tr_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save({
                "epoch":   epoch,
                "state":   model.state_dict(),
                "val_acc": val_acc,
                "seed":    seed,
            }, MODEL_PATH)
            print(f"    ✓ saved  best_val_acc={val_acc:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop: patience={patience}")
                break

    # ── Финальная оценка ──────────────────────────────────────────
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state"])
    model.eval()

    te_met = evaluate(model, te_loader, device)
    print(f"\n  Test dir_acc:  {te_met['dir_acc']:.4f}")
    print(f"  Val  best_acc: {best_val_acc:.4f}")
    print(f"  Baseline (always-UP): {always_up_acc:.4f}")

    # ── B-9 + B-12: раздельные val/test + полный набор для MetaLearner ──
    val_preds  = predict_proba(model, va_loader,  device)
    test_preds = predict_proba(model, te_loader, device)
    train_preds = predict_proba(model, tr_loader, device)

    np.savez(
        PRED_PATH,
        dir_prob  = val_preds["dir_prob"].astype(np.float32),
        vol_pred  = val_preds["vol_pred"].astype(np.float32),
        y_true    = val_preds["y_true"].astype(np.int8),
        dates     = val_preds["dates"],
        tickers   = val_preds["tickers"],
        val_acc   = np.array(best_val_acc),
        split     = np.array("val"),
    )
    np.savez(
        PRED_TEST_PATH,
        dir_prob  = test_preds["dir_prob"].astype(np.float32),
        vol_pred  = test_preds["vol_pred"].astype(np.float32),
        y_true    = test_preds["y_true"].astype(np.int8),
        dates     = test_preds["dates"],
        tickers   = test_preds["tickers"],
        test_acc  = np.array(te_met["dir_acc"]),
        split     = np.array("test"),
    )

    # B-12: полный датасет с метками split. MetaLearner может фильтровать
    # train-сэмплы при необходимости (split != 'train'), но по умолчанию
    # использует все, чтобы максимизировать пересечение с V3 test_dates.
    # B-13: добавлен tickers для (date, ticker) join в meta_features.
    all_dir     = np.concatenate([train_preds["dir_prob"], val_preds["dir_prob"], test_preds["dir_prob"]])
    all_vol     = np.concatenate([train_preds["vol_pred"], val_preds["vol_pred"], test_preds["vol_pred"]])
    all_y       = np.concatenate([train_preds["y_true"],   val_preds["y_true"],   test_preds["y_true"]])
    all_dates   = np.concatenate([train_preds["dates"],    val_preds["dates"],    test_preds["dates"]])
    all_tickers = np.concatenate([train_preds["tickers"],  val_preds["tickers"],  test_preds["tickers"]])
    all_split = np.concatenate([
        np.full(len(train_preds["dir_prob"]), "train", dtype="U5"),
        np.full(len(val_preds["dir_prob"]),   "val",   dtype="U5"),
        np.full(len(test_preds["dir_prob"]),  "test",  dtype="U5"),
    ])
    np.savez(
        PRED_ALL_PATH,
        dir_prob = all_dir.astype(np.float32),
        vol_pred = all_vol.astype(np.float32),
        y_true   = all_y.astype(np.int8),
        dates    = all_dates,
        tickers  = all_tickers,
        split    = all_split,
        val_acc  = np.array(best_val_acc),
        test_acc = np.array(te_met["dir_acc"]),
    )

    print(f"\n  Val  predictions: {PRED_PATH}  (N={len(val_preds['dir_prob'])})")
    print(f"  Test predictions: {PRED_TEST_PATH}  (N={len(test_preds['dir_prob'])})")
    print(f"  All  predictions: {PRED_ALL_PATH}  (N={len(all_dir)} = "
          f"{len(train_preds['dir_prob'])}+{len(val_preds['dir_prob'])}+{len(test_preds['dir_prob'])})")

    return best_val_acc, te_met["dir_acc"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--rebuild",    action="store_true")
    parser.add_argument("--smoke",      action="store_true",
                        help="Тест на 3 тикерах, 3 эпохи")
    parser.add_argument("--tickers",    nargs="+",  default=None)
    args = parser.parse_args()

    if args.smoke:
        args.tickers = args.tickers or ["SBER", "LKOH", "GAZP"]
        args.epochs  = min(args.epochs, 3)

    train(
        epochs     = args.epochs,
        lr         = args.lr,
        batch_size = args.batch_size,
        seed       = args.seed,
        rebuild    = args.rebuild,
        tickers    = args.tickers,
        patience   = args.patience,
    )
