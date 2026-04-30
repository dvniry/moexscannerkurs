"""Sprint 4: MetaLearner + автоматический пайплайн.

Автоматически запускает нужные этапы в зависимости от состояния файлов:
  1. Обучить HourlySpecialist    (если hourly_specialist.pt отсутствует)
  2. Собрать meta-features       (если meta_features.npz отсутствует или устарел)
  3. Обучить MetaLearner         (если meta_learner.pt отсутствует или устарел)
  4. Оценить MetaEnsemble

Запуск:
    python -m ml.meta_ensemble                  # авто-пайплайн
    python -m ml.meta_ensemble --rebuild hourly # перетренировать HourlySpecialist
    python -m ml.meta_ensemble --rebuild meta   # пересобрать meta-features и модель
    python -m ml.meta_ensemble --rebuild all    # всё с нуля
    python -m ml.meta_ensemble --eval-only      # только оценка (всё должно быть готово)
    python -m ml.meta_ensemble --smoke          # 3 тикера, 3 эпохи
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

ENSEMBLE_DIR      = os.path.join(os.path.dirname(__file__), "ensemble")
HOURLY_MODEL_PATH = os.path.join(ENSEMBLE_DIR, "hourly_specialist.pt")
HOURLY_PRED_PATH  = os.path.join(ENSEMBLE_DIR, "hourly_val_predictions.npz")
DAILY_PRED_PATH   = os.path.join(ENSEMBLE_DIR, "ensemble_predictions.npz")
META_FEAT_PATH    = os.path.join(ENSEMBLE_DIR, "meta_features.npz")
META_MODEL_PATH   = os.path.join(ENSEMBLE_DIR, "meta_learner.pt")


# ══════════════════════════════════════════════════════════════════
# Утилиты проверки зависимостей
# ══════════════════════════════════════════════════════════════════

def _mtime(path: str) -> float:
    """Время последней модификации файла, или 0 если не существует."""
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def _is_fresh(target: str, *deps: str) -> bool:
    """True если target существует и новее всех deps."""
    if not os.path.exists(target):
        return False
    t = _mtime(target)
    return all(_mtime(d) <= t for d in deps if d)


def _status(path: str, label: str) -> str:
    if os.path.exists(path):
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(_mtime(path)))
        return f"  ✓  {label:40s}  [{ts}]"
    return f"  ✗  {label:40s}  [отсутствует]"


def print_status():
    """Показывает текущее состояние всех артефактов пайплайна."""
    print("\n  Статус артефактов:")
    print(_status(HOURLY_MODEL_PATH, "hourly_specialist.pt"))
    print(_status(HOURLY_PRED_PATH,  "hourly_val_predictions.npz"))
    print(_status(DAILY_PRED_PATH,   "ensemble_predictions.npz"))
    print(_status(META_FEAT_PATH,    "meta_features.npz"))
    print(_status(META_MODEL_PATH,   "meta_learner.pt"))
    print()


# ══════════════════════════════════════════════════════════════════
# Этап 1: HourlySpecialist
# ══════════════════════════════════════════════════════════════════

def _run_hourly_training(epochs: int, tickers: list | None, rebuild_ds: bool):
    """Запускает trainer_hourly.train() напрямую."""
    from ml.trainer_hourly import train as hourly_train
    print("\n" + "─" * 60)
    print("  ЭТАП 1/3: Обучение HourlySpecialist")
    print("─" * 60)
    hourly_train(
        epochs   = epochs,
        tickers  = tickers,
        rebuild  = rebuild_ds,
        seed     = 42,
    )


# ══════════════════════════════════════════════════════════════════
# Этап 2: Meta-features
# ══════════════════════════════════════════════════════════════════

def _load_daily_dates() -> np.ndarray | None:
    """Восстанавливает даты тест-выборки V3.

    Приоритет:
    1. ensemble_predictions.npz → ключ 'test_dates' (после rebuild)
    2. ml/cache_v3/dates_{ticker}.npy → собирает через V3 датасет
    """
    # Приоритет 1: уже сохранено в npz (после rebuild с новым кодом)
    if os.path.exists(DAILY_PRED_PATH):
        d = np.load(DAILY_PRED_PATH, allow_pickle=True)
        if "test_dates" in d.files:
            arr = d["test_dates"]
            if arr is not None and len(arr) > 0 and any(str(x) != "" for x in arr):
                print(f"  Даты V3 из ensemble_predictions.npz: {len(arr)} сэмплов")
                return arr

    # Приоритет 2: dates_{ticker}.npy в кэше (после rebuild с новым кодом dataset_v3)
    try:
        from ml.dataset_v3 import (
            build_full_multiscale_dataset_v3, temporal_split as tsplit,
            _dates_path,
        )
        print("  Загрузка дат тест-выборки из V3 датасета (может занять время)...")
        ds, _, _, ticker_lengths = build_full_multiscale_dataset_v3()
        _, _, te_idx = tsplit(ticker_lengths)
        dates = []
        n_ok = 0
        for global_idx in te_idx:
            ticker, local_idx = ds.records[int(global_idx)]
            dp = _dates_path(ticker)
            if os.path.exists(dp):
                d_arr = np.load(dp, allow_pickle=True)
                if local_idx < len(d_arr):
                    dates.append(str(d_arr[local_idx]))
                    n_ok += 1
                    continue
            dates.append("")
        if n_ok > 0:
            print(f"  Даты V3 из кэша: {n_ok}/{len(dates)} сэмплов с датами")
            return np.array(dates, dtype=object)
        print("  [WARN] dates_{ticker}.npy отсутствует — запустите --rebuild ensemble")
        return None
    except Exception as e:
        print(f"  [WARN] Даты V3 недоступны: {e}")
        return None


def build_meta_features() -> bool:
    """Шаг 2: выравнивает hourly + daily предсказания → meta_features.npz."""
    print("\n" + "─" * 60)
    print("  ЭТАП 2/3: Сборка meta-features")
    print("─" * 60)

    if not os.path.exists(HOURLY_PRED_PATH):
        print(f"  ERROR: {HOURLY_PRED_PATH} не найден.")
        return False
    if not os.path.exists(DAILY_PRED_PATH):
        print(f"  ERROR: {DAILY_PRED_PATH} не найден.")
        return False

    h = np.load(HOURLY_PRED_PATH, allow_pickle=True)
    h_dir_prob = h["dir_prob"].astype(np.float32)
    h_vol_pred = h["vol_pred"].astype(np.float32)
    h_y_true   = h["y_true"].astype(np.int8)
    h_dates    = np.array([str(d)[:10] for d in h["dates"]])
    print(f"  Hourly: {len(h_dir_prob)} сэмплов")

    d = np.load(DAILY_PRED_PATH, allow_pickle=True)
    d_dir_prob = d["dir_prob"].astype(np.float32)
    d_mfe      = d["mfe_mae_pred"][:, 0].astype(np.float32)
    d_fill     = d["fill_prob"][:, 0].astype(np.float32)
    d_edge     = d["edge_pred"][:, 0].astype(np.float32)
    d_y        = d["y_test"].astype(np.int8)

    d_dates_raw = _load_daily_dates()
    aligned     = (d_dates_raw is not None and len(d_dates_raw) == len(d_dir_prob))

    if aligned:
        d_dates = np.array([str(x)[:10] for x in d_dates_raw])
        print(f"  Daily:  {len(d_dir_prob)} сэмплов")

        h_df = pd.DataFrame({
            "date": h_dates,
            "h_dir": h_dir_prob,
            "h_vol": h_vol_pred,
        }).drop_duplicates("date").set_index("date")

        d_df = pd.DataFrame({
            "date":   d_dates,
            "d_dir":  d_dir_prob,
            "d_mfe":  d_mfe,
            "d_fill": d_fill,
            "d_edge": d_edge,
            "y":      d_y,
        }).drop_duplicates("date").set_index("date")

        merged = h_df.join(d_df, how="inner")
        n_aligned = len(merged)
        print(f"  После выравнивания: {n_aligned} сэмплов")
    else:
        n_aligned = 0

    if n_aligned >= 50:
        h_dir  = merged["h_dir"].values.astype(np.float32)
        h_vol  = merged["h_vol"].values.astype(np.float32)
        d_dir  = merged["d_dir"].values.astype(np.float32)
        X = np.stack([
            h_dir,
            np.tanh(h_vol),
            np.abs(h_dir - 0.5) * 2,
            d_dir,
            np.clip(merged["d_edge"].values.astype(np.float32) * 100, -5, 5),
            np.clip(merged["d_mfe"].values.astype(np.float32)  * 100,  0, 5),
            merged["d_fill"].values.astype(np.float32),
        ], axis=-1)
        y         = merged["y"].values.astype(np.int8)
        dates_out = np.array(merged.index, dtype=object)
        print(f"  Режим: полное выравнивание (7 features, N={n_aligned})")
    else:
        # Fallback: только hourly features
        print("  [WARN] Дневные даты недоступны → только hourly features (4 из 7)")
        X = np.stack([
            h_dir_prob,
            np.tanh(h_vol_pred),
            np.abs(h_dir_prob - 0.5) * 2,
            h_dir_prob,
            np.zeros_like(h_dir_prob),
            np.zeros_like(h_dir_prob),
            np.zeros_like(h_dir_prob),
        ], axis=-1)
        y         = h_y_true
        dates_out = h_dates

    print(f"  X.shape={X.shape}  y.shape={y.shape}")
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    np.savez(META_FEAT_PATH, X=X, y=y, dates=dates_out)
    print(f"  Сохранено: {META_FEAT_PATH}")
    return True


# ══════════════════════════════════════════════════════════════════
# Этап 3: MetaLearner
# ══════════════════════════════════════════════════════════════════

class MetaLearner(nn.Module):
    """Tiny MLP 7 → 32 → 16 → 1."""

    def __init__(self, n_feat: int = 7, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_meta(epochs: int = 100, lr: float = 1e-3, seed: int = 42) -> float:
    """Шаг 3: обучает MetaLearner."""
    print("\n" + "─" * 60)
    print("  ЭТАП 3/3: Обучение MetaLearner")
    print("─" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    data  = np.load(META_FEAT_PATH, allow_pickle=True)
    X     = torch.tensor(data["X"], dtype=torch.float32)
    y     = torch.tensor(data["y"], dtype=torch.float32)
    n     = len(X)
    n_tr  = int(n * 0.6)
    X_tr, X_va = X[:n_tr], X[n_tr:]
    y_tr, y_va = y[:n_tr], y[n_tr:]

    n_up = int(y_tr.sum()); n_dn = n_tr - n_up
    pos_w = torch.tensor(n_dn / max(n_up, 1)).clamp(0.5, 3.0)
    print(f"  train={n_tr} val={n - n_tr}  UP={n_up} DOWN={n_dn}  pos_w={pos_w.item():.2f}")

    model = MetaLearner()
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_val_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        logit = model(X_tr)
        loss  = F.binary_cross_entropy_with_logits(logit, y_tr, pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                v_logit = model(X_va)
                v_acc   = ((v_logit >= 0).long() == y_va.long()).float().mean().item()
                v_loss  = F.binary_cross_entropy_with_logits(v_logit, y_va).item()
            print(f"  E{ep:03d}  loss={loss.item():.4f}  val_acc={v_acc:.4f}  val_loss={v_loss:.4f}")
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                os.makedirs(ENSEMBLE_DIR, exist_ok=True)
                torch.save(model.state_dict(), META_MODEL_PATH)

    print(f"\n  Best val_acc: {best_val_acc:.4f}  → {META_MODEL_PATH}")
    return best_val_acc


# ══════════════════════════════════════════════════════════════════
# Оценка
# ══════════════════════════════════════════════════════════════════

def evaluate_meta():
    """Оценивает MetaEnsemble vs специалисты."""
    for path in [META_FEAT_PATH, META_MODEL_PATH]:
        if not os.path.exists(path):
            print(f"  Файл не найден: {path}"); return

    data = np.load(META_FEAT_PATH, allow_pickle=True)
    X    = torch.tensor(data["X"], dtype=torch.float32)
    y    = data["y"].astype(np.int64)

    model = MetaLearner()
    model.load_state_dict(torch.load(META_MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        meta_logit = model(X)

    meta_acc = float(((meta_logit >= 0).long() == torch.tensor(y)).float().mean())
    h_acc    = float(((X[:, 0] >= 0.5).long() == torch.tensor(y)).float().mean())
    d_acc    = float(((X[:, 3] >= 0.5).long() == torch.tensor(y)).float().mean())
    baseline = max(float((y == 1).mean()), float((y == 0).mean()))

    print(f"\n{'═'*50}")
    print(f"  MetaEnsemble Evaluation  N={len(y)}")
    print(f"{'═'*50}")
    print(f"  Baseline (majority):   {baseline:.4f}")
    print(f"  HourlySpecialist:      {h_acc:.4f}")
    print(f"  DailySpecialist:       {d_acc:.4f}")
    print(f"  MetaEnsemble:          {meta_acc:.4f}  ← цель ≥ 0.545")
    print(f"{'═'*50}")
    winner = "MetaEnsemble" if meta_acc >= max(h_acc, d_acc) else f"лучший = {max(h_acc, d_acc):.4f}"
    print(f"  {'✓ ' if meta_acc >= max(h_acc, d_acc) else '✗ '}{winner}")
    agree = float(((X[:, 0] >= 0.5) == (X[:, 3] >= 0.5)).float().mean())
    print(f"  Pairwise H↔D agreement: {agree:.4f}  (меньше → больший ансамблевый gain)")


# ══════════════════════════════════════════════════════════════════
# Главный пайплайн
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    epochs_hourly: int  = 30,
    epochs_meta:   int  = 100,
    tickers:       list | None = None,
    rebuild:       str  = "",   # "hourly" | "meta" | "all" | ""
    eval_only:     bool = False,
):
    """
    Авто-пайплайн: проверяет зависимости и запускает только нужные этапы.

    Граф зависимостей:
        hourly_specialist.pt
            └─► hourly_val_predictions.npz  (создаётся trainer_hourly)
                    └─► meta_features.npz
                            └─► meta_learner.pt
                                    └─► evaluate
    """
    print("\n" + "═" * 60)
    print("  Sprint 4 MetaEnsemble Pipeline")
    print("═" * 60)
    print_status()

    if eval_only:
        evaluate_meta()
        return

    force_hourly = rebuild in ("hourly", "all")
    force_meta   = rebuild in ("meta", "all") or force_hourly

    # ── Этап 1: HourlySpecialist ──────────────────────────────────
    need_hourly = (
        force_hourly
        or not _is_fresh(HOURLY_MODEL_PATH)
        or not _is_fresh(HOURLY_PRED_PATH, HOURLY_MODEL_PATH)
    )
    if need_hourly:
        _run_hourly_training(
            epochs     = epochs_hourly,
            tickers    = tickers,
            rebuild_ds = force_hourly,
        )
        force_meta = True   # predictions изменились → нужно пересобрать features
    else:
        print("  ЭТАП 1/3: HourlySpecialist — актуален, пропускаем")

    # ── Этап 2: Meta-features ─────────────────────────────────────
    need_features = (
        force_meta
        or not _is_fresh(META_FEAT_PATH, HOURLY_PRED_PATH, DAILY_PRED_PATH)
    )
    if need_features:
        ok = build_meta_features()
        if not ok:
            print("  Пайплайн прерван: ошибка сборки meta-features")
            return
        force_train_meta = True
    else:
        print("  ЭТАП 2/3: meta_features.npz — актуален, пропускаем")
        force_train_meta = False

    # ── Этап 3: MetaLearner ───────────────────────────────────────
    need_meta = (
        force_train_meta
        or not _is_fresh(META_MODEL_PATH, META_FEAT_PATH)
    )
    if need_meta:
        train_meta(epochs=epochs_meta)
    else:
        print("  ЭТАП 3/3: meta_learner.pt — актуален, пропускаем")

    # ── Оценка ────────────────────────────────────────────────────
    evaluate_meta()
    print()
    print_status()


# ══════════════════════════════════════════════════════════════════
# Inference pipeline (runtime use)
# ══════════════════════════════════════════════════════════════════

class MetaEnsembleInference:
    """Runtime inference: часовые бары → meta-сигнал."""

    def __init__(self, device: str = "cpu"):
        self.device   = torch.device(device)
        self._h_model = None
        self._m_model = None

    def _load_hourly_model(self):
        from ml.hourly_specialist import build_hourly_specialist
        model = build_hourly_specialist().to(self.device)
        ckpt  = torch.load(HOURLY_MODEL_PATH, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["state"])
        model.eval()
        return model

    def _load_meta_model(self):
        model = MetaLearner()
        model.load_state_dict(torch.load(META_MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        hourly_x:   np.ndarray,   # [45, 37]
        d_dir_prob: float,
        d_edge:     float,
        d_mfe:      float,
        d_fill:     float,
    ) -> dict:
        if self._h_model is None:
            self._h_model = self._load_hourly_model()
        if self._m_model is None:
            self._m_model = self._load_meta_model()

        x     = torch.tensor(hourly_x, dtype=torch.float32).unsqueeze(0).to(self.device)
        h_out = self._h_model(x)
        h_dir = float(torch.sigmoid(h_out["dir_logit"]).item())
        h_vol = float(h_out["vol_pred"].item())

        feat = torch.tensor([[
            h_dir,
            float(np.tanh(h_vol)),
            abs(h_dir - 0.5) * 2,
            float(d_dir_prob),
            float(np.clip(d_edge * 100, -5, 5)),
            float(np.clip(d_mfe  * 100,  0, 5)),
            float(d_fill),
        ]], dtype=torch.float32)

        meta_logit = self._m_model(feat).item()
        meta_prob  = float(torch.sigmoid(torch.tensor(meta_logit)).item())
        return {
            "meta_dir_prob":  meta_prob,
            "meta_dir_logit": meta_logit,
            "h_dir_prob":     h_dir,
            "d_dir_prob":     d_dir_prob,
            "signal":         "BUY" if meta_prob >= 0.5 else "SELL",
        }


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MetaEnsemble pipeline (Sprint 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python -m ml.meta_ensemble                          # авто-пайплайн
  python -m ml.meta_ensemble --smoke                  # тест на 3 тикерах
  python -m ml.meta_ensemble --rebuild hourly         # перетренировать HourlySpecialist
  python -m ml.meta_ensemble --rebuild meta           # пересобрать meta-features + модель
  python -m ml.meta_ensemble --rebuild all            # всё с нуля
  python -m ml.meta_ensemble --eval-only              # только оценка
        """,
    )
    parser.add_argument("--rebuild",       choices=["hourly", "meta", "all"], default="",
                        nargs="?", const="all",
                        help="Что пересобрать (hourly/meta/all, по умолчанию all)")
    parser.add_argument("--eval-only",     action="store_true",
                        help="Только оценка — не обучать ничего")
    parser.add_argument("--smoke",         action="store_true",
                        help="Быстрый тест: 3 тикера, 3 эпохи hourly, 10 эпох meta")
    parser.add_argument("--epochs-hourly", type=int,   default=30)
    parser.add_argument("--epochs-meta",   type=int,   default=100)
    parser.add_argument("--tickers",       nargs="+",  default=None)
    args = parser.parse_args()

    if args.smoke:
        args.tickers      = args.tickers or ["SBER", "LKOH", "GAZP"]
        args.epochs_hourly = min(args.epochs_hourly, 3)
        args.epochs_meta   = min(args.epochs_meta, 10)
        args.rebuild       = args.rebuild or "all"

    run_pipeline(
        epochs_hourly = args.epochs_hourly,
        epochs_meta   = args.epochs_meta,
        tickers       = args.tickers,
        rebuild       = args.rebuild or "",
        eval_only     = args.eval_only,
    )
