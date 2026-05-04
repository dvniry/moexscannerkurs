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
import json
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

# ── B-25 helper: нормализация h_vol_pred в [0,1] ─────────────────────
# vol_head обучен на range_norm = (high-low)/atr — типичный диапазон 0..0.05.
# np.tanh(0.03) ≈ 0.03 (почти identity) → сигнал волатильности не различим
# для модели. Линейная нормализация раскрывает диапазон в [0,1].
H_VOL_NORM_CAP = 0.05

def _normalize_h_vol(v):
    """Линейная нормализация h_vol_pred → [0, 1] (B-25 fix)."""
    return float(np.clip(v, 0.0, H_VOL_NORM_CAP) / H_VOL_NORM_CAP) \
        if np.isscalar(v) else \
        np.clip(np.asarray(v, dtype=np.float32), 0.0, H_VOL_NORM_CAP) / H_VOL_NORM_CAP


ENSEMBLE_DIR        = os.path.join(os.path.dirname(__file__), "ensemble")
HOURLY_MODEL_PATH   = os.path.join(ENSEMBLE_DIR, "hourly_specialist.pt")
# B-12: приоритет читать hourly_all (train+val+test со split-меткой) для большего overlap
# с V3 test_dates. Fallback на hourly_val_predictions.npz (старый формат).
HOURLY_ALL_PATH     = os.path.join(ENSEMBLE_DIR, "hourly_all_predictions.npz")
HOURLY_PRED_PATH    = os.path.join(ENSEMBLE_DIR, "hourly_val_predictions.npz")
DAILY_PRED_PATH     = os.path.join(ENSEMBLE_DIR, "ensemble_predictions.npz")
META_FEAT_PATH      = os.path.join(ENSEMBLE_DIR, "meta_features.npz")
META_MODEL_PATH     = os.path.join(ENSEMBLE_DIR, "meta_learner.pt")
# Sprint 9 — MetaLearner v3 (расширенный feature stack)
META_FEAT_PATH_V3   = os.path.join(ENSEMBLE_DIR, "meta_features_v3.npz")
META_MODEL_PATH_V3  = os.path.join(ENSEMBLE_DIR, "meta_learner_v3.pt")
META_V3_CONFIG_PATH = os.path.join(ENSEMBLE_DIR, "meta_v3_config.json")
FUND_MAP_PATH       = os.path.join(ENSEMBLE_DIR, "fundamentals_map.json")
DIV_MAP_PATH        = os.path.join(ENSEMBLE_DIR, "dividends_map.json")


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
    print(_status(HOURLY_ALL_PATH,   "hourly_all_predictions.npz (B-12)"))
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

    # B-12: предпочитаем hourly_all (train+val+test) для overlap с V3 test.
    # Если его нет — fallback на val-only (старый формат).
    if os.path.exists(HOURLY_ALL_PATH):
        h_path = HOURLY_ALL_PATH
        print(f"  Используем hourly_all_predictions.npz (B-12)")
    elif os.path.exists(HOURLY_PRED_PATH):
        h_path = HOURLY_PRED_PATH
        print(f"  [WARN] hourly_all_predictions.npz отсутствует — используем val-only.")
        print(f"         Перетренируйте hourly: py -m ml.trainer_hourly")
    else:
        print(f"  ERROR: ни {HOURLY_ALL_PATH}, ни {HOURLY_PRED_PATH} не найдены.")
        return False
    if not os.path.exists(DAILY_PRED_PATH):
        print(f"  ERROR: {DAILY_PRED_PATH} не найден.")
        return False

    h = np.load(h_path, allow_pickle=True)
    h_dir_prob = h["dir_prob"].astype(np.float32)
    h_vol_pred = h["vol_pred"].astype(np.float32)
    # Hourly y: уже бинарная {0=DOWN, 1=UP}
    h_y_true   = h["y_true"].astype(np.int8)
    h_dates    = np.array([str(d)[:10] for d in h["dates"]])
    h_tickers  = (np.array([str(t) for t in h["tickers"]])
                  if "tickers" in h.files else
                  np.full(len(h_dir_prob), "_unknown_", dtype="U16"))
    h_split    = (np.array([str(s) for s in h["split"]])
                  if "split" in h.files else
                  np.full(len(h_dir_prob), "val", dtype="U5"))
    if "split" in h.files:
        cnts = {s: int((h_split == s).sum()) for s in ("train", "val", "test")}
        print(f"  Hourly: {len(h_dir_prob)} сэмплов "
              f"(train={cnts.get('train',0)} val={cnts.get('val',0)} test={cnts.get('test',0)})")
    else:
        print(f"  Hourly: {len(h_dir_prob)} сэмплов")
    if "tickers" not in h.files:
        print(f"  [WARN B-13] hourly npz без tickers — fallback на join по date только. "
              f"Перетренируйте: py -m ml.trainer_hourly")

    d = np.load(DAILY_PRED_PATH, allow_pickle=True)
    d_dir_prob = d["dir_prob"].astype(np.float32)
    d_mfe      = d["mfe_mae_pred"][:, 0].astype(np.float32)
    d_fill     = d["fill_prob"][:, 0].astype(np.float32)
    d_edge     = d["edge_pred"][:, 0].astype(np.float32)
    # Daily y_test: 3-классная {0=UP, 1=FLAT, 2=DOWN}.
    # B-2 фикс: бинаризуем под BCE/dir_acc → 1=UP, 0=DOWN; FLAT отфильтруем ниже.
    d_y_raw    = d["y_test"].astype(np.int8)
    d_y_bin    = (d_y_raw == 0).astype(np.int8)        # UP=1, иначе 0
    d_is_flat  = (d_y_raw == 1)
    # B-13: читаем tickers из ensemble_predictions.npz
    d_tickers  = (np.array([str(t) for t in d["test_tickers"]])
                  if "test_tickers" in d.files else None)
    # Sprint 5 + MetaLearner v2: подцепляем regime tag для per-regime обучения
    d_regime   = (d["test_regime"].astype(np.int8)
                  if "test_regime" in d.files else
                  np.full(len(d_dir_prob), -1, dtype=np.int8))

    d_dates_raw = _load_daily_dates()
    aligned     = (d_dates_raw is not None and len(d_dates_raw) == len(d_dir_prob))
    use_ticker_key = (
        aligned and d_tickers is not None
        and len(d_tickers) == len(d_dir_prob)
        and "tickers" in h.files
    )
    if use_ticker_key:
        print(f"  Используем (date, ticker) join (B-13) — макс. overlap")
    else:
        print(f"  [WARN] Используем join только по date — overlap будет мал. "
              f"Чтобы активировать B-13: trainer_v3_ensemble --rebuild и trainer_hourly")

    merged = None
    if aligned:
        d_dates = np.array([str(x)[:10] for x in d_dates_raw])
        print(f"  Daily:  {len(d_dir_prob)} сэмплов "
              f"(UP={int((d_y_raw==0).sum())} FLAT={int(d_is_flat.sum())} DOWN={int((d_y_raw==2).sum())})")

        # B-12: при дубликатах ключа предпочитаем test > val > train —
        # split из hourly_all даёт "чистые" предсказания на test-датах.
        split_priority = {"test": 0, "val": 1, "train": 2}
        h_pri = np.array([split_priority.get(str(s), 9) for s in h_split])

        # B-13: ключ join — (date, ticker) при наличии тикеров с обеих сторон,
        # иначе fallback на просто date (мало overlap, см. WARN выше).
        h_df = pd.DataFrame({
            "date":    h_dates,
            "ticker":  h_tickers,
            "h_dir":   h_dir_prob,
            "h_vol":   h_vol_pred,
            "h_split": h_split,
            "_pri":    h_pri,
        })
        d_df = pd.DataFrame({
            "date":    d_dates,
            "ticker":  d_tickers if use_ticker_key else np.full(len(d_dates), "_unknown_", dtype="U16"),
            "d_dir":   d_dir_prob,
            "d_mfe":   d_mfe,
            "d_fill":  d_fill,
            "d_edge":  d_edge,
            "y":       d_y_bin,
            "is_flat": d_is_flat,
            "regime":  d_regime,    # MetaLearner v2: 0=bear, 1=side, 2=bull, -1=unknown
        })

        key_cols = ["date", "ticker"] if use_ticker_key else ["date"]
        h_df = (h_df.sort_values(key_cols + ["_pri"])
                    .drop_duplicates(key_cols, keep="first")
                    .drop(columns=["_pri"]))
        d_df = d_df.drop_duplicates(key_cols, keep="first")

        merged = h_df.merge(d_df, on=key_cols, how="inner")
        # B-4 фикс: явная сортировка по дате (тикер вторичен) — pandas merge не гарантирует
        # порядок, без сортировки последующий X[:n_tr]/X[n_tr:] split смешает train/val.
        merged = merged.sort_values(key_cols).reset_index(drop=True)
        # Отфильтровать FLAT — они шум для бинарной задачи UP-vs-DOWN
        n_before = len(merged)
        merged = merged[~merged["is_flat"].astype(bool)].reset_index(drop=True)
        n_aligned = len(merged)
        n_uniq_dates  = merged["date"].nunique()
        n_uniq_ticks  = merged["ticker"].nunique() if use_ticker_key else 0
        print(f"  После выравнивания: {n_aligned} сэмплов "
              f"(отфильтровано FLAT: {n_before - n_aligned}; "
              f"уникальных дат: {n_uniq_dates}"
              + (f", тикеров: {n_uniq_ticks}" if use_ticker_key else "")
              + ")")
    else:
        n_aligned = 0

    # MetaLearner v2 формат X (14 features):
    #   X[:, 0]  = h_dir_prob
    #   X[:, 1]  = tanh(h_vol_pred)
    #   X[:, 2]  = |h_dir - 0.5| * 2           # confidence
    #   X[:, 3]  = d_dir_prob
    #   X[:, 4]  = clip(d_edge * 100, -5, 5)
    #   X[:, 5]  = clip(d_mfe  * 100,  0, 5)
    #   X[:, 6]  = d_fill_long_prob
    #   X[:, 7]  = h_dir × d_dir              # both UP confidence
    #   X[:, 8]  = (1-h_dir) × (1-d_dir)      # both DOWN confidence
    #   X[:, 9]  = |h_dir - d_dir|            # disagreement magnitude
    #   X[:, 10] = sign agreement bit         # 1 if both > 0.5 or both < 0.5
    #   X[:, 11] = is_bear (one-hot)
    #   X[:, 12] = is_side
    #   X[:, 13] = is_bull
    META_FEAT_DIM = 14
    if n_aligned >= 50 and merged is not None:
        h_dir   = merged["h_dir"].values.astype(np.float32)
        h_vol   = merged["h_vol"].values.astype(np.float32)
        d_dir   = merged["d_dir"].values.astype(np.float32)
        d_edge  = np.clip(merged["d_edge"].values.astype(np.float32) * 100, -5, 5)
        d_mfe   = np.clip(merged["d_mfe"].values.astype(np.float32)  * 100,  0, 5)
        d_fill  = merged["d_fill"].values.astype(np.float32)

        # interaction features
        agree_up   = h_dir * d_dir
        agree_dn   = (1.0 - h_dir) * (1.0 - d_dir)
        disagree   = np.abs(h_dir - d_dir)
        sign_agree = ((h_dir > 0.5) == (d_dir > 0.5)).astype(np.float32)

        # regime one-hot (-1/unknown → all zeros)
        reg = merged["regime"].values.astype(np.int8) if "regime" in merged.columns \
              else np.full(len(h_dir), -1, dtype=np.int8)
        is_bear = (reg == 0).astype(np.float32)
        is_side = (reg == 1).astype(np.float32)
        is_bull = (reg == 2).astype(np.float32)

        X = np.stack([
            h_dir, _normalize_h_vol(h_vol), np.abs(h_dir - 0.5) * 2,  # B-25 fix
            d_dir, d_edge, d_mfe, d_fill,
            agree_up, agree_dn, disagree, sign_agree,
            is_bear, is_side, is_bull,
        ], axis=-1)

        y          = merged["y"].values.astype(np.int8)
        dates_out  = merged["date"].values.astype("U10")
        tickers_out = (merged["ticker"].values.astype("U16")
                       if "ticker" in merged.columns else
                       np.full(n_aligned, "_unknown_", dtype="U16"))
        h_split_out = (merged["h_split"].values.astype("U5")
                       if "h_split" in merged.columns else
                       np.full(n_aligned, "val", dtype="U5"))
        cnts = {s: int((h_split_out == s).sum()) for s in ("train", "val", "test")}
        n_known_regime = int((reg >= 0).sum())
        print(f"  Режим: полное выравнивание (v2: {META_FEAT_DIM} features, N={n_aligned}) "
              f"UP={int(y.sum())} DOWN={int((1 - y).sum())}  "
              f"split: train={cnts.get('train',0)} val={cnts.get('val',0)} test={cnts.get('test',0)}")
        print(f"  Regime coverage: bear={int(is_bear.sum())} side={int(is_side.sum())} "
              f"bull={int(is_bull.sum())} unknown={n_aligned - n_known_regime}")
    else:
        # Fallback: только hourly features. h_y_true уже бинарная.
        print("  [WARN] Дневные даты недоступны → только hourly features (degenerate v2)")
        zero = np.zeros_like(h_dir_prob)
        X = np.stack([
            h_dir_prob, _normalize_h_vol(h_vol_pred), np.abs(h_dir_prob - 0.5) * 2,  # B-25
            h_dir_prob, zero, zero, zero,
            zero, zero, zero, zero,   # interactions = 0 без daily
            zero, zero, zero,          # regime = unknown
        ], axis=-1)
        y           = h_y_true
        dates_out   = h_dates.astype("U10")
        tickers_out = h_tickers.astype("U16")
        h_split_out = h_split.astype("U5")

    print(f"  X.shape={X.shape}  y.shape={y.shape}  unique(y)={np.unique(y).tolist()}")
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    np.savez(META_FEAT_PATH, X=X, y=y, dates=dates_out, tickers=tickers_out, h_split=h_split_out)
    print(f"  Сохранено: {META_FEAT_PATH}")
    return True


# ══════════════════════════════════════════════════════════════════
# Этап 3: MetaLearner
# ══════════════════════════════════════════════════════════════════

class MetaLearner(nn.Module):
    """MetaLearner v2: 14 → 64 → 32 → 1 с LayerNorm.

    v1: 7 → 32 → 16 → 1 (loss train не падал, val_acc=0.5256 < DailySpec 0.5340)
    v2: добавлены 4 interaction features + 3 regime one-hot, LayerNorm для стабильности.
    """

    def __init__(self, n_feat: int = 14, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_meta(epochs: int = 150, lr: float = 3e-3, seed: int = 42,
               holdout_only: bool = False) -> float:
    """Шаг 3: обучает MetaLearner.

    y бинарный: 1=UP, 0=DOWN (бинаризовано в build_meta_features).
    Split — temporal: первые 60% по дате — train, последние 40% — val.

    holdout_only: если True — фильтруем meta-features до h_split=='test'
        (только сэмплы, на которых HourlySpec НЕ обучался). Даёт честную
        upper-bound оценку без оптимизма из train/val-предсказаний.
    """
    print("\n" + "─" * 60)
    print("  ЭТАП 3/3: Обучение MetaLearner"
          + ("  [HOLDOUT-ONLY]" if holdout_only else ""))
    print("─" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    data  = np.load(META_FEAT_PATH, allow_pickle=True)

    X_full = data["X"]
    y_full = data["y"]

    # holdout-only: оставляем только hourly test-предсказания
    if holdout_only:
        if "h_split" in data.files:
            mask = (data["h_split"] == "test")
            X_full = X_full[mask]
            y_full = y_full[mask]
            print(f"  holdout_only: {int(mask.sum())} сэмплов из {len(mask)} (h_split=='test')")
        else:
            print(f"  [WARN] h_split отсутствует в meta_features.npz — игнорируем holdout_only")

    # B-4: гарантируем сортировку по дате перед split (на случай если меta_features
    # был записан без sort_index, или dates изменились).
    dates_arr = data["dates"] if "dates" in data.files else None
    if holdout_only and "h_split" in data.files:
        dates_arr = dates_arr[data["h_split"] == "test"]
    if dates_arr is not None and len(dates_arr) == len(X_full):
        order = np.argsort(np.array([str(d) for d in dates_arr]))
        X_np  = X_full[order]
        y_np  = y_full[order]
    else:
        X_np  = X_full
        y_np  = y_full

    # B-2: y должна быть бинарной {0,1}. Проверяем и при необходимости падаем явно.
    uniq = np.unique(y_np)
    assert set(uniq.tolist()).issubset({0, 1}), \
        f"[BUG B-2] meta y должна быть бинарной {{0,1}}, получено {uniq.tolist()}. " \
        f"Запустите: python -m ml.meta_ensemble --rebuild meta"

    X     = torch.tensor(X_np, dtype=torch.float32)
    y     = torch.tensor(y_np, dtype=torch.float32)
    n     = len(X)
    n_tr  = int(n * 0.6)
    X_tr, X_va = X[:n_tr], X[n_tr:]
    y_tr, y_va = y[:n_tr], y[n_tr:]

    # Корректный подсчёт UP/DOWN на бинарной y
    n_up = int(y_tr.sum().item())
    n_dn = int(n_tr - n_up)
    pos_w = torch.tensor(n_dn / max(n_up, 1)).clamp(0.5, 3.0)
    print(f"  train={n_tr} val={n - n_tr}  UP={n_up} DOWN={n_dn}  pos_w={pos_w.item():.2f}")

    n_feat = X.shape[1]
    model = MetaLearner(n_feat=n_feat, hidden=64)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    best_val_acc = 0.0
    patience_cnt = 0
    patience_max = 30

    # Mini-batch training (вместо full-batch GD который застревал на underfit)
    batch_size = 256
    n_batches = max(1, (n_tr + batch_size - 1) // batch_size)
    print(f"  MetaLearner v2: n_feat={n_feat}  hidden=64  "
          f"lr={lr}  wd=5e-4  batch={batch_size} ({n_batches} batches/epoch)")

    for ep in range(1, epochs + 1):
        model.train()
        # shuffle inside epoch
        perm = torch.randperm(n_tr)
        ep_loss = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch_size: (bi + 1) * batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            logit = model(xb)
            loss  = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= max(n_tr, 1)
        scheduler.step()

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                v_logit = model(X_va)
                v_acc   = ((v_logit >= 0).long() == y_va.long()).float().mean().item()
                v_loss  = F.binary_cross_entropy_with_logits(v_logit, y_va).item()
            cur_lr = opt.param_groups[0]['lr']
            print(f"  E{ep:03d}  loss={ep_loss:.4f}  val_acc={v_acc:.4f}  "
                  f"val_loss={v_loss:.4f}  lr={cur_lr:.5f}")
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                patience_cnt = 0
                os.makedirs(ENSEMBLE_DIR, exist_ok=True)
                # B-20 fix: сохраняем n_feat/hidden рядом со state_dict — чтобы инференс
                # мог восстановить правильную архитектуру при изменении размера фич
                # (напр. после Sprint 9 добавления fundamentals/dividends).
                torch.save({
                    "state":  model.state_dict(),
                    "n_feat": n_feat,
                    "hidden": 64,
                }, META_MODEL_PATH)
            else:
                patience_cnt += 10
                if patience_cnt >= patience_max:
                    print(f"  Early stop on E{ep:03d} (no val improvement {patience_max} epochs)")
                    break

    print(f"\n  Best val_acc: {best_val_acc:.4f}  → {META_MODEL_PATH}")
    return best_val_acc


# ══════════════════════════════════════════════════════════════════
# Оценка
# ══════════════════════════════════════════════════════════════════

def evaluate_meta(holdout_only: bool = False):
    """Оценивает MetaEnsemble vs специалисты.

    Все метрики в бинарной задаче UP-vs-DOWN: y∈{0,1}, predict = (logit≥0).
    holdout_only: фильтрует до h_split=='test' для честной оценки.
    """
    for path in [META_FEAT_PATH, META_MODEL_PATH]:
        if not os.path.exists(path):
            print(f"  Файл не найден: {path}"); return

    data = np.load(META_FEAT_PATH, allow_pickle=True)
    X_np = data["X"]
    y_np = data["y"].astype(np.int64)
    if holdout_only and "h_split" in data.files:
        mask = (data["h_split"] == "test")
        X_np = X_np[mask]
        y_np = y_np[mask]
        print(f"  [holdout_only] {int(mask.sum())} сэмплов из {len(mask)}")
    X    = torch.tensor(X_np, dtype=torch.float32)
    y    = torch.tensor(y_np, dtype=torch.long)

    # B-2: страховка от старых meta_features.npz, где y ∈ {0,1,2}
    uniq = np.unique(y_np)
    if not set(uniq.tolist()).issubset({0, 1}):
        print(f"  [WARN B-2] meta y={uniq.tolist()} — не бинарная. "
              f"Перестройте: python -m ml.meta_ensemble --rebuild meta")
        return

    # B-20: ckpt теперь dict {state, n_feat, hidden} — обратно-совместим со старым форматом
    ckpt = torch.load(META_MODEL_PATH, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state" in ckpt:
        model = MetaLearner(n_feat=ckpt.get("n_feat", X.shape[1]),
                            hidden=ckpt.get("hidden", 64))
        model.load_state_dict(ckpt["state"])
    else:
        # legacy формат — голый state_dict
        model = MetaLearner(n_feat=X.shape[1], hidden=64)
        model.load_state_dict(ckpt)
    model.eval()
    with torch.no_grad():
        meta_logit = model(X)

    meta_pred = (meta_logit >= 0).long()
    h_pred    = (X[:, 0] >= 0.5).long()
    d_pred    = (X[:, 3] >= 0.5).long()

    meta_acc = float((meta_pred == y).float().mean())
    h_acc    = float((h_pred    == y).float().mean())
    d_acc    = float((d_pred    == y).float().mean())
    # Baseline always-UP / always-DOWN — лучший из двух «тривиальных» классификаторов
    baseline = max(float((y_np == 1).mean()), float((y_np == 0).mean()))

    print(f"\n{'═'*50}")
    print(f"  MetaEnsemble Evaluation  N={len(y_np)}  "
          f"(UP={int(y_np.sum())} DOWN={int((1 - y_np).sum())})")
    print(f"{'═'*50}")
    print(f"  Baseline (majority):   {baseline:.4f}")
    print(f"  HourlySpecialist:      {h_acc:.4f}")
    print(f"  DailySpecialist:       {d_acc:.4f}")
    print(f"  MetaEnsemble:          {meta_acc:.4f}  ← цель ≥ 0.545")
    print(f"{'═'*50}")
    winner = "MetaEnsemble" if meta_acc >= max(h_acc, d_acc) else f"лучший = {max(h_acc, d_acc):.4f}"
    print(f"  {'✓ ' if meta_acc >= max(h_acc, d_acc) else '✗ '}{winner}")
    agree = float((h_pred == d_pred).float().mean())
    print(f"  Pairwise H↔D agreement: {agree:.4f}  (меньше → больший ансамблевый gain)")


# ══════════════════════════════════════════════════════════════════
# Sprint 9: MetaLearner v3 — расширенный feature stack
# ══════════════════════════════════════════════════════════════════
#
# Состав фич (34 dims, фиксированный порядок):
#   [0:14]   v2 features (h_dir, _normalize_h_vol(h_vol), conf, d_dir, d_edge,
#            d_mfe, d_fill, agree_up, agree_dn, disagree, sign_agree,
#            is_bear, is_side, is_bull)                                       — 14
#   [14:17]  daily cls_probs[UP, FLAT, DOWN] из ensemble_predictions.npz     —  3
#   [17:29]  fundamentals (sector z-score) из FUND_MAP_PATH                  — 12
#   [29:34]  dividends features из DIV_MAP_PATH                              —  5
#
# Размерности должны совпадать со SCHEMA_V3 ниже — в _ckpt сохраняем n_feat,
# чтобы B-20 защитил от drift'а.

V3_FEATURE_NAMES = (
    # ── v2 base (14) ───────────────────────────────────────────
    "h_dir", "h_vol_norm", "h_conf",
    "d_dir", "d_edge", "d_mfe", "d_fill",
    "agree_up", "agree_dn", "disagree", "sign_agree",
    "is_bear", "is_side", "is_bull",
    # ── daily cls_probs (3) ────────────────────────────────────
    "cls_up", "cls_flat", "cls_down",
    # ── fundamentals (12) ──────────────────────────────────────
    "f_pe", "f_ps", "f_pb", "f_ev_ebitda", "f_d2e", "f_curr_ratio",
    "f_roe", "f_roa", "f_net_margin", "f_div_yield",
    "f_rev_growth", "f_free_float",
    # ── dividends (5) ──────────────────────────────────────────
    "div_days_to_record", "div_is_ex_today", "div_gap_pct",
    "div_dy_ttm", "div_density",
)
META_V3_FEAT_DIM = len(V3_FEATURE_NAMES)
assert META_V3_FEAT_DIM == 34, f"V3 schema mismatch: expected 34, got {META_V3_FEAT_DIM}"


class MetaLearnerV3(nn.Module):
    """MetaLearner v3: 34 → 128 → 64 → 1 с LayerNorm.

    Глубже v2 (14→64→32) пропорционально росту входной размерности,
    плюс bigger dropout для устойчивости к шуму fundamentals (часть полей =0).
    """

    def __init__(self, n_feat: int = META_V3_FEAT_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden // 2, hidden // 4),
            nn.LayerNorm(hidden // 4),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden // 4, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _load_fundamentals_map() -> dict[str, np.ndarray]:
    """Загружает {ticker: 12-vec} либо тянет через API при отсутствии."""
    if not os.path.exists(FUND_MAP_PATH):
        print(f"  [V3] {FUND_MAP_PATH} не найден — генерирую через API...")
        from ml.fundamentals_loader import build_fundamentals_map, _bootstrap_env
        from data.tinkoff_client import TinkoffDataClient
        from ml.config import CFG
        _bootstrap_env()
        client = TinkoffDataClient(os.getenv("TINKOFF_TOKEN", ""))
        fmap = build_fundamentals_map(client, list(CFG.tickers), log=True)
        os.makedirs(os.path.dirname(FUND_MAP_PATH), exist_ok=True)
        with open(FUND_MAP_PATH, "w", encoding="utf-8") as fh:
            json.dump({k: v.tolist() for k, v in fmap.items()}, fh,
                      ensure_ascii=False, indent=2)
        return fmap

    with open(FUND_MAP_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def _load_dividends_map() -> dict[str, list[dict]]:
    """Загружает {ticker: [div_records]} либо тянет через API."""
    if not os.path.exists(DIV_MAP_PATH):
        print(f"  [V3] {DIV_MAP_PATH} не найден — генерирую через API...")
        from ml.dividends_loader import build_dividends_map, _bootstrap_env
        from data.tinkoff_client import TinkoffDataClient
        from ml.config import CFG
        _bootstrap_env()
        client = TinkoffDataClient(os.getenv("TINKOFF_TOKEN", ""))
        dmap = build_dividends_map(client, list(CFG.tickers), log=True)
        os.makedirs(os.path.dirname(DIV_MAP_PATH), exist_ok=True)
        with open(DIV_MAP_PATH, "w", encoding="utf-8") as fh:
            json.dump(dmap, fh, ensure_ascii=False, indent=2)
        return dmap

    with open(DIV_MAP_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_meta_features_v3() -> bool:
    """Sprint 9: расширяет v2 features fundamentals + dividends + cls_probs.

    Зависит от:
      - meta_features.npz (v2)         — переиспользуем h_*/d_*/regime
      - ensemble_predictions.npz       — добавляем cls_probs
      - FUND_MAP_PATH                  — 12 fundamentals/тикер
      - DIV_MAP_PATH                   — 5 div features/(тикер, дата)
    """
    print("\n" + "─" * 60)
    print("  Sprint 9: Сборка meta-features V3 (34 dims)")
    print("─" * 60)

    # 0) Гарантируем v2 существует
    if not os.path.exists(META_FEAT_PATH):
        print(f"  [V3] требует {META_FEAT_PATH} — запускаю v2 build...")
        if not build_meta_features():
            print("  [V3] v2 build failed — прерывание")
            return False

    v2 = np.load(META_FEAT_PATH, allow_pickle=True)
    X_v2     = v2["X"].astype(np.float32)         # [N, 14]
    y        = v2["y"].astype(np.int8)
    dates    = np.array([str(d)[:10] for d in v2["dates"]])
    tickers  = (np.array([str(t) for t in v2["tickers"]])
                if "tickers" in v2.files else
                np.full(len(X_v2), "_unknown_", dtype="U16"))
    h_split  = (np.array([str(s) for s in v2["h_split"]])
                if "h_split" in v2.files else
                np.full(len(X_v2), "val", dtype="U5"))
    n = len(X_v2)
    print(f"  v2 base: N={n}  dim={X_v2.shape[1]}")

    # 1) cls_probs из ensemble_predictions.npz по (date, ticker)
    if not os.path.exists(DAILY_PRED_PATH):
        print(f"  ERROR: {DAILY_PRED_PATH} отсутствует"); return False
    daily = np.load(DAILY_PRED_PATH, allow_pickle=True)
    if "cls_probs" not in daily.files:
        print(f"  ERROR: cls_probs отсутствует в {DAILY_PRED_PATH}"); return False
    cls_full = daily["cls_probs"].astype(np.float32)
    d_dates_full = np.array([str(d)[:10] for d in daily["test_dates"]])
    d_tickers_full = (np.array([str(t) for t in daily["test_tickers"]])
                      if "test_tickers" in daily.files else None)

    # Индекс (date, ticker) → cls_probs[3]
    cls_lookup: dict[tuple[str, str], np.ndarray] = {}
    if d_tickers_full is not None and len(d_tickers_full) == len(cls_full):
        for i, (d, t) in enumerate(zip(d_dates_full, d_tickers_full)):
            cls_lookup[(d, t)] = cls_full[i]
        print(f"  cls_probs lookup: {len(cls_lookup)} (date,ticker) ключей")
    else:
        print("  [WARN] cls_probs без tickers — fallback по date только")
        # one-shot по дате — берём первый совпадающий
        for i, d in enumerate(d_dates_full):
            cls_lookup.setdefault((d, "_any_"), cls_full[i])

    cls_block = np.zeros((n, 3), dtype=np.float32)
    miss_cls = 0
    for i in range(n):
        key = (str(dates[i]), str(tickers[i]))
        v = cls_lookup.get(key)
        if v is None:
            v = cls_lookup.get((str(dates[i]), "_any_"))
        if v is None:
            miss_cls += 1
            cls_block[i] = np.array([0.33, 0.34, 0.33], dtype=np.float32)
        else:
            cls_block[i] = v
    if miss_cls:
        print(f"  [WARN] cls_probs не найдены для {miss_cls}/{n} сэмплов "
              f"(заменены на uniform)")

    # 2) Fundamentals: тикер → 12-vec (статические, не зависят от даты в первой версии)
    fund_map = _load_fundamentals_map()
    fund_block = np.zeros((n, 12), dtype=np.float32)
    miss_fund = 0
    for i, t in enumerate(tickers):
        v = fund_map.get(str(t))
        if v is None:
            miss_fund += 1
        else:
            fund_block[i] = v.astype(np.float32)
    if miss_fund:
        print(f"  [WARN] fundamentals не найдены для {miss_fund}/{n} сэмплов")

    # 3) Dividends: (тикер, дата) → 5-vec
    from ml.dividends_loader import featurize_for_dates
    div_map = _load_dividends_map()
    # Группируем по тикеру для batched featurize
    div_block = np.zeros((n, 5), dtype=np.float32)
    by_ticker: dict[str, list[int]] = {}
    for i, t in enumerate(tickers):
        by_ticker.setdefault(str(t), []).append(i)
    for t, idxs in by_ticker.items():
        divs = div_map.get(t)
        if not divs:
            continue
        sub_dates = [str(dates[i]) for i in idxs]
        feats = featurize_for_dates(divs, sub_dates, closes=None)
        for k, i in enumerate(idxs):
            div_block[i] = feats[k]
    n_with_div = int((div_block.sum(axis=1) > 0).sum())
    print(f"  dividend features: {n_with_div}/{n} сэмплов с ненулевыми флагами")

    # 4) Сборка финального X
    X_v3 = np.concatenate([X_v2, cls_block, fund_block, div_block], axis=1).astype(np.float32)
    assert X_v3.shape[1] == META_V3_FEAT_DIM, \
        f"V3 dim mismatch: {X_v3.shape[1]} vs {META_V3_FEAT_DIM}"
    print(f"  X_v3.shape={X_v3.shape}  y={np.unique(y).tolist()}")

    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    np.savez(META_FEAT_PATH_V3,
             X=X_v3, y=y, dates=dates, tickers=tickers, h_split=h_split,
             feature_names=np.array(V3_FEATURE_NAMES, dtype="U32"))
    with open(META_V3_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "n_feat":  int(META_V3_FEAT_DIM),
            "feature_names": list(V3_FEATURE_NAMES),
            "n_samples": int(n),
            "miss_cls":  int(miss_cls),
            "miss_fund": int(miss_fund),
        }, fh, ensure_ascii=False, indent=2)
    print(f"  Сохранено: {META_FEAT_PATH_V3}")
    return True


def train_meta_v3(epochs: int = 200, lr: float = 2e-3, seed: int = 42,
                  holdout_only: bool = False) -> float:
    """Обучает MetaLearner v3 на 34-фичевом X. Чекпоинт сохраняется с n_feat."""
    print("\n" + "─" * 60)
    print(f"  Sprint 9: Обучение MetaLearner v3 (34 fts)"
          + ("  [HOLDOUT-ONLY]" if holdout_only else ""))
    print("─" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(META_FEAT_PATH_V3, allow_pickle=True)
    X_full = data["X"]
    y_full = data["y"]

    if holdout_only and "h_split" in data.files:
        mask = (data["h_split"] == "test")
        X_full = X_full[mask]
        y_full = y_full[mask]
        print(f"  holdout_only: {int(mask.sum())} сэмплов")

    # Сортировка по дате — temporal split
    dates_arr = data["dates"] if "dates" in data.files else None
    if holdout_only and "h_split" in data.files:
        dates_arr = dates_arr[data["h_split"] == "test"]
    if dates_arr is not None and len(dates_arr) == len(X_full):
        order = np.argsort(np.array([str(d) for d in dates_arr]))
        X_np  = X_full[order]
        y_np  = y_full[order]
    else:
        X_np  = X_full; y_np = y_full

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    n = len(X)
    n_tr = int(n * 0.6)
    X_tr, X_va = X[:n_tr], X[n_tr:]
    y_tr, y_va = y[:n_tr], y[n_tr:]

    n_up = int(y_tr.sum().item())
    n_dn = int(n_tr - n_up)
    pos_w = torch.tensor(n_dn / max(n_up, 1)).clamp(0.5, 3.0)
    print(f"  train={n_tr} val={n - n_tr}  UP={n_up} DOWN={n_dn}  pos_w={pos_w.item():.2f}")

    n_feat = X.shape[1]
    hidden = 128
    model = MetaLearnerV3(n_feat=n_feat, hidden=hidden)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    best_val_acc = 0.0
    patience_cnt = 0
    patience_max = 40
    batch_size = 256
    n_batches = max(1, (n_tr + batch_size - 1) // batch_size)
    print(f"  MetaLearner v3: n_feat={n_feat}  hidden={hidden}  "
          f"lr={lr}  wd=1e-3  batch={batch_size} ({n_batches} b/ep)")

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        ep_loss = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch_size: (bi + 1) * batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= max(n_tr, 1)
        scheduler.step()

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                v_logit = model(X_va)
                v_acc = ((v_logit >= 0).long() == y_va.long()).float().mean().item()
                v_loss = F.binary_cross_entropy_with_logits(v_logit, y_va).item()
            cur_lr = opt.param_groups[0]['lr']
            print(f"  E{ep:03d}  loss={ep_loss:.4f}  val_acc={v_acc:.4f}  "
                  f"val_loss={v_loss:.4f}  lr={cur_lr:.5f}")
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                patience_cnt = 0
                os.makedirs(ENSEMBLE_DIR, exist_ok=True)
                torch.save({
                    "state":  model.state_dict(),
                    "n_feat": n_feat,
                    "hidden": hidden,
                    "version": "v3",
                }, META_MODEL_PATH_V3)
            else:
                patience_cnt += 10
                if patience_cnt >= patience_max:
                    print(f"  Early stop on E{ep:03d}")
                    break

    print(f"\n  Best val_acc: {best_val_acc:.4f}  → {META_MODEL_PATH_V3}")
    return best_val_acc


def evaluate_meta_v3(holdout_only: bool = False):
    """Honest eval V3 vs специалисты vs v2 baseline."""
    for path in [META_FEAT_PATH_V3, META_MODEL_PATH_V3]:
        if not os.path.exists(path):
            print(f"  Файл не найден: {path}"); return

    data = np.load(META_FEAT_PATH_V3, allow_pickle=True)
    X_np = data["X"]
    y_np = data["y"].astype(np.int64)
    if holdout_only and "h_split" in data.files:
        mask = (data["h_split"] == "test")
        X_np = X_np[mask]; y_np = y_np[mask]
        print(f"  [holdout_only] {int(mask.sum())} сэмплов")

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    ckpt = torch.load(META_MODEL_PATH_V3, map_location="cpu", weights_only=True)
    model = MetaLearnerV3(n_feat=ckpt.get("n_feat", X.shape[1]),
                          hidden=ckpt.get("hidden", 128))
    model.load_state_dict(ckpt["state"])
    model.eval()
    with torch.no_grad():
        meta_logit = model(X)

    meta_pred = (meta_logit >= 0).long()
    h_pred = (X[:, 0] >= 0.5).long()       # h_dir
    d_pred = (X[:, 3] >= 0.5).long()       # d_dir

    meta_acc = float((meta_pred == y).float().mean())
    h_acc    = float((h_pred    == y).float().mean())
    d_acc    = float((d_pred    == y).float().mean())
    baseline = max(float((y_np == 1).mean()), float((y_np == 0).mean()))

    print(f"\n{'═'*52}")
    print(f"  MetaEnsemble V3 Evaluation  N={len(y_np)}  "
          f"(UP={int(y_np.sum())} DOWN={int((1 - y_np).sum())})")
    print(f"{'═'*52}")
    print(f"  Baseline (majority):   {baseline:.4f}")
    print(f"  HourlySpecialist:      {h_acc:.4f}")
    print(f"  DailySpecialist:       {d_acc:.4f}")
    print(f"  MetaEnsemble V3:       {meta_acc:.4f}  ← цель ≥ 0.560")
    delta_v2 = "—"
    if os.path.exists(META_MODEL_PATH):
        try:
            v2_data = np.load(META_FEAT_PATH, allow_pickle=True)
            v2_y = v2_data["y"]
            mask2 = (v2_data["h_split"] == "test") if (holdout_only and "h_split" in v2_data.files) else slice(None)
            v2_y_holdout = v2_y[mask2]
            v2_X = torch.tensor(v2_data["X"][mask2], dtype=torch.float32)
            v2_ckpt = torch.load(META_MODEL_PATH, map_location="cpu", weights_only=True)
            from_state = v2_ckpt["state"] if isinstance(v2_ckpt, dict) and "state" in v2_ckpt else v2_ckpt
            n_v2 = v2_X.shape[1]
            v2_model = MetaLearner(n_feat=n_v2, hidden=64)
            v2_model.load_state_dict(from_state)
            v2_model.eval()
            with torch.no_grad():
                v2_pred = (v2_model(v2_X) >= 0).long()
            v2_acc = float((v2_pred.numpy() == v2_y_holdout.astype(np.int64)).mean())
            delta_v2 = f"{(meta_acc - v2_acc) * 100:+.2f} pp (v2 = {v2_acc:.4f})"
        except Exception as e:
            delta_v2 = f"(eval failed: {e})"
    print(f"  Δ vs MetaLearner v2:   {delta_v2}")
    print(f"{'═'*52}\n")


# ══════════════════════════════════════════════════════════════════
# Главный пайплайн
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    epochs_hourly: int  = 30,
    epochs_meta:   int  = 100,
    tickers:       list | None = None,
    rebuild:       str  = "",   # "hourly" | "meta" | "all" | ""
    eval_only:     bool = False,
    holdout_only:  bool = False,
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
        evaluate_meta(holdout_only=holdout_only)
        return

    force_hourly = rebuild in ("hourly", "all")
    force_meta   = rebuild in ("meta", "all") or force_hourly

    # ── Этап 1: HourlySpecialist ──────────────────────────────────
    need_hourly = (
        force_hourly
        or not _is_fresh(HOURLY_MODEL_PATH)
        or not _is_fresh(HOURLY_PRED_PATH, HOURLY_MODEL_PATH)
        or not os.path.exists(HOURLY_ALL_PATH)   # B-12
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
        or not _is_fresh(META_FEAT_PATH, HOURLY_PRED_PATH, DAILY_PRED_PATH, HOURLY_ALL_PATH)
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
        train_meta(epochs=epochs_meta, holdout_only=holdout_only)
    else:
        print("  ЭТАП 3/3: meta_learner.pt — актуален, пропускаем")

    # ── Оценка ────────────────────────────────────────────────────
    evaluate_meta(holdout_only=holdout_only)
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
        # B-20 fix: читаем n_feat/hidden из ckpt вместо хардкода (default=14, 64).
        # При смене размерности (напр. после Sprint 9) state_dict shape mismatch
        # заменён на корректную автоматическую реконструкцию.
        ckpt = torch.load(META_MODEL_PATH, map_location=self.device, weights_only=True)
        if isinstance(ckpt, dict) and "state" in ckpt:
            model = MetaLearner(n_feat=ckpt.get("n_feat", 14),
                                hidden=ckpt.get("hidden", 64)).to(self.device)
            model.load_state_dict(ckpt["state"])
        else:
            model = MetaLearner().to(self.device)
            model.load_state_dict(ckpt)
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
        regime:     int = -1,     # MetaLearner v2: 0=bear, 1=side, 2=bull, -1=unknown
    ) -> dict:
        # B-19 fix: при unknown regime (-1) one-hot был [0,0,0] — OOD относительно
        # тренировочного распределения (там все сэмплы имели регим). Fallback на 1=side
        # как наиболее нейтральный режим, чтобы не вносить искусственный bias.
        if regime not in (0, 1, 2):
            regime = 1
        if self._h_model is None:
            self._h_model = self._load_hourly_model()
        if self._m_model is None:
            self._m_model = self._load_meta_model()

        x     = torch.tensor(hourly_x, dtype=torch.float32).unsqueeze(0).to(self.device)
        h_out = self._h_model(x)
        h_dir = float(torch.sigmoid(h_out["dir_logit"]).item())
        h_vol = float(h_out["vol_pred"].item())

        # MetaLearner v2: 14 features (см. build_meta_features документацию)
        d_dir_v   = float(d_dir_prob)
        d_edge_v  = float(np.clip(d_edge * 100, -5, 5))
        d_mfe_v   = float(np.clip(d_mfe  * 100,  0, 5))
        d_fill_v  = float(d_fill)
        feat = torch.tensor([[
            h_dir,
            _normalize_h_vol(h_vol),                    # B-25 fix: было np.tanh
            abs(h_dir - 0.5) * 2,
            d_dir_v, d_edge_v, d_mfe_v, d_fill_v,
            h_dir * d_dir_v,                            # agree_up
            (1.0 - h_dir) * (1.0 - d_dir_v),            # agree_dn
            abs(h_dir - d_dir_v),                       # disagree
            float((h_dir > 0.5) == (d_dir_v > 0.5)),    # sign_agree
            float(regime == 0),                         # is_bear
            float(regime == 1),                         # is_side
            float(regime == 2),                         # is_bull
        ]], dtype=torch.float32).to(self.device)

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
    parser.add_argument("--holdout-only",  action="store_true",
                        help="Использовать только сэмплы где h_split=='test' (B-13: чистый holdout)")
    parser.add_argument("--smoke",         action="store_true",
                        help="Быстрый тест: 3 тикера, 3 эпохи hourly, 10 эпох meta")
    parser.add_argument("--epochs-hourly", type=int,   default=30)
    parser.add_argument("--epochs-meta",   type=int,   default=100)
    parser.add_argument("--tickers",       nargs="+",  default=None)
    parser.add_argument("--version",       choices=["v2", "v3"], default="v2",
                        help="MetaLearner architecture: v2 (14 fts) или v3 (34 fts, fundamentals+dividends)")
    args = parser.parse_args()

    if args.smoke:
        args.tickers      = args.tickers or ["SBER", "LKOH", "GAZP"]
        args.epochs_hourly = min(args.epochs_hourly, 3)
        args.epochs_meta   = min(args.epochs_meta, 10)
        args.rebuild       = args.rebuild or "all"

    if args.version == "v3":
        # Sprint 9 pipeline — переиспользует v2 build для общих фич, добавляет fund+div
        print("\n" + "═" * 60)
        print("  Sprint 9 MetaLearner V3 Pipeline")
        print("═" * 60)
        if args.eval_only:
            evaluate_meta_v3(holdout_only=args.holdout_only)
        else:
            need_v2 = (not os.path.exists(META_FEAT_PATH)) or args.rebuild in ("meta", "all")
            if need_v2:
                # пересобрать v2 base — нужен для extraction h_*/d_*/regime
                ok = build_meta_features()
                if not ok:
                    print("  v2 build failed — прерывание"); sys.exit(1)
            need_v3_feats = args.rebuild in ("meta", "all") or \
                            not _is_fresh(META_FEAT_PATH_V3, META_FEAT_PATH, FUND_MAP_PATH, DIV_MAP_PATH)
            if need_v3_feats:
                if not build_meta_features_v3():
                    print("  v3 build failed — прерывание"); sys.exit(1)
            train_meta_v3(epochs=args.epochs_meta, holdout_only=args.holdout_only)
            evaluate_meta_v3(holdout_only=args.holdout_only)
    else:
        run_pipeline(
            epochs_hourly = args.epochs_hourly,
            epochs_meta   = args.epochs_meta,
            tickers       = args.tickers,
            rebuild       = args.rebuild or "",
            eval_only     = args.eval_only,
            holdout_only  = args.holdout_only,
        )
