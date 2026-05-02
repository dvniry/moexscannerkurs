# ML Trading System — MOEX | План проекта
**Обновлено:** 2026-05-01

---

## ГЛАВНАЯ ЦЕЛЬ

Превратить задачу прогнозирования в **экономически целесообразную decision system**:
- **BUY / SELL / HOLD**, где **HOLD** = ожидаемый edge не покрывает costs (комиссия + спред + slippage).
- Дневной прогноз — главное торговое решение.
- Часовые свечи текущего дня — intraday feedback loop (вход / выход / TP-SL / cancel).

### Слои системы
| Слой | Назначение |
|------|-----------|
| **Python** | research backtest, симуляция, walk-forward, sweep, генерация targets |
| **MT5** | execution layer, live/demo, проверка лимиток |
| **T-Bank API** | data source, потенциальный live path |

---

## ЧЕК-ЛИСТ 15 ИДЕЙ С КАРДИНАЛЬНЫМ ЭФФЕКТОМ

| # | Идея | Статус | Что и где |
|---|------|--------|-----------|
| **P0** | | | |
| 1 | Decision-aware training | ✅ | `EconomicHeads` + `EconomicLoss` (MFE/MAE/fill/edge) в `multiscale_cnn_v3.py` |
| 2 | No-trade / abstain (HOLD) | ✅ | `DecisionLayer.decide` → `SIG_HOLD` если edge < cost_ratio в `decision_layer.py` |
| 3 | **Regime detection** | ✅ | HMM(3) `bear/side/bull` в `context_loader.py` + `RegimeAwareDecisionLayer` с per-regime порогами + `patch_ensemble_regime.py` для добавления regime в npz |
| 4 | Intraday path targets | ✅ | MFE/MAE в econ-heads + `HourlyFeedbackEncoder` + `intraday_feats` в кэше v3.7.0 |
| 5 | Cost-aware training | ✅ | `EconomicLoss(cost_roundtrip)` + `econ_commission/slippage/spread` из CFG |
| **P1** | | | |
| 6 | Walk-forward validation | ❌ | `temporal_split` — однократный 70/15/15. Нужно скользящее окно |
| 7 | Confidence calibration | ❌ | `calibrate_threshold.py` это threshold sweep, не temperature scaling / isotonic |
| 8 | Intraday order management | ⚠️ | `simulate_intraday_refinement(cancel_threshold)` есть, но не интегрирован в live-loop |
| 9 | Market context / RS | ✅ | `RS_5d/RS_20d/IMOEX_*` в INDICATOR_COLS + breadth/trend/vol_regime в ctx (21 dim) |
| 10 | Multi-objective optimization | ⚠️ | `val_metric = dir_acc + 0.3·max(dec_hit−0.5, 0)` — есть multi-obj, но без Sharpe/PnL/DD |
| **P2** | | | |
| 11 | Dynamic position sizing | ✅ | `position_size = max_position_pct × clip(conf,0,1)` в `simulate_decision_strategy` |
| 12 | Ensemble by trading quality | ❌ | Веса по `val_dir_acc - 0.5`, а не по Sharpe / PnL / drawdown |
| 13 | Separate research / execution backtest | ✅ | Python: `backtest_strategy.py`. MT5: `ML_Strategy_Backtest.mq5` |
| 14 | Early exit / intraday invalidation | ⚠️ | `cancel_threshold=0.6` в `simulate_intraday_refinement`, но без live-trigger |
| 15 | Feature importance | ❌ | Нет SHAP / permutation importance |

**Итого:** 8/15 ✅, 3/15 ⚠️, 4/15 ❌. **#3 Regime detection ЗАКРЫТО (Sprint 5)**.

---

## АРХИТЕКТУРА (3 стадии)

```
Stage 1 — DailySpecialist (V3 ансамбль 3 seeds)
  Input:  imgs[4 scales] + nums + ctx + hourly + intraday_feats
  Heads:  cls_head(3) + ohlc_head(20) + dir_head(1) + econ_heads(4+2+2) + extremes(2)
  Output: ensemble_predictions.npz (cls_probs, dir_prob, mfe_mae_pred, fill_prob, edge_pred,
                                    decision_signal, decision_confidence, test_dates, test_tickers)

Stage 2 — HourlySpecialist (BiLSTM + multi-scale CNN)
  Input:  hourly window [B, 45, 37]
  Heads:  dir_head + vol_head (B-3: учится через MSE на range_norm)
  Output: hourly_{val|test|all}_predictions.npz с tickers и split-метками (B-12, B-13)

Stage 3 — MetaLearner (tiny MLP)
  Input:  7 features (h_dir, h_vol, h_conf, d_dir, d_edge, d_mfe, d_fill)
  Output: meta_features.npz + meta_learner.pt
  Join:   (date, ticker) для overlap (B-13: 14 226 сэмплов)

DecisionLayer (B-15): cost-aware фильтр
  Пороги: edge_r=5.0, dir=0.75, sell=0.55, fill=0.40, rr=1.2
```

---

## КЛЮЧЕВЫЕ ПРАВКИ (B-1 … B-17)

| ID | Severity | Описание | Статус |
|----|----------|----------|--------|
| B-1 | 🔴 P0 | `dates_arr` через `_align_arrays` — синхронизация | ✅ |
| B-2 | 🔴 P0 | Бинаризация y в meta_ensemble (UP=1, FLAT отфильтрован) | ✅ |
| B-3 | 🟠 P1 | `vol_head` обучается через MSE на `range_norm` | ✅ |
| B-4 | 🟠 P1 | Sort_index + temporal split в train_meta | ✅ |
| B-5 | 🟠 P1 | Verified: `_sharpe_daily` используется в main backtest | ✅ |
| B-6 | 🟡 P2 | Лог `n_dropped_oversize` в simulate_decision_strategy | ✅ |
| B-7 | 🟡 P2 | Упрощён `is_buy/is_sell` в DecisionLayer.decide | ✅ |
| B-8 | 🟡 P2 | `self.device` в `MetaEnsembleInference._load_meta_model` | ✅ |
| B-9 | 🟠 P1 | Раздельные `hourly_val/test_predictions.npz` (анти-лик) | ✅ |
| B-10 | 🟡 P2 | `dtype='U10'` для dates + `allow_pickle=True` в np.load | ✅ |
| B-11 | 🟠 P1 | cls_head fix: `cls_weights` без max-norm (FLAT capped), `direction_weight=0.40`, `gamma=(2,0.5,2)` | ✅ |
| B-12 | 🔴 P0 | `hourly_all_predictions.npz` (train+val+test) для max overlap | ✅ |
| B-13 | 🔴 P0 | Join по `(date, ticker)` → 14 226 сэмплов вместо 346 | ✅ |
| B-14 | 🟡 P2 | Cache-path `_align_arrays` 14-tuple unpack | ✅ |
| **B-15** | 🔴 P0 | **Sweep**: пороги `5.0/0.75/0.55` (было `4.0/0.70/0.85`). Expectancy `−1.84% → +0.10%/trade` | ✅ |
| B-16 | 🟠 P1 | EconomicHeads не учится → edge_pred ≈ const. **Фикс:** gain финального слоя 0.01→0.1, edge_bias 0.002→0, beta smooth_l1 0.001→0.005, w_edge 0.5→1.0. Smoke-тест: edge_pred std 0.0066 → 0.056 (×8.5). Требует rebuild V3 для полной валидации. | ✅ fixed (pending rebuild) |
| B-17 | 🟡 P2 | `ohlc_test` в ATR-z-units → `decision_sweep` денормализует через `× atr_ratio` | ✅ |
| **S5-1** | 🔴 P0 | Sprint 5: `RegimeAwareDecisionLayer` + per-regime пороги (bear: aggressive, side: light, bull: **disabled**). **Backtest перевёрнут с −2.21% / Sharpe −1.05 на +1.61% / Sharpe +1.19** | ✅ |
| **S5-2** | 🔴 P0 | **MetaLearner v2**: 7 → 14 features (4 interaction + 3 regime one-hot), hidden 32→64, LayerNorm, batch=256, lr 1e-3→3e-3, cosine scheduler. **Holdout val_acc 0.5075 → 0.5462 (+3.87pp, 6.3σ)** — впервые превзошёл DailySpec и majority baseline на честном OOS | ✅ |
| **S6-1** | 🟠 P1 | **Sprint 6 walk-forward**: 5 фолдов на dates 2024-11 → 2026-04. Подтверждает: regime-aware OOS Sharpe −1.40 vs static −4.30 (×3 улучшение, направление верное), но **Sprint 5 in-sample прибыль (+1.19) не воспроизводится** на OOS (+0.010 ± 0.269). Stable инсайт: `bull=OFF` устойчив на всех фолдах. Per-regime пороги нестабильны (Sharpe std=6.00). | ✅ |

---

## ТЕКУЩИЕ МЕТРИКИ (после B-1…B-17, 2026-05-01)

### V3 ансамбль (3 seeds × 30 epochs)
```
Test:           19 061 сэмплов (53 тикера)
dir_acc:        0.5341  (baseline always-BUY: 0.4676, edge +6.7%)
Pairwise:       0.6579  (хорошее разнообразие)
F1 UP/DOWN:     0.36 / 0.49  (после B-11; было 0.10 / 0.10)
F1 FLAT:        0.18  (было 0.40 в коллапсе — теперь корректно)
```

### HourlySpecialist v1 (восстановлен после v2 отката)
```
Param count:    223 266 (v2 500k не дал прироста)
Val acc:        0.5378–0.5389 (val plateau)
Test acc:       0.5475–0.5496
Vol_mse:        0.00006–0.00165 (B-3 работает)
```

### MetaEnsemble (после v2 апгрейда: 14 features, 64 hidden, LayerNorm)
```
Full eval (N=14 226):
  Hourly:        0.5103
  Daily:         0.5340
  MetaEnsemble:  0.5326  (v1 было 0.5256, +0.7pp)

Holdout-only (N=6 774 — только hourly test ∩ V3 test):
  Baseline:      0.5407
  Hourly:        0.5114
  Daily:         0.5272
  MetaEnsemble:  0.5462 ✅ (v1: 0.5075, +3.87pp, 6.3σ статистически значимо)
                          впервые превзошёл baseline и обоих специалистов на чистом OOS

Что сделало разницу:
  - 4 interaction features (h_dir × d_dir, |h_dir - d_dir|, sign agree, agree_dn)
  - 3 regime one-hot — критически важно после Sprint 5
  - LayerNorm + cosine LR scheduler — выход из underfit
  - Loss train: v1 0.7275→0.7266 (stagnation) vs v2 0.7219→0.7156 (учится)
```

### Backtest E_decision_layer

**B-15 (единые пороги):**
```
Сделок:    1060 (BUY=247, SELL=813)
Win rate:  42.08%   Avg gross +0.111%   2×fee=0.200%
NET/trade: −0.089%  Total −2.21%   Sharpe −1.05
```

**Sprint 5 (regime-aware, bull DISABLED):** ✅ **прибыльная стратегия**
```
Сделок:    353 (BUY=174, SELL=179)
Win rate:  49.86%   Avg gross +0.442%   2×fee=0.200%
NET/trade: +0.242%  Total +1.61%   Sharpe +1.19

E2 close-only:
NET/trade: +0.373%  Total +2.49%   Sharpe +1.67   Win rate 51.84%
Max DD:    −0.38%
```

### 🚨 Текущий диагноз
Модель **прибыльна в bear/side, убыточна в bull** (DOWN-bias не работает в восходящем рынке).
Решение: **отключить торговлю в bull-режиме**, агрессивные пороги в bear, лёгкие в side.

### Walk-forward validation (Sprint 6, 5 folds OOS)
```
                       coverage    hit_rate    exp%/trade    Sharpe (daily)
B-15 static:           3.37±2.26%  0.4228      −0.197±0.111  −4.30±2.18
Sprint 5 regime-aware: 0.52±0.28%  0.4700      +0.010±0.269  −1.40±6.00

Выводы:
  ✅ Regime-aware ×3 улучшил Sharpe vs static (направление верное)
  ✅ Stable инсайт: bull=OFF на всех 5 фолдах (DOWN-bias модели реален)
  ❌ Sprint 5 in-sample +1.19 Sharpe НЕ воспроизводится на OOS
  ❌ Per-regime пороги нестабильны (Sharpe std=6.00)
  ❌ Coverage упало до 0.5% — модель почти не торгует на OOS

Корневая блокирующая проблема: B-16 (edge_pred ≈ const). Без работающего
EconomicHeads DecisionLayer фильтрует только по dir_prob/mfe/mae, что
недостаточно для устойчивой OOS-стратегии.
```

---

## ROADMAP

### Sprint 5 — Regime detection (✅ ЗАКРЫТО 2026-05-01)
- [x] Сохранить regime[N] в `ensemble_predictions.npz` → `patch_ensemble_regime.py`
- [x] Per-regime sweep в `decision_sweep.py --by-regime`
- [x] `RegimeAwareDecisionLayer` с per-regime порогами в `decision_layer.py`
- [x] `patch_decision_signal.py --regime-aware` для применения к existing npz
- [x] Backtest подтвердил прибыльность: Total +1.61% / Sharpe +1.19

### Sprint 6 — Walk-forward (✅ ВЫПОЛНЕН 2026-05-02)
- [x] `ml/walk_forward.py` — 5-fold rolling window на post-processing уровне
- [x] B-15 static и Sprint 5 regime-aware протестированы
- [x] **Главный вывод:** Sprint 5 in-sample +1.19 Sharpe не воспроизводится OOS
- [x] **Stable инсайт сохранён:** `bull=OFF` устойчив на всех фолдах
- [x] B-16 фикс EconomicHeads (gain/bias/beta/w_edge) — pending rebuild V3 для полной валидации
- [ ] **Следующий шаг:** rebuild V3 с B-16 фиксом → walk-forward retest. Если edge_pred теперь
      имеет реальную дисперсию, OOS Sharpe должен подняться

### Sprint 7 — Calibration + Multi-objective
- [ ] Temperature scaling per ticker для `dir_prob` (#7)
- [ ] Заменить ensemble weights с `val_dir_acc` на per-seed Sharpe (#12)
- [ ] `val_metric = dir_acc + α·hit_rate + β·sharpe_proxy` (#10)
- [ ] Permutation importance на 37 INDICATOR_COLS (#15)

### Sprint 8 — Intraday execution (1-day path-based trading)
**Идея:** дневная свеча предсказывает H/L/C → внутри дня покупаем у low, продаём у high.
Часовые свечи в реальном времени корректируют прогноз и точки входа/выхода.

- [ ] **High/Low ordering head** — добавить в V3 бинарную голову `high_first_prob`:
      P(high достигается раньше low в течение дня)
      - Если `low_first` (high позже): идеальный сценарий для long в один день
      - Если `high_first` (low позже): нужен carry в следующий день, либо отказ
- [ ] **Path-aware execution simulator** в `backtest_strategy.py`:
      ```
      pred_low, pred_high, pred_close = predicted OHLC bar1
      if high_first_prob > 0.6:
          # high раньше → классический long с carry
          entry_at = next_day_open + 0.5*(pred_low - open_today)
          exit_at  = limit at pred_high  → если не закрылось на close
      else:
          # low раньше → можно сделать round-trip за день
          entry_at = limit at pred_low
          exit_at  = limit at pred_high (тот же день)
      ```
- [ ] **Hourly feedback for entry timing**:
      - В 10:30 (после 1 часа торгов): `update_entry_price(hourly_low_so_far)`
      - В 13:00 (4 часа): `update_tp(refined_pred_high)`
      - В 17:00 (7 часов): `force_close_if_dir_inverted()`
- [ ] **Intraday extremes head уже есть** (`extremes_head` в multiscale_cnn_v3.py:692)
      → подключить к decision flow, сейчас output не используется

### Sprint 9 — Multi-scale temporal hierarchy: 15min → Hour → Day
**Идея:** дополнить существующую иерархию Hour→Day новым уровнем 15min→Hour.
Это даст feedback loop внутри часа для коррекции часового прогноза.

- [ ] **MinuteSpecialist** (по аналогии с HourlySpecialist):
      - Input: [B, 60, 25] — 60×15min bars (= 15h истории) × ~25 признаков
      - Target: P(next_hour up vs down)
      - Архитектура: BiLSTM(48) + multi-scale CNN(5,15,30) + 2 heads (dir + vol)
- [ ] **Cascading predictions:**
      ```
      15min bars → MinuteSpec → next_hour_pred ─┐
                                                 ├→ HourlySpec → next_day_pred ─┐
      Daily bars → V3 Daily ────────────────────┘                                ├→ MetaLearner v3
                                                                  ↑               │
                                              regime + cls + econ + dir ──────────┘
      ```
- [ ] **Trade-off:** 15min cache в ~4× больше hourly. Эффект:
      - +200% capacity for short-term momentum
      - −50% training speed
      - Возможен diminishing returns если HourlySpec уже на data ceiling (test_acc ~0.55)
- [ ] **Условие запуска:** только если Sprint 8 (intraday execution) показал value-add
      от 15min-уровня данных. Иначе откладывать в P3.

### Sprint 10 — Усиление модели
- [ ] Расширение тикеров (53 → 100 эшелон-2)
- [ ] Time-shift jitter (±2 бара) — anti-overfitting
- [ ] Mixup для последовательностей (alpha=0.2)
- [ ] Random indicator dropout (p=0.10) на тренинге
- [ ] Live intraday order management loop (#8, #14)
- [ ] Kronos fine-tuning (после B-4 fix CAWR NaN)

---

## КОМАНДЫ (актуальное состояние)

```bash
# Sprint 5 — Regime-aware pipeline (без перетренировки):
py -m ml.patch_ensemble_regime             # добавить test_regime в npz
py -m ml.decision_sweep --by-regime        # per-regime sweep
py -m ml.patch_decision_signal --regime-aware  # применить per-regime пороги
py -m ml.backtest_strategy --future-bars 5     # backtest → in-sample Sharpe +1.19

# Sprint 6 — Walk-forward (честная OOS-проверка):
py -m ml.walk_forward --folds 5            # 5-fold rolling на post-processing уровне

# Альтернативно: единые пороги B-15
py -m ml.patch_decision_signal             # default DecisionLayer (B-15)

# Полные перетренировки (опционально):
py -m ml.trainer_v3_ensemble               # V3 (~30-60 мин)
py -m ml.trainer_hourly                    # HourlySpec v1 (~7 мин)
py -m ml.meta_ensemble --rebuild meta      # MetaLearner с (date,ticker) join
py -m ml.meta_ensemble --rebuild meta --holdout-only  # честная оценка
```

---

## ЖЕЛЕЗО

```
GPU:  RTX 4050 Laptop 6GB
CPU:  AMD Ryzen 7 7735HS
OS:   Windows 11
Env:  Python 3.12 (venv), PyTorch 2.x, CUDA 12.x
Run:  py -m ml.<module>
```
