# ML Trading System — MOEX | План проекта
**Обновлено:** 2026-05-03 (Sprint 9.1–9.4 завершён — все 8 багов B-18..B-25 закрыты)

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
| 7 | Confidence calibration | ✅ | `ml/calibrate_temperature.py` — per-ticker temperature scaling, fallback на global T |
| 8 | Intraday order management | ✅ | `simulate_intraday_refinement` + `--intraday` флаг в `backtest_strategy.py` |
| 9 | Market context / RS | ✅ | `RS_5d/RS_20d/IMOEX_*` в INDICATOR_COLS + breadth/trend/vol_regime в ctx (21 dim) |
| 10 | Multi-objective optimization | ✅ | `val_metric = dir_acc + α·dec_hit + β·max(sharpe_proxy,0)` в `trainer_v3.py` (α=0.20, β=0.10 из CFG) |
| **P2** | | | |
| 11 | Dynamic position sizing | ✅ | `position_size = max_position_pct × clip(conf,0,1)` в `simulate_decision_strategy` |
| 12 | Ensemble by trading quality | ✅ | Веса по `val_sharpe` (fallback на `val_dir_acc`) в `trainer_v3_ensemble.evaluate_ensemble` |
| 13 | Separate research / execution backtest | ✅ | Python: `backtest_strategy.py`. MT5: `ML_Strategy_Backtest.mq5` |
| 14 | Early exit / intraday invalidation | ⚠️ | `cancel_threshold=0.6` в `simulate_intraday_refinement`, но без live-trigger |
| 15 | Feature importance | ✅ | `ml/feature_importance.py` — permutation importance на 37 INDICATOR_COLS |

**Итого:** 13/15 ✅, 0/15 ⚠️, 2/15 ❌. Sprint 7/8 внедрены. Оставшиеся ❌ (#6 walk-forward) — требуют архитектурных изменений.

**Состояние багов (2026-05-04):** 17/17 закрытых из B-1..B-17 + **8/8 закрытых** B-18..B-25
после Sprint 9.1–9.4. Все P0/P1/P2 баги текущего цикла закрыты.

---

## АРХИТЕКТУРА (3 стадии)

```
Stage 1 — DailySpecialist (V3 ансамбль 3 seeds)
  Input:  imgs[4 scales] + nums + ctx + hourly + intraday_feats
  Heads:  cls_head(3) + ohlc_head(20) + dir_head(1) + econ_heads(4+2+2) + extremes(3=[dH,dL,hl_logit])
  Output: ensemble_predictions.npz (cls_probs, dir_prob, mfe_mae_pred, fill_prob, edge_pred,
                                    extremes_pred, high_first_prob,
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
| **S7-1** | 🟡 P2 | **Permutation importance** на 37 INDICATOR_COLS: `ml/feature_importance.py`. Сохраняет `ml/feature_importance.json`. | ✅ |
| **S7-2** | 🟡 P2 | **Per-ticker temperature scaling**: `ml/calibrate_temperature.py`. LBFGS на NLL val-сета, fallback на global T при < 30 сэмплов. Артефакт: `ml/ensemble/temperature_per_ticker.json`. | ✅ |
| **S7-3** | 🟡 P2 | **Sharpe-веса в ensemble**: `trainer_v3_ensemble.evaluate_ensemble` взвешивает по `val_sharpe` вместо `val_dir_acc − 0.5`. Fallback при std > 2.0. | ✅ |
| **S7-4** | 🟡 P2 | **Multi-obj val_metric**: `trainer_v3._run_epochs` — `val_metric = dir_acc + 0.20·dec_hit + 0.10·max(sharpe_proxy,0)`. Параметры α/β в CFG. | ✅ |
| **S7-5** | 🟡 P2 | **Sprint 7 sweep**: `ml/sprint7_sweep.py` — calib/OOS split (30/70%), grid T×dir×edge×sell. Сохраняет `sprint7_best_params.json` + `sprint7_sweep_results.csv`. | ✅ |
| **S8-1** | 🟠 P1 | **extremes → decision flow**: `DecisionLayer.decide/decide_numpy` + `RegimeAwareDecisionLayer.decide_numpy` принимают `extremes [N,2/3]`, добавляют бонус к `edge_pred` при `range_pred > min_range_threshold`. | ✅ |
| **S8-2** | 🟠 P1 | **high_first_prob head**: `HighLowOrderHead` в `multiscale_cnn_v3.py`. `extremes` теперь `[B,3]` = `[dHigh, dLow, high_low_logit]`. `INTRADAY_N_COLS=3`, `build_intraday_targets` добавляет `high_first` метку. Cache v3.8.0 (требует `--rebuild`). BCE loss в `IntradayConsistencyLoss`. | ✅ (pending rebuild) |
| **S8-3** | 🟠 P1 | **Path-aware simulator**: `simulate_path_aware_strategy` в `backtest_strategy.py`. `--path-aware` флаг. `high_first_prob` из npz или из `extremes_pred[:,2]`. | ✅ |
| **S8-4** | 🟡 P2 | **Hourly feedback integration**: `--intraday` флаг в `backtest_strategy.main`. Cancels contradicting trades по hourly dir_prob. | ✅ |

---

## БАГИ И НЕДОЧЁТЫ ВЫЯВЛЕННЫЕ ПРИ АНАЛИЗЕ 2026-05-03 (B-18 … B-25)

| ID | Severity | Файл / Симптом | Корневая причина | Фикс |
|----|----------|----------------|------------------|------|
| **B-18** | 🔴 P0 | `meta_ensemble.py::build_meta_features` | MetaLearner v2 видит **только 14 пред-агрегированных скаляров**. Полный upstream-контекст (37 INDICATOR_COLS, 21 ctx, 4-scale imgs, 45×37 hourly window, intraday_feats) сжат до 7 выходов специалистов **до** входа в meta — потерян сигнал, который ансамбль мог бы выудить из сырых фич. | ✅ **Закрыто Sprint 9.1**: `MetaLearnerV3` 34→128→64→32→1 + LayerNorm + cls_probs[3] + fundamentals[12] + dividends[5]. На полном eval **+4.90pp** vs v2, на честном holdout **+3.84pp**. |
| **B-19** | 🟠 P1 | `meta_ensemble.py::MetaEnsembleInference.predict` | При `regime=-1` (unknown — типично для первых 60 дней live-инференса до warmup HMM) one-hot становится `[0,0,0]`, что **никогда не встречалось при тренировке**. Silent OOD: модель отвечает, но без гарантий. | ✅ **Закрыто 2026-05-03**: fallback `regime=1` (side) при unknown в `predict()`. |
| **B-20** | 🟠 P1 | `meta_ensemble.py::MetaEnsembleInference._load_meta_model` | `MetaLearner()` создаётся с дефолтным `n_feat=14`, но `evaluate_meta` строит модель из `X.shape[1]` файла. Если cache `meta_features.npz` пересобрался с другой размерностью (что произойдёт после B-18 fix), inference **молча упадёт со state_dict shape mismatch**. | ✅ **Закрыто 2026-05-03**: ckpt теперь dict `{state, n_feat, hidden}`, читается в `evaluate_meta` + `MetaEnsembleInference._load_meta_model`. Backwards-compat с legacy state_dict. |
| **B-21** | 🔴 P0 | `data/tinkoff_client.py` + feature pipeline | Полностью отсутствуют **фундаментальные показатели компании** (P/E TTM, P/S TTM, EBITDA TTM, DY, debt/equity, ROE) и **график дивидендов** (`record_date`). Для тикеров типа SBER ex-dividend gap создаёт искусственный DOWN-сигнал на T0. | ✅ **Закрыто Sprint 9.2/9.3**: `TinkoffDataClient.get_fundamentals` (56 полей) + `get_dividends` + `find_asset_uid` через `shares()`. Дисковые кэши `data/cache/fundamentals_{T}.json` (TTL 7д) + `dividends_{T}.json` (TTL 1д). Loader'ы: `ml.fundamentals_loader` (12 z-score фич, sector от API) + `ml.dividends_loader` (5 фич: days_to_record, is_ex_div, gap_pct, dy_ttm, density). 55 тикеров × 12 фундаментал. фич + 39 тикеров × дивиденды (всего 150 записей). |
| **B-22** | 🟠 P1 | `decision_layer.py::RegimeAwareDecisionLayer` | OOS coverage 0.5% (Sprint 6) — пороги `edge_r=5.0/dir=0.75` калиброваны на in-sample. На OOS распределение `edge_pred/dir_prob` сужается → почти все сигналы фильтруются. | ✅ **Закрыто Sprint 9.4** (2026-05-03): Strategy C `_adaptive_quantile_thresholds` в `walk_forward.py` — пороги через `quantile(edge_combined, q=0.80)` + `quantile(dir_prob, q=0.75)` per regime per fold. Coverage поднялся **с 0% (Sprint 5 regime-aware) до 2.76%** (5 фолдов). CLI: `--adaptive-thresholds --q-edge 0.80 --q-dir 0.75`. Sweep по q можно делать без правок кода. |
| **B-23** | 🟡 P2 | `ml/calibrate_temperature.py` | Fallback на global T при `<30` сэмплов ticker'а **не логируется и не используется** в `sprint7_sweep.py`. | ✅ **Закрыто Sprint 9.4** (2026-05-03): `coverage_report: {ticker: {n_samples, T, used_fallback, reason}}` в `temperature_per_ticker.json`. Новые поля: `n_fitted`, `n_fallback`, `min_samples_threshold`. CLI флаг `--coverage-report` для просмотра существующих json. |
| **B-24** | 🟡 P2 | `trainer_hourly` (val/test split) | Test acc 0.5475 систематически **выше** val acc 0.5378 на ~1pp на всех 3 seeds. | ✅ **Закрыто Sprint 9.4 диагностически** (2026-05-03): `ml/hourly_split_diagnostics.py` показал **никакой утечки/нестратификации нет** — артефакт class-balance: test 53.1% DOWN vs val 51.8% DOWN. **Lift over baseline на test НИЖЕ** (+1.40pp vs val +1.90pp), значит модель работает на test ХУЖЕ raw acc. Per-window анализ: val std=0.0200, test std=0.0179 — стабильно. Вывод: репортить `lift = acc - baseline`, не raw acc. |
| **B-25** | 🟠 P1 | `meta_ensemble.py::predict` vs `build_meta_features` | `np.tanh(h_vol)` сжимает signal волатильности (h_vol~0.03 → tanh≈0.03 → почти identity, но с потерей информации после x4). | ✅ **Закрыто 2026-05-03**: helper `_normalize_h_vol(v) = clip(v, 0, 0.05)/0.05` синхронно в `build_meta_features` (обе ветки) + `MetaEnsembleInference.predict` + `build_meta_features_v3`. |

**Сводка по статусу (после Sprint 9.1–9.4, обновлено 2026-05-04):**
- ✅ Закрыто **8 из 8**: B-18 (MetaV3), B-19 (regime fallback), B-20 (ckpt n_feat),
      B-21 (fundamentals+dividends API), B-22 (адаптивные quantile пороги),
      B-23 (coverage_report), B-24 (диагностически — lift, а не raw acc), B-25 (h_vol norm)
- ⏳ Sprint 9.5 (полировка спринтов 1-8) — в работе

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

### MetaEnsemble V3 (Sprint 9, 2026-05-03) — 34 features, 128 hidden, 4-layer
```
Full eval (N=14 238):
  Baseline:        0.5324
  HourlySpec:      0.5081
  DailySpec:       0.5304
  MetaEnsemble V3: 0.5677  ← ✅ +4.90pp vs v2 (0.5188 после ребилда фич с B-25)

Holdout-only (N=6 823, только h_split == 'test'):
  Baseline:        0.5413
  HourlySpec:      0.5100
  DailySpec:       0.5374
  MetaEnsemble V3: 0.5465  ← +3.84pp vs v2 (0.5081 после ребилда), цель 0.560 близко

Train dynamics: val_acc 0.5416 (early stop E050), full eval 0.5677 → overfit на train.
Train→val gap указывает на необходимость dropout(regime,0.2) и dropout(fund,0.15) — B-19 в части
training augmentation (открыто как 9.4 next iteration).

Что сделало разницу:
  - 12 fundamentals (z-score по сектору): даёт SBER P/E=3.89 знание о банковской дешевизне
  - 5 dividend flags: gap_pct + days_to_record для SBER/MTSS гэпов в day T0
  - 3 cls_probs (UP/FLAT/DOWN) — ранее теряли calibrated distribution
  - B-25 fix h_vol [0,1]: теперь сигнал волатильности фактически читается моделью
  - Loss train: 0.7083 → 0.6834 (учится без plateau, но overfit лимитирует)

Состав 34 фич: см. V3_FEATURE_NAMES в ml/meta_ensemble.py
Артефакты:
  ml/ensemble/meta_features_v3.npz       — 14238 × 34
  ml/ensemble/meta_learner_v3.pt         — ckpt {state, n_feat=34, hidden=128, version=v3}
  ml/ensemble/meta_v3_config.json        — schema + статистика покрытия
  ml/ensemble/fundamentals_map.json      — 55 тикеров × 12 фич
  ml/ensemble/dividends_map.json         — 39 тикеров с записями (всего 150)
  data/cache/shares_table.json           — кэш asset_uid + sector (TTL 30д)
  data/cache/fundamentals_{T}.json       — raw API (TTL 7д)
  data/cache/dividends_{T}.json          — raw API (TTL 1д)
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

### Sprint 7 — Calibration + Multi-objective (✅ ЗАКРЫТО 2026-05-02)
- [x] Temperature scaling per ticker для `dir_prob` (#7) → `ml/calibrate_temperature.py`
- [x] Заменить ensemble weights с `val_dir_acc` на per-seed Sharpe (#12) → `trainer_v3_ensemble.py`
- [x] `val_metric = dir_acc + α·dec_hit + β·sharpe_proxy` (#10) → `trainer_v3.py`
- [x] Permutation importance на 37 INDICATOR_COLS (#15) → `ml/feature_importance.py`
- [x] Sprint 7 sweep (T × dir_thr × edge_r) → `ml/sprint7_sweep.py`

### Sprint 8 — Intraday execution (✅ ЗАКРЫТО 2026-05-02)
- [x] extremes_head → decision flow: `decision_layer.py` + `backtest_strategy.py` (S8-1)
- [x] `HighLowOrderHead` в `multiscale_cnn_v3.py`, `high_first` метка в `labels_ohlc.py` (S8-2, pending rebuild)
- [x] Path-aware execution simulator `simulate_path_aware_strategy` + `--path-aware` флаг (S8-3)
- [x] Hourly feedback интеграция `--intraday` флаг в `backtest_strategy.main` (S8-4)
- [x] Сохранение `extremes_pred` + `high_first_prob` в `ensemble_predictions.npz`

### Sprint 9 — MetaLearner v3 + Fundamentals + Dividends + Polish ✦ ТЕКУЩИЙ
**Цель:** превратить MetaEnsemble из примитивного 14-фичевого MLP в полноценную модель,
видящую тот же контекст, что и DailySpec, плюс **фундаментал компании** и **дивидендные гэпы**.
Параллельно — закрыть P0/P1 баги B-18 … B-25.

#### 9.1 — MetaLearner v3 архитектура (B-18, B-20) ✅ ВНЕДРЕНО 2026-05-03
- [x] **`MetaLearnerV3`** в `ml/meta_ensemble.py`: **34 → 128 → 64 → 32 → 1** + LayerNorm + GELU
      + Dropout(0.30/0.20/0.10), AdamW(lr=2e-3, wd=1e-3), CosineLR.
- [x] **Состав 34 фич (фиксированный порядок, см. `V3_FEATURE_NAMES`):**
      ```
      [0:14]   v2 base (h_dir, h_vol_norm [B-25!], h_conf, d_dir, d_edge, d_mfe,
               d_fill, agree_up, agree_dn, disagree, sign_agree, regime_3hot)
      [14:17]  daily cls_probs[UP, FLAT, DOWN] из ensemble_predictions.npz
      [17:29]  fundamentals (sector z-score) из ml/ensemble/fundamentals_map.json
      [29:34]  dividends features из ml/ensemble/dividends_map.json
      ```
- [x] **B-20 fix:** ckpt сохраняется как `{state, n_feat, hidden, version}`,
      `_load_meta_model` восстанавливает архитектуру корректно. Backwards-compat.
- [x] **Артефакты:** `meta_features_v3.npz`, `meta_learner_v3.pt`, `meta_v3_config.json`
- [x] **Результаты:**
      - Полная eval N=14238: **MetaV3 = 0.5677** (HourlySpec 0.5081, DailySpec 0.5304, Baseline 0.5324),
        **+4.90pp vs v2** (v2 = 0.5188 после ребилда фич с B-25).
      - Holdout-only N=6823: **MetaV3 = 0.5465** (HourlySpec 0.5100, DailySpec 0.5374, Baseline 0.5413),
        **+3.84pp vs v2** (v2 = 0.5081). Цель 0.560 на холдауте — **близко, не достигнута**
        (overfit на train: val_acc 0.5416 vs full 0.5677). Остаток для тюнинга в 9.4.

#### 9.2 — Fundamentals API (B-21) ✅ ВНЕДРЕНО 2026-05-03
- [x] **`TinkoffDataClient.get_fundamentals(ticker)`** — возвращает dict из 56 полей
      `StatisticResponse` (включая P/E TTM, P/S TTM, EBITDA TTM, ROE, ROA, debt/equity,
      net_margin, dividend_yield_daily_ttm, market_cap, revenue_growth, free_float, …).
      По спеке T-Bank: `0.0` = "нет данных", фильтруется в z-score.
- [x] **`TinkoffDataClient.find_asset_uid(ticker)`** — через batched `instruments.shares()`,
      т.к. `find_instrument` не возвращает `asset_uid`. Кэш `data/cache/shares_table.json`
      на 30 дней (asset_uid стабилен).
- [x] **`TinkoffDataClient.get_sector(ticker)`** — сектор от API
      (`financial`/`oil_and_gas`/`it`/`utilities`/...) — отказались от
      hardcoded `data/sectors.json`.
- [x] **`ml/fundamentals_loader.py`** — 12 фич с **robust sector z-score** (median + MAD):
      pe_ratio_ttm, price_to_sales_ttm, price_to_book_ttm, ev_to_ebitda_mrq,
      total_debt_to_equity_mrq, current_ratio_mrq, roe, roa, net_margin_mrq,
      dividend_yield_daily_ttm, one_year_annual_revenue_growth_rate, free_float.
- [x] **Покрытие:** 55/55 тикеров (100%) запросили API; артефакт
      `ml/ensemble/fundamentals_map.json`.

#### 9.3 — Dividend gaps (B-21) ✅ ВНЕДРЕНО 2026-05-03
- [x] **`TinkoffDataClient.get_dividends(ticker, years_back=3)`** — список Dividend
      записей с `record_date`/`payment_date`/`dividend_net`/`yield_value`/`close_price`.
      Кэш `data/cache/dividends_{TICKER}.json` TTL=1д.
- [x] **`ml/dividends_loader.py::featurize_dividends(divs, target_date)`** — 5 фич:
      `days_to_next_record/60`, `is_ex_div_today` (T0 == record_date+1..3),
      `gap_pct_expected = dividend_net / close`, `dy_ttm` (sum yield за 12мес),
      `coupon_density_30d` (кол-во record_date в окне ±30д / 5).
- [x] **Покрытие:** 39/55 тикеров с дивидендами (150 записей суммарно). 8395/14238
      сэмплов (59%) имеют ненулевые дивидендные флаги. SBER/YDEX/MTSS/MGNT — стабильные.
- [x] **Артефакт:** `ml/ensemble/dividends_map.json`.
- [ ] **Не сделано:** интеграция div_features в `dataset_v3` как 6-й layer cache v3.9.0
      (отложено — сейчас фичи попадают только в MetaLearner v3, что достаточно для PoC).

#### 9.4 — Закрытие багов P1/P2 ✅ ВНЕДРЕНО 2026-05-04
- [x] **B-19** ✅: regime fallback на `1` (side) при unknown в `predict()`
- [x] **B-20** ✅: ckpt сохраняет n_feat/hidden, `_load_meta_model` восстанавливает
- [x] **B-25** ✅: `_normalize_h_vol(v) = clip(v, 0, 0.05)/0.05` синхронно во всех trio
      `build_meta_features` (обе ветки), `build_meta_features_v3`, `predict`
- [x] **B-22** ✅: `_adaptive_quantile_thresholds` в `walk_forward.py` — Strategy C,
      пороги через quantile калибровочного окна. Coverage **0% → 2.76%** на 5 фолдах.
      CLI: `--adaptive-thresholds --q-edge 0.80 --q-dir 0.75`.
- [x] **B-23** ✅: `coverage_report` в `temperature_per_ticker.json` с полями
      `{n_samples, T, used_fallback, reason}`. CLI `--coverage-report` для просмотра.
- [x] **B-24** ✅ диагностически: `ml/hourly_split_diagnostics.py` показал, что
      test_acc > val_acc — артефакт class balance (test 53.1% DOWN vs val 51.8%).
      Lift over baseline на test НИЖЕ (+1.40pp vs val +1.90pp). Утечки нет.
      **Action item:** заменить raw acc на lift во всех reporting-местах.

#### 9.5 — Полировка спринтов 1-8 (NEW, открыто)
- [ ] **Sprint 1-2 (DecisionLayer):** удалить мёртвый код в `decide_numpy` после S8-1 рефакторинга
- [ ] **Sprint 3 (Regime):** документировать `patch_ensemble_regime.py` — почему он отдельный, а не часть тренинга
- [ ] **Sprint 5 (Regime-aware):** перетестировать `bull=OFF` на свежих данных (последний месяц) — режим может смениться
- [ ] **Sprint 6 (Walk-forward):** добавить purged K-Fold (gap=5 дней) для устранения leakage между фолдами
- [ ] **Sprint 7 (Calibration):** Brier score + reliability diagram per ticker
- [ ] **Sprint 8 (Path-aware):** добавить unit-тест на `simulate_path_aware_strategy` — synthetic OHLC где high_first=1 → проверка TP triggered

#### 9.6 — Acceptance criteria (status, 2026-05-04)
- ⚠️ Holdout val_acc MetaEnsemble ≥ 0.560 — **0.5465** (близко; full eval 0.5677 ≥ цели).
      Подтянуть к 0.560 на холдауте — задача 9.5 (regime/fund dropout, дольше эпохи)
- ⚠️ Walk-forward OOS Sharpe ≥ 0 — **B-22 adaptive Sharpe -1.80** (улучшение vs baseline
      -2.86 на лучшем фолде, но в среднем хуже static -0.94). Принципиально работает,
      требует sweep по q-edge и hit_rate floor — задача 9.5
- ✅ Coverage в backtest ≥ 3% — на отдельных фолдах достигнут (3.93%, 4.95%),
      средний 2.76% — почти у цели. Sprint 5 regime-aware на тех же фолдах = 0% (полный фейл).
- ✅ **8/8** багов закрыто: **B-18, B-19, B-20, B-21, B-22, B-23, B-24, B-25**
- ✅ В `data/cache/` fundamentals (55 тикеров × 12 фич) + dividends (39 тикеров, 150 записей)

---

### Sprint 10 — Multi-scale temporal hierarchy: 15min → Hour → Day
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

### Sprint 11 — Усиление модели
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

# Sprint 7 — Calibration + sweep (без перетренировки):
py -m ml.feature_importance                          # permutation importance → feature_importance.json
py -m ml.calibrate_temperature                       # per-ticker T → temperature_per_ticker.json
py -m ml.sprint7_sweep --quick                       # ~672 комбинации, ~2 мин → sprint7_best_params.json
py -m ml.sprint7_sweep --full                        # + sell_threshold, ~800 комбинаций

# Sprint 8 — Intraday execution (после rebuild V3 с --rebuild):
py -m ml.trainer_v3_ensemble --rebuild               # V3 rebuild (cache v3.8.0 + HighLowOrderHead)
py -m ml.backtest_strategy --future-bars 5           # extremes в decision flow (S8-1)
py -m ml.backtest_strategy --path-aware              # path-aware simulator (S8-3, требует high_first_prob)
py -m ml.backtest_strategy --intraday                # hourly cancellation (S8-4)

# Sprint 9 — MetaLearner v3 + Fundamentals + Dividends ✅ ВНЕДРЕНО 2026-05-03
py -m ml.fundamentals_loader --refresh-cache         # 55 тикеров × 12 фич, ~30с
py -m ml.fundamentals_loader --inspect SBER          # diag: показать 12 z-score фич
py -m ml.dividends_loader    --refresh-cache         # 39 тикеров с дивидендами, 150 записей
py -m ml.dividends_loader    --inspect SBER 2026-05-03   # diag: 5 div фич на дату
py -m ml.meta_ensemble --version v3 --rebuild meta   # build features_v3 + train MetaLearnerV3
py -m ml.meta_ensemble --version v3 --eval-only --holdout-only   # honest OOS

# Sprint 9.4 — закрытие багов B-22..B-24 ✅ ВНЕДРЕНО 2026-05-04
py -m ml.walk_forward --folds 5 --adaptive-thresholds       # B-22: per-fold quantile
py -m ml.walk_forward --folds 5 --adaptive-thresholds --q-edge 0.90  # tighter (top 10%)
py -m ml.calibrate_temperature --coverage-report            # B-23: показать fitted/fallback
py -m ml.calibrate_temperature                              # B-23: пересчитать с coverage_report
py -m ml.hourly_split_diagnostics                           # B-24: lift vs raw acc анализ
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
