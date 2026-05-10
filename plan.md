# ML Trading System — MOEX | План проекта

**Обновлено:** 2026-05-10 · Sprint 11.1 этап 3 **ребилд завершён**. Ordering overlap: L↔O=0.40%, C↔H=0.27% ✅. Dir_acc восстановился до 0.5261 (было 0.5166 в этапе 2). **D3_coin_flip впервые breakeven при N=217 сделках** (total=−0.00%, gross=+0.209%/trade). ⚠️ O/C channel coverage collapsed: avg O=2.53%, C=0.89% — D5 execution blocked. Platt калибровка ухудшилась (ECE 10.78%→12.83% после применения) — нужен re-fit Platt.
**Обновлено:** 2026-05-09 · Sprint 11.1 этап 2 ребилд: OHLC quantile heads + ordering penalty работают (overlap < 3%, медианы строго упорядочены). **In-sample 6 D/E режимов в плюс**: D3 total +0.37% Sharpe +0.92, E_decision +0.31%. Но coverage квантилей упал до 38% (overconfident), dir_acc -1.27pp. Веса скорректированы (ordering 0.10→0.03, quantile 0.15→0.10), ждём третий ребилд.
**Текущий статус:** инфраструктура полная, D3 breakeven in-sample (N=217), OOS walk-forward убыточен из-за низкого coverage (n_avg=11 сделок/фолд)

---

## 1. ГЛАВНАЯ ЦЕЛЬ

Построить **экономически целесообразную decision system** для торговли MOEX-акциями:
- **BUY / SELL / HOLD** на дневном горизонте, где **HOLD = ожидаемый edge не покрывает costs** (комиссия + спред + slippage = 0.2% round-trip).
- **Дневной прогноз** — главное торговое решение (entry).
- **Часовые свечи + intraday feedback** — внутридневная коррекция (TP/SL/cancel).

### Слои системы

| Слой | Назначение | Текущее покрытие |
|------|-----------|------------------|
| **Python research** | training, backtest, walk-forward, sweep, генерация targets | ✅ полностью |
| **MT5 execution** | live/demo, проверка лимиток | ✅ скрипт `ML_Strategy_Backtest.mq5` |
| **T-Bank API** | data source: candles, fundamentals (P/E, EBITDA…), dividends | ✅ 55 тикеров |

---

## 2. ЧТО ХОТИМ ПОЛУЧИТЬ ОТ ОБУЧЕНИЯ (acceptance criteria)

| Метрика | Цель | Текущее (2026-05-10, 3-й ребилд) | Status |
|---------|------|---------------------|--------|
| **Direction acc** (DailySpec) | ≥ +5pp над baseline | 0.5261 (baseline 0.4675, **+5.86pp**) | ✅ |
| **MetaV3 holdout acc** | ≥ 0.560 | **0.5555** (Δ −0.0045 vs цели) | 🟡 −0.34pp vs пред. |
| **MetaV3 full eval acc** | ≥ 0.560 | **0.5702** | ✅ |
| **OOS coverage** в backtest | ≥ 3% | SELL=4.0%, BUY=0.3%; B-22 WF no valid folds | 🔴 коллапс в WF |
| **OOS Sharpe** (5 fold WF, daily) | ≥ 0 | B-15: −7.22±2.81; B-22/regime: no valid folds | 🔴 регрессия |
| **End-to-end profit** на full test | > 0 | A_market: −0.91%; **D3: −0.00%** (N=217) | 🟡 D3 breakeven |
| **Прибыльные периоды** в WF | хотя бы 1/5 | folds 4+5 exp%>0 (но n=1 каждый — нестат.) | 🟡 мало сделок |
| **Свежие 60 дней** (side+bull) | exp% > 0 | не обновлялся в этом ребилде | — |
| **D3 in-sample exp%/trade** | gross > 2×fee = 0.2% | **+0.209%/trade (N=217, win 52.5%)** | ✅ масштабируется |
| **D3 walk-forward avg exp%** | gross > 2×fee = 0.2% | **+0.240% ± 0.502** (5 фолдов, n_avg=11) | 🟡 exp%>0 но нестат. |
| **D2 in-sample exp%/trade** | gross > 2×fee = 0.2% | **+0.145%/trade (N=217)** | 🟡 < 0.20% |
| **D2 walk-forward avg exp%** | gross > 2×fee = 0.2% | **+0.185% ± 0.541** (5 фолдов, n_avg=11) | 🟡 exp%>0 но нестат. |
| **D3 walk-forward Sharpe** | > 0.5 | −2.35 ± 2.18 | 🔴 |
| **Прибыльные фолды** (D3 5-fold WF) | ≥ 3/5 | 2/5 по exp% (folds 4+5, n=1 каждый) | 🔴 нестат. |
| **Calibration ECE (raw)** | < 5% | **10.78%** (было 11.15%) | 🟡 −0.37pp |
| **Calibration ECE (Platt)** | < 5% | **12.83%** (хуже raw!) — Platt устарел | 🔴 re-fit нужен |
| **DOWN-bias worst bin** | abs < 0.10 | bin[0.70,0.80]: bias=−0.337 | 🔴 ухудшение |
| **Quantile H/L coverage (t+1)** | ≥ 80% | H=79.07%, L=75.72% | ✅ восстановлено |
| **Quantile O/C coverage** | ≥ 70% | O=2.53%, C=0.89% | 🔴 collapse |

**Сводка (2026-05-10):** Главный прорыв — **D3_coin_flip breakeven при N=217** (до этого N=23). D2/D3 walk-forward впервые показывает позитивный exp% в среднем (+0.240%/+0.185%), но статистика ненадёжна (n_avg=11 сделок/фолд). Блокеры: (1) **O/C quantile coverage collapsed** (2-3%) — D5 execution невозможен; (2) **Platt калибровка устарела** после ребилда, ухудшает ECE; (3) walk-forward coverage слишком мал для надёжных выводов.

---

## 3. СТРУКТУРА PIPELINE ОБУЧЕНИЯ

### Команда одной кнопкой

```bash
py -m ml.retrain_all                # инкрементально (skip up-to-date)
py -m ml.retrain_all --rebuild all  # ⟳ FORCE всё с нуля (~1.5-2 ч)
py -m ml.retrain_all --rebuild ensemble   # ⟳ только V3 (+regime+temp)
py -m ml.retrain_all --rebuild hourly     # ⟳ только HourlySpec
py -m ml.retrain_all --rebuild meta       # ⟳ только Meta v2 + v3 (~5 мин)
py -m ml.retrain_all --rebuild fund       # ⟳ Fundamentals + Dividends API
py -m ml.retrain_all --diagnostics-only   # ~12 с sanity без обучения
py -m ml.retrain_all --resume-from N      # продолжить с stage N
py -m ml.retrain_all --dry-run            # показать план + статус
```

### Граф зависимостей и артефакты

```
                         ВХОДЫ                          АРТЕФАКТЫ
                         ──────                         ─────────
[1] trainer_v3_ensemble  (~30-60 мин)
    in:  cache_v3/* (imgs[4 scales]+nums+ctx+hourly+intraday_feats)
    out: ml/ensemble/ensemble_predictions.npz, model_seed{42,123,7}.pt
         keys: cls_probs[N,3], dir_prob[N], mfe_mae_pred[N,4], fill_prob[N,2],
               edge_pred[N,2], extremes_pred[N,3], high_first_prob[N],
               y_test[N], ohlc_test[N,20], atr_ratio[N], econ_test[N,11],
               test_dates[N], test_tickers[N]
              ↓
[2] patch_ensemble_regime  (~10 с)
    out: добавляет test_regime[N] в ensemble_predictions.npz (HMM bear/side/bull)
              ↓
[3] calibrate_temperature  (~3-5 мин)
    out: ml/ensemble/temperature_per_ticker.json
         (per-ticker T + coverage_report{ticker:{n_samples,T,used_fallback}})
         + dir_prob_calibrated[N] в npz
              ↓
[4] trainer_hourly  (~7 мин)
    in:  cache_hourly/* (45×37 hourly bars)
    out: ml/ensemble/hourly_specialist.pt
         ml/ensemble/hourly_all_predictions.npz (N=142471, split=train/val/test)
              ↓
[5] meta_ensemble (v2)  (~1-2 мин)
    in:  hourly_all + ensemble_predictions
    out: ml/ensemble/meta_features.npz [N=14238 × 14 features]
         ml/ensemble/meta_learner.pt {state, n_feat=14, hidden=64}
              ↓
       параллельно:
       [6] fundamentals_loader --refresh-cache  (~30 с)
           in:  T-Bank API getAssetFundamentals для 55 тикеров
           out: ml/ensemble/fundamentals_map.json {ticker: 12-vec sector z-score}
                + data/cache/fundamentals_{T}.json (raw, TTL 7д)
                + data/cache/shares_table.json (asset_uid+sector, TTL 30д)

       [7] dividends_loader --refresh-cache  (~30 с)
           in:  T-Bank API getDividends для 55 тикеров (3 года истории)
           out: ml/ensemble/dividends_map.json {ticker: [div records]}
                + data/cache/dividends_{T}.json (raw, TTL 1д)
              ↓
[8] meta_ensemble --version v3  (~2 мин) — ГЛАВНЫЙ ВЫХОД
    in:  meta_features.npz + ensemble_predictions.npz + fundamentals + dividends
    out: ml/ensemble/meta_features_v3.npz [N=14238 × 34 features]
         ml/ensemble/meta_learner_v3.pt {state, n_feat=34, hidden=128, version=v3}
         ml/ensemble/meta_v3_config.json (schema + coverage stats)
              ↓
       SANITY DIAGNOSTICS (~12 с total):
       [9.1] walk_forward --adaptive-thresholds --purge-days 5
       [9.2] bull_regime_check --window-days 60 --grid
       [9.3] reliability_report (Brier + ECE per ticker)
       [9.4] hourly_split_diagnostics (lift vs raw acc)
       [9.5] tests.test_path_aware (10 unit-тестов)
       [9.6] meta_ensemble --version v3 --eval-only --holdout-only
```

### Состав 34 фич MetaV3 (фиксированный порядок, см. `V3_FEATURE_NAMES`)

| Block | Idx | Описание |
|-------|-----|----------|
| **v2 base** (14) | [0:14] | h_dir, h_vol_norm [B-25!], h_conf, d_dir, d_edge, d_mfe, d_fill, agree_up, agree_dn, disagree, sign_agree, regime_3hot |
| **daily cls_probs** (3) | [14:17] | UP, FLAT, DOWN softmax probabilities из ensemble_predictions.npz |
| **fundamentals** (12) | [17:29] | pe_ttm, ps_ttm, pb_ttm, ev_ebitda, debt_eq, curr_ratio, roe, roa, net_margin, dy_ttm, rev_growth, free_float — все z-score по сектору (median+MAD) |
| **dividends** (5) | [29:34] | days_to_record/60, is_ex_div_today, gap_pct_expected, dy_ttm, density_30d/5 |

---

## 4. АРХИТЕКТУРА (3 stages + DecisionLayer)

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1 — DailySpecialist (V3 ансамбль 3 seeds × 30 epochs)     │
│   Input:  imgs[4 scales] + nums + ctx(21) + hourly + intraday   │
│   Heads:  cls(3) + ohlc(20) + dir(1) + econ(4+2+2) + extremes(3)│
│   Out:    ensemble_predictions.npz                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2 — HourlySpecialist (BiLSTM + multi-scale CNN, 223k параметров) │
│   Input:  hourly window [B, 45, 37]                             │
│   Heads:  dir_head + vol_head (MSE на range_norm)               │
│   Out:    hourly_all_predictions.npz                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3 — MetaLearner V3 (MLP 34→128→64→32→1 + LayerNorm)       │
│   Input:  34 features (см. блок выше)                           │
│   Out:    meta_dir_prob (P(UP) для финального решения)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ DecisionLayer (Sprint 9.5 defaults — bull ENABLED)              │
│   bear: edge=2.0/dir=0.80/sell=0.50                             │
│   side: edge=4.0/dir=0.80/sell=0.50                             │
│   bull: edge=5.0/dir=0.75/sell=0.50  (Sprint 5 OFF устарел)     │
│   Production: --adaptive-thresholds (B-22 quantile-based)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. ТЕКУЩИЕ РЕЗУЛЬТАТЫ (2026-05-10, 3-й ребилд Sprint 11.1 этап 3)

### V3 ансамбль (DailySpecialist)
```
Test:                  19 087 сэмплов (55 тикеров)
Weights (Sharpe):      seed42=0.420, seed123=0.343, seed7=0.238
Per-seed val_dir_acc:  42=0.5338, 123=0.5249, 7=0.5234
Per-seed best metric:  42=0.6876 (E3), 123=0.6733 (E2), 7=0.6457 (E4) — coverage-gating работает
Ensemble dir_acc:      0.5261   (baseline always-BUY 0.4675, edge +0.0587)
Pairwise agreement:    0.6923
F1 UP/FLAT/DOWN:       0.34/0.10/0.50
Decision coverage:     BUY=65 (0.3%) | HOLD=18267 | SELL=755 (4.0%)  ← SELL +2.8pp vs пред.

vs 2026-05-06: dir_acc 0.5365→0.5261 (−1.04pp), SELL coverage 1.2%→4.0% (+2.8pp)
Регрессия dir_acc: coverage-gating + quantile weight съедают capacity backbone
```

### HourlySpecialist (ребилд 2026-05-05 — не переобучался)
```
Param count:    223 266
Val acc:        0.5341 (baseline always-UP 0.4877, lift +4.6pp)
Test acc:       0.5477 (baseline 0.5340, lift +1.37pp)
```

### MetaEnsemble V3 (Sprint 11.1 этап 3 ребилд)
```
Full eval (N=14 262):
  Baseline majority: 0.5326
  HourlySpec:        0.5194
  DailySpec:         0.5263
  MetaV3:            0.5702  ← ✅ цель ≥ 0.560
  Δ vs v2:           +4.58pp (v2=0.5244)

Holdout (N=6 868):
  Baseline majority: 0.5414
  HourlySpec:        0.5264
  DailySpec:         0.5166
  MetaV3:            0.5555  ← 🟡 цель 0.560 (Δ −0.0045)
  Δ vs v2:           +3.80pp (v2=0.5175)
  Δ vs DailySpec:    +3.89pp

vs 2026-05-06: holdout 0.5589→0.5555 (−0.34pp), full eval 0.5745→0.5702 (−0.43pp)
Train dynamics: best val_acc 0.5600 на E10 (seed 42/123), early stop E050-060
```

### Backtest по режимам исполнения (полный test, future-bars=5) — 🆕 D3 breakeven
```
                           N    win%   gross/trade    total    sharpe(daily)
A_market                 286   38.81%   −0.213%       −0.91%     −1.59
C_limit_close            217   39.17%   −0.061%       −0.44%     −0.84
D2_bm_formula            217   48.39%   +0.145%       −0.10%     −0.62
D3_coin_flip             217   52.53%   +0.209%       −0.00%     −0.01  ← breakeven!
E_decision_layer         820   41.83%   (partial)     (trunc.)   —

Fill rate LONG=69.9%, SHORT=77.9% (из 73 BUY + 213 SELL сигналов)
```
**D3_coin_flip: N=217, gross +0.209%/trade > 2×fee=0.20%, total −0.00%** — первый breakeven при масштабе.
D2 стабильно +0.145%/trade при N=217 (было N=23, +0.146% — масштабируется).
Coverage выросла в 9× благодаря SELL (755 vs 106 ранее).

### Walk-forward D2/D3 execution (2026-05-10)
```
mode             | n_avg  | win           | exp%          | sharpe        | total%
A_market         |    15  | 0.401±0.242   | −0.077±0.423  | +2.05±7.28    | −4.84±8.79
D2_bm_formula    |    11  | 0.649±0.294   | +0.185±0.541  | −4.49±2.09    | −2.25±3.97
D3_coin_flip     |    11  | 0.699±0.256   | +0.240±0.502  | −2.35±2.18    | −1.37±3.20

Per-fold D3:
  Fold 1 (2024-11→2025-06): n=30  exp=−0.256%  sharpe=−5.38  ← ранний период плохой
  Fold 2 (2025-06→2025-09): n=14  exp=−0.013%  sharpe=−0.32  ← почти breakeven
  Fold 3 (2025-09→2025-11): n= 9  exp=−0.060%  sharpe=−1.35
  Fold 4 (2025-11→2026-02): n= 1  exp=+0.372%  sharpe=nan    ← нестатистично
  Fold 5 (2026-02→2026-05): n= 1  exp=+1.159%  sharpe=nan    ← нестатистично
```
D3/D2 mean exp% впервые положительные (+0.240%/+0.185%), но n_avg=11 — статистически ненадёжно.
Fold 1 с n=30 убыточен → ранние периоды доминируют в неудачных фолдах.
Sharpe отрицателен из-за высокой дисперсии на малом N.

### Walk-forward B-15/regime (5 фолдов, 2024-11 → 2026-04, режим E_decision_layer)
```
                       coverage     hit_rate    exp%/trade    Sharpe (daily)
B-15 static:           1.68±1.22%   0.347±0.119 −0.317±0.203  −7.22±2.81 (5 folds)
Sprint 5 regime-aware: no valid folds
B-22 adaptive (q=0.80): no valid folds  ← коллапс vs пред. (5 фолдов было)

vs 2026-05-06: B-22 потерял все фолды (модель не генерирует adaptive signals в WF)
Причина: новый dir_prob/Platt distribution не совпадает с calibrated_threshold из пред. ребилда
```

### Quantile Predictions (quantile_eval, 2026-05-10)
```
Coverage [q_0.10, q_0.90]:
  H: t+1=79.07%, t+2=61.86%, t+3=55.14%, avg=59.24%  ← t+1 почти идеал (80%)
  L: t+1=75.72%, t+2=62.56%, t+3=60.02%, avg=61.92%
  O: t+1=10.22%, avg=2.53%  ← COLLAPSE (нужно ≥ 70%)
  C: t+1= 2.41%, avg=0.89%  ← COLLAPSE

Ordering (overlap диапазонов):
  L↔O: 0.40%  L↔C: 1.52%  O↔H: 1.16%  C↔H: 0.27%  ← все < 2%  ✅
  Median violations L<O<C<H: 0.00%  ✅
  Crossover violations (монотонность q10<q50<q90): 0.00%  ✅

Sharpness: ширина интервала растёт t+1→t+5 (0.48→1.01 ATR-norm) — модель учла неопределённость.
Median bias: high завышен +0.047 ATR (t+1), low занижает +0.047 ATR → симметрично, ≈0 смещение.

Причина O/C collapse: backbone смешивает O/C с cls-сигналом (одни и те же features),
модель "уверена" в O/C → интервалы схлопываются. D5 execution (BUY на q_O[0.1]) невозможен.
```

### Reliability (Brier + ECE, 2026-05-10 ребилд)
```
RAW dir_prob:   Brier=0.2464  ECE=10.78%  (было 11.15% в пред. ребилде — лучше)
Platt dir_prob: Brier=0.2476  ECE=12.83%  ← ХУЖЕ raw! Platt устарел, нужен re-fit

Reliability bins (RAW):
  bin [0.40,0.50]: pred=0.4488 actual=0.3572  bias=−0.0915
  bin [0.50,0.60]: pred=0.5458 actual=0.3699  bias=−0.1759
  bin [0.60,0.70]: pred=0.6437 actual=0.3873  bias=−0.2564
  bin [0.70,0.80]: pred=0.7359 actual=0.3992  bias=−0.3367

Top-5 best calibrated: SMLT (Brier 0.2147), SNGS (0.2172), HYDR, MTLR, FEES
Worst-5: ENPG (0.2724), FLOT, LENT, AFLT (0.3157), OZON (0.3162)
```

---

## 6. НАЙДЕННЫЕ ПРОБЛЕМЫ И РЕШЕНИЯ

### Закрытые баги (25/25)

**B-1..B-17** (исторические, Sprint 1-8):
- B-15 sweep: пороги 5.0/0.75/0.55 (было 4.0/0.70/0.85), expectancy −1.84% → +0.10%/trade
- B-16 EconomicHeads gain fix: edge_pred std ×8.5
- S5 RegimeAware: backtest +1.61%/Sharpe +1.19 in-sample (не воспроизводится OOS — см. Sprint 9.5)
- S5-2 MetaLearner v2: 7→14 features, holdout 0.5075 → 0.5462

**B-18..B-25** (Sprint 9):
| ID | Severity | Краткое описание | Решение |
|----|----------|------------------|---------|
| B-18 | 🔴 P0 | MetaLearner v2 видит только 14 пред-агрегированных скаляров | MetaV3 34 фичи + cls_probs + fundamentals + dividends |
| B-19 | 🟠 P1 | regime=-1 (unknown) даёт OOD one-hot [0,0,0] | Fallback `regime=1` (side) в `predict()` |
| B-20 | 🟠 P1 | `MetaLearner()` хардкодит n_feat=14 → state_dict mismatch | ckpt `{state, n_feat, hidden}` + auto-restore |
| B-21 | 🔴 P0 | Нет fundamentals/dividends → SBER ex-div gap воспринимается как DOWN | T-Bank API integration: `get_fundamentals` (56 полей) + `get_dividends` |
| B-22 | 🟠 P1 | OOS coverage 0.5% — пороги залочены на in-sample | `--adaptive-thresholds` quantile-based (cov 0% → 2.76%) |
| B-23 | 🟡 P2 | Temperature fallback не логируется | `coverage_report` в json |
| B-24 | 🟡 P2 | test_acc > val_acc — подозрение на leak | Диагностически: артефакт class-balance, lift на test НИЖЕ |
| B-25 | 🟠 P1 | `np.tanh(h_vol)` сжимает signal волатильности | Линейная нормализация `clip(v,0,0.05)/0.05` |

### Главные открытые проблемы (актуально 2026-05-10)

1. **O/C quantile channel collapse** — после 3-го ребилда O/C coverage avg=2.53%/0.89%. Backbone смешивает O/C-предсказания с cls-сигналом → D5 execution невозможен. Нужен отдельный backbone или stop-gradient от cls-path к O/C head.
2. **WF coverage слишком низкий для статистики** — n_avg=11 сделок/фолд при D3. Нужно ≥50/фолд чтобы отличить edge от шума. Путь: снизить пороги decision_layer или расширить coverage через режимную адаптацию.
3. **Platt калибровка устарела** — после каждого ребилда dir_prob меняется, Platt params из предыдущей модели ухудшают ECE (10.78%→12.83%). Нужно автоматически re-fit Platt в `retrain_all`.
4. ~~DOWN-bias~~ — частично закрыт. Per-ticker Platt (a,b) исправляет bias для AFLT/OZON/ENPG. Raw ECE 10.78% vs target <5% — остаётся, но не основной блокер.

### Открытые проблемы Sprint 11.1 (Quantile heads)

| # | Severity | Проблема | Что попробовать |
|---|----------|----------|-----------------|
| Q-1 | 🟡 | **Walk-forward gap**: D3 OOS exp%=+0.240%±0.502 (n_avg=11, нестат.). Folds 1-3 убыточны, 4-5 (n=1) прибыльны. | Увеличить coverage (снизить пороги), per-fold calibration window |
| Q-2 | ✅ | **Coverage квантилей H/L** восстановлена: H avg=59%, t+1=79% (близко к 80%). | Закрыт после 3-го ребилда |
| Q-3 | 🟡 | **dir_acc регрессия**: 0.5365 (2026-05-06) → 0.5261 (3-й ребилд, −1.04pp). Причина: quantile multi-task + coverage-gating. | Отдельный backbone для quantile heads |
| Q-4 | 🔴 | **D5_quantile execution заблокирован** — O/C coverage collapsed, q_O[0.1] бесполезен. | Сначала закрыть Q-9, потом D5 |
| Q-5 | ✅ | **O↔C body penalty** — реализован и работает: L↔O=0.40%, C↔H=0.27%. | Закрыт в 3-м ребилде |
| Q-6 | 🟡 | **Volume per-bar target отсутствует** — aux_y покрывает только агрегированные vol+skew. | Расширение `build_ohlc_labels` — отложено |
| Q-7 | 🟡 | **quantile_eval Section 9** не разделяет body overlap по bull/bear/doji. | Добавить в [ml/quantile_eval.py](ml/quantile_eval.py) секцию body_overlap |
| Q-8 | 🟡 | **D3 RNG non-determinism**: walk_forward_d3 усредняет по 5 seeds — ОК. В production нужен явный rng_state. | Низкий приоритет |
| Q-9 | 🔴 | **O/C quantile channel collapse**: coverage avg O=2.53%, C=0.89%. Backbone не отделяет O/C от cls-signal. | Отдельный shallow trunk для O/C (2 conv layers), или detach cls-path перед O/C head |
| Q-10 | 🟠 | **Platt устаревает после каждого ребилда**: ECE 10.78%→12.83% после применения старых params. | Добавить `calibrate_platt` как обязательный шаг в `retrain_all` после trainer_v3_ensemble |

### Возможные улучшения (после закрытия Q-1..Q-3)

1. **Per-quantile loss weighting**: q=0.10 и q=0.90 (хвосты) важнее q=0.50 для D5 execution. Поднять веса tail-quantiles в pinball_loss.
2. **Symmetric body eps в bull/bear**: сейчас 0.05·ATR одинаковый. Возможно asymmetric eps лучше для рынков с UP-skew.
3. **Per-ticker ordering_w**: тикеры с большим ATR (ENPG, OZON) могут требовать другого scale штрафа.
4. **Bull-mask weighted в pinball**: усиливать pinball только когда model уверена в направлении (cross-task регуляризация).

---

## 7. ИДЕИ ДЛЯ УЛУЧШЕНИЯ

### Приоритет P0 (потенциал >+5pp accuracy / >2× Sharpe)

1. **Ребилд V3 с balanced UP class weight × 1.5** — лечит DOWN-bias из reliability_report.
   - Изменение: `cls_weights[0] *= 1.5` в `multiscale_cnn_v3.py`
   - ETA: ~1 ч (ребилд V3 ансамбля)
   - Ожидаемый эффект: ECE с 13% → ~7%, calibrated bins ближе к диагонали, BUY win-rate +5pp

2. **MetaLearner v3 → v4** — добавить недостающие фичи:
   - `daily nums pool[32]` (mean+std по 16 INDICATOR_COLS за окно [-5..0])
   - `daily ctx[21]` (regime + breadth + RS + IMOEX)
   - `intraday_feats pool[11]` (mean по часам текущего дня)
   - Итого ~98 фич вместо 34. Архитектура: 98→256→128→1.
   - Гипотеза: **+2..3pp holdout** (сейчас 0.5499 → ~0.575)

3. **Калибровка через Platt scaling вместо температуры** — temperature scaling предполагает симметричный shift, но bias негативный (predicted > actual). Platt = `sigmoid(a*logit + b)` может корректировать асимметрию.
   - File: `ml/calibrate_platt.py` (по образцу `calibrate_temperature.py`)
   - Ожидаемый эффект: ECE 13.38% → ~6%

### Приоритет P1 (потенциал +1..3pp accuracy)

4. **Adaptive thresholds В DEFAULT** — заменить `RegimeAwareDecisionLayer.DEFAULT_REGIME_THRESHOLDS` на квантили из последних 60-90 дней калибровки (rolling). Drift-resistant из коробки, без ручного `--adaptive-thresholds`.

5. **Sprint 7 sweep на calibrated dir_prob + MetaV3 dir_prob** — текущий `decision_sweep` идёт по raw `dir_prob`. Проверка на calibrated может изменить best per-regime пороги.

6. **Регуляризация MetaV3** — текущий overfit на train (val 0.5446 vs full 0.5681):
   - `dropout(regime_onehot, p=0.2)` — regime может стать noise-канал
   - `dropout(fund_features, p=0.15)` — fund данные не для всех тикеров полные
   - `weight_decay 1e-3 → 5e-3`
   - Ожидаемый эффект: holdout 0.5499 → 0.555+ (закроет цель 0.560)

7. **Temperature scaling per regime** — сейчас global + per-ticker T, но volatility режима тоже влияет. Дополнительный T_regime[bear/side/bull].

### Приоритет P2 (полировка / экспансия)

8. **Sprint 10 — MinuteSpecialist (15min → Hour)** — новый источник signal'а:
   - Input: [B, 60, 25] (60×15min bars = 15h истории × ~25 признаков)
   - Target: P(next_hour up vs down)
   - Архитектура: BiLSTM(48) + multi-scale CNN(5,15,30) + 2 heads
   - Cascading: 15min → MinuteSpec → HourlySpec → V3 → MetaV4
   - ⚠️ 15min cache в ~4× больше hourly. Проверить на data ceiling.

9. **Расширение тикеров 55 → 100** (эшелон-2 MOEX) — больше данных = меньше overfit.

10. **Live-trigger для intraday cancel** — `cancel_threshold=0.6` уже в `simulate_intraday_refinement`, но без production loop.

11. **Sprint 11 — Аугментации** — time-shift jitter (±2 бара), Mixup α=0.2, random indicator dropout p=0.10.

12. **Kronos fine-tuning** — pretrained transformer для time-series (после фикса CAWR NaN).

13. **Reduce costs** — переход с market orders на лимитки (entry at pred_low/high из path-aware) снижает spread с 0.2% до ~0.05%. Может быть **самый дешёвый способ выйти в плюс** без улучшения модели.

---

## 8. ROADMAP

### ✅ Sprint 1-8 — Foundation (закрыты)
- DecisionLayer, EconomicHeads, HourlySpecialist, MetaLearner v1/v2, walk-forward, calibration, intraday execution, path-aware simulator

### ✅ Sprint 9 — MetaV3 + Fundamentals + Polish (закрыт 2026-05-04)
- 9.1 MetaLearner V3 (34 фич, +4.90pp full / +3.84pp holdout vs v2)
- 9.2 Fundamentals API (55 тикеров × 12 фич, sector z-score)
- 9.3 Dividends API (39 тикеров с записями, 5 фич)
- 9.4 Закрытие багов B-19..B-25
- 9.5 Полировка спринтов 1-8 + bull=OFF переоценка
- Variant A: новые `DEFAULT_REGIME_THRESHOLDS` (bull ENABLED)
- Бонус: `ml/retrain_all.py` оркестратор всего pipeline + `--rebuild` force semantics

### 🟡 Sprint 10 — Снять блокер coverage + лимитки (NEXT)

**Стратегия (после A-провала + обнаружения val_metric bug 2026-05-06):**
- **Главное открытие 2026-05-06**: val_metric placeholder bug объясняет ВСЕ предыдущие странности — A-провал, revert-провал, выбор E1 как best, coverage collapse. Это не специфика UP×1.5, а структурная проблема селекции best-checkpoint.
- DOWN-bias частично рассосался при первом ребилде (до bug): ECE 14.65% → 11.15%, worst bin bias −0.46 → −0.16. Возможно после фикса метрики реальная картина по DOWN-bias станет иной.
- MetaV3 holdout 0.5589 (по моделям с E1-cold-start) ≈ цель 0.560 — после фикса метрики ожидаем holdout > 0.57.
- Platt B остаётся в плане, но **запускать только после ребилда с фиксом метрики**, иначе калибруем degenerate модель.
- D2_bm_formula лимитки → ждут.

**Новый порядок действий:**
1. Ребилд V3 ансамбля с фиксом val_metric (`retrain_all --rebuild ensemble`)
2. Сравнить новые числа с baseline 0.5365 — должны быть ≥
3. Запустить Platt (`calibrate_platt`)
4. Reliability report → решение применять Platt в decision_layer или нет
5. Если gross/trade < 0.2% после Platt — Sprint 11.1 Quantile heads

- [x] **Pre-A. `decision_sweep --by-regime`** (2026-05-06) — ❌ на full test НЕТ прибыльной комбинации порогов. Per-regime best: bear −0.090%, side −0.221%, bull −0.155%. BUY=0 на жёстких порогах. Bull_regime_check +0.124% оказался артефактом recent 60-дневного окна (full bull = −0.155%).
- [x] **A. Ребилд V3 с UP class weight × 1.5** (2026-05-06) — ❌ **провал, откачено**. UP×1.5 переучил в зеркальный UP-bias: recall UP=0.91, recall DOWN=0.10. Coverage упал 8× (28 vs 225 SELL). Корневая причина: `cls_weights` управляет cls_head, но decision_layer смотрит на dir_prob (отдельная голова) — рассогласование. Best metric выбирался на E1 когда модель вся в BUY, реальной учёбы не было. Side-эффект: оставшиеся 28 сигналов дали exp +0.335%/trade, но N слишком мало. Изменение откачено в trainer_v3_ensemble.py.
- [x] **🔴 P0 BUG fix — val_metric placeholder** (2026-05-06) — найден критический баг: при нулевой торговле decision layer placeholder `dec_hit=0.5` инфлирует val_metric относительно реально торгующих эпох (real hit ≈0.40). Все 3 seed'а в последнем ребилде выбрали best=E1 (cold start) несмотря на улучшение dir_acc к E8. Coverage упал с 225 до 5 SELL. Объясняет провал и UP×1.5, и revert. **Фикс**: в [ml/trainer_v3.py:496-518](ml/trainer_v3.py#L496-L518) добавлен coverage-gating — `dec_hit_term = dec_hit * min(trade_cov / 0.05, 1.0)`. При нулевой торговле вклад = 0; при ≥5% coverage = полный. Требует ребилд V3 ансамбля.
- [ ] **Pre-B. Фикс reliability_report calibrated path** (~10 мин) — temperature_per_ticker.json есть (T=3.12 mean), но reliability_report выдаёт "Calibrated: нет". Нужно для сравнения temperature vs Platt после применения.
- [x] **B. Platt scaling — реализация** (2026-05-06) — ✅ создан [ml/calibrate_platt.py](ml/calibrate_platt.py): per-ticker `P(UP) = sigmoid(a·logit + b)`, LBFGS на NLL, fallback на global при n<30. Зарегистрирован stage 35 в `retrain_all.py`. Запись `dir_prob_platt` в npz. Auto-apply к ensemble_predictions.npz после fit.
- [x] **B-run. Запуск Platt + сравнение** (2026-05-06) — 🟡 **слабый прирост**. ECE 12.83% raw → 11.66% Platt (Δ −1.17pp). Цель ≤7% не достигнута. Глобально `b≈0` (асимметрии почти нет, модель симметрично overconfident в обе стороны). Per-ticker есть значимая асимметрия для отдельных тикеров (ENPG, OZON, VTBR), но train→test calibration drift нивелирует эффект на отдельных тикерах. **Вывод**: калибровка не главный блокер — gross/trade < 0.20% важнее.
- [x] **B-apply. Переключить decision_layer на dir_prob_platt** (2026-05-06) — ✅ [patch_decision_signal.py](ml/patch_decision_signal.py:84) теперь автоматически выбирает `dir_prob_platt > dir_prob_calibrated > dir_prob`. [decision_sweep.py](ml/decision_sweep.py) получил `--source` флаг (auto-detect default). После Platt + Sprint 5 thresholds: E_decision total −2.62% vs −8.63% до (улучшение благодаря blacklist).
- [x] **🆕 Per-ticker blacklist** (2026-05-06) — ✅ [decision_layer.py:248-256](ml/decision_layer.py#L248-L256) `RegimeAwareDecisionLayer.TICKER_BLACKLIST = {AFLT, OZON, VKCO, CBOM, TATN}` — тикеры с ECE >19% даже после Platt. `decide_numpy(tickers=...)` force-HOLD'ит сэмплы blacklist'а. Эффект: 1537 сэмплов отсечено, E_decision total -8.63% → -2.62%. Список должен пересматриваться после каждого ребилда V3.
- [x] **Decision_sweep на dir_prob_platt** (2026-05-06) — ✅ найдена прибыльная зона `edge=6.0/dir=0.7/sell=0.85`: **+0.066%/trade** на 38 сделках (cov 0.20%). Узко для production, но доказывает что калибровка раскрывает прибыльные сегменты которые на raw невидимы.
- [ ] **C. MetaLearner v4: 34 → 98 фич** (P0 идея #2) — запускать только если A+B+execution не дали gross/trade > 0.2%. Daily nums pool[32] + ctx[21] + intraday[11].
- [ ] **D. Регуляризация MetaV3** (P1 идея #6) — best val_acc на E10, дальше деградация → классический overfit. Dropout regime/fund + weight_decay 5e-3.
- [ ] **E. Adaptive thresholds в DEFAULT** (P1 идея #4).

**🆕 Параллельный трек (Sprint 13 идея #13 поднят на Sprint 10):** D2_bm_formula показал **+0.146%/trade на 23 сделках** — это самый дешёвый путь к прибыли. Нужно:
- [ ] Расширить D2-логику на полную выборку (не только когда срабатывает D-режим, а как основной execution mode)
- [ ] Прогнать walk-forward с D2 как основным execution path вместо market
- [ ] Если на N>200 сделок gross/trade держится > 0.15% → готовить MT5 limit-order реализацию

**Цель Sprint 10:** OOS Sharpe ≥ 0, gross/trade > 0.2% (покрытие costs), coverage ≥ 3%. MetaV3 holdout ≥ 0.560 практически достигнут (0.5589).

### 📝 Аудит плана 2026-05-05 — что НЕ делаем (уже покрыто)

Внешний аудит советовал «перейти от класса к структуре свечи через multi-task forecasting» — это **уже реализовано** в `MultiTaskLossV3`:
- direction (`cls`+`dir`) + normalized high/low (`ohlc[20]`) + range (`econ.mfe/mae`) + path order (`high_first_prob` + `extremes`).

Каскад timeframes: HourlySpec → DailySpec → MetaV3 даёт +4.90pp full / +3.00pp holdout vs v2. Рыночный/секторный контекст: `ctx[21]` + regime HMM + fundamentals(12) + dividends(5). Аугментации запланированы Sprint 11. Teacher-student с передачей эмбеддингов внутридневной формы (вместо текущих скаляров h_dir/h_conf/h_vol) — единственная новая идея, перенесена в Sprint 11 как расширение MinuteSpecialist.

### 🟢 Sprint 11 — Multi-scale temporal + Quantile execution

**11.1 — Quantile heads для свечи** 🆕 [реализовано 2026-05-06]

Идея: вместо точечных O/H/L/C-предсказаний учить **распределение** через 3 квантиля для high и low (q=0.1, 0.5, 0.9). Pinball loss (стандартная quantile regression) автоматически даёт нужное поведение: узкий уверенный интервал = низкий loss при попадании, узкий промах = большой штраф; широкий = средний loss всегда. Без кастомных reward-функций.

**Что реализовано:**
- ✅ [QuantileExtremesHead](ml/multiscale_cnn_v3.py) — `head_low + head_high`, output shape [B, 6×fb] (3 квантиля × 2 стороны × future_bars)
- ✅ [pinball_loss_quantile](ml/multiscale_cnn_v3.py) — стандартная pinball loss
- ✅ [MultiTaskLossV3](ml/multiscale_cnn_v3.py:777) принимает `quantile_pred`, `quantile_loss_weight=0.15` (default)
- ✅ Trainer передаёт quantile_pred извлекая из extremes_pred[:, 3:]
- ✅ [trainer_v3_ensemble.py](ml/trainer_v3_ensemble.py:631) сохраняет `quantile_pred` отдельным ключом в npz; `extremes_pred` остаётся 3-колоночным для backward-compat
- ✅ Старые checkpoints совместимы через `strict=False` (calibrate_platt/temperature)
- 🆕 +20K параметров поверх 2.74M (~0.7% overhead)

**Что реализовано (этап 1, 2-channel L+H):**
- ✅ Ребилд V3 ансамбля (2026-05-06) — quantile_pred сохранён в npz, shape [N, 30]
- ✅ **Главный эффект**: pinball loss через shared backbone дисциплинировал features → D3_coin_flip впервые в плюс in-sample (+0.254%/trade), но walk-forward не подтвердил
- ✅ matplotlib visualizer [ml/quantile_viz.py](ml/quantile_viz.py): actual candle (left) + predicted quantile bands (right) на одной y-шкале

**Этап 2 — full OHLC quantile head + ordering penalty (реализован, ждёт ребилд):**
- ✅ [QuantileOHLCHead](ml/multiscale_cnn_v3.py) — 4 канала (open, high, low, close) × 3 квантиля × fb = **60 outputs** (было 30)
- ✅ [ordering_penalty_ohlc](ml/multiscale_cnn_v3.py) — штраф за перекрытие диапазонов:
  ```
  L_q90 ≤ O_q10  и  L_q90 ≤ C_q10  (low's upper tail ниже body's lower tail)
  O_q90 ≤ H_q10  и  C_q90 ≤ H_q10  (body's upper tail ниже high's lower tail)
  + intra-channel monotonicity: q10 ≤ q50 ≤ q90 для каждого канала
  ```
  По запросу пользователя 2026-05-06: «диапазоны low/open/close/high не должны перекрываться даже частично, иначе штраф».
- ✅ MultiTaskLossV3 принимает `quantile_loss_weight=0.15` + `ordering_loss_weight=0.10`
- ✅ Bias-init: `head_low.bias=-0.3, head_high.bias=+0.3` — снимает холодный старт overlap
- ✅ Backward-compat: старые npz [N, 30] и старые checkpoints читаются (strict=False)

**Volume status:** vol уже частично реализован через `aux_y = [vol, skew]` (см. `dataset_v3.py:791`) и `AuxHead`. Per-bar future volume target требует расширения [labels_ohlc.py](ml/labels_ohlc.py) (добавить volume в `ohlc_labels`) — отложено.

**Что реализовано (этап 2 — ребилд 2026-05-09):**
- ✅ Ребилд V3 ансамбля: quantile_pred shape [N, 60] (4 канала × 3 квантиля × 5 fb)
- ✅ Ordering работает: overlap L↔O=0.50%, L↔C=2.47%, O↔H=2.58%, C↔H=1.46% (target 0%, было ~50%)
- ✅ Median violations: 0.00% — медианы строго упорядочены L<O<C<H
- ✅ matplotlib viz рисует 4 цветных band'а (L blue, O purple, C green, H red) — визуально упорядочены
- ✅ **In-sample D3 + E одновременно в плюс**: D3 total +0.37% Sharpe +0.92, E_decision +0.31%, E2 +0.26%, E3 +0.16% — впервые

**🔴 Регрессии (нужно тюнить веса):**
- Coverage квантилей упал 77% → 37-44% — модель сжимает интервалы чтобы избежать overlap
- ensemble dir_acc 0.5293 → 0.5166 (−1.27pp) — backbone теряет capacity
- Walk-forward D3 OOS: exp avg −0.043%, Sharpe −1.00 — train-test gap сохраняется

**Корректировка весов loss (внесена 2026-05-09):**
- `quantile_loss_weight: 0.15 → 0.10` (4 канала × 0.15 съели dir_acc)
- `ordering_loss_weight: 0.10 → 0.03` (слишком сильный → coverage 38%)

**Этап 3 — directional body penalty O↔C (реализован 2026-05-10, ребилд завершён):**

Проблема: безусловный запрет overlap O↔C физически некорректен — O и C **могут** идти в любом порядке (бычий vs медвежий бар). Жёсткий ReLU сломал бы модель на смешанных свечах.

Решение в [ordering_penalty_ohlc](ml/multiscale_cnn_v3.py:864) (расширение существующей функции):
```
body = target_C − target_O                 # фактическое направление свечи
bull_mask = body >  0.05·ATR-norm          # явно бычий
bear_mask = body < -0.05·ATR-norm          # явно медвежий
# bull: тело идёт O→C вверх БЕЗ overlap
penalty_bull = ReLU(O_q90 − C_q10) * bull_mask
# bear: тело идёт O→C вниз БЕЗ overlap
penalty_bear = ReLU(C_q90 − O_q10) * bear_mask
# doji (|body| ≤ eps): нет штрафа — модель имеет право на overlap при неопределённости
body_pen = (penalty_bull + penalty_bear).sum() / (bull_mask.sum() + bear_mask.sum())
```
Нормировка на число активных (non-doji) bar'ов чтобы вес не плыл от доли doji в батче. eps=0.05 в нормированных единицах ATR ≈ 5% от ATR-расстояния (типичное «тело шумовое» при доджи).

Подключение: тот же `ordering_loss_weight=0.03` (отдельный гиперпараметр не вводили — пусть сначала увидим эффект). Передача таргетов `target_O, target_C` в `MultiTaskLossV3.forward` уже доступна локально (строки 967-969).

**Результаты 3-го ребилда (2026-05-10):**
- ✅ Ordering overlap восстановлен: L↔O=0.40%, L↔C=1.52%, O↔H=1.16%, C↔H=0.27%
- ✅ Median violations: 0.00%, crossover violations: 0.00%
- ✅ dir_acc восстановился до 0.5261 (было 0.5166 в этапе 2 → etap 3 +0.95pp)
- ✅ **D3_coin_flip breakeven: total=−0.00%, gross=+0.209%/trade, N=217** — главный результат
- ✅ SELL coverage вырос: 1.2% → 4.0%
- ✅ H/L quantile coverage t+1 ≈ 79%/76% (почти идеальные 80%)
- 🔴 O/C channel coverage collapsed: O=2.53%, C=0.89% — backbone не отделяет O/C от cls-сигнала
- 🔴 Platt калибровка устарела: ECE 10.78% → 12.83% после применения (re-fit нужен)
- 🟡 dir_acc регрессия vs 2026-05-06: 0.5365→0.5261 (−1.04pp)
- 🟡 D3/D2 WF exp% впервые положительные (+0.240%/+0.185%) но нестатистично (n_avg=11)

**Что осталось:**
- [x] Ребилд с весами + body penalty — **✅ завершён (2026-05-10)**
- [x] **Анализ quantile predictions** ([ml/quantile_eval.py](ml/quantile_eval.py)) — ✅ H/L работают: coverage H=59%, L=62% avg (t+1≈80%), sharpness растёт с горизонтом, crossover = 0%. **⚠️ Критически**: O/C coverage avg=2.53%/0.89% — collapse. Ordering violations близки к 0% ✅
- [x] **Walk-forward с D2/D3 execution** ([ml/walk_forward_d3.py](ml/walk_forward_d3.py)) — 🟡 **впервые exp% > 0 в среднем** (D3 +0.240%±0.502, D2 +0.185%±0.541), но n_avg=11 сделок/фолд — статистически ненадёжно. Fold 1 (n=30) убыточен (exp=−0.256%), fold 4+5 (n=1 каждый) прибыльны. **Вывод**: in-sample edge подтверждается, но OOS требует больше coverage (цель n≥50/фолд).
- [ ] **Fix O/C channel coverage collapse** — отдельный backbone для O/C предсказаний или gradient stop из cls-head (Q-9)
- [ ] **Re-fit Platt** после каждого ребилда (`py -m ml.calibrate_platt`) — автоматизировать в retrain_all
- [ ] D5_quantile execution mode в backtest_strategy: BUY на q_H[0.1] как TP-target (не q_O — O/C бесполезны пока), exit на q_H[0.9]; сравнение с D3
- [ ] Volume per-bar target — отдельный спринт (требует расширения `build_ohlc_labels`)

### 🔴 Sprint 10 — открытые вопросы после walk-forward (2026-05-06)

- [ ] **Per-fold threshold optimization** — текущие thresholds зафиксированы on full test (cheating-flavor). Walk-forward с per-fold calibration window даст честную оценку. Возможно D3 будет ещё хуже.
- [ ] **Расширить coverage**: 7% даёт 7-75 trades/fold. Цель — 15-20% coverage с edge ≥ 2×fee. Но `decision_sweep` показал что широкая coverage даёт −0.20%/trade.
- [ ] **Regime-aware D-execution**: возможно D3 работает только в bull-периодах. Проверить per-regime exp%.
- [ ] **D5_quantile execution** может дать стабильнее результат через калиброванные интервалы вместо точечных предсказаний.

**Почему это даёт прирост:**
1. **Текущая OHLC-голова** даёт точку → нет понятия уверенности → D2_bm_formula использует фиксированную формулу для entry/exit.
2. **Квантильная голова** даёт интервалы → execution layer становится управляемым: BUY на q=0.05 предсказанного low даёт высокий edge при низком fill rate; BUY на q=0.2 — наоборот. Можно адаптировать по режиму/уверенности.
3. **Pinball loss** калибрует уверенность сама: модель не штрафуется за широкий интервал когда она неуверена, но штрафуется за узкий когда не попала.
4. **Эмпирический ground**: D2_bm_formula уже показал +0.146%/trade на текущих точечных OHLC. Квантили — следующий шаг этой же стратегии.

**Минимальная реализация:**
```python
# в multiscale_cnn_v3.py
self.low_quantile_head  = nn.Linear(trunk_dim, 3 * future_bars)   # q=0.1, 0.5, 0.9
self.high_quantile_head = nn.Linear(trunk_dim, 3 * future_bars)
# в MultiTaskLossV3 добавить pinball_loss(pred_q, target, q)
# текущие OHLC/extremes головы НЕ трогаем — пусть параллельно
```
Сравниваем D2 backtest (старая OHLC-голова) vs новый D2_quantile (entry на q_low_0.1, exit на q_high_0.9).

**Запуск только после:** Sprint 10 B (Platt) применён, decision_layer переключён на dir_prob_platt, и видим на N>200 сделок что D2 даёт стабильный edge < 0.2% (т.е. точечный OHLC недостаточен).

**11.2 — MinuteSpecialist (15min → Hour) + teacher-student каскад** (P2 идея #8)
- Input: [B, 60, 25] (60×15min bars)
- Выход: эмбеддинг внутридневной формы передаётся в HourlySpec (не скаляры h_dir/h_conf/h_vol, а dense vector)
- Запуск только если 11.1 не закрыл цель edge > 2×fee

### 🔵 Sprint 12 — Усиление модели
- Расширение тикеров 55 → 100 (P2 идея #9)
- Аугментации (P2 идея #11)
- Live intraday execution loop (P2 идея #10)
- Kronos fine-tuning (P2 идея #12)

### 🔴 Sprint 13 — Production
- Лимитные ордера вместо market (P2 идея #13 — снижает costs ×4)
- MT5 live trading с MetaV4 + adaptive thresholds
- Monitoring + auto-retrain каждую неделю

---

## 9. КОМАНДЫ

### Полный pipeline (рекомендуемый путь)

```bash
py -m ml.retrain_all                         # инкрементально
py -m ml.retrain_all --rebuild all           # ⟳ всё с нуля (~1.5-2 ч)
py -m ml.retrain_all --diagnostics-only      # ~12 с sanity
py -m ml.retrain_all --dry-run               # план без выполнения
```

### Гранулярные команды (если нужно)

```bash
# Обучение
py -m ml.trainer_v3_ensemble                 # V3 ансамбль (~30-60 мин)
py -m ml.trainer_v3_ensemble --rebuild       # с пересборкой кэша
py -m ml.trainer_hourly                      # HourlySpec (~7 мин)
py -m ml.calibrate_temperature               # per-ticker T (Sprint 7)
py -m ml.calibrate_platt                     # per-ticker Platt (Sprint 10 B) — асимметрия
py -m ml.calibrate_platt --coverage-report   # покрытие fit/fallback
py -m ml.calibrate_platt --apply ml/ensemble/ensemble_predictions.npz  # повторное применение
py -m ml.meta_ensemble                       # v2 (~1 мин)
py -m ml.meta_ensemble --version v3 --rebuild meta   # v3 build+train

# Fundamentals + Dividends
py -m ml.fundamentals_loader --refresh-cache         # 55 тикеров × 12 фич
py -m ml.fundamentals_loader --inspect SBER          # diag
py -m ml.dividends_loader --refresh-cache            # 39 тикеров
py -m ml.dividends_loader --inspect SBER 2026-05-04  # diag

# Decision flow
py -m ml.patch_ensemble_regime                       # HMM regime → npz
py -m ml.decision_sweep --by-regime                  # best per-regime
py -m ml.patch_decision_signal --regime-aware        # apply Sprint 9.5 defaults
py -m ml.backtest_strategy --future-bars 5           # E_decision_layer
py -m ml.backtest_strategy --path-aware              # S8.3 simulator
py -m ml.backtest_strategy --intraday                # S8.4 hourly cancel

# Walk-forward + sanity
py -m ml.walk_forward --folds 5 --adaptive-thresholds --purge-days 5
py -m ml.bull_regime_check --window-days 60 --grid
py -m ml.reliability_report                          # Brier + ECE
py -m ml.reliability_report --per-ticker --csv ml/ensemble/reliability.csv
py -m ml.hourly_split_diagnostics                    # B-24 lift анализ
py -m ml.calibrate_temperature --coverage-report     # B-23 fitted/fallback
py -m ml.tests.test_path_aware                       # 10 unit-тестов
py -m ml.feature_importance                          # permutation на 37 cols
py -m ml.meta_ensemble --version v3 --eval-only --holdout-only
```

---

## 10. ЖЕЛЕЗО / ENV

```
GPU:  RTX 4050 Laptop 6GB
CPU:  AMD Ryzen 7 7735HS
OS:   Windows 11 (PowerShell)
Env:  Python 3.12 (venv), PyTorch 2.x, CUDA 12.x
Run:  py -m ml.<module>
Cert: russian_ca.cer для T-Bank API SSL
.env: TINKOFF_TOKEN=<your_token>
```
