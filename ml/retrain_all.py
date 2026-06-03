"""Orchestrator: запускает полный pipeline переобучения одной командой.

Граф зависимостей (см. plan.md):
    1.  trainer_v3_ensemble      (~30-60 мин)
    2.  patch_ensemble_regime    (~10 с)
    3.  calibrate_temperature    (~3-5 мин)
    35. calibrate_platt          (~3-5 мин)  ← Sprint 10 B
    4.  trainer_hourly           (~7 мин)
    5. meta_ensemble v2          (~1 мин)
    6. fundamentals_loader       (~30 с)
    7. dividends_loader          (~30 с)
    8. meta_ensemble v3          (~2 мин)
    9. sanity diagnostics        (~5 мин total)

Каждый этап:
    • Проверяет artefact freshness — пропускает если up-to-date (idempotent)
    • Печатает START/END с timestamp + elapsed
    • Сохраняет статус в `ml/ensemble/retrain_status.json` (для resume)
    • Если падает — НЕ запускает зависимые этапы, печатает чёткий exit code

Запуск:
    py -m ml.retrain_all                       # авто-инкрементальный (skip up-to-date)
    py -m ml.retrain_all --rebuild all         # ⟳ FORCE всё с нуля (~1.5-2 ч)
    py -m ml.retrain_all --rebuild ensemble    # ⟳ только V3 ансамбль + regime
    py -m ml.retrain_all --rebuild hourly      # ⟳ только HourlySpec
    py -m ml.retrain_all --rebuild meta        # ⟳ только Meta v2 + v3 (~5 мин)
    py -m ml.retrain_all --rebuild fund        # ⟳ обновить fundamentals/dividends API
    py -m ml.retrain_all --resume-from 7       # продолжить с этапа N
    py -m ml.retrain_all --skip-stages 1,2,3   # пропустить указанные
    py -m ml.retrain_all --diagnostics-only    # только этапы 9.x
    py -m ml.retrain_all --no-diagnostics      # без sanity-блока
    py -m ml.retrain_all --dry-run             # показать план, ничего не делать

⟳ = FORCE rebuild — `--rebuild X` обходит is_fresh() check для соответствующих stages.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENSEMBLE_DIR  = os.path.join(os.path.dirname(__file__), "ensemble")
STATUS_PATH   = os.path.join(ENSEMBLE_DIR, "retrain_status.json")

# ── Артефакты для freshness-check ─────────────────────────────────────
A_V3_NPZ        = os.path.join(ENSEMBLE_DIR, "ensemble_predictions.npz")
A_V3_SEED42     = os.path.join(ENSEMBLE_DIR, "model_seed42.pt")
A_TEMP_JSON     = os.path.join(ENSEMBLE_DIR, "temperature_per_ticker.json")
A_PLATT_JSON    = os.path.join(ENSEMBLE_DIR, "platt_per_ticker.json")
A_HOURLY_PT     = os.path.join(ENSEMBLE_DIR, "hourly_specialist.pt")
A_HOURLY_ALL    = os.path.join(ENSEMBLE_DIR, "hourly_all_predictions.npz")
A_META_V2_NPZ   = os.path.join(ENSEMBLE_DIR, "meta_features.npz")
A_META_V2_PT    = os.path.join(ENSEMBLE_DIR, "meta_learner.pt")
A_FUND_MAP      = os.path.join(ENSEMBLE_DIR, "fundamentals_map.json")
A_DIV_MAP       = os.path.join(ENSEMBLE_DIR, "dividends_map.json")
A_META_V3_NPZ   = os.path.join(ENSEMBLE_DIR, "meta_features_v3.npz")
A_META_V3_PT    = os.path.join(ENSEMBLE_DIR, "meta_learner_v3.pt")

# ANSI colors
GREEN  = "\033[92m"; RED   = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; RESET = "\033[0m";  BOLD   = "\033[1m"
DIM    = "\033[2m"; MAG   = "\033[95m"


def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _newer(target: str, *deps: str) -> bool:
    """True если target существует и новее всех deps. Если dep отсутствует — игнорируется."""
    if not _exists(target):
        return False
    t = _mtime(target)
    for d in deps:
        if _exists(d) and _mtime(d) > t:
            return False
    return True


def _fmt_time(secs: float) -> str:
    if secs < 60:  return f"{secs:.1f}s"
    if secs < 3600: return f"{secs/60:.1f}m"
    return f"{secs/3600:.1f}h"


# ──────────────────────────────────────────────────────────────────────
# Stage definition
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Stage:
    n:        int
    name:     str
    desc:     str
    cmd:      list[str]
    artefacts: list[str]      # для freshness/exists check
    deps:     list[str] = field(default_factory=list)  # apaths которые должны быть
    eta:      str = ""        # ETA для отображения
    optional: bool = False    # если True — fail не блокирует pipeline
    force:    bool = False    # FORCE rebuild — обойти is_fresh() check

    def is_fresh(self) -> bool:
        """Все артефакты существуют и новее всех зависимостей.
        Stages без артефактов (диагностика) считаются ВСЕГДА требующими запуска.
        """
        if not self.artefacts:
            return False
        if not all(_exists(a) for a in self.artefacts):
            return False
        for a in self.artefacts:
            if not _newer(a, *self.deps):
                return False
        return True

    def has_deps(self) -> bool:
        return all(_exists(d) for d in self.deps)


def _build_stages(rebuild: str) -> list[Stage]:
    """Собирает список этапов с учётом --rebuild флагов.

    --rebuild семантика:
      ensemble (alias: v3) — force stages 1, 2 (V3 ансамбль + regime patch)
      hourly               — force stage 4 (HourlySpec)
      meta                 — force stages 5, 8 (Meta v2 + v3)
      fund                 — force stages 6, 7 (Fundamentals + Dividends API)
      all                  — force ВСЕ 8 стейджей (включая temperature)
    """
    # Конкретные --rebuild флаги (`v3` = legacy alias для `ensemble`)
    ensemble_rebuild = (rebuild in ("ensemble", "v3", "all"))
    meta_rebuild     = (rebuild in ("meta", "all"))
    hourly_reb       = (rebuild in ("hourly", "all"))
    fund_refresh     = (rebuild in ("fund", "all"))
    temp_force       = (rebuild == "all")   # T-калибровка перетягивается только при --rebuild all

    py = sys.executable

    return [
        Stage(
            n=1, name="V3 ensemble",
            desc="MultiScaleHybridV3 ансамбль из 3 seeds",
            cmd=[py, "-m", "ml.trainer_v3_ensemble"]
                + (["--rebuild"] if ensemble_rebuild else []),
            artefacts=[A_V3_NPZ, A_V3_SEED42],
            eta="~30-60 мин",
            force=ensemble_rebuild,
        ),
        Stage(
            n=2, name="Patch regime",
            desc="HMM regime tag → ensemble_predictions.npz",
            cmd=[py, "-m", "ml.patch_ensemble_regime"],
            artefacts=[A_V3_NPZ],   # тот же файл, но обогащён
            deps=[A_V3_NPZ],
            eta="~10 с",
            force=ensemble_rebuild,
        ),
        Stage(
            n=3, name="Temperature calibration",
            desc="Per-ticker T (Sprint 7) + dir_prob_calibrated в npz",
            cmd=[py, "-m", "ml.calibrate_temperature"],
            artefacts=[A_TEMP_JSON],
            deps=[A_V3_NPZ, A_V3_SEED42],
            eta="~3-5 мин",
            force=ensemble_rebuild or temp_force,
        ),
        Stage(
            n=35, name="Platt calibration",
            desc="Per-ticker Platt (Sprint 10 B) + dir_prob_platt в npz — лечит асимметричный DOWN-bias",
            cmd=[py, "-m", "ml.calibrate_platt"],
            artefacts=[A_PLATT_JSON],
            deps=[A_V3_NPZ, A_V3_SEED42],
            eta="~3-5 мин",
            force=ensemble_rebuild or temp_force,
        ),
        Stage(
            n=4, name="HourlySpecialist",
            desc="BiLSTM + multi-scale CNN hourly (Sprint 4)",
            cmd=[py, "-m", "ml.trainer_hourly"]
                + (["--rebuild"] if hourly_reb else []),
            artefacts=[A_HOURLY_PT, A_HOURLY_ALL],
            eta="~7 мин",
            force=hourly_reb,
        ),
        Stage(
            n=5, name="Meta v2 features+train",
            desc="Базовый MetaLearner v2 (нужен для v3 build)",
            cmd=[py, "-m", "ml.meta_ensemble"]
                + (["--rebuild", "meta"] if meta_rebuild else []),
            artefacts=[A_META_V2_NPZ, A_META_V2_PT],
            deps=[A_V3_NPZ, A_HOURLY_ALL],
            eta="~1-2 мин",
            force=meta_rebuild,
        ),
        Stage(
            n=6, name="Fundamentals API",
            desc="getAssetFundamentals для CFG.tickers (Sprint 9.2)",
            cmd=[py, "-m", "ml.fundamentals_loader"]
                + (["--refresh-cache"] if fund_refresh or not _exists(A_FUND_MAP) else
                   ["--inspect", "SBER"]),  # noop-инспект если кэш свежий
            artefacts=[A_FUND_MAP],
            eta="~30 с",
            optional=True,  # API может быть недоступен временно
            force=fund_refresh,
        ),
        Stage(
            n=7, name="Dividends API",
            desc="getDividends для CFG.tickers (Sprint 9.3)",
            cmd=[py, "-m", "ml.dividends_loader"]
                + (["--refresh-cache"] if fund_refresh or not _exists(A_DIV_MAP) else
                   ["--inspect", "SBER", "2026-01-01"]),
            artefacts=[A_DIV_MAP],
            eta="~30 с",
            optional=True,
            force=fund_refresh,
        ),
        Stage(
            n=8, name="MetaLearner v3",
            desc="34-features MetaV3 (Sprint 9.1) — главная цель",
            cmd=[py, "-m", "ml.meta_ensemble", "--version", "v3"]
                + (["--rebuild", "meta"] if meta_rebuild else []),
            artefacts=[A_META_V3_NPZ, A_META_V3_PT],
            deps=[A_META_V2_NPZ, A_FUND_MAP, A_DIV_MAP, A_V3_NPZ],
            eta="~2 мин",
            force=meta_rebuild,
        ),
    ]


def _build_diagnostics() -> list[Stage]:
    py = sys.executable
    # Diagnostic stages — всегда optional (не блокируют), не имеют артефактов
    return [
        Stage(n=9, name="Walk-forward (purged + adaptive)",
              desc="Sprint 6 + 9.4 + 9.5 — OOS sanity",
              cmd=[py, "-m", "ml.walk_forward", "--folds", "5",
                   "--adaptive-thresholds", "--purge-days", "5"],
              artefacts=[], optional=True, eta="~10 с"),
        Stage(n=10, name="Bull regime check",
              desc="Sprint 9.5 — устаревание bull=OFF default",
              cmd=[py, "-m", "ml.bull_regime_check", "--window-days", "60", "--grid"],
              artefacts=[], optional=True, eta="~5 с"),
        Stage(n=11, name="Reliability + Brier",
              desc="Sprint 9.5 — калибровка raw vs calibrated",
              cmd=[py, "-m", "ml.reliability_report"],
              artefacts=[], optional=True, eta="~2 с"),
        Stage(n=12, name="Hourly split diagnostics",
              desc="B-24 — lift over baseline вместо raw acc",
              cmd=[py, "-m", "ml.hourly_split_diagnostics"],
              artefacts=[], optional=True, eta="~2 с"),
        Stage(n=13, name="Path-aware unit tests",
              desc="Sprint 9.5 — 10 unit-тестов",
              cmd=[py, "-m", "ml.tests.test_path_aware"],
              artefacts=[], optional=True, eta="~2 с"),
        Stage(n=14, name="MetaV3 holdout eval",
              desc="Финальная honest OOS оценка v3",
              cmd=[py, "-m", "ml.meta_ensemble", "--version", "v3",
                   "--eval-only", "--holdout-only"],
              artefacts=[], optional=True, eta="~2 с"),
    ]


# ──────────────────────────────────────────────────────────────────────
# Status persistence
# ──────────────────────────────────────────────────────────────────────

def _load_status() -> dict:
    if not os.path.exists(STATUS_PATH):
        return {"runs": []}
    try:
        with open(STATUS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"runs": []}


def _save_status(status: dict) -> None:
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as fh:
            json.dump(status, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  {YELLOW}WARN: не удалось сохранить status: {e}{RESET}")


# ──────────────────────────────────────────────────────────────────────
# Stage runner
# ──────────────────────────────────────────────────────────────────────

def _print_header(s: Stage, total: int):
    ts = time.strftime("%H:%M:%S")
    print(f"\n{BOLD}{CYAN}━━━ Stage {s.n}/{total} · {s.name} ━━━{RESET}  "
          f"{DIM}[{ts}, ETA {s.eta}]{RESET}")
    print(f"  {DIM}{s.desc}{RESET}")
    print(f"  {DIM}cmd: {' '.join(os.path.basename(s.cmd[0]) if i==0 else c for i,c in enumerate(s.cmd))}{RESET}")


def _run_stage(s: Stage, *, dry_run: bool = False) -> tuple[str, float]:
    """Возвращает (status, elapsed_s). status ∈ {OK, SKIP, FAIL, MISSING_DEP}."""
    if not s.has_deps() and s.deps:
        missing = [d for d in s.deps if not _exists(d)]
        print(f"  {RED}MISSING DEP{RESET}: " + ", ".join(os.path.basename(m) for m in missing))
        return ("MISSING_DEP", 0.0)

    # Skip if fresh — НО force-rebuild обходит этот check
    if s.force:
        print(f"  {YELLOW}FORCE REBUILD{RESET}: --rebuild флаг пересиливает is_fresh()")
    elif s.is_fresh():
        ages = []
        for a in s.artefacts:
            age_s = time.time() - _mtime(a)
            ages.append(f"{os.path.basename(a)} ({_fmt_time(age_s)} ago)")
        print(f"  {GREEN}SKIP{RESET}: артефакты свежие — " + ", ".join(ages))
        return ("SKIP", 0.0)

    if dry_run:
        print(f"  {YELLOW}DRY-RUN{RESET}: пропуск выполнения")
        return ("SKIP", 0.0)

    t0 = time.time()
    try:
        # cwd=ROOT, чтобы относительные пути работали
        result = subprocess.run(
            s.cmd, cwd=ROOT, check=False,
            stdout=None, stderr=None,  # стримим напрямую
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            tag = f"{YELLOW}OPTIONAL FAIL{RESET}" if s.optional else f"{RED}FAIL{RESET}"
            print(f"  {tag} (exit={result.returncode}, {_fmt_time(elapsed)})")
            return ("FAIL", elapsed)
        # Verify artefacts existence post-run
        if s.artefacts and not all(_exists(a) for a in s.artefacts):
            missing = [a for a in s.artefacts if not _exists(a)]
            print(f"  {RED}MISSING ARTEFACT после успешного выполнения{RESET}: " +
                  ", ".join(os.path.basename(m) for m in missing))
            return ("FAIL", elapsed)
        print(f"  {GREEN}OK{RESET}  ({_fmt_time(elapsed)})")
        return ("OK", elapsed)
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n  {YELLOW}INTERRUPTED by user{RESET} ({_fmt_time(elapsed)})")
        raise
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {RED}EXCEPTION{RESET}: {type(e).__name__}: {e}")
        return ("FAIL", elapsed)


# ──────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(args) -> int:
    skip_stages = set(int(x) for x in args.skip_stages.split(",") if x.strip()) \
        if args.skip_stages else set()

    if args.diagnostics_only:
        train_stages = []
        diag_stages  = _build_diagnostics()
    else:
        train_stages = _build_stages(args.rebuild)
        diag_stages  = [] if args.no_diagnostics else _build_diagnostics()

    all_stages = train_stages + diag_stages
    total = len(all_stages)

    if args.resume_from:
        all_stages = [s for s in all_stages if s.n >= args.resume_from]

    print(f"\n{BOLD}{MAG}{'═' * 70}{RESET}")
    print(f"{BOLD}{MAG}  Retrain pipeline orchestrator{RESET}")
    print(f"{BOLD}{MAG}{'═' * 70}{RESET}")
    print(f"  Stages to run: {len(all_stages)} / {total} total")
    print(f"  Rebuild mode:  {args.rebuild or 'incremental (only stale artefacts)'}")
    if skip_stages:
        print(f"  Skip stages:   {sorted(skip_stages)}")
    if args.dry_run:
        print(f"  {YELLOW}DRY-RUN: показать план, ничего не запускать{RESET}")

    # Plan summary
    print(f"\n{BOLD}  План:{RESET}")
    print(f"    {DIM}legend: ✓=fresh(skip)  ·=will run  ⟳=FORCE rebuild  ✗=no deps  -=user-skipped{RESET}")
    for s in all_stages:
        marker = "·"
        if s.n in skip_stages:    marker = f"{DIM}-{RESET}"
        elif s.force:             marker = f"{YELLOW}⟳{RESET}"   # force overrides freshness
        elif s.is_fresh():        marker = f"{GREEN}✓{RESET}"
        elif not s.has_deps():    marker = f"{RED}✗{RESET}"
        print(f"    {marker}  [{s.n:>2}] {s.name:<35s} {DIM}({s.eta}){RESET}")

    if args.dry_run:
        return 0

    print()

    # Run
    summary = []
    overall_t0 = time.time()
    try:
        for s in all_stages:
            if s.n in skip_stages:
                print(f"\n{DIM}━━━ Stage {s.n} · {s.name} ━━━  SKIP (--skip-stages){RESET}")
                summary.append((s, "SKIP", 0.0))
                continue
            _print_header(s, total)
            status, elapsed = _run_stage(s, dry_run=args.dry_run)
            summary.append((s, status, elapsed))
            if status == "FAIL" and not s.optional:
                print(f"\n  {RED}{BOLD}Pipeline остановлен: stage {s.n} ({s.name}) failed{RESET}")
                print(f"  {DIM}Resume: py -m ml.retrain_all --resume-from {s.n}{RESET}")
                break
    except KeyboardInterrupt:
        print(f"\n  {YELLOW}{BOLD}Pipeline прерван пользователем{RESET}")

    overall_elapsed = time.time() - overall_t0

    # Final summary
    print(f"\n{BOLD}{MAG}{'═' * 70}{RESET}")
    print(f"{BOLD}{MAG}  Pipeline summary{RESET}  {DIM}(total: {_fmt_time(overall_elapsed)}){RESET}")
    print(f"{BOLD}{MAG}{'═' * 70}{RESET}")
    n_ok = n_skip = n_fail = n_miss = 0
    for s, status, t in summary:
        if status == "OK":           color, n = GREEN, "OK"; n_ok += 1
        elif status == "SKIP":       color, n = DIM, "SKIP"; n_skip += 1
        elif status == "FAIL":       color, n = (YELLOW if s.optional else RED), \
                                              ("OPTIONAL FAIL" if s.optional else "FAIL")
        elif status == "MISSING_DEP": color, n = RED, "NO DEPS"; n_miss += 1
        else:                         color, n = RED, status
        if status == "FAIL" and not s.optional: n_fail += 1
        t_s = _fmt_time(t) if t > 0 else "—"
        print(f"  [{s.n:>2}] {s.name:<35s} {color}{n:<14s}{RESET} {DIM}{t_s:>8}{RESET}")

    # Persist run log
    status = _load_status()
    status["runs"].append({
        "ts":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "rebuild": args.rebuild or "incremental",
        "elapsed_s": round(overall_elapsed, 1),
        "stages":  [{"n": s.n, "name": s.name, "status": st, "elapsed_s": round(t, 1)}
                    for s, st, t in summary],
    })
    status["runs"] = status["runs"][-20:]  # keep last 20
    _save_status(status)

    print(f"\n  Status log: {STATUS_PATH}")
    if n_fail > 0:
        print(f"  {RED}{BOLD}Pipeline failed: {n_fail} stage(s) crashed{RESET}")
        return 1
    print(f"  {GREEN}{BOLD}OK: {n_ok} ran, {n_skip} skipped{RESET}")
    return 0


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Полный pipeline переобучения (Sprint 9.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  py -m ml.retrain_all                     # инкрементально (skip up-to-date)
  py -m ml.retrain_all --rebuild all       # ВСЁ с нуля (~1.5-2 часа)
  py -m ml.retrain_all --rebuild meta      # только meta v2 + v3 (~5 мин если ансамбль свежий)
  py -m ml.retrain_all --rebuild fund      # только обновить fundamentals/dividends API
  py -m ml.retrain_all --diagnostics-only  # только sanity (~30 с)
  py -m ml.retrain_all --resume-from 5     # продолжить с этапа 5
  py -m ml.retrain_all --dry-run           # показать план, ничего не делать
        """,
    )
    p.add_argument("--rebuild",
                   choices=["", "fund", "hourly", "meta", "ensemble", "v3", "all"],
                   default="",
                   help="FORCE rebuild конкретных этапов (обходит is_fresh check):\n"
                        "  ensemble (alias v3) — V3 ансамбль + regime patch (stages 1,2)\n"
                        "  hourly              — HourlySpec (stage 4)\n"
                        "  meta                — MetaLearner v2 + v3 (stages 5, 8)\n"
                        "  fund                — Fundamentals + Dividends API (stages 6, 7)\n"
                        "  all                 — ВСЁ с нуля, включая temperature (~1.5-2 ч)\n"
                        "  (пусто)             — инкрементально (skip up-to-date)")
    p.add_argument("--resume-from", type=int, default=0,
                   help="Начать с указанного stage N (1-14 для diagnostics)")
    p.add_argument("--skip-stages", default="",
                   help="Пропустить указанные stages, через запятую (напр. '1,3')")
    p.add_argument("--diagnostics-only", action="store_true",
                   help="Только sanity-этапы (walk_forward, reliability, и т.д.)")
    p.add_argument("--no-diagnostics", action="store_true",
                   help="Без sanity-блока в конце")
    p.add_argument("--dry-run", action="store_true",
                   help="Показать план, ничего не запускать")
    args = p.parse_args()

    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
