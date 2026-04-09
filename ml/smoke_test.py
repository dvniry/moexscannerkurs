# ml/smoke_test.py
"""Smoke-test v1.2 — быстрая проверка ОДНОГО тикера (без загрузки всех).

Изменения v1.2:
- _sanitize_grads: зануляем Inf/NaN в градиентах (и модели, и criterion) перед clip
- AMP-путь: scheduler.step() только если scaler не пропустил шаг
- CPU-путь: после sanitize шаг делаем всегда без проверок isfinite
- total_steps исправлен (было n_steps+1, стало n_steps)

Запуск:
    python -m ml.smoke_test --ticker SBER
    python -m ml.smoke_test --ticker GAZP --steps 20 --no-hourly
"""
import os, sys, argparse, time, importlib
os.environ['GRPC_DNS_RESOLVER'] = 'native'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from collections import Counter
from sklearn.metrics import f1_score

from ml.config import CFG, SCALES


# ── Цвета для терминала ────────────────────────────────────────────
GREEN = '\033[92m'; RED = '\033[91m'; YELLOW = '\033[93m'
CYAN  = '\033[96m'; RESET = '\033[0m'; BOLD = '\033[1m'

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗ FAIL: {msg}{RESET}")
def info(msg):  print(f"  {CYAN}·{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def section(s): print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}\n{BOLD}  {s}{RESET}")


# ══════════════════════════════════════════════════════════════════
# FIX v1.2: sanitize градиентов — зануляем Inf/NaN перед clip
# ══════════════════════════════════════════════════════════════════

def _sanitize_grads(params) -> int:
    """Заменяет Inf/NaN градиенты на 0.0. Возвращает число затронутых параметров."""
    n_bad = 0
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            n_bad += 1
    return n_bad


# ══════════════════════════════════════════════════════════════════
# Загрузка данных одного тикера (3 стратегии)
# ══════════════════════════════════════════════════════════════════

def _load_one_ticker(ticker: str, use_hourly: bool):
    import ml.dataset_v3 as ds_mod
    build_fn = ds_mod.build_full_multiscale_dataset_v3

    try:
        result = build_fn(force_rebuild=False, use_hourly=use_hourly, ticker_filter=[ticker])
        ok(f"Загружен через ticker_filter=['{ticker}']")
        return result
    except TypeError as e:
        if 'ticker_filter' not in str(e):
            raise

    CANDIDATE_ATTRS = [
        'TICKERS', 'tickers', 'ALL_TICKERS', 'SYMBOLS', 'symbols',
        'TICKER_LIST', 'ticker_list', '_TICKERS', '_tickers',
    ]
    patched_attr = None; original_val = None
    for attr in CANDIDATE_ATTRS:
        val = getattr(ds_mod, attr, None)
        if isinstance(val, (list, tuple)) and len(val) > 1:
            val_upper = [str(t).upper() for t in val]
            if ticker.upper() in val_upper:
                patched_attr = attr; original_val = val; break

    if patched_attr is not None:
        orig_list  = list(original_val)
        ticker_obj = next(t for t in orig_list if str(t).upper() == ticker.upper())
        warn(f"ticker_filter не поддерживается → monkey-patch {patched_attr}=['{ticker}']")
        setattr(ds_mod, patched_attr, [ticker_obj])
        try:
            result = build_fn(force_rebuild=False, use_hourly=use_hourly)
        finally:
            setattr(ds_mod, patched_attr, original_val)
        ok(f"Загружен через monkey-patch {patched_attr}")
        return result

    print(f"\n{RED}{'═'*55}{RESET}")
    print(f"{RED}  Не удалось загрузить только один тикер.{RESET}")
    print(f"""
  Добавь параметр ticker_filter в dataset_v3.py:

  {CYAN}# найди функцию build_full_multiscale_dataset_v3 и добавь аргумент:{RESET}
  def build_full_multiscale_dataset_v3(
      force_rebuild=False,
      use_hourly=True,
      {YELLOW}ticker_filter=None,{RESET}
  ):
      tickers_to_use = (
          TICKERS if ticker_filter is None
          else [t for t in TICKERS if t in ticker_filter]
      )

  {CYAN}Или скинь dataset_v3.py — добавим за минуту.{RESET}
""")
    print(f"{RED}{'═'*55}{RESET}\n")
    sys.exit(2)


# ══════════════════════════════════════════════════════════════════
# Утилиты проверки тензоров
# ══════════════════════════════════════════════════════════════════

def check_tensor(t: torch.Tensor, name: str) -> bool:
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    rng = f"[{t.float().min():.3f}, {t.float().max():.3f}]"
    if has_nan or has_inf:
        fail(f"{name}: shape={tuple(t.shape)} nan={has_nan} inf={has_inf} range={rng}")
        return False
    ok(f"{name}: shape={tuple(t.shape)} range={rng}")
    return True

def check_ohlc(ohlc_y: torch.Tensor) -> bool:
    mn = ohlc_y.float().min().item(); mx = ohlc_y.float().max().item()
    abs_max = max(abs(mn), abs(mx))
    has_nan = torch.isnan(ohlc_y).any().item()
    has_inf = torch.isinf(ohlc_y).any().item()
    if has_nan or has_inf:
        fail(f"ohlc_y: nan={has_nan} inf={has_inf}"); return False
    note = "OK" if abs_max <= 10.0 else "⚠ большой диапазон — ATR-нормировка?"
    (ok if abs_max <= 10.0 else warn)(f"ohlc_y: range=[{mn:.3f}, {mx:.3f}]  {note}")
    if abs_max > 50.0:
        fail(f"ohlc_y abs_max={abs_max:.1f} — ATR-нормировка нарушена!"); return False
    return True


# ══════════════════════════════════════════════════════════════════
# Основная функция
# ══════════════════════════════════════════════════════════════════

def run_smoke_test(
    ticker:     str,
    n_steps:    int  = 10,
    use_hourly: bool = True,
    device_str: str  = 'auto',
) -> bool:
    t0 = time.time(); errors = []

    print(f"\n{BOLD}{'═'*55}")
    print(f"  SMOKE TEST  —  тикер: {ticker}")
    print(f"{'═'*55}{RESET}")

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))
    info(f"Device: {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        info(f"GPU: {p.name}  {getattr(p,'total_memory',0)/1024**2:.0f} MB VRAM")

    # ══ [DATA] Загрузка ═══════════════════════════════════════════
    section("[DATA] Загрузка датасета (1 тикер)")
    try:
        dataset, y_all, ctx_dim, ticker_lengths = _load_one_ticker(ticker, use_hourly)
        n_samples = len(y_all)
        ok(f"Сэмплов: {n_samples}  ctx_dim={ctx_dim}")
    except SystemExit:
        raise
    except Exception as e:
        fail(f"Ошибка загрузки: {e}"); return _finish([str(e)], t0)

    if n_samples < 10:
        fail(f"Слишком мало сэмплов: {n_samples}")
        return _finish([f"too few samples: {n_samples}"], t0)

    from ml.dataset_v3 import class_distribution
    counts = Counter(y_all.tolist())
    for i, name in enumerate(['UP', 'FLAT', 'DOWN']):
        pct = counts.get(i, 0) / max(n_samples, 1) * 100
        info(f"  {name}: {counts.get(i,0)} ({pct:.1f}%)")

    from ml.dataset_v3 import temporal_split
    from ml.multiscale_cnn_v3 import _make_loader_v3
    from torch.utils.data import Subset

    idx_tr, idx_val, _ = temporal_split(
        ticker_lengths, val_ratio=0.15, test_ratio=0.10, purge_bars=CFG.future_bars)
    ok(f"Split → train={len(idx_tr)}  val={len(idx_val)}")

    BS = min(CFG.batch_size, 32)
    tr_loader  = _make_loader_v3(Subset(dataset, idx_tr.tolist()),  BS, shuffle=True)
    val_loader = _make_loader_v3(Subset(dataset, idx_val.tolist()), BS, shuffle=False)

    # ══ [DATA] Первый батч ════════════════════════════════════════
    section("[DATA] Проверка первого батча")
    batch = next(iter(tr_loader))
    imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
    hourly_data = hourly_opt[0] if hourly_opt else None

    batch_ok = True
    for W, t in imgs_dict.items():
        if not check_tensor(t, f"imgs[{W}]"):
            batch_ok = False; errors.append(f"imgs[{W}] NaN/Inf")
    if num_dict:
        for W, t in num_dict.items():
            if not check_tensor(t, f"nums[{W}]"):
                batch_ok = False; errors.append(f"nums[{W}] NaN/Inf")
    if not check_tensor(ctx, "ctx"):
        batch_ok = False; errors.append("ctx NaN/Inf")
    if hourly_data is not None:
        if not check_tensor(hourly_data, "hourly"):
            batch_ok = False; errors.append("hourly NaN/Inf")
    else:
        info("hourly: None")
    if not check_ohlc(ohlc_y):
        batch_ok = False; errors.append("ohlc_y out of range")
    ok(f"cls_y unique: {cls_y.unique().tolist()}")
    if not batch_ok:
        return _finish(errors, t0)

    # ══ [MODEL] Forward pass ═════════════════════════════════════
    section("[MODEL] Forward pass")
    from ml.multiscale_cnn_v3 import MultiScaleHybridV3, MultiTaskLossV3

    model = MultiScaleHybridV3(
        ctx_dim=ctx_dim, n_indicator_cols=30,
        future_bars=CFG.future_bars, use_hourly=use_hourly,
    ).to(device)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'cls_head' in name and 'bias' in name:
                p.zero_(); info(f"cls_head bias обнулён: {name}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok(f"Модель: {n_params:,} параметров")

    imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
    ctx_t = ctx.to(device) if ctx_dim > 0 else None
    ht    = hourly_data.to(device) if (use_hourly and hourly_data is not None) else None
    nums  = ({W: num_dict[W].to(device) for W in SCALES} if num_dict else None)

    model_ok = True
    try:
        with torch.no_grad():
            logits, ohlc_out = model(imgs, nums, ctx_t, hourly=ht)
            logits_c = logits.clamp(-15.0, 15.0)
        if not check_tensor(logits_c, "logits"):
            model_ok = False; errors.append("logits NaN/Inf")
        if not check_tensor(ohlc_out, "ohlc_out"):
            model_ok = False; errors.append("ohlc_out NaN/Inf")
        nan_w = [n for n, p in model.named_parameters() if torch.isnan(p).any()]
        if nan_w:
            fail(f"NaN в весах: {nan_w[:5]}")
            model_ok = False; errors.append("NaN weights")
        else:
            ok("Веса: ОК (нет NaN)")
    except Exception as e:
        fail(f"Forward упал: {e}")
        errors.append(f"forward: {e}"); model_ok = False

    if not model_ok:
        return _finish(errors, t0)

    # ══ [LOSS] Первый батч ════════════════════════════════════════
    section("[LOSS] Первый батч")
    y_tr_arr = y_all[idx_tr]
    c = Counter(y_tr_arr.tolist()); total_tr = len(y_tr_arr)
    raw_w = [total_tr / (c.get(i, 1) * 3) for i in range(3)]
    max_w = max(raw_w)
    cls_weights = torch.tensor(
        [w / max_w * 2.0 for w in raw_w], dtype=torch.float32).to(device)
    info(f"cls_weights: UP={cls_weights[0]:.3f} FLAT={cls_weights[1]:.3f} DOWN={cls_weights[2]:.3f}")

    criterion = MultiTaskLossV3(
        cls_weight=cls_weights,
        gamma_per_class=(2.5, 0.3, 2.5),
        label_smoothing=0.02,
        huber_delta=0.5,
        direction_weight=0.3,
        reg_loss_weight=0.15,
    ).to(device)

    loss_ok = True
    try:
        with torch.no_grad():
            cy_d = cls_y.to(device)
            oy_d = ohlc_y.to(device).clamp(-5.0, 5.0)
            loss, lcls, lreg = criterion(
                logits_c.float(), cy_d, ohlc_out.float(), oy_d.float())
        if not torch.isfinite(loss):
            fail(f"loss={loss.item()} — не конечное!")
            errors.append(f"loss={loss.item()}"); loss_ok = False
        else:
            ok(f"loss={loss.item():.4f}  cls={lcls.item():.4f}  reg={lreg.item():.4f}")
            if loss.item() > 20.0:
                warn(f"loss={loss.item():.2f} очень большой")
    except Exception as e:
        fail(f"Criterion упал: {e}")
        errors.append(f"criterion: {e}"); loss_ok = False

    if not loss_ok:
        return _finish(errors, t0)

    # ══ [TRAIN] N шагов ══════════════════════════════════════════
    section(f"[TRAIN] {n_steps} шагов обучения")

    # FIX v1.2: разделяем параметры — sanitize нужен для ВСЕХ, clip только для модели
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    crit_params      = list(criterion.parameters())
    all_opt_params   = trainable_params + crit_params

    optimizer = AdamW(all_opt_params, lr=1e-4, weight_decay=1e-3)
    # FIX v1.2: total_steps=n_steps (было n_steps+1), pct_start=0.2
    scheduler = OneCycleLR(
        optimizer, max_lr=3e-4,
        total_steps=max(n_steps, 1),
        pct_start=0.2, anneal_strategy='cos')
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    train_ok = True; nan_steps = []; grad_norms = []; total_bad_grads = 0
    loader_it = iter(tr_loader)

    for step in range(1, n_steps + 1):
        try:
            try:
                b = next(loader_it)
            except StopIteration:
                loader_it = iter(tr_loader); b = next(loader_it)

            im_d, nu_d, cy, oy, cx, *ho = b
            ho_d = ho[0] if ho else None

            im   = {W: im_d[W].to(device) for W in SCALES}
            cy   = cy.to(device)
            oy   = oy.to(device).clamp(-5.0, 5.0)
            cx_t = cx.to(device) if ctx_dim > 0 else None
            ht_t = ho_d.to(device) if (use_hourly and ho_d is not None) else None
            nu   = ({W: nu_d[W].to(device) for W in SCALES} if nu_d else None)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    lo, op = model(im, nu, cx_t, hourly=ht_t)
                lo = lo.float().clamp(-15.0, 15.0); op = op.float()
                loss, lcls, lreg = criterion(lo, cy, op, oy.float())
                if not torch.isfinite(loss):
                    nan_steps.append(step)
                    warn(f"step {step:3d}: loss=nan — пропуск"); continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # FIX v1.2: sanitize ВСЕ параметры (модель + criterion)
                n_bad = _sanitize_grads(all_opt_params)
                total_bad_grads += n_bad
                gn = nn.utils.clip_grad_norm_(trainable_params, 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                new_scale = scaler.get_scale()
                # FIX v1.2: scheduler только если scaler не пропустил шаг
                if new_scale >= old_scale:
                    scheduler.step()
                else:
                    info(f"step {step:3d}: scaler drop {old_scale:.0f}→{new_scale:.0f}, LR не двигаем")
            else:
                lo, op = model(im, nu, cx_t, hourly=ht_t)
                lo = lo.clamp(-15.0, 15.0)
                loss, lcls, lreg = criterion(lo, cy, op, oy)
                if not torch.isfinite(loss):
                    nan_steps.append(step)
                    warn(f"step {step:3d}: loss=nan — пропуск"); continue
                loss.backward()
                # FIX v1.2: sanitize ВСЕ параметры (модель + criterion)
                n_bad = _sanitize_grads(all_opt_params)
                total_bad_grads += n_bad
                gn = nn.utils.clip_grad_norm_(trainable_params, 1.0)
                # После sanitize grad_norm всегда конечный → шаг делаем всегда
                optimizer.step()
                scheduler.step()

            grad_norms.append(float(gn))
            bad_str = f" sanitized={n_bad}" if n_bad else ""
            ok(f"step {step:3d}: loss={loss.item():.4f} "
               f"cls={lcls.item():.4f} reg={lreg.item():.4f} "
               f"grad={gn:.3f}{bad_str}")

        except Exception as e:
            fail(f"step {step}: {e}")
            errors.append(f"step {step}: {e}"); train_ok = False; break

    nan_rate = len(nan_steps) / max(n_steps, 1)
    if nan_steps:
        (fail if nan_rate > 0.3 else warn)(
            f"NaN на {len(nan_steps)}/{n_steps} шагах ({nan_rate*100:.0f}%)")
        if nan_rate > 0.3:
            errors.append(f"high nan rate {nan_rate:.2f}"); train_ok = False
    else:
        ok(f"Все {n_steps} шагов без NaN ✓")

    if total_bad_grads > 0:
        warn(f"Исправлено плохих градиентов за {n_steps} шагов: {total_bad_grads}")

    if grad_norms:
        avg_gn = np.mean(grad_norms)
        max_gn = np.max(grad_norms)
        (ok if max_gn < 5.0 else warn)(
            f"grad_norm: avg={avg_gn:.3f} max={max_gn:.3f} "
            f"{'OK' if max_gn < 5.0 else 'WARNING'}")

    # ══ [VAL] ════════════════════════════════════════════════════
    section("[VAL] Быстрая валидация")
    try:
        model.eval(); criterion.eval()
        vp_list = []; vt_list = []
        with torch.no_grad():
            for vb in val_loader:
                vi, vn, vc, _, vcx, *vho = vb
                vho_t = vho[0].to(device) if (vho and vho[0] is not None) else None
                vi    = {W: vi[W].to(device) for W in SCALES}
                vcx_t = vcx.to(device) if ctx_dim > 0 else None
                vnu   = ({W: vn[W].to(device) for W in SCALES} if vn else None)
                lo_v, _ = model(vi, vnu, vcx_t, hourly=vho_t)
                vp_list.extend(lo_v.argmax(1).cpu().numpy())
                vt_list.extend(vc.numpy())

        vp = np.array(vp_list); vt = np.array(vt_list)
        acc  = (vp == vt).mean()
        mf1  = f1_score(vt, vp, average='macro', zero_division=0)
        f1pc = f1_score(vt, vp, average=None, labels=[0, 1, 2], zero_division=0)
        ok(f"val_acc={acc:.4f}  macro_f1={mf1:.4f}  "
           f"UP={f1pc[0]:.3f} FLAT={f1pc[1]:.3f} DOWN={f1pc[2]:.3f}")
        n_zero = sum(1 for f in f1pc if f < 0.01)
        if n_zero >= 2:
            warn(f"Mode collapse: {n_zero}/3 классов с F1≈0")
        elif n_zero == 1:
            warn("1 класс с F1≈0 — норм при маленьком датасете")
        else:
            ok("Все три класса предсказываются ✓")
    except Exception as e:
        fail(f"Validation: {e}")
        errors.append(f"val: {e}")

    return _finish(errors, t0)


def _finish(errors, t0):
    elapsed = time.time() - t0
    print(f"\n{BOLD}{'═'*55}{RESET}")
    if not errors:
        print(f"{BOLD}{GREEN}  ✓ SMOKE TEST PASSED  ({elapsed:.1f}s){RESET}")
    else:
        print(f"{BOLD}{RED}  ✗ SMOKE TEST FAILED  ({elapsed:.1f}s){RESET}")
        for e in errors:
            print(f"    {RED}· {e}{RESET}")
    print(f"{'═'*55}\n")
    return not errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smoke-test одного тикера — быстрая проверка данных и обучения')
    parser.add_argument('--ticker',    required=True, help='Тикер, например SBER')
    parser.add_argument('--steps',     type=int, default=10)
    parser.add_argument('--no-hourly', action='store_true')
    parser.add_argument('--device',    default='auto', help='cpu / cuda / auto')
    args = parser.parse_args()

    success = run_smoke_test(
        ticker     = args.ticker.upper(),
        n_steps    = args.steps,
        use_hourly = not args.no_hourly,
        device_str = args.device,
    )
    sys.exit(0 if success else 1)