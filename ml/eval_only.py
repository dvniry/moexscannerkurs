# ml/eval_only.py
"""eval_only.py v3.4

Новое:
- TTA (Test Time Augmentation): 5 проходов с малым гауссовым шумом на nums,
  усреднение вероятностей → более стабильные предсказания.
- Temperature Scaling: post-hoc калибровка на val-сете (LBFGS),
  применяется к logits перед argmax.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ──────────────────────────────────────────────────────────────────
# Temperature Scaling
# ──────────────────────────────────────────────────────────────────

class TemperatureScaler:
    """Post-hoc калибровка: logits / T, T подбирается на val-сете через LBFGS."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, model, val_loader, device, ctx_dim, use_hourly, scales):
        import torch, torch.nn.functional as F
        from ml.trainer_v3 import _forward_unpack

        model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                imgs_dict, num_dict, cls_y, _, ctx, hourly_data, *_ = batch
                imgs   = {W: imgs_dict[W].to(device) for W in scales}
                ctx_t  = ctx.to(device) if ctx_dim > 0 else None
                ht     = hourly_data.to(device) if use_hourly else None
                nums   = {W: num_dict[W].to(device) for W in scales} \
                         if num_dict is not None else None
                lo, _, _, _, _, _ = _forward_unpack(model, imgs, nums, ctx_t, ht)
                all_logits.append(lo.cpu())
                all_labels.append(cls_y)

        logits_cat = torch.cat(all_logits).float()
        labels_cat = torch.cat(all_labels)

        T = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits_cat / T.clamp(min=0.1), labels_cat)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(T.clamp(min=0.1).item())
        print(f"  Temperature Scaling: T = {self.temperature:.4f}")
        return self

    def __call__(self, logits):
        import torch
        return logits / self.temperature


# ──────────────────────────────────────────────────────────────────
# TTA evaluate
# ──────────────────────────────────────────────────────────────────

def evaluate_with_tta(model, te_ds, y_test, ctx_dim,
                      use_hourly=True, save_json=None,
                      scaler: TemperatureScaler = None,
                      tta_passes: int = 5,
                      tta_noise_std: float = 0.015):
    """evaluate_multiscale_v3 + TTA + опциональный Temperature Scaling.

    TTA: tta_passes проходов с гауссовым шумом на nums (σ=tta_noise_std),
    вероятности усредняются.
    """
    import torch, json, numpy as np
    from sklearn.metrics import classification_report, f1_score
    from ml.multiscale_cnn_v3 import _make_loader_v3
    from ml.config import SCALES
    from ml.trainer_v3 import _forward_unpack

    device = next(model.parameters()).device
    loader = _make_loader_v3(te_ds, batch_size=256, shuffle=False, num_workers=0)
    model.eval()

    all_preds, all_trues = [], []
    all_ohlc_pred, all_ohlc_true = [], []

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, hourly_data, *_ = batch
            imgs   = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t  = ctx.to(device) if ctx_dim > 0 else None
            ht     = hourly_data.to(device) if use_hourly else None
            nums   = {W: num_dict[W].to(device) for W in SCALES} \
                     if num_dict is not None else None

            # ── TTA loop ──────────────────────────────────────────
            prob_sum = None
            for t in range(tta_passes):
                if t == 0:
                    n = nums
                else:
                    # добавляем малый гауссовый шум к числовым признакам
                    n = {W: nums[W] + torch.randn_like(nums[W]) * tta_noise_std
                         for W in SCALES} if nums is not None else None

                lo, op, _, _, _, _ = _forward_unpack(model, imgs, n, ctx_t, ht)

                # применяем Temperature Scaling если есть
                if scaler is not None:
                    lo = torch.tensor(lo.cpu().numpy() / scaler.temperature,
                                      device=device)

                probs = torch.softmax(lo, dim=-1)
                prob_sum = probs if prob_sum is None else prob_sum + probs

            avg_probs = prob_sum / tta_passes
            all_preds.extend(avg_probs.argmax(1).cpu().numpy())
            all_trues.extend(cls_y.numpy())
            all_ohlc_pred.append(op.cpu().float().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    preds  = np.array(all_preds)
    trues  = np.array(all_trues)
    ohlc_p = np.concatenate(all_ohlc_pred, 0)
    ohlc_t = np.concatenate(all_ohlc_true, 0)

    print(f"\n  [TTA x{tta_passes}, noise_std={tta_noise_std}]"
          + (f"  [T={scaler.temperature:.3f}]" if scaler else ""))
    print(classification_report(trues, preds,
                                 target_names=['UP', 'FLAT', 'DOWN'],
                                 digits=4, zero_division=0))

    n_bars = ohlc_p.shape[1] // 4
    print(f'\n  OHLC MAE по барам ({n_bars} bars):')
    print(f'  {"Bar":>4} {"ΔOpen":>8} {"ΔHigh":>8} {"ΔLow":>8} {"ΔClose":>8}')
    for bar in range(n_bars):
        s = bar * 4
        p_b = ohlc_p[:, s:s + 4]; t_b = ohlc_t[:, s:s + 4]
        if p_b.shape[1] < 4: break
        mae_b = np.abs(p_b - t_b).mean(0)
        print(f'  {bar + 1:>4} {mae_b[0]:>8.4f} {mae_b[1]:>8.4f} '
              f'{mae_b[2]:>8.4f} {mae_b[3]:>8.4f}')

    dir_acc = 0.5
    if ohlc_p.shape[1] >= 4:
        dir_acc = float((np.sign(ohlc_p[:, 3]) == np.sign(ohlc_t[:, 3])).mean())
        print(f'\n  Direction accuracy (ΔClose bar1): {dir_acc:.4f}')
        O_p, H_p, L_p, C_p = ohlc_p[:, 0], ohlc_p[:, 1], ohlc_p[:, 2], ohlc_p[:, 3]
        h_viol = (H_p < np.maximum(O_p, C_p)).mean()
        l_viol = (L_p > np.minimum(O_p, C_p)).mean()
        print(f'  H violation: {h_viol:.2%}  L violation: {l_viol:.2%}')

    if save_json:
        result = {
            'accuracy':  float((preds == trues).mean()),
            'macro_f1':  float(f1_score(trues, preds, average='macro', zero_division=0)),
            'dir_acc':   dir_acc,
            'temperature': scaler.temperature if scaler else 1.0,
            'tta_passes':  tta_passes,
            'ohlc_mae':  np.abs(ohlc_p[:, :4] - ohlc_t[:, :4]).mean(0).tolist()
                         if ohlc_p.shape[1] >= 4 else [],
        }
        with open(save_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  Saved → {save_json}')

    return preds, trues


# ──────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    from ml.config import CFG
    from ml.dataset_v3 import (
        build_full_multiscale_dataset_v3, temporal_split, INDICATOR_COLS,
    )
    from ml.multiscale_cnn_v3 import (
        MultiScaleHybridV3, _make_loader_v3,
    )
    from ml.visualize_predictions import predict_and_plot
    from torch.utils.data import Subset

    MODEL_PATH = 'ml/model_multiscale_v3.pt'

    # ── Данные ────────────────────────────────────────────────────
    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=False, use_hourly=True
    )
    idx_train, idx_val, idx_test = temporal_split(ticker_lengths)
    y_test = y_all[idx_test]
    te_ds  = Subset(dataset, idx_test.tolist())
    val_ds = Subset(dataset, idx_val.tolist())

    # ── Модель ────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiScaleHybridV3(
        ctx_dim=ctx_dim,
        n_indicator_cols=len(INDICATOR_COLS),
        future_bars=CFG.future_bars,
        use_hourly=True,
        in_channels=4,           # [3.4] Heikin-Ashi канал
    ).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    # ── Temperature Scaling на val-сете ───────────────────────────
    val_loader = _make_loader_v3(val_ds, batch_size=256, shuffle=False, num_workers=0)
    scaler = TemperatureScaler().fit(
        model, val_loader, device, ctx_dim,
        use_hourly=True, scales=CFG.SCALES if hasattr(CFG, 'SCALES') else [5,10,20,30]
    )

    # ── Evaluate с TTA ────────────────────────────────────────────
    evaluate_with_tta(
        model, te_ds, y_test, ctx_dim,
        use_hourly=True,
        save_json=MODEL_PATH.replace('.pt', '_eval.json'),
        scaler=scaler,
        tta_passes=5,
        tta_noise_std=0.015,
    )

    predict_and_plot(MODEL_PATH, te_ds, y_test, ctx_dim,
                     use_hourly=True, n_examples=8)