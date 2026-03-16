"""Запуск обучения — MLP, MultiScale CNN, Ensemble."""
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from ml.config  import CFG, SCALES
from ml.dataset import build_full_dataset, class_distribution



def run_mlp_training():
    print("=" * 50 + "\nMLP Baseline\n" + "=" * 50)
    X_flat, X_img, y = build_full_dataset()
    print(f"\nВсего сэмплов: {len(y)}")
    class_distribution(y)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_flat, y, test_size=0.3, random_state=CFG.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=CFG.seed, stratify=y_tmp)
    print(f"  Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")

    from ml.mlp_model import train_mlp, evaluate_mlp
    model = train_mlp(X_tr, y_tr, X_val, y_val)
    print("\n" + "=" * 50 + "\nОценка MLP\n" + "=" * 50)
    evaluate_mlp(model, X_test, y_test)

    np.save("ml/X_flat_train.npy", X_tr)
    np.save("ml/X_flat_val.npy",   X_val)
    np.save("ml/X_flat_test.npy",  X_test)
    np.save("ml/y_train.npy", y_tr)
    np.save("ml/y_val.npy",   y_val)
    np.save("ml/y_test.npy",  y_test)
    print("  Данные сохранены.")


def run_multiscale_training():
    print("=" * 50 + "\nMultiScale CNN + Context + TL\n" + "=" * 50)

    from ml.dataset_v2     import build_full_multiscale_dataset
    from ml.multiscale_cnn import train_multiscale, finetune_multiscale, evaluate_multiscale

    imgs_by_scale, y, ctx, ctx_dim = build_full_multiscale_dataset()
    print(f"\nВсего сэмплов: {len(y)}")
    print(f"Контекст: {ctx.shape if ctx is not None else 'нет'}")
    class_distribution(y)

    idx = np.arange(len(y))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y, test_size=0.3, random_state=CFG.seed, stratify=y)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, random_state=CFG.seed, stratify=y_tmp)

    tr_s  = {W: imgs_by_scale[W][idx_tr]   for W in SCALES}
    val_s = {W: imgs_by_scale[W][idx_val]  for W in SCALES}
    te_s  = {W: imgs_by_scale[W][idx_test] for W in SCALES}
    tr_ctx  = ctx[idx_tr]   if ctx is not None else None
    val_ctx = ctx[idx_val]  if ctx is not None else None
    te_ctx  = ctx[idx_test] if ctx is not None else None

    print(f"  Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # Сохранить для ансамбля
    np.save("ml/ctx_train.npy", tr_ctx  if tr_ctx  is not None else np.array([]))
    np.save("ml/ctx_val.npy",   val_ctx if val_ctx is not None else np.array([]))
    np.save("ml/ctx_test.npy",  te_ctx  if te_ctx  is not None else np.array([]))
    for W in SCALES:
        np.save(f"ml/imgs_{W}_train.npy", tr_s[W])
        np.save(f"ml/imgs_{W}_test.npy",  te_s[W])
    np.save("ml/y_ms_train.npy", y_tr)
    np.save("ml/y_ms_test.npy",  y_test)

    model = train_multiscale(tr_s, y_tr, val_s, y_val,
                             tr_ctx, val_ctx, ctx_dim or 0)
    model = finetune_multiscale(model, tr_s, y_tr, val_s, y_val,
                                tr_ctx, val_ctx)

    print("\n" + "=" * 50 + "\nОценка MultiScale CNN\n" + "=" * 50)
    evaluate_multiscale(model, te_s, y_test, te_ctx)


def run_ensemble():
    print("=" * 50 + "\nАнсамбль MLP + CNN\n" + "=" * 50)
    from ml.mlp_model      import MLP
    from ml.multiscale_cnn import MultiScaleCNN
    from ml.ensemble       import evaluate_ensemble
    from ml.dataset import INDICATOR_COLS
    import torch

    def get_device() -> torch.device:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            props = torch.cuda.get_device_properties(0)
            print(f"  Устройство: {props.name} "
                f"({props.total_memory // 1024**2} MB VRAM)")
        else:
            dev = torch.device("cpu")
            print("  Устройство: cpu")
        return dev

    device = get_device()

    mlp = MLP(input_dim=len(INDICATOR_COLS) * CFG.window)
    mlp.load_state_dict(torch.load("ml/model_mlp.pt", map_location=device))
    mlp.to(device)

    # Загрузить CNN
    ctx_train = np.load("ml/ctx_train.npy", allow_pickle=True)
    ctx_dim   = ctx_train.shape[1] if ctx_train.ndim == 2 else 0
    cnn = MultiScaleCNN(ctx_dim=ctx_dim)
    cnn.load_state_dict(torch.load("ml/model_multiscale.pt", map_location=device))
    cnn.to(device)

    # Загрузить тест данные
    X_flat_test = np.load("ml/X_flat_test.npy")
    y_test      = np.load("ml/y_ms_test.npy")
    te_s        = {W: np.load(f"ml/imgs_{W}_test.npy") for W in SCALES}
    te_ctx      = np.load("ml/ctx_test.npy", allow_pickle=True)
    te_ctx      = te_ctx if te_ctx.ndim == 2 else None

    print("\nОценка ансамбля (MLP 35% + CNN 65%):")
    evaluate_ensemble(mlp, cnn, X_flat_test, te_s, y_test, te_ctx)


if __name__ == "__main__":
    if "--ensemble" in sys.argv:
        run_ensemble()
    elif "--multiscale" in sys.argv:
        run_multiscale_training()
    else:
        run_mlp_training()
