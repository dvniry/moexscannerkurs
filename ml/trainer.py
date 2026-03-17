"""Точка входа: python -m ml.trainer [--mlp|--multiscale|--deep|--ensemble]"""
import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'
import sys
import numpy as np
from sklearn.model_selection import train_test_split
_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.config import CFG
from ml.dataset import class_distribution


def run_mlp_training():
    print("=" * 50 + "\nMLP обучение\n" + "=" * 50)
    from ml.dataset   import build_full_dataset
    from ml.mlp_model import train_mlp, evaluate_mlp

    X_flat, X_img, y = build_full_dataset()
    print(f"\nВсего сэмплов: {len(y)}")
    class_distribution(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_flat, y, test_size=0.2, random_state=CFG.seed, stratify=y)
    model = train_mlp(X_tr, y_tr, X_te, y_te)
    evaluate_mlp(model, X_te, y_te)


def run_multiscale_training(mode: str = "standard"):
    """
    mode: 'standard' (--multiscale) | 'deep' (--deep)
    """
    label = {"standard": "Standard CNN", "deep": "Deep Hybrid (TCN+BiLSTM+Attention)"}
    print("=" * 50 + f"\nMultiScale [{label[mode]}]\n" + "=" * 50)

    from ml.dataset_v2     import build_full_multiscale_dataset
    from ml.multiscale_cnn import (train_multiscale, train_multiscale_deep,
                                   evaluate_multiscale)
    from torch.utils.data  import Subset

    dataset, y_all, ctx_dim = build_full_multiscale_dataset()

    print(f"\nВсего сэмплов: {len(y_all)}")
    print(f"Контекст: dim={ctx_dim}")
    class_distribution(y_all)

    idx = np.arange(len(y_all))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y_all, test_size=0.3, random_state=CFG.seed, stratify=y_all)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, random_state=CFG.seed, stratify=y_tmp)

    tr_ds  = Subset(dataset, idx_tr)
    val_ds = Subset(dataset, idx_val)
    te_ds  = Subset(dataset, idx_test)
    print(f"  Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")

    if mode == "deep":
        save_path = 'ml/model_multiscale_deep.pt'
        model     = train_multiscale_deep(tr_ds, y_tr, val_ds, y_val,
                                          ctx_dim, save_path=save_path)
        hybrid    = True
    else:
        save_path = 'ml/model_multiscale.pt'
        model     = train_multiscale(tr_ds, y_tr, val_ds, y_val,
                                     ctx_dim, save_path=save_path)
        hybrid    = False

    print("\n" + "=" * 50 + "\nОценка\n" + "=" * 50)
    evaluate_multiscale(model, te_ds, y_test, ctx_dim,
                        save_json=save_path.replace('.pt', '_eval.json'),
                        hybrid=hybrid)


def run_ensemble():
    print("=" * 50 + "\nEnsemble\n" + "=" * 50)
    from ml.ensemble import run_ensemble as _run
    _run()


if __name__ == "__main__":
    if "--ensemble" in sys.argv:
        run_ensemble()
    elif "--deep" in sys.argv:
        run_multiscale_training(mode="deep")
    elif "--multiscale" in sys.argv:
        run_multiscale_training(mode="standard")
    else:
        run_mlp_training()
