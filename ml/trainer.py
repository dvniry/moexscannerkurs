"""Запуск обучения моделей."""
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from ml.config  import CFG, SCALES
from ml.dataset import build_full_dataset, class_distribution


# ── MLP ───────────────────────────────────────────────────

def run_mlp_training():
    print("=" * 50)
    print("Шаг 1 — Сборка датасета (MLP)")
    print("=" * 50)
    X_flat, X_img, y = build_full_dataset()

    print(f"\nВсего сэмплов: {len(y)}")
    class_distribution(y)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_flat, y, test_size=0.3, random_state=CFG.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=CFG.seed, stratify=y_tmp)
    print(f"  Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")

    from ml.mlp_model import train_mlp, evaluate_mlp
    print("\n" + "=" * 50 + "\nОбучение MLP\n" + "=" * 50)
    model = train_mlp(X_tr, y_tr, X_val, y_val)

    print("\n" + "=" * 50 + "\nОценка MLP на тесте\n" + "=" * 50)
    evaluate_mlp(model, X_test, y_test)

    # Сохранить индексные сплиты для совместимости
    np.save("ml/X_img_train.npy", X_img[:len(y_tr)])
    np.save("ml/X_img_val.npy",   X_img[len(y_tr):len(y_tr)+len(y_val)])
    np.save("ml/X_img_test.npy",  X_img[len(y_tr)+len(y_val):])
    np.save("ml/y_train.npy", y_tr)
    np.save("ml/y_val.npy",   y_val)
    np.save("ml/y_test.npy",  y_test)
    print("\n  Данные сохранены для CNN.")


# ── MultiScale CNN ────────────────────────────────────────

def run_multiscale_training():
    print("=" * 50)
    print("MultiScale CNN + LSTM + Transfer Learning")
    print("=" * 50)

    from ml.dataset_v2      import build_full_multiscale_dataset
    from ml.multiscale_cnn  import (train_multiscale, finetune_multiscale,
                                     evaluate_multiscale)

    print("\nШаг 1 — Сборка мультимасштабного датасета...")
    imgs_by_scale, y = build_full_multiscale_dataset()
    print(f"\nВсего сэмплов: {len(y)}")
    class_distribution(y)

    idx = np.arange(len(y))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y, test_size=0.3, random_state=CFG.seed, stratify=y)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, random_state=CFG.seed, stratify=y_tmp)

    tr_s   = {W: imgs_by_scale[W][idx_tr]   for W in SCALES}
    val_s  = {W: imgs_by_scale[W][idx_val]  for W in SCALES}
    test_s = {W: imgs_by_scale[W][idx_test] for W in SCALES}
    print(f"  Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")

    print("\nШаг 2 — Pretrain на всех тикерах...")
    model = train_multiscale(tr_s, y_tr, val_s, y_val)

    print("\nШаг 3 — Fine-tune (TL-1)...")
    model = finetune_multiscale(model, tr_s, y_tr, val_s, y_val)

    print("\n" + "=" * 50 + "\nОценка MultiScale CNN на тесте\n" + "=" * 50)
    evaluate_multiscale(model, test_s, y_test)


# ── Точка входа ───────────────────────────────────────────

if __name__ == "__main__":
    if "--multiscale" in sys.argv:
        run_multiscale_training()
    else:
        run_mlp_training()
