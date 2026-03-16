"""Ансамбль MLP + MultiScale CNN."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from ml.mlp_model      import MLP
from ml.multiscale_cnn import MultiScaleCNN, to_tensors
from ml.config         import SCALES


def predict_ensemble(
    mlp:        MLP,
    cnn:        MultiScaleCNN,
    x_flat:     np.ndarray,
    scale_imgs: dict,
    context:    np.ndarray = None,
    weights:    tuple = (0.35, 0.65),   # MLP, CNN
) -> dict:
    """
    Взвешенный ансамбль предсказаний MLP и MultiScale CNN.
    weights = (w_mlp, w_cnn), сумма должна быть 1.0
    """
    device = next(mlp.parameters()).device
    mlp.eval()
    cnn.eval()

    # MLP вероятности
    x = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        p_mlp = F.softmax(mlp(x), dim=1).cpu().numpy()[0]

    # CNN вероятности
    tensors = {
        W: torch.tensor(img, dtype=torch.float32)
               .unsqueeze(0).unsqueeze(0).to(device)
        for W, img in scale_imgs.items()
    }
    ctx_t = torch.tensor(context, dtype=torch.float32) \
                 .unsqueeze(0).to(device) if context is not None else None
    with torch.no_grad():
        p_cnn = torch.softmax(cnn(tensors, ctx_t), dim=1).cpu().numpy()[0]

    # Взвешенное среднее
    p = weights[0] * p_mlp + weights[1] * p_cnn
    labels = ["BUY", "HOLD", "SELL"]
    return {
        "signal":       labels[p.argmax()],
        "confidence":   float(p.max()),
        "probs":        {l: float(p[i]) for i, l in enumerate(labels)},
        "mlp_signal":   labels[p_mlp.argmax()],
        "cnn_signal":   labels[p_cnn.argmax()],
    }


def evaluate_ensemble(
    mlp, cnn, X_flat_test, test_scales, y_test,
    test_ctx=None, weights=(0.35, 0.65),
):
    device   = next(mlp.parameters()).device
    mlp.eval()
    cnn.eval()

    te_t     = to_tensors(test_scales, device)
    te_ctx_t = torch.tensor(test_ctx, dtype=torch.float32).to(device) \
               if test_ctx is not None else None
    X_te     = torch.tensor(X_flat_test, dtype=torch.float32).to(device)

    preds = []
    BS    = 64
    with torch.no_grad():
        for s in range(0, len(y_test), BS):
            e      = min(s + BS, len(y_test))
            p_mlp  = F.softmax(mlp(X_te[s:e]), dim=1)
            vb     = {W: te_t[W][s:e] for W in SCALES}
            vc     = te_ctx_t[s:e] if te_ctx_t is not None else None
            p_cnn  = torch.softmax(cnn(vb, vc), dim=1)
            p      = weights[0] * p_mlp + weights[1] * p_cnn
            preds.extend(p.argmax(1).cpu().numpy())

    print(classification_report(y_test, preds,
          target_names=["BUY", "HOLD", "SELL"], digits=4))
    return preds
