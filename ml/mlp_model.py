"""Персептрон (MLP) — baseline классификатор."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report
from ml.config import CFG


class MLP(nn.Module):
    def __init__(self, input_dim: int = 225, hidden: list = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or CFG.mlp_hidden
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts  = np.bincount(y)
    weights = 1.0 / counts[y]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights), replacement=True,
    )


def train_mlp(X_train, y_train, X_val, y_val,
              save_path="ml/model_mlp.pt") -> MLP:
    torch.manual_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Устройство: {device}")

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val,   dtype=torch.long).to(device)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=CFG.batch_size, sampler=make_sampler(y_train))
    val_dl   = DataLoader(TensorDataset(X_vl, y_vl), batch_size=CFG.batch_size)

    model     = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.mlp_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs_pre)
    criterion = nn.CrossEntropyLoss()
    best_val  = 0.0

    for epoch in range(1, CFG.epochs_pre + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                correct += (model(xb).argmax(1) == yb).sum().item()

        val_acc = correct / len(y_val)
        scheduler.step()
        print(f"  Epoch {epoch:3d}/{CFG.epochs_pre} | "
              f"loss={train_loss/len(train_dl):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Сохранено (val_acc={best_val:.4f})")

    print(f"\n  Лучший val_acc: {best_val:.4f}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def evaluate_mlp(model, X_test, y_test):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)
                      .to(device)).argmax(1).cpu().numpy()
    print(classification_report(y_test, preds,
          target_names=["BUY", "HOLD", "SELL"], digits=4))
    return preds


def predict_mlp(model, x_flat: np.ndarray) -> dict:
    device = next(model.parameters()).device
    model.eval()
    x = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    labels = ["BUY", "HOLD", "SELL"]
    return {"signal": labels[probs.argmax()], "confidence": float(probs.max())}
