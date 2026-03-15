"""MultiScale CNN + LSTM память + Transfer Learning."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.metrics import classification_report
from ml.config   import CFG, SCALES
from ml.mlp_model import make_sampler


# ── Энкодер одного масштаба ───────────────────────────────

class ScaleEncoder(nn.Module):
    """
    use_pretrained=True  → ResNet18 (ImageNet TL) — для крупных масштабов
    use_pretrained=False → лёгкая кастомная CNN   — для мелких масштабов
    """
    def __init__(self, out_dim: int = 64, use_pretrained: bool = False):
        super().__init__()

        if use_pretrained:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            for param in base.parameters():
                param.requires_grad = False   # заморозить backbone
            # Адаптировать под grayscale (1 канал)
            base.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            base.fc = nn.Sequential(
                nn.Linear(512, out_dim), nn.ReLU(), nn.Dropout(0.3),
            )
            self.net = base
        else:
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
                nn.ReLU(), nn.MaxPool2d(2),

                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(), nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(), nn.AdaptiveAvgPool2d(4),

                nn.Flatten(),
                nn.Linear(64 * 4 * 4, out_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

    def forward(self, x):
        return self.net(x)

    def unfreeze(self):
        """Fine-tune: разморозить последний блок ResNet."""
        for name, param in self.net.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True


# ── Мультимасштабная модель ───────────────────────────────

class MultiScaleCNN(nn.Module):
    """
    Transfer Learning на трёх уровнях:
    - TL-0: ImageNet → ScaleEncoder(W=20, W=30)
    - TL-1: все тикеры → fine-tune на конкретном тикере
    - TL-2: совместное обучение масштабов через shared LSTM
    """
    def __init__(self):
        super().__init__()
        dim = CFG.scale_dim

        self.encoders = nn.ModuleDict({
            "5":  ScaleEncoder(dim, use_pretrained=False),  # микро паттерн
            "10": ScaleEncoder(dim, use_pretrained=False),  # малый паттерн
            "20": ScaleEncoder(dim, use_pretrained=True),   # средний + TL
            "30": ScaleEncoder(dim, use_pretrained=True),   # тренд + TL
        })

        # LSTM — память последовательности паттернов
        self.lstm = nn.LSTM(
            input_size  = len(SCALES) * dim,   # 4 * 64 = 256
            hidden_size = CFG.lstm_hidden,      # 128
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.3,
        )

        self.head = nn.Sequential(
            nn.Linear(CFG.lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 3),    # BUY / HOLD / SELL
        )

    def forward(self, scale_imgs: dict) -> torch.Tensor:
        feats = [self.encoders[str(W)](scale_imgs[W]) for W in SCALES]
        x     = torch.cat(feats, dim=1).unsqueeze(1)   # (B, 1, 256)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])

    def unfreeze_for_finetune(self):
        """TL-1: разморозить layer4 ResNet энкодеров для fine-tune."""
        for W in ["20", "30"]:
            self.encoders[str(W)].unfreeze()


# ── Подготовка тензоров ───────────────────────────────────

def to_tensors(imgs_by_scale: dict, device) -> dict:
    return {
        W: torch.tensor(imgs, dtype=torch.float32).unsqueeze(1).to(device)
        for W, imgs in imgs_by_scale.items()
    }


# ── Pretrain ──────────────────────────────────────────────

def train_multiscale(
    train_scales, y_train,
    val_scales,   y_val,
    save_path="ml/model_multiscale.pt",
) -> MultiScaleCNN:
    torch.manual_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Устройство: {device}")

    y_tr_t   = torch.tensor(y_train, dtype=torch.long)
    y_vl_t   = torch.tensor(y_val,   dtype=torch.long).to(device)
    idx_ds   = TensorDataset(torch.arange(len(y_train)), y_tr_t)
    sampler  = make_sampler(y_train)
    train_dl = DataLoader(idx_ds, batch_size=CFG.batch_size, sampler=sampler)

    tr_t = to_tensors(train_scales, torch.device("cpu"))
    vl_t = to_tensors(val_scales,   device)

    model     = MultiScaleCNN().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.cnn_lr, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=4, factor=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val, patience_cnt = 0.0, 0
    PATIENCE = 10

    print("\n  ── Фаза 1: Pretrain (backbone заморожен) ──")
    for epoch in range(1, CFG.epochs_pre + 1):
        model.train()
        train_loss = 0.0
        for idx_b, yb in train_dl:
            batch = {W: tr_t[W][idx_b].to(device) for W in SCALES}
            yb    = yb.to(device)
            optimizer.zero_grad()
            loss  = criterion(model(batch), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(y_val), 64):
                e  = min(s + 64, len(y_val))
                vb = {W: vl_t[W][s:e] for W in SCALES}
                correct += (model(vb).argmax(1) == y_vl_t[s:e]).sum().item()

        val_acc  = correct / len(y_val)
        avg_loss = train_loss / len(train_dl)
        scheduler.step(avg_loss)
        print(f"  Epoch {epoch:3d}/{CFG.epochs_pre} | "
              f"loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val, patience_cnt = val_acc, 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Сохранено (val_acc={best_val:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  ⏹ Ранняя остановка (epoch {epoch})")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ── Fine-tune (TL-1) ──────────────────────────────────────

def finetune_multiscale(
    model,
    train_scales, y_train,
    val_scales,   y_val,
    save_path="ml/model_multiscale.pt",
) -> MultiScaleCNN:
    """
    TL-1: дообучить на данных конкретного тикера.
    Разморозить layer4 ResNet энкодеров.
    """
    device = next(model.parameters()).device
    model.unfreeze_for_finetune()
    print(f"\n  ── Фаза 2: Fine-tune (layer4 разморожен, LR={CFG.cnn_finetune_lr}) ──")

    y_tr_t   = torch.tensor(y_train, dtype=torch.long)
    y_vl_t   = torch.tensor(y_val,   dtype=torch.long).to(device)
    idx_ds   = TensorDataset(torch.arange(len(y_train)), y_tr_t)
    sampler  = make_sampler(y_train)
    train_dl = DataLoader(idx_ds, batch_size=32, sampler=sampler)

    tr_t = to_tensors(train_scales, torch.device("cpu"))
    vl_t = to_tensors(val_scales,   device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.cnn_finetune_lr, weight_decay=1e-3,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val  = 0.0

    for epoch in range(1, CFG.epochs_fine + 1):
        model.train()
        train_loss = 0.0
        for idx_b, yb in train_dl:
            batch = {W: tr_t[W][idx_b].to(device) for W in SCALES}
            yb    = yb.to(device)
            optimizer.zero_grad()
            loss  = criterion(model(batch), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(y_val), 64):
                e  = min(s + 64, len(y_val))
                vb = {W: vl_t[W][s:e] for W in SCALES}
                correct += (model(vb).argmax(1) == y_vl_t[s:e]).sum().item()

        val_acc = correct / len(y_val)
        print(f"  Fine-tune {epoch:2d}/{CFG.epochs_fine} | "
              f"loss={train_loss/len(train_dl):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Сохранено (val_acc={best_val:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ── Оценка ────────────────────────────────────────────────

def evaluate_multiscale(model, test_scales, y_test):
    device = next(model.parameters()).device
    te_t   = to_tensors(test_scales, device)
    model.eval()
    preds  = []
    with torch.no_grad():
        for s in range(0, len(y_test), 64):
            e  = min(s + 64, len(y_test))
            vb = {W: te_t[W][s:e] for W in SCALES}
            preds.extend(model(vb).argmax(1).cpu().numpy())
    print(classification_report(y_test, preds,
          target_names=["BUY", "HOLD", "SELL"], digits=4))
    return preds


# ── Inference ─────────────────────────────────────────────

def predict_multiscale(model, scale_imgs: dict) -> dict:
    """scale_imgs: {5: (64,64), 10: (64,64), 20: (64,64), 30: (64,64)}"""
    device  = next(model.parameters()).device
    model.eval()
    tensors = {
        W: torch.tensor(img, dtype=torch.float32)
               .unsqueeze(0).unsqueeze(0).to(device)
        for W, img in scale_imgs.items()
    }
    with torch.no_grad():
        probs = torch.softmax(model(tensors), dim=1).cpu().numpy()[0]
    labels = ["BUY", "HOLD", "SELL"]
    return {"signal": labels[probs.argmax()], "confidence": float(probs.max())}
