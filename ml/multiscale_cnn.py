"""MultiScale EfficientNet-B0 с контекстом и Transfer Learning."""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from ml.config import CFG, SCALES
from torchvision import models


class MultiScaleCNN(nn.Module):
    def __init__(self, ctx_dim: int = 0, n_scales: int = len(SCALES)):
        super().__init__()
        _net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.backbone = nn.Sequential(
            _net.features,
            _net.avgpool,
            nn.Flatten(),
        )
        feat_dim  = 1280
        self.scale_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.SiLU(),
                nn.Dropout(0.4),
            ) for _ in SCALES
        ])
        fused_dim = 128 * n_scales + ctx_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.5),          # было 0.4
            nn.Linear(256, 128),      # добавить промежуточный слой
            nn.SiLU(),  
            nn.Dropout(0.5),
            nn.Linear(128, 3),
        )

    def forward(self, imgs_by_scale: dict, ctx: torch.Tensor = None):
        feats = []
        for i, W in enumerate(SCALES):
            x = imgs_by_scale[W]                    # (B, 3, 224, 224)
            e = self.backbone(x)                     # (B, 1280)
            feats.append(self.scale_proj[i](e))      # (B, 128)  ← был typo
        fused = torch.cat(feats, dim=1)
        if ctx is not None:
            fused = torch.cat([fused, ctx], dim=1)
        return self.head(fused)


def get_device():
    if torch.cuda.is_available():
        d = torch.device('cuda')
        p = torch.cuda.get_device_properties(0)
        print(f"  Устройство: {p.name} ({p.total_memory // 1024**2} MB VRAM)")
    else:
        d = torch.device('cpu')
        print("  Устройство: cpu")
    return d


def _collate(batch):
    """Собрать список (imgs_dict, y, ctx) → батч."""
    imgs_list, y_list, ctx_list = zip(*batch)
    imgs_batch = {W: torch.stack([x[W] for x in imgs_list]) for W in SCALES}
    return imgs_batch, torch.tensor(y_list, dtype=torch.long), torch.stack(ctx_list)


def _make_loader(ds, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, collate_fn=_collate)


def _step(model, batch, device, ctx_dim):
    imgs_dict, y, ctx = batch
    imgs = {W: imgs_dict[W].to(device) for W in SCALES}
    y    = y.to(device)
    ctx  = ctx.to(device) if ctx_dim > 0 else None
    return model(imgs, ctx), y


def train_multiscale(tr_ds, y_tr, val_ds, y_val, ctx_dim,
                     save_path='ml/model_multiscale.pt'):
    device = get_device()
    model  = MultiScaleCNN(ctx_dim=ctx_dim).to(device)

    counts   = np.bincount(y_tr)
    weights  = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    tr_loader  = _make_loader(tr_ds,  CFG.batch_size, True)
    val_loader = _make_loader(val_ds, CFG.batch_size, False)

    print("\n  ── Фаза 1: Pretrain (backbone заморожен) ──")
    for p in model.backbone[0].parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2   # рестарт каждые 10 эпох, потом 20
    )
    best_acc, patience = 0.0, 0

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            logits, y = _step(model, batch, device, ctx_dim)
            optimizer.zero_grad()
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                logits, y = _step(model, batch, device, ctx_dim)
                correct += (logits.argmax(1) == y).sum().item()
        val_acc = correct / len(y_val)
        print(f"  Epoch {epoch:3d}/30 | loss={total_loss/len(tr_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Сохранено (val_acc={val_acc:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print("  ⏹ Ранняя остановка")
                break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model


def finetune_multiscale(model, tr_ds, y_tr, val_ds, y_val, ctx_dim,
                        save_path='ml/model_multiscale.pt'):
    device    = next(model.parameters()).device
    counts    = np.bincount(y_tr)
    weights   = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    print("\n  ── Фаза 2: Fine-tune (features[8] only + head) ──")
    for name, p in model.backbone[0].named_parameters():
        if name.split('.')[0] == '8':
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,            # было 3e-5 — чуть выше, меньше слоёв
        weight_decay=5e-3,  # было 1e-3 — сильнее L2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    tr_loader  = _make_loader(tr_ds,  CFG.batch_size, True)
    val_loader = _make_loader(val_ds, CFG.batch_size, False)
    best_acc, patience = 0.0, 0

    for epoch in range(1, 26):
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            logits, y = _step(model, batch, device, ctx_dim)
            optimizer.zero_grad()
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                logits, y = _step(model, batch, device, ctx_dim)
                correct += (logits.argmax(1) == y).sum().item()
        val_acc = correct / len(y_val)
        print(f"  Fine-tune {epoch:2d}/25 | loss={total_loss/len(tr_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Сохранено (val_acc={val_acc:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print("  ⏹ Ранняя остановка")
                break

    print(f"\n  Лучший val_acc: {best_acc:.4f}")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model


def evaluate_multiscale(model, te_ds, y_test, ctx_dim):
    device = next(model.parameters()).device
    loader = _make_loader(te_ds, 64, False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            logits, _ = _step(model, batch, device, ctx_dim)
            preds.extend(logits.argmax(1).cpu().numpy())
    print(classification_report(y_test, preds,
                                 target_names=['BUY', 'HOLD', 'SELL'], digits=4))
