"""MultiScale EfficientNet-B0 с контекстом и Transfer Learning."""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from ml.config import CFG, SCALES
from torchvision import models

class MultiScaleCNN(nn.Module):
    def __init__(self, ctx_dim: int = 0, n_scales: int = len(SCALES)):
        super().__init__()

        # EfficientNet-B0 из torchvision — качается с dl.fbaipublicfiles.com
        _net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # Убираем голову, оставляем backbone + avgpool
        self.backbone = nn.Sequential(
            _net.features,   # свёрточная часть
            _net.avgpool,    # global avg pool → (B, 1280, 1, 1)
            nn.Flatten(),    # → (B, 1280)
        )
        feat_dim = 1280

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
            nn.Dropout(0.4),
            nn.Linear(256, 3),
        )

    def forward(self, imgs_by_scale: dict, ctx: torch.Tensor = None):
        feats = []
        for i, W in enumerate(SCALES):
            x = imgs_by_scale[W]   # уже (B, 3, 224, 224) CHW — нормализация в render
            e = self.backbone(x)
            feats.append(self.scale_proj[i](e))

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


def _make_loader(imgs_s, y, ctx, batch_size, shuffle):
    tensors = [torch.tensor(imgs_s[W]).float() for W in SCALES]
    tensors.append(torch.tensor(y).long())
    if ctx is not None:
        tensors.append(torch.tensor(ctx).float())
    return DataLoader(TensorDataset(*tensors),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def _unpack(batch, ctx_dim, device):
    if ctx_dim > 0:
        *img_tensors, y, ctx = batch
        ctx = ctx.to(device)
    else:
        *img_tensors, y = batch
        ctx = None
    imgs = {W: img_tensors[i].to(device) for i, W in enumerate(SCALES)}
    return imgs, y.to(device), ctx


def train_multiscale(tr_s, y_tr, val_s, y_val, tr_ctx, val_ctx, ctx_dim,
                     save_path='ml/model_multiscale.pt'):
    device = get_device()
    model  = MultiScaleCNN(ctx_dim=ctx_dim).to(device)

    counts  = np.bincount(y_tr)
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    tr_loader  = _make_loader(tr_s,  y_tr,  tr_ctx,  CFG.batch_size, True)
    val_loader = _make_loader(val_s, y_val, val_ctx, CFG.batch_size, False)

    # Фаза 1: заморозить только features (backbone[0]), не avgpool/flatten
    print("\n  ── Фаза 1: Pretrain (backbone заморожен) ──")
    for p in model.backbone[0].parameters():   # только features
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_acc, patience = 0.0, 0

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            imgs, y, ctx = _unpack(batch, ctx_dim, device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, ctx), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, y, ctx = _unpack(batch, ctx_dim, device)
                correct += (model(imgs, ctx).argmax(1) == y).sum().item()
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


def finetune_multiscale(model, tr_s, y_tr, val_s, y_val, tr_ctx, val_ctx,
                        save_path='ml/model_multiscale.pt'):
    device   = next(model.parameters()).device
    # ctx_dim из реальных данных, не из модели
    _ctx_dim = tr_ctx.shape[1] if tr_ctx is not None else 0

    counts  = np.bincount(y_tr)
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # Фаза 2: разморозить последние 3 блока features (6, 7, 8)
    # torchvision EfficientNet-B0: features = Sequential[0..8]
    print("\n  ── Фаза 2: Fine-tune (features[6,7,8] + head) ──")
    for name, p in model.backbone[0].named_parameters():
        block_idx = name.split('.')[0]   # '0', '1', ... '8'
        if block_idx in ('6', '7', '8'):
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    tr_loader  = _make_loader(tr_s,  y_tr,  tr_ctx,  CFG.batch_size, True)
    val_loader = _make_loader(val_s, y_val, val_ctx, CFG.batch_size, False)

    best_acc, patience = 0.0, 0

    for epoch in range(1, 26):
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            imgs, y, ctx = _unpack(batch, _ctx_dim, device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, ctx), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, y, ctx = _unpack(batch, _ctx_dim, device)
                correct += (model(imgs, ctx).argmax(1) == y).sum().item()
        val_acc = correct / len(y_val)
        print(f"  Fine-tune {epoch:2d}/25 | loss={total_loss/len(tr_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Сохранено (val_acc={val_acc:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                print("  ⏹ Ранняя остановка")
                break

    print(f"\n  Лучший val_acc: {best_acc:.4f}")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model

def evaluate_multiscale(model, te_s, y_test, te_ctx):
    _ctx_dim = te_ctx.shape[1] if te_ctx is not None else 0
    device   = next(model.parameters()).device
    loader   = _make_loader(te_s, y_test, te_ctx, 128, False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            imgs, y, ctx = _unpack(batch, _ctx_dim, device)
            preds.extend(model(imgs, ctx).argmax(1).cpu().numpy())
    print(classification_report(y_test, preds,
                                 target_names=['BUY', 'HOLD', 'SELL'], digits=4))
