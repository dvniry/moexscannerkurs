"""MultiScale EfficientNet-B0 с контекстом и Transfer Learning."""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from ml.config import CFG, SCALES
from torchvision import models


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            weight=weight, reduction='none', label_smoothing=label_smoothing
        )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


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
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
        )

    def forward(self, imgs_by_scale: dict, ctx: torch.Tensor = None):
        feats = []
        for i, W in enumerate(SCALES):
            x = imgs_by_scale[W]
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


def _collate(batch):
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


def _mixup_batch(imgs_dict, y, alpha=0.2):
    """MixUp: смешивает два случайных сэмпла — снижает переобучение."""
    if alpha <= 0:
        return imgs_dict, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B   = y.shape[0]
    idx = torch.randperm(B, device=y.device)
    imgs_mixed = {W: lam * imgs_dict[W] + (1 - lam) * imgs_dict[W][idx]
                  for W in imgs_dict}
    return imgs_mixed, y, y[idx], lam


def train_multiscale(tr_ds, y_tr, val_ds, y_val, ctx_dim,
                     save_path='ml/model_multiscale.pt'):
    device = get_device()
    model  = MultiScaleCNN(ctx_dim=ctx_dim).to(device)

    counts    = np.bincount(y_tr)
    weights   = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=2.0, weight=weights, label_smoothing=0.1)

    tr_loader  = _make_loader(tr_ds,  CFG.batch_size, True)
    val_loader = _make_loader(val_ds, CFG.batch_size, False)

    print("\n  ── Фаза 1: Pretrain (backbone заморожен) ──")
    for p in model.backbone[0].parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(tr_loader),
        epochs=30,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100,
    )
    best_acc, patience = 0.0, 0

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            # ── MixUp ──────────────────────────────────────────
            imgs_dict, y, ctx = batch
            imgs_dict = {W: imgs_dict[W].to(device) for W in SCALES}
            y   = y.to(device)
            ctx = ctx.to(device) if ctx_dim > 0 else None

            imgs_m, y_a, y_b, lam = _mixup_batch(imgs_dict, y, alpha=0.2)
            logits = model(imgs_m, ctx)
            loss   = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            # ───────────────────────────────────────────────────

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

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
    criterion = FocalLoss(gamma=2.0, weight=weights, label_smoothing=0.1)

    print("\n  ── Фаза 2: Fine-tune (features[8] only + head) ──")
    for name, p in model.backbone[0].named_parameters():
        if name.split('.')[0] == '8':
            p.requires_grad = True

    tr_loader  = _make_loader(tr_ds,  CFG.batch_size, True)
    val_loader = _make_loader(val_ds, CFG.batch_size, False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=5e-3,
    )
    # ── OneCycleLR вместо CosineAnnealingLR ──────────────────────
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(tr_loader),
        epochs=25,
        pct_start=0.2,
        div_factor=5,
        final_div_factor=50,
    )
    best_acc, patience = 0.0, 0

    for epoch in range(1, 26):
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            # ── MixUp ──────────────────────────────────────────
            imgs_dict, y, ctx = batch
            imgs_dict = {W: imgs_dict[W].to(device) for W in SCALES}
            y   = y.to(device)
            ctx = ctx.to(device) if ctx_dim > 0 else None

            imgs_m, y_a, y_b, lam = _mixup_batch(imgs_dict, y, alpha=0.2)
            logits = model(imgs_m, ctx)
            loss   = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            # ───────────────────────────────────────────────────

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

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
            if patience >= 8:   # было 5 → 8
                print("  ⏹ Ранняя остановка")
                break

    print(f"\n  Лучший val_acc: {best_acc:.4f}")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model


def evaluate_multiscale(model, te_ds, y_test, ctx_dim,
                        save_json: str = 'ml/eval_results.json',
                        confidence_thr: float = 0.0):
    import json, datetime
    device = next(model.parameters()).device
    loader = _make_loader(te_ds, 64, False)
    model.eval()

    preds, probs_all = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = _step(model, batch, device, ctx_dim)
            softmax = torch.softmax(logits, dim=1)
            probs_all.append(softmax.cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())

    probs_all = np.concatenate(probs_all, axis=0)

    if confidence_thr > 0.0:
        max_conf = probs_all.max(axis=1)
        preds = np.array(preds)
        preds[max_conf < confidence_thr] = 1
        print(f"  Отфильтровано как HOLD (conf < {confidence_thr}): "
              f"{(max_conf < confidence_thr).sum()} / {len(preds)}")

    report_str  = classification_report(y_test, preds,
                                        target_names=['BUY', 'HOLD', 'SELL'], digits=4)
    report_dict = classification_report(y_test, preds,
                                        target_names=['BUY', 'HOLD', 'SELL'], output_dict=True)
    print(report_str)

    print("  Среднее max_confidence:")
    for cls_idx, name in enumerate(['BUY', 'HOLD', 'SELL']):
        mask = np.array(y_test) == cls_idx
        if mask.sum() > 0:
            print(f"    {name}: {probs_all[mask].max(axis=1).mean():.3f}")

    results = {
        "timestamp":      datetime.datetime.now().isoformat(),
        "accuracy":       float(report_dict['accuracy']),
        "macro_f1":       float(report_dict['macro avg']['f1-score']),
        "confidence_thr": confidence_thr,
        "per_class": {
            name: {
                "precision": float(report_dict[name]['precision']),
                "recall":    float(report_dict[name]['recall']),
                "f1":        float(report_dict[name]['f1-score']),
            } for name in ['BUY', 'HOLD', 'SELL']
        }
    }
    with open(save_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Метрики → {save_json}")

    return results

# ══════════════════════════════════════════════════════════════════
#  DEEP MODE: MultiScale + Bidirectional LSTM
#  Масштабы [5, 10, 20, 30] → последовательность → BiLSTM → head
#  "Короткая память" = scale 5, "Длинная память" = scale 30
# ══════════════════════════════════════════════════════════════════

class MultiScaleLSTM(nn.Module):
    """
    Каждый масштаб = один шаг последовательности.
    BiLSTM видит паттерн от короткого к длинному таймфрейму.
    """
    def __init__(self, ctx_dim: int = 0, n_scales: int = len(SCALES),
                 lstm_hidden: int = 256, lstm_layers: int = 2):
        super().__init__()
        _net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.backbone = nn.Sequential(
            _net.features,
            _net.avgpool,
            nn.Flatten(),
        )
        feat_dim = 1280
        self.scale_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.SiLU(),
                nn.Dropout(0.3),
            ) for _ in SCALES
        ])

        # BiLSTM: input=128, hidden=256, 2 слоя, двунаправленный
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # После BiLSTM: lstm_hidden * 2 (bi) + ctx_dim
        lstm_out_dim = lstm_hidden * 2
        fused_dim    = lstm_out_dim + ctx_dim

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3),
        )

    def forward(self, imgs_by_scale: dict, ctx: torch.Tensor = None):
        # Каждый масштаб → (B, 128), собираем в (B, n_scales, 128)
        feats = []
        for i, W in enumerate(SCALES):
            e = self.backbone(imgs_by_scale[W])     # (B, 1280)
            feats.append(self.scale_proj[i](e))     # (B, 128)
        seq = torch.stack(feats, dim=1)              # (B, 4, 128)

        # BiLSTM → берём последний шаг (длинная память) + первый (короткая)
        lstm_out, (h_n, _) = self.lstm(seq)          # lstm_out: (B, 4, 512)
        # Конкатенируем последний и первый выход — оба направления
        out = torch.cat([lstm_out[:, -1, :], lstm_out[:, 0, :]], dim=1)  # (B, 1024)
        # Но нам нужно lstm_hidden*2, берём только последний шаг
        out = lstm_out[:, -1, :]                     # (B, 512)

        if ctx is not None:
            out = torch.cat([out, ctx], dim=1)
        return self.head(out)


def train_multiscale_deep(tr_ds, y_tr, val_ds, y_val, ctx_dim,
                          save_path='ml/model_multiscale_deep.pt'):
    """
    Глубокое обучение: больше эпох, BiLSTM, постепенный разморозка backbone.
    Фаза 1 (30 эпох): только head + lstm, backbone заморожен
    Фаза 2 (30 эпох): features[6,7,8] разморожены
    Фаза 3 (20 эпох): весь backbone, очень маленький LR
    """
    device = get_device()
    model  = MultiScaleLSTM(ctx_dim=ctx_dim).to(device)

    counts    = np.bincount(y_tr)
    weights   = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=2.0, weight=weights, label_smoothing=0.05)

    tr_loader  = _make_loader(tr_ds,  CFG.batch_size, True)
    val_loader = _make_loader(val_ds, CFG.batch_size, False)

    def _run_phase(phase_name, n_epochs, max_lr, wd, patience_limit):
        nonlocal model
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=max_lr / 10, weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr,
            steps_per_epoch=len(tr_loader),
            epochs=n_epochs,
            pct_start=0.2,
            div_factor=10,
            final_div_factor=200,
        )
        best_acc, patience = 0.0, 0

        for epoch in range(1, n_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in tr_loader:
                imgs_dict, y, ctx = batch
                imgs_dict = {W: imgs_dict[W].to(device) for W in SCALES}
                y   = y.to(device)
                ctx = ctx.to(device) if ctx_dim > 0 else None

                imgs_m, y_a, y_b, lam = _mixup_batch(imgs_dict, y, alpha=0.3)
                logits = model(imgs_m, ctx)
                loss   = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            model.eval()
            correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    logits, y = _step(model, batch, device, ctx_dim)
                    correct += (logits.argmax(1) == y).sum().item()
            val_acc = correct / len(y_val)
            print(f"  [{phase_name}] Epoch {epoch:3d}/{n_epochs} | "
                  f"loss={total_loss/len(tr_loader):.4f} | val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Сохранено (val_acc={val_acc:.4f})")
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    print(f"  ⏹ Ранняя остановка [{phase_name}]")
                    break

        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print(f"  └─ Лучший val_acc [{phase_name}]: {best_acc:.4f}")

    # ── Фаза 1: только LSTM + head ────────────────────────────────
    print("\n  ── Фаза 1/3: Pretrain — backbone заморожен ──")
    for p in model.backbone[0].parameters():
        p.requires_grad = False
    _run_phase("F1-pretrain", n_epochs=30, max_lr=1e-3, wd=1e-4, patience_limit=12)

    # ── Фаза 2: features[6,7,8] разморожены ──────────────────────
    print("\n  ── Фаза 2/3: Fine-tune — features 6/7/8 ──")
    for name, p in model.backbone[0].named_parameters():
        if name.split('.')[0] in ('6', '7', '8'):
            p.requires_grad = True
    _run_phase("F2-finetune", n_epochs=30, max_lr=3e-5, wd=1e-3, patience_limit=12)

    # ── Фаза 3: весь backbone, очень маленький LR ─────────────────
    print("\n  ── Фаза 3/3: Deep fine-tune — весь backbone ──")
    for p in model.backbone[0].parameters():
        p.requires_grad = True
    _run_phase("F3-deep", n_epochs=20, max_lr=5e-6, wd=5e-3, patience_limit=8)

    return model
