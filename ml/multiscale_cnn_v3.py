# ml/multiscale_cnn_v3.py
"""MultiScale CNN Hybrid v3.11

Изменения v3.11:
- RevIN (Reversible Instance Normalization) в SingleScaleBackbone и BiLSTMBranch
  → устраняет distribution shift между тикерами/режимами БЕЗ утечки будущего
  → статистики считаются только по входному окну (история), не по targets
- OHLCVAugment: amplitude_jitter + volatility_regime_swap + num_jitter
  → применяется только к history-окну; метки (cls_y/ohlc_y) — из будущего, не трогаются
- temporal_cutmix вместо mixup_data
  → вырезаем ТОЛЬКО из [0:T] (исторические бары); targets берём от исходного сэмпла
  → нет утечки: imgs[W] ∈ прошлое, cls_y/ohlc_y ∈ будущее (t+future_bars)
- TemporalBalancedSampler: каждый батч ≈ равные доли из разных тикеров
  → уменьшает overfit на конкретных эмитентов/периоды
- _make_loader_v3: добавлен аргумент sampler
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import numpy as np

try:
    from ml.config import CFG, SCALES
except ImportError:
    from config import CFG, SCALES

TRUNK_OUT = 128


# ────────────────────────────────────────────────────────────
# RevIN — Reversible Instance Normalization
# ────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """Нормировка по статистикам ВХОДНОГО окна.

    Ключевое свойство: mean/std считаются по dim=-1 (T) входного батча.
    Это только прошлые бары — никакой информации о будущем.
    Affine-параметры (weight, bias) обучаются — модель может обратить нормировку.

    Input/Output: [B, C, T]  (C — каналы/признаки, T — время)
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps     = eps
        self.affine  = affine
        self._mean   = None
        self._std    = None
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        # x: [B, C, T]
        if mode == 'norm':
            self._mean = x.mean(dim=-1, keepdim=True).detach()
            self._std  = x.std(dim=-1, keepdim=True).detach() + self.eps
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.weight[None, :, None] + self.bias[None, :, None]
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias[None, :, None]) / (self.weight[None, :, None] + self.eps)
            x = x * self._std + self._mean
        return x


# ────────────────────────────────────────────────────────────
# Blocks
# ────────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, k, stride=stride,
                      padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            ConvBnAct(c, c, k=3, dilation=dilation),
            nn.Conv1d(c, c, 1, bias=False),
            nn.BatchNorm1d(c),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.body(x))


class SingleScaleBackbone(nn.Module):
    """1D-CNN backbone для одного масштаба.
    Input:  [B, 3, T]   (T = 64 для всех масштабов после resize в датасете)
    Output: [B, TRUNK_OUT]

    RevIN применяется ДО backbone — нормирует по статистикам текущего окна.
    Это устраняет distribution shift: SBER@300руб и GAZP@150руб дают одинаковый
    нормированный паттерн, если форма свечей одинакова.
    """
    def __init__(self, base_ch=64):
        super().__init__()
        # RevIN по 3 каналам (O, H/L, C или аналогичное кодирование изображения)
        self.revin = RevIN(3, affine=True)
        self.stem = ConvBnAct(3, base_ch, k=5)
        self.blocks = nn.Sequential(
            ResBlock(base_ch, dilation=1),
            ConvBnAct(base_ch, base_ch*2, k=3, stride=2),
            ResBlock(base_ch*2, dilation=2),
            ConvBnAct(base_ch*2, TRUNK_OUT, k=3, stride=2),
            ResBlock(TRUNK_OUT, dilation=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Нормируем по статистикам текущего окна — нет утечки будущего
        x = self.revin(x, mode='norm')
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)   # [B, TRUNK_OUT]


class GRN(nn.Module):
    """Gated Residual Network."""
    def __init__(self, d_in, d_out, dropout=0.3):
        super().__init__()
        self.fc1  = nn.Linear(d_in,  d_out)
        self.fc2  = nn.Linear(d_out, d_out)
        self.gate = nn.Linear(d_in,  d_out)
        self.proj = nn.Linear(d_in,  d_out) if d_in != d_out else nn.Identity()
        self.ln   = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        h = F.gelu(self.fc1(x))
        h = self.drop(self.fc2(h))
        return self.ln(self.proj(x) + g * h)


class BiLSTMBranch(nn.Module):
    """Обрабатывает числовые фичи длинного масштаба (scale=30).
    Input:  [B, T, n_ind]
    Output: [B, TRUNK_OUT]

    RevIN нормирует каждый индикатор по временному окну отдельно:
    RSI@70 и RSI@45 превращаются в одинаковый relative-паттерн.
    Нет утечки: T — исторические бары, targets — из будущего.
    """
    def __init__(self, n_ind, hidden=128, num_layers=2):
        super().__init__()
        self.hidden = hidden
        # RevIN по n_ind признакам, нормировка по оси T
        self.revin  = RevIN(n_ind, affine=True)
        self.bilstm = nn.LSTM(
            input_size=n_ind, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 4, TRUNK_OUT),
            nn.LayerNorm(TRUNK_OUT),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, T, n_ind]
        x = x.clamp(-8.0, 8.0).nan_to_num(nan=0.0, posinf=8.0, neginf=-8.0)
        # RevIN: транспонируем → нормируем по T → транспонируем обратно
        x = x.transpose(1, 2)          # [B, n_ind, T]
        x = self.revin(x, mode='norm') # нормировка по T для каждого индикатора
        x = x.transpose(1, 2)          # [B, T, n_ind]
        _, (h_n, _) = self.bilstm(x)
        h = torch.cat([h_n[-2], h_n[-1],
                       h_n[-4] if h_n.shape[0] >= 4 else h_n[0],
                       h_n[-3] if h_n.shape[0] >= 4 else h_n[1]], dim=-1)
        h = h.nan_to_num(nan=0.0, posinf=10.0, neginf=-10.0)
        return self.proj(h)


class HourlyEncoder(nn.Module):
    """Кодировщик часовых OHLCV данных.
    Input:  [B, n_days, n_hours, n_feats]
    Output: [B, TRUNK_OUT]
    """
    def __init__(self, n_feats=9, n_hours=11, n_days=5, out_dim=TRUNK_OUT):
        super().__init__()
        flat_dim = n_days * n_hours * n_feats
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        x = x.nan_to_num(nan=0.0, posinf=5.0, neginf=-5.0)
        return self.net(x)


# ────────────────────────────────────────────────────────────
# Heads
# ────────────────────────────────────────────────────────────

class CalibratedClsHead(nn.Module):
    """LayerNorm → SiLU → Linear(→3). Без Dropout."""
    def __init__(self, in_dim=TRUNK_OUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, 3),
        )

    def forward(self, x):
        return self.head(x)


class OHLCHead(nn.Module):
    def __init__(self, in_dim=TRUNK_OUT, n_out=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ────────────────────────────────────────────────────────────
# Full Model
# ────────────────────────────────────────────────────────────

class MultiScaleHybridV3(nn.Module):
    def __init__(self, ctx_dim=0, n_indicator_cols=30,
                 future_bars=5, use_hourly=True):
        super().__init__()
        self.use_hourly = use_hourly
        self.ctx_dim    = ctx_dim

        self.backbones = nn.ModuleDict(
            {str(W): SingleScaleBackbone() for W in SCALES}
        )
        self.bilstm_branch = BiLSTMBranch(n_ind=n_indicator_cols)
        self.num_grn = nn.ModuleDict(
            {str(W): GRN(n_indicator_cols, TRUNK_OUT)
             for W in SCALES if W < max(SCALES)}
        )
        if use_hourly:
            self.hourly_enc = HourlyEncoder()
        if ctx_dim > 0:
            self.ctx_proj = nn.Sequential(
                nn.Linear(ctx_dim, TRUNK_OUT),
                nn.LayerNorm(TRUNK_OUT),
                nn.GELU(),
            )

        n_streams  = len(SCALES) + 1
        n_streams += len([W for W in SCALES if W < max(SCALES)])
        if use_hourly:  n_streams += 1
        if ctx_dim > 0: n_streams += 1

        self.fusion = nn.Sequential(
            GRN(TRUNK_OUT * n_streams, TRUNK_OUT * 2),
            GRN(TRUNK_OUT * 2, TRUNK_OUT),
        )
        self.cls_head  = CalibratedClsHead(TRUNK_OUT)
        self.ohlc_head = OHLCHead(TRUNK_OUT, n_out=4 * future_bars)
        self.backbone  = self.backbones[str(min(SCALES))]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, imgs, nums, ctx=None, hourly=None):
        feats = []

        for W in SCALES:
            x = imgs[W].float()
            feats.append(self.backbones[str(W)](x))

        long_W = max(SCALES)
        if nums is not None and long_W in nums:
            lstm_feat = self.bilstm_branch(nums[long_W].float())
        else:
            lstm_feat = torch.zeros(imgs[min(SCALES)].shape[0], TRUNK_OUT,
                                    device=imgs[min(SCALES)].device)
        feats.append(lstm_feat)

        for W in SCALES:
            if W < long_W and nums is not None and W in nums:
                x_num  = nums[W].float().nan_to_num(nan=0.0, posinf=5.0, neginf=-5.0)
                x_mean = x_num.mean(dim=1)
                feats.append(self.num_grn[str(W)](x_mean))

        if self.use_hourly and hourly is not None:
            feats.append(self.hourly_enc(hourly.float()))

        if self.ctx_dim > 0 and ctx is not None:
            feats.append(self.ctx_proj(ctx.float()))

        cat = torch.cat(feats, dim=-1)
        cat = cat.nan_to_num(nan=0.0, posinf=10.0, neginf=-10.0)
        cat = F.dropout(cat, p=0.2, training=self.training)
        h   = self.fusion(cat)

        return self.cls_head(h), self.ohlc_head(h)


# ────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma_per_class=(2.0, 2.0, 2.0),
                 label_smoothing=0.15):
        super().__init__()
        if weight is not None:
            self.register_buffer('cls_weight', weight)
        else:
            self.cls_weight = None
        self.gamma = gamma_per_class
        self.ls    = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits.float(), targets,
            weight=self.cls_weight,
            label_smoothing=self.ls,
            reduction='none'
        )
        pt      = torch.exp(-ce.detach()).clamp(0.0, 1.0 - 1e-6)
        gamma_t = torch.tensor(self.gamma, device=logits.device,
                               dtype=logits.dtype)[targets]
        return ((1.0 - pt) ** gamma_t * ce).mean()


class MultiTaskLossV3(nn.Module):
    def __init__(self, cls_weight=None, gamma_per_class=(2.0, 2.0, 2.0),
                 label_smoothing=0.1, huber_delta=0.5,
                 direction_weight=0.3, reg_loss_weight=0.1):
        super().__init__()
        self.focal = AsymmetricFocalLoss(
            weight=cls_weight,
            gamma_per_class=gamma_per_class,
            label_smoothing=label_smoothing,
        )
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.dir_w = direction_weight
        self.reg_loss_weight = reg_loss_weight

    def forward(self, logits, cls_y, ohlc_pred, ohlc_true):
        cls_loss = self.focal(logits, cls_y)
        n = ohlc_pred.shape[1]
        reg_loss = self.huber(ohlc_pred, ohlc_true[:, :n])
        if self.dir_w > 0 and ohlc_pred.shape[-1] >= 4:
            dir_loss = F.binary_cross_entropy_with_logits(
                ohlc_pred[:, 3],
                (ohlc_true[:, 3] > 0).float(),
            )
            reg_loss = reg_loss + self.dir_w * dir_loss
        return cls_loss + self.reg_loss_weight * reg_loss, cls_loss, reg_loss


# ────────────────────────────────────────────────────────────
# Augmentation
# ────────────────────────────────────────────────────────────

class OHLCVAugment:
    """Аугментации ТОЛЬКО для входного (исторического) окна.

    Гарантия отсутствия утечки будущего:
    - imgs[W] ∈ [0, T-1] → только прошлые бары
    - nums[W] ∈ [0, T-1] → только прошлые индикаторы
    - cls_y, ohlc_y никогда не изменяются — они из будущего (t + future_bars)

    Три типа аугментации (выбор случайный):
    1. amplitude_jitter   — малое масштабирование ±1.5%: "тот же паттерн, другой масштаб цены"
    2. volatility_swap    — смена режима волатильности [0.75×, 1.25×]: "тот же паттерн, другой ATR"
    3. num_jitter         — гауссовский шум в числовых признаках ±0.8%: устойчивость к шуму индикаторов
    """
    def __init__(self, amplitude_sigma=0.015, vol_range=(0.75, 1.25),
                 num_jitter_sigma=0.008, p=0.5):
        self.amp_sigma = amplitude_sigma
        self.vol_range = vol_range
        self.num_sigma = num_jitter_sigma
        self.p         = p

    def __call__(self, imgs: dict, nums) -> tuple:
        if np.random.random() > self.p:
            return imgs, nums
        r = np.random.random()
        if r < 0.4:
            # Amplitude jitter
            scale = 1.0 + float(np.random.randn() * self.amp_sigma)
            imgs  = {W: (x * scale).clamp(0.0, 1.0) for W, x in imgs.items()}
        elif r < 0.75:
            # Volatility regime swap — масштабируем отклонение от среднего
            factor = float(np.random.uniform(*self.vol_range))
            imgs_aug = {}
            for W, x in imgs.items():
                # mid: [B, 1, 1] — средний уровень свечей в окне
                mid = x.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
                imgs_aug[W] = (mid + (x - mid) * factor).clamp(0.0, 1.0)
            imgs = imgs_aug
        else:
            # Числовой шум
            if nums is not None:
                nums = {W: n + torch.randn_like(n) * self.num_sigma
                        for W, n in nums.items()}
        return imgs, nums


# ────────────────────────────────────────────────────────────
# Temporal CutMix
# ────────────────────────────────────────────────────────────

def temporal_cutmix(imgs, nums, cls_y, ohlc_y, ctx, alpha=0.3, hourly=None):
    """Temporal CutMix — безопасная замена mixup_data.

    Что делаем: вырезаем случайный отрезок [cut_start : cut_start+cut_len]
    из сэмпла B и вставляем в сэмпл A. Только в историческом окне.

    Почему нет утечки будущего:
    - imgs[W].shape = [B, C, T], T = 64 исторических бара — прошлое
    - nums[W].shape = [B, T_w, n_ind], T_w = W баров — прошлое
    - cls_y, ohlc_y — из t+future_bars (будущее), к ним не прикасаемся
    - Метки смешиваются пропорционально lam (доля оригинального окна)
    """
    if alpha <= 0:
        return imgs, nums, cls_y, cls_y, ohlc_y, ctx, 1.0, hourly

    lam = float(np.random.beta(alpha, alpha))
    B   = cls_y.shape[0]
    idx = torch.randperm(B, device=cls_y.device)

    W_min = min(imgs.keys())
    T     = imgs[W_min].shape[-1]          # 64 (только история)
    cut_len   = max(1, int(T * (1.0 - lam)))
    cut_start = int(np.random.randint(0, max(1, T - cut_len)))

    # Смешиваем изображения
    m_imgs = {}
    for W, x in imgs.items():
        x_new = x.clone()
        x_new[:, :, cut_start:cut_start + cut_len] =             x[idx, :, cut_start:cut_start + cut_len]
        m_imgs[W] = x_new

    # Смешиваем числовые признаки (пропорциональная проекция временной оси)
    m_nums = None
    if nums is not None:
        m_nums = {}
        for W, n in nums.items():
            T_w = n.shape[1]
            cs  = int(cut_start * T_w / T)
            cl  = max(1, int(cut_len   * T_w / T))
            n_new = n.clone()
            n_new[:, cs:cs + cl, :] = n[idx, cs:cs + cl, :]
            m_nums[W] = n_new

    # Смешиваем OHLC-таргеты пропорционально (регрессия)
    m_ohlc   = lam * ohlc_y + (1 - lam) * ohlc_y[idx]
    m_ctx    = (lam * ctx + (1 - lam) * ctx[idx])    if ctx    is not None else None
    m_hourly = (lam * hourly + (1 - lam) * hourly[idx]) if hourly is not None else None

    # Возвращаем оба набора меток (cls_a, cls_b) — trainer решает как взвешивать
    return m_imgs, m_nums, cls_y, cls_y[idx], m_ohlc, m_ctx, lam, m_hourly


# backward-compat alias
def mixup_data(imgs, nums, cls_y, ohlc_y, ctx, alpha=0.2, hourly=None):
    return temporal_cutmix(imgs, nums, cls_y, ohlc_y, ctx, alpha, hourly)


# ────────────────────────────────────────────────────────────
# Ticker-Balanced Sampler
# ────────────────────────────────────────────────────────────

class TemporalBalancedSampler(Sampler):
    """Каждый батч содержит примерно равные доли из разных тикеров.

    Аргументы:
        ticker_to_local_idx: dict {ticker_key: [local_idx, ...]}
            local_idx — позиция в Subset (0..len(tr_subset)-1)
        batch_size: размер батча
        drop_last:  обрезать последний неполный батч
    """
    def __init__(self, ticker_to_local_idx: dict, batch_size: int, drop_last: bool = True):
        self.groups      = [np.array(v, dtype=np.int64)
                            for v in ticker_to_local_idx.values()]
        self.batch_size  = batch_size
        self.n_groups    = len(self.groups)
        self.per_group   = max(1, batch_size // self.n_groups)
        self.drop_last   = drop_last

    def __iter__(self):
        shuffled = [np.random.permutation(g) for g in self.groups]
        ptrs     = [0] * self.n_groups
        result   = []
        max_iter = max(len(g) for g in self.groups)

        for _ in range(0, max_iter, self.per_group):
            for gi, g in enumerate(shuffled):
                p   = ptrs[gi]
                end = p + self.per_group
                chunk = g[p:end].tolist()
                if len(chunk) < self.per_group:
                    # Дозаполняем с начала (с новым перемешиванием)
                    extra = self.per_group - len(chunk)
                    shuffled[gi] = np.random.permutation(self.groups[gi])
                    chunk += shuffled[gi][:extra].tolist()
                    ptrs[gi] = extra
                else:
                    ptrs[gi] = end
                result.extend(chunk)

        if self.drop_last:
            result = result[:(len(result) // self.batch_size) * self.batch_size]
        return iter(result)

    def __len__(self):
        n = sum(len(g) for g in self.groups)
        return (n // self.batch_size) * self.batch_size if self.drop_last else n


# ────────────────────────────────────────────────────────────
# DataLoader helper
# ────────────────────────────────────────────────────────────

def _make_loader_v3(dataset, batch_size, shuffle=False, num_workers=4, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=(shuffle or sampler is not None),
    )


# ────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────

def evaluate_multiscale_v3(model, te_ds, y_test, ctx_dim,
                            use_hourly=True, save_json=None):
    from sklearn.metrics import classification_report
    import json

    device = next(model.parameters()).device
    loader = _make_loader_v3(te_ds, batch_size=256, shuffle=False)
    model.eval()

    all_preds, all_trues = [], []
    all_ohlc_pred, all_ohlc_true = [], []

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx, *hourly_opt = batch
            hourly_data = hourly_opt[0] if hourly_opt else None
            imgs  = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t = ctx.to(device) if ctx_dim > 0 else None
            ht    = (hourly_data.to(device)
                     if (use_hourly and hourly_data is not None) else None)
            nums  = ({W: num_dict[W].to(device) for W in SCALES}
                     if num_dict is not None else None)
            lo, op = model(imgs, nums, ctx_t, hourly=ht)
            all_preds.extend(lo.argmax(1).cpu().numpy())
            all_trues.extend(cls_y.numpy())
            all_ohlc_pred.append(op.cpu().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    preds  = np.array(all_preds)
    trues  = np.array(all_trues)
    ohlc_p = np.concatenate(all_ohlc_pred, 0)
    ohlc_t = np.concatenate(all_ohlc_true, 0)

    print(classification_report(trues, preds, target_names=['UP','FLAT','DOWN'],
                                 digits=4, zero_division=0))
    mae = np.abs(ohlc_p - ohlc_t).mean(0)
    print("\n  OHLC MAE:")
    for i, name in enumerate(['ΔOpen','ΔHigh','ΔLow','ΔClose']):
        print(f"    {name}: {mae[i]:.4f}")
    dir_acc = (np.sign(ohlc_p[:, 3]) == np.sign(ohlc_t[:, 3])).mean()
    print(f"  Direction accuracy: {dir_acc:.4f}")

    if save_json:
        from sklearn.metrics import f1_score
        result = {
            'accuracy': float((preds==trues).mean()),
            'macro_f1': float(f1_score(trues, preds, average='macro', zero_division=0)),
            'dir_acc': float(dir_acc),
            'ohlc_mae': mae.tolist(),
        }
        with open(save_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {save_json}")
    return preds, trues
