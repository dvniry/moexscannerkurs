# ml/multiscale_cnn_v4.py
"""MultiScale CNN Hybrid v4.1 — с Kronos backbone + DirectionHead (v3.17 changes).

Изменения v4.1:
- DirectionHead, SeqStatsPool и 4-кортеж из v3.17.
- get_param_groups добавляет dir_head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List

from ml.multiscale_cnn_v3 import (
    TRUNK_OUT,
    ConvBnAct, ResBlock, DropPath,
    SingleScaleBackbone,
    GRN,
    SeqStatsPool,
    WaveletDenoise,
    xLSTMBranch,
    VariableSelectionNetwork,
    HourlyEncoder,
    CalibratedClsHead,
    OHLCHeadV2,
    OHLCLossV2,
    AuxHead,
    DirectionHead,             # NEW v4.1
    AsymmetricFocalLoss,
    PinballLoss,
    AuxLoss,
    MultiTaskLossV3,
    mixup_data,
    _make_loader_v3,
    evaluate_multiscale_v3,
)

try:
    from ml.config import CFG, SCALES
except ImportError:
    from config import CFG, SCALES


class MultiScaleHybridV4(nn.Module):
    def __init__(
        self,
        ctx_dim: int = 0,
        n_indicator_cols: int = 37,
        future_bars: int = 5,
        use_hourly: bool = True,
        in_channels: int = 4,
        use_kronos: bool = True,
        kronos_model: str = "amazon/chronos-t5-tiny",
        kronos_seq_len: int = 64,
        kronos_n_unfreeze: int = 2,
        kronos_grad_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_hourly = use_hourly
        self.ctx_dim = ctx_dim
        self.use_kronos = use_kronos

        self.backbones = nn.ModuleDict({
            str(W): SingleScaleBackbone(in_channels=in_channels)
            for W in SCALES
        })

        self.wavelet = WaveletDenoise(threshold=0.08, levels=1)

        self.seq_branch = xLSTMBranch(
            n_ind=n_indicator_cols,
            context_length=max(SCALES),
        )

        # v4.1: SeqStatsPool
        self.num_stats = nn.ModuleDict({
            str(W): SeqStatsPool(n_indicator_cols, TRUNK_OUT)
            for W in SCALES if W < max(SCALES)
        })

        if use_hourly:
            self.hourly_enc = HourlyEncoder()

        if ctx_dim > 0:
            self.ctx_proj = nn.Sequential(
                nn.Linear(ctx_dim, TRUNK_OUT),
                nn.LayerNorm(TRUNK_OUT),
                nn.GELU(),
            )

        if use_kronos:
            from ml.kronos_adapter import KronosAdapter
            self.kronos = KronosAdapter(
                n_indicator_cols=n_indicator_cols,
                img_channels=in_channels,
                kronos_seq_len=kronos_seq_len,
                model_name=kronos_model,
                n_unfreeze_layers=kronos_n_unfreeze,
                use_grad_checkpoint=kronos_grad_checkpoint,
                out_dim=TRUNK_OUT,
            )

        n_streams = (
            len(SCALES)
            + 1
            + len([W for W in SCALES if W < max(SCALES)])
        )
        if use_hourly: n_streams += 1
        if ctx_dim > 0: n_streams += 1
        if use_kronos: n_streams += 1

        self.vsn = VariableSelectionNetwork(
            n_streams=n_streams, d_model=TRUNK_OUT, dropout=0.15,
        )

        # v4.1: 4 головы (+ dir_head)
        self.cls_head  = CalibratedClsHead(TRUNK_OUT)
        self.ohlc_head = OHLCHeadV2(TRUNK_OUT, future_bars=future_bars)
        self.aux_head  = AuxHead(TRUNK_OUT)
        self.dir_head  = DirectionHead(TRUNK_OUT)

        self.backbone = self.backbones[str(min(SCALES))]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def get_param_groups(self, max_lr: float = 3e-4) -> List[Dict]:
        kronos_backbone_ids = set()
        kronos_adapter_ids  = set()
        backbone_ids        = set()
        hourly_ids          = set()
        cls_head_ids        = set()
        dir_head_ids        = set()

        for p in self.backbones.parameters():
            backbone_ids.add(id(p))

        if self.use_kronos:
            for p in self.kronos.get_backbone_params():
                kronos_backbone_ids.add(id(p))
            for p in self.kronos.get_adapter_params():
                kronos_adapter_ids.add(id(p))
            for p in self.kronos.grn.parameters():
                kronos_adapter_ids.add(id(p))

        if self.use_hourly and hasattr(self, 'hourly_enc'):
            for p in self.hourly_enc.parameters():
                hourly_ids.add(id(p))

        for p in self.cls_head.parameters():
            cls_head_ids.add(id(p))
        for p in self.dir_head.parameters():
            dir_head_ids.add(id(p))

        all_special = (kronos_backbone_ids | kronos_adapter_ids
                       | backbone_ids | hourly_ids
                       | cls_head_ids | dir_head_ids)

        other_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in all_special
        ]

        groups = [
            {'params': [p for p in self.backbones.parameters()
                        if p.requires_grad],
             'lr': max_lr * 0.15, 'weight_decay': 5e-4,
             'name': 'cnn_backbone'},
            {'params': [p for p in self.cls_head.parameters()
                        if p.requires_grad],
             'lr': max_lr * 0.5, 'weight_decay': 1e-4,
             'name': 'cls_head'},
            {'params': [p for p in self.dir_head.parameters()
                        if p.requires_grad],
             'lr': max_lr, 'weight_decay': 1e-4,
             'name': 'dir_head'},
            {'params': other_params,
             'lr': max_lr, 'weight_decay': 5e-3,
             'name': 'other'},
        ]

        if self.use_hourly and hourly_ids:
            groups.append({
                'params': [p for p in self.hourly_enc.parameters()
                           if p.requires_grad],
                'lr': max_lr * 0.1, 'weight_decay': 5e-4,
                'name': 'hourly'})

        if self.use_kronos:
            kb_params = [p for p in self.kronos.get_backbone_params()
                         if p.requires_grad]
            if kb_params:
                groups.append({
                    'params': kb_params,
                    'lr': 1e-5,
                    'weight_decay': 1e-4,
                    'name': 'kronos_backbone'})
            ka_params = list(self.kronos.get_adapter_params())
            ka_params += list(self.kronos.grn.parameters())
            ka_params = [p for p in ka_params if p.requires_grad]
            if ka_params:
                groups.append({
                    'params': ka_params,
                    'lr': max_lr * 0.1,
                    'weight_decay': 1e-4,
                    'name': 'kronos_adapter'})

        return groups

    def forward(
        self,
        imgs:   Dict[int, torch.Tensor],
        nums:   Optional[Dict[int, torch.Tensor]],
        ctx:    Optional[torch.Tensor] = None,
        hourly: Optional[torch.Tensor] = None,
    ):
        feats = []

        for W in SCALES:
            x = imgs[W].float()
            feats.append(self.backbones[str(W)](x))

        long_W = max(SCALES)
        if nums is not None and long_W in nums:
            x_long = nums[long_W].float()
            x_long = self.wavelet(x_long)
            feats.append(self.seq_branch(x_long))
        else:
            B = imgs[min(SCALES)].shape[0]
            dev = imgs[min(SCALES)].device
            feats.append(torch.zeros(B, TRUNK_OUT, device=dev))

        for W in SCALES:
            if W < long_W and nums is not None and W in nums:
                feats.append(self.num_stats[str(W)](nums[W].float()))

        if self.use_hourly and hourly is not None:
            feats.append(self.hourly_enc(hourly.float()))

        if self.ctx_dim > 0 and ctx is not None:
            feats.append(self.ctx_proj(ctx.float()))

        if self.use_kronos:
            if nums is not None and long_W in nums:
                nums_long = nums[long_W].float()
            else:
                B   = imgs[min(SCALES)].shape[0]
                dev = imgs[min(SCALES)].device
                nums_long = torch.zeros(B, long_W, 37, device=dev)
            short_W    = min(SCALES)
            imgs_short = imgs[short_W].float()
            kronos_feat = self.kronos(nums_long, imgs_short)
            feats.append(kronos_feat)

        feats = [f.nan_to_num(nan=0., posinf=10., neginf=-10.) for f in feats]
        h = self.vsn(feats)

        logits    = self.cls_head(h)
        ohlc      = self.ohlc_head(h)
        aux       = self.aux_head(h)
        dir_logit = self.dir_head(h)            # NEW v4.1
        return logits, ohlc, aux, dir_logit

    def init_kronos(self, device: torch.device):
        if self.use_kronos:
            self.kronos.extractor.load_now(device)