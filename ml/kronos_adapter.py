# ml/kronos_adapter.py
"""Kronos Adapter v1.1 — интеграция Chronos/Kronos как backbone.

Изменения v1.1:
- _input_proj создаётся в load_now() (ДО создания оптимизатора),
  чтобы его параметры попадали в param_groups.
- get_adapter_params() всегда возвращает _input_proj если он создан.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Optional, Tuple

try:
    from chronos import ChronosPipeline, ChronosConfig, ChronosModel
    HAS_KRONOS = True
except ImportError:
    try:
        from transformers import AutoModel, AutoConfig
        HAS_KRONOS = False
        HAS_HF_CHRONOS = True
    except ImportError:
        HAS_KRONOS = False
        HAS_HF_CHRONOS = False

KRONOS_OUT_DIM = 128


class KronosInputProjector(nn.Module):
    def __init__(
        self,
        n_indicator_cols: int = 37,
        img_channels: int = 4,
        target_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_seq_len = target_seq_len

        self.ind_weights = nn.Parameter(torch.zeros(n_indicator_cols))
        with torch.no_grad():
            self.ind_weights[-1] = 2.0  # close_rel
            self.ind_weights[0] = 0.5   # ema9
        self.ind_scale = nn.Parameter(torch.ones(1))

        self.img_proj = nn.Sequential(
            nn.Conv1d(img_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.mix = nn.Parameter(torch.tensor([0.7, 0.3]))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(1)

    def forward(self, nums: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
        B = nums.shape[0]
        w = torch.softmax(self.ind_weights, dim=0)
        ind_series = (nums * w.unsqueeze(0).unsqueeze(0)).sum(-1)
        ind_series = ind_series * self.ind_scale

        img_series = self.img_proj(imgs).squeeze(1)

        ind_resampled = F.interpolate(
            ind_series.unsqueeze(1),
            size=self.target_seq_len,
            mode='linear',
            align_corners=False,
        ).squeeze(1)

        img_resampled = F.interpolate(
            img_series.unsqueeze(1),
            size=self.target_seq_len,
            mode='linear',
            align_corners=False,
        ).squeeze(1)

        mix_w = torch.softmax(self.mix, dim=0)
        combined = mix_w[0] * ind_resampled + mix_w[1] * img_resampled

        mean = combined.mean(dim=-1, keepdim=True)
        std  = combined.std(dim=-1, keepdim=True).clamp(min=1e-6)
        combined = (combined - mean) / std

        return self.drop(combined)


class KronosFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-tiny",
        n_unfreeze_layers: int = 2,
        use_grad_checkpoint: bool = True,
        out_dim: int = KRONOS_OUT_DIM,
        kronos_seq_len: int = 64,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_grad_checkpoint = use_grad_checkpoint
        self.out_dim = out_dim
        self.kronos_seq_len = kronos_seq_len
        self._loaded = False

        self.encoder = None
        self.embed_dim = None
        self._out_proj = None
        self._input_proj = None
        self._n_unfreeze = n_unfreeze_layers
        self._is_fallback = False

    def _lazy_load(self, device: torch.device):
        if self._loaded:
            return

        print(f"\n  [Kronos] Загрузка {self.model_name}...")

        try:
            if HAS_KRONOS:
                self._load_via_chronos(device)
            elif HAS_HF_CHRONOS:
                self._load_via_hf(device)
            else:
                self._load_fallback(device)
        except Exception as e:
            print(f"  [Kronos] Ошибка загрузки: {e} → fallback режим")
            self._load_fallback(device)

        # v1.1: создаём _input_proj ЗДЕСЬ, до создания оптимизатора
        if not self._is_fallback:
            self._input_proj = nn.Linear(1, self.embed_dim, bias=True).to(
                device=device, dtype=torch.float32)
            nn.init.kaiming_normal_(
                self._input_proj.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(self._input_proj.bias)
            self.add_module('_input_proj_module', self._input_proj)

        self._loaded = True
        print(f"  [Kronos] Загружен: embed_dim={self.embed_dim}, "
              f"out_dim={self.out_dim}")

    def _load_via_chronos(self, device: torch.device):
        pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        model = pipeline.model
        self.encoder = model.model.encoder
        self.embed_dim = model.model.config.d_model
        self._tokenizer = pipeline.tokenizer
        self._freeze_and_setup(device)

    def _load_via_hf(self, device: torch.device):
        config = AutoConfig.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )
        if hasattr(model, 'encoder'):
            self.encoder = model.encoder.to(device)
            self.embed_dim = config.d_model
        else:
            self.encoder = model.to(device)
            self.embed_dim = getattr(config, 'd_model',
                                     getattr(config, 'hidden_size', 256))
        self._freeze_and_setup(device)

    def _load_fallback(self, device: torch.device):
        print("  [Kronos] Используется fallback Transformer encoder")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.encoder = self.encoder.to(device)
        self.embed_dim = 64
        self._is_fallback = True

        self._out_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.GELU(),
        ).to(device)

    def _freeze_and_setup(self, device: torch.device):
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        if hasattr(self.encoder, 'block'):
            blocks = list(self.encoder.block)
            n_to_unfreeze = min(self._n_unfreeze, len(blocks))
            for block in blocks[-n_to_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad_(True)
            if hasattr(self.encoder, 'final_layer_norm'):
                for param in self.encoder.final_layer_norm.parameters():
                    param.requires_grad_(True)
        elif hasattr(self.encoder, 'layers'):
            layers = list(self.encoder.layers)
            n_to_unfreeze = min(self._n_unfreeze, len(layers))
            for layer in layers[-n_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad_(True)

        if self.use_grad_checkpoint and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()

        frozen = sum(1 for p in self.encoder.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.encoder.parameters() if p.requires_grad)
        print(f"  [Kronos] Заморожено: {frozen} параметров, "
              f"обучаемо: {trainable} параметров")

        self._out_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.GELU(),
        ).to(device)
        self._is_fallback = False

    def _input_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fallback:
            return x.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        # v1.1: _input_proj уже создан в _lazy_load
        return self._input_proj(x.float().unsqueeze(-1))

    def _run_encoder(self, embeds: torch.Tensor) -> torch.Tensor:
        if self._is_fallback:
            if self.use_grad_checkpoint and self.training:
                return checkpoint(self.encoder, embeds, use_reentrant=False)
            return self.encoder(embeds)

        if self.use_grad_checkpoint and self.training:
            encoder_ref = self.encoder

            def _t5_forward(inp_embeds):
                with torch.amp.autocast('cuda', enabled=False):
                    return encoder_ref(
                        inputs_embeds=inp_embeds.float(),
                        attention_mask=None,
                        return_dict=True,
                    ).last_hidden_state

            out = checkpoint(_t5_forward, embeds, use_reentrant=False)
        else:
            out = self.encoder(
                inputs_embeds=embeds,
                attention_mask=None,
                return_dict=True,
            ).last_hidden_state
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        self._lazy_load(device)

        if self.encoder is not None and next(self.encoder.parameters()).device != device:
            self.encoder = self.encoder.to(device)
        if self._out_proj is not None and next(self._out_proj.parameters()).device != device:
            self._out_proj = self._out_proj.to(device)
        if self._input_proj is not None and next(self._input_proj.parameters()).device != device:
            self._input_proj = self._input_proj.to(device)

        x_typed = x.float()
        embeds = self._input_embedding(x_typed)
        hidden = self._run_encoder(embeds)

        cls_token = hidden[:, 0, :]
        mean_pool = hidden.mean(dim=1)
        pooled = 0.5 * cls_token + 0.5 * mean_pool
        pooled = pooled.clamp(-10., 10.)

        out = self._out_proj(pooled.float())
        return out.nan_to_num(nan=0., posinf=5., neginf=-5.)

    def load_now(self, device: torch.device):
        self._lazy_load(device)


class KronosAdapter(nn.Module):
    def __init__(
        self,
        n_indicator_cols: int = 37,
        img_channels: int = 4,
        kronos_seq_len: int = 64,
        model_name: str = "amazon/chronos-t5-tiny",
        n_unfreeze_layers: int = 2,
        use_grad_checkpoint: bool = True,
        out_dim: int = KRONOS_OUT_DIM,
    ):
        super().__init__()
        self.projector = KronosInputProjector(
            n_indicator_cols=n_indicator_cols,
            img_channels=img_channels,
            target_seq_len=kronos_seq_len,
        )
        self.extractor = KronosFeatureExtractor(
            model_name=model_name,
            n_unfreeze_layers=n_unfreeze_layers,
            use_grad_checkpoint=use_grad_checkpoint,
            out_dim=out_dim,
            kronos_seq_len=kronos_seq_len,
        )

        from ml.multiscale_cnn_v3 import GRN, TRUNK_OUT
        self.grn = GRN(out_dim, TRUNK_OUT, dropout=0.2)

    def forward(self, nums: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
        series = self.projector(nums, imgs)
        kronos_feat = self.extractor(series)
        return self.grn(kronos_feat)

    def get_backbone_params(self):
        return list(self.extractor.encoder.parameters()) if self.extractor.encoder else []

    def get_adapter_params(self):
        """Параметры адаптера (НЕ включая grn — его обрабатываем отдельно)."""
        params = list(self.projector.parameters())
        if self.extractor._out_proj is not None:
            params += list(self.extractor._out_proj.parameters())
        if self.extractor._input_proj is not None:
            params += list(self.extractor._input_proj.parameters())
        return params