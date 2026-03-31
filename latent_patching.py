from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class LatentPatching(nn.Module):
    def __init__(self, model_dim: int, patch_size: int, patch_temperature: float):
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"latent_patch_size must be positive, got {patch_size}")
        if patch_temperature <= 0.0:
            raise ValueError(f"latent_patch_temperature must be positive, got {patch_temperature}")

        self.patch_size = patch_size
        self.gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(patch_temperature), dtype=torch.float32))
        self.predictor_norm = RMSNorm()
        self.predictor = CastedLinear(model_dim, model_dim, bias=False)

    def reshape(self, x: Tensor) -> Tensor:
        usable = (x.size(1) // self.patch_size) * self.patch_size
        if usable <= 0:
            return x[:, :0, :].reshape(x.size(0), 0, self.patch_size, x.size(-1))
        return x[:, :usable, :].reshape(x.size(0), usable // self.patch_size, self.patch_size, x.size(-1))

    def pool_decoder(self, patches: Tensor) -> Tensor:
        if patches.size(1) == 0:
            return patches[:, :, 0, :]
        if patches.size(2) == 1:
            return patches.squeeze(2)
        gate_inputs = F.rms_norm(patches, (patches.size(-1),))
        logits = (gate_inputs * self.gate.to(dtype=patches.dtype)[None, None, None, :]).sum(dim=-1)
        weights = F.softmax(logits * self.logit_scale.exp().to(dtype=patches.dtype), dim=-1)
        return (weights[..., None] * patches).sum(dim=2)

    def loss(self, encoder_latent: Tensor, decoder_latent: Tensor, loss_ref: Tensor) -> Tensor:
        encoder_patches = self.reshape(encoder_latent)
        decoder_patches = self.reshape(decoder_latent)
        if encoder_patches.size(1) <= 1 or decoder_patches.size(1) <= 1:
            return loss_ref.new_zeros(())

        pred_patches = self.pool_decoder(decoder_patches)
        pred_patches = self.predictor(self.predictor_norm(pred_patches))
        pred = F.normalize(pred_patches[:, :-1, :].float(), dim=-1)
        tgt = F.normalize(encoder_patches.mean(dim=2)[:, 1:, :].detach().float(), dim=-1)
        return (1.0 - F.cosine_similarity(pred, tgt, dim=-1)).mean()