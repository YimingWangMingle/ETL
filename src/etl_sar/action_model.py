from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ActionModelOutput:
    etl_action: Tensor
    raw_sar_action: Tensor
    sar_action: Tensor
    full_action: Tensor
    contribution_ratio: Tensor


class ETLSARActionModel(nn.Module):
    def __init__(
        self,
        *,
        decoder: nn.Module,
        synergy_basis: Tensor,
        latent_dim: int,
        action_low: Tensor,
        action_high: Tensor,
        rho: float = 0.20,
        enabled_scale: float = 1.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if synergy_basis.ndim != 2:
            raise ValueError("synergy basis must have shape [action_dim, components]")
        if action_low.shape != action_high.shape or action_low.numel() != synergy_basis.shape[0]:
            raise ValueError("action bounds must match the synergy action dimension")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.rho = float(rho)
        self.enabled_scale = float(enabled_scale)
        self.epsilon = float(epsilon)
        self.synergy_head = nn.Linear(latent_dim, synergy_basis.shape[1], bias=False)
        nn.init.zeros_(self.synergy_head.weight)
        self.register_buffer("synergy_basis", synergy_basis.detach().clone().float())
        self.register_buffer("action_low", action_low.detach().clone().float())
        self.register_buffer("action_high", action_high.detach().clone().float())

    def forward(self, latent: Tensor) -> ActionModelOutput:
        etl_action = self.decoder(latent)
        coefficients = self.synergy_head(latent)
        raw_sar_action = (
            self.enabled_scale * coefficients @ self.synergy_basis.transpose(0, 1)
        )
        etl_norm = etl_action.norm(dim=-1, keepdim=True)
        raw_norm = raw_sar_action.norm(dim=-1, keepdim=True).clamp_min(self.epsilon)
        max_scale = self.rho * etl_norm / raw_norm
        gate = torch.minimum(torch.ones_like(max_scale), max_scale)
        sar_action = raw_sar_action * gate
        full_action = torch.maximum(
            torch.minimum(etl_action + sar_action, self.action_high),
            self.action_low,
        )
        contribution_ratio = sar_action.norm(dim=-1) / etl_norm.squeeze(-1).clamp_min(
            self.epsilon
        )
        return ActionModelOutput(
            etl_action=etl_action,
            raw_sar_action=raw_sar_action,
            sar_action=sar_action,
            full_action=full_action,
            contribution_ratio=contribution_ratio,
        )
