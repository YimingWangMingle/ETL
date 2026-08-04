from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mlp(dimensions: Sequence[int], *, final_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(input_dim, output_dim))
        if index < len(dimensions) - 2:
            layers.append(nn.ReLU())
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class GMVAEOutput:
    reconstruction: Tensor
    latent: Tensor
    component_probs: Tensor
    component_log_probs: Tensor
    q_mean: Tensor
    q_logvar: Tensor
    prior_mean: Tensor
    prior_logvar: Tensor


@dataclass(frozen=True)
class GMVAELoss:
    total: Tensor
    reconstruction: Tensor
    latent_kl: Tensor
    categorical_kl: Tensor


class GMVAE(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        latent_dim: int = 20,
        components: int = 20,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.components = components
        trunk_dim = hidden_dims[-1]
        self.encoder = _mlp((action_dim, *hidden_dims))
        self.component_logits = nn.Linear(trunk_dim, components)
        self.posterior_mean = nn.Linear(trunk_dim, components * latent_dim)
        self.posterior_logvar = nn.Linear(trunk_dim, components * latent_dim)
        self.prior_mean = nn.Parameter(torch.zeros(components, latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(components, latent_dim))
        self.decoder = _mlp(
            (latent_dim, *reversed(hidden_dims), action_dim),
            final_activation=nn.Tanh(),
        )

    def forward(self, actions: Tensor) -> GMVAEOutput:
        features = self.encoder(actions)
        logits = self.component_logits(features)
        component_log_probs = F.log_softmax(logits, dim=-1)
        component_probs = component_log_probs.exp()
        batch_size = actions.shape[0]
        q_mean = self.posterior_mean(features).reshape(
            batch_size,
            self.components,
            self.latent_dim,
        )
        q_logvar = self.posterior_logvar(features).reshape_as(q_mean).clamp(-12.0, 12.0)
        noise = torch.randn_like(q_mean)
        component_latents = q_mean + noise * torch.exp(0.5 * q_logvar)
        latent = (component_probs.unsqueeze(-1) * component_latents).sum(dim=1)
        reconstruction = self.decode(latent)
        return GMVAEOutput(
            reconstruction=reconstruction,
            latent=latent,
            component_probs=component_probs,
            component_log_probs=component_log_probs,
            q_mean=q_mean,
            q_logvar=q_logvar,
            prior_mean=self.prior_mean.unsqueeze(0).expand_as(q_mean),
            prior_logvar=self.prior_logvar.unsqueeze(0).expand_as(q_logvar),
        )

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)


def gmvae_loss(
    output: GMVAEOutput,
    target: Tensor,
    *,
    reconstruction_weight: float = 1.0,
    latent_kl_weight: float = 1.0,
    categorical_kl_weight: float = 1.0,
) -> GMVAELoss:
    reconstruction = F.mse_loss(output.reconstruction, target)
    variance_ratio = torch.exp(output.q_logvar - output.prior_logvar)
    mean_term = (
        output.q_mean - output.prior_mean
    ).square() * torch.exp(-output.prior_logvar)
    component_kl = 0.5 * (
        output.prior_logvar
        - output.q_logvar
        + variance_ratio
        + mean_term
        - 1.0
    ).sum(dim=-1)
    latent_kl = (output.component_probs * component_kl).sum(dim=-1).mean()
    uniform_log_probability = -math.log(output.component_probs.shape[-1])
    categorical_kl = (
        output.component_probs
        * (output.component_log_probs - uniform_log_probability)
    ).sum(dim=-1).mean()
    total = (
        reconstruction_weight * reconstruction
        + latent_kl_weight * latent_kl
        + categorical_kl_weight * categorical_kl
    )
    return GMVAELoss(
        total=total,
        reconstruction=reconstruction,
        latent_kl=latent_kl,
        categorical_kl=categorical_kl,
    )
