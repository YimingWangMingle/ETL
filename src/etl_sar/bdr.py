from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


class StateEncoder(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        dimensions = (observation_dim, *hidden_dims, latent_dim)
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(observations)


@dataclass(frozen=True)
class BehaviorMetricLoss:
    total: Tensor
    metric: Tensor
    discriminability: Tensor


def sample_unit_directions(count: int, dimension: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(count, dimension, generator=generator)
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def directional_bonus(phi_s: Tensor, phi_next: Tensor, direction: Tensor) -> Tensor:
    return ((phi_next - phi_s) * direction).sum(dim=-1)


def discriminability_hinge(
    embeddings: Tensor,
    rewards: Tensor,
    *,
    epsilon: float,
    margin: float,
) -> Tensor:
    if embeddings.shape[0] % 2:
        raise ValueError("BDR requires an even number of paired embeddings")
    distances = torch.linalg.vector_norm(
        embeddings[0::2] - embeddings[1::2],
        dim=-1,
    )
    reward_gaps = (rewards[0::2] - rewards[1::2]).abs()
    active = (reward_gaps < epsilon).to(embeddings.dtype)
    return (active * torch.relu(margin - distances).square()).mean()


def behavior_metric_loss(
    embeddings: Tensor,
    next_embeddings: Tensor,
    rewards: Tensor,
    *,
    gamma: float,
    bdr_weight: float,
    epsilon: float,
    margin: float,
) -> BehaviorMetricLoss:
    if embeddings.shape[0] % 2:
        raise ValueError("behavior metric loss requires paired samples")
    current_distance = torch.linalg.vector_norm(
        embeddings[0::2] - embeddings[1::2],
        dim=-1,
    )
    next_distance = torch.linalg.vector_norm(
        next_embeddings[0::2] - next_embeddings[1::2],
        dim=-1,
    )
    reward_gap = (rewards[0::2] - rewards[1::2]).abs()
    target = reward_gap + gamma * next_distance.detach()
    metric = 0.5 * (current_distance - target).square().mean()
    discriminability = discriminability_hinge(
        embeddings,
        rewards,
        epsilon=epsilon,
        margin=margin,
    )
    total = metric + bdr_weight * discriminability
    return BehaviorMetricLoss(total=total, metric=metric, discriminability=discriminability)
