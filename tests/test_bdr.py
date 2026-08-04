from __future__ import annotations

import pytest
import torch

from etl_sar.bdr import (
    StateEncoder,
    behavior_metric_loss,
    directional_bonus,
    discriminability_hinge,
    sample_unit_directions,
)


def test_directional_bonus_matches_etl_equation_one() -> None:
    phi_s = torch.tensor([[0.0, 1.0]])
    phi_next = torch.tensor([[2.0, 1.0]])
    direction = torch.tensor([[1.0, 0.0]])

    assert directional_bonus(phi_s, phi_next, direction).item() == pytest.approx(2.0)


def test_sampled_directions_are_unit_vectors() -> None:
    directions = sample_unit_directions(count=20, dimension=20, seed=7)

    assert directions.shape == (20, 20)
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones(20))


def test_bdr_hinge_penalizes_collapsed_low_reward_pair() -> None:
    collapsed = torch.zeros(2, 3)
    reward = torch.zeros(2)

    loss = discriminability_hinge(
        collapsed,
        reward,
        epsilon=0.1,
        margin=1.0,
    )

    assert loss.item() == pytest.approx(1.0)


def test_bdr_hinge_ignores_large_reward_gap() -> None:
    collapsed = torch.zeros(2, 3)
    reward = torch.tensor([0.0, 1.0])

    loss = discriminability_hinge(
        collapsed,
        reward,
        epsilon=0.1,
        margin=1.0,
    )

    assert loss.item() == pytest.approx(0.0)


def test_behavior_metric_loss_backpropagates_through_encoder() -> None:
    encoder = StateEncoder(observation_dim=5, latent_dim=3, hidden_dims=(8,))
    observations = torch.randn(4, 5)
    next_observations = torch.randn(4, 5)
    rewards = torch.tensor([0.0, 0.0, 0.3, 0.8])
    embeddings = encoder(observations)
    next_embeddings = encoder(next_observations)

    loss = behavior_metric_loss(
        embeddings,
        next_embeddings,
        rewards,
        gamma=0.99,
        bdr_weight=0.1,
        epsilon=0.05,
        margin=0.5,
    )
    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert all(parameter.grad is not None for parameter in encoder.parameters())
