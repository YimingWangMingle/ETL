from __future__ import annotations

import numpy as np
import torch

from etl_sar.lattice.distributions import (
    LatticeNoiseDistribution,
    LatticeStateDependentNoiseDistribution,
)


def test_non_state_dependent_covariance_matches_official_equation() -> None:
    distribution = LatticeNoiseDistribution(action_dim=2)
    mean_net, _ = distribution.proba_distribution_net(
        latent_dim=3, log_std_init=np.log(0.5)
    )
    with torch.no_grad():
        mean_net.weight.copy_(
            torch.tensor([[1.0, 2.0, -1.0], [0.5, -1.0, 3.0]])
        )
    mean = torch.tensor([[0.1, -0.2]])
    log_std = torch.log(torch.tensor([0.4, 0.6, 0.3, 0.5, 0.7]))

    distribution.proba_distribution(mean, log_std)

    std = log_std.exp() * 0.5
    action_variance = std[:2].square()
    latent_variance = std[2:].square()
    expected = (mean_net.weight * latent_variance).matmul(mean_net.weight.T)
    expected.diagonal().add_(action_variance)
    torch.testing.assert_close(distribution.distribution.covariance_matrix[0], expected)


def test_state_dependent_covariance_matches_official_equation() -> None:
    distribution = LatticeStateDependentNoiseDistribution(
        action_dim=2,
        full_std=True,
        use_expln=False,
        std_clip=(1e-3, 10.0),
        std_reg=0.1,
        alpha=1.0,
    )
    mean_net, log_std = distribution.proba_distribution_net(
        latent_dim=3, log_std_init=0.0
    )
    with torch.no_grad():
        mean_net.weight.copy_(
            torch.tensor([[1.0, 0.25, -0.5], [-0.75, 0.5, 1.25]])
        )
    latent = torch.tensor([[0.2, -0.4, 0.8]])
    mean = mean_net(latent)

    distribution.proba_distribution(mean, log_std, latent)

    corr_std, ind_std = distribution.get_std(log_std)
    corr_variance = latent.square().matmul(corr_std.square())
    ind_variance = latent.square().matmul(ind_std.square()) + 0.1**2
    expected = (mean_net.weight * corr_variance[:, None, :]).matmul(mean_net.weight.T)
    expected[:, range(2), range(2)] += ind_variance
    torch.testing.assert_close(distribution.distribution.covariance_matrix, expected)


def test_fixed_seed_sampling_is_reproducible() -> None:
    first = LatticeNoiseDistribution(action_dim=2)
    second = LatticeNoiseDistribution(action_dim=2)
    first_net, _ = first.proba_distribution_net(3, 0.0)
    second_net, _ = second.proba_distribution_net(3, 0.0)
    second_net.load_state_dict(first_net.state_dict())
    mean = torch.zeros(1, 2)
    log_std = torch.full((5,), -0.5)
    first.proba_distribution(mean, log_std)
    second.proba_distribution(mean, log_std)

    torch.manual_seed(91)
    first_action = first.sample()
    torch.manual_seed(91)
    second_action = second.sample()

    torch.testing.assert_close(first_action, second_action, rtol=0, atol=0)
    torch.testing.assert_close(first.log_prob(first_action), second.log_prob(second_action))
