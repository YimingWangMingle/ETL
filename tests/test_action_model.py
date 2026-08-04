from __future__ import annotations

import torch
from torch import nn

from etl_sar.action_model import ETLSARActionModel


def make_action_model(*, enabled_scale: float = 1.0) -> ETLSARActionModel:
    torch.manual_seed(2)
    decoder = nn.Sequential(nn.Linear(20, 39), nn.Tanh())
    basis = torch.randn(39, 20)
    model = ETLSARActionModel(
        decoder=decoder,
        synergy_basis=basis,
        latent_dim=20,
        action_low=-torch.ones(39),
        action_high=torch.ones(39),
        rho=0.20,
        enabled_scale=enabled_scale,
    )
    with torch.no_grad():
        model.synergy_head.weight.fill_(10.0)
    return model


def test_sar_residual_never_exceeds_twenty_percent() -> None:
    model = make_action_model()

    result = model(torch.randn(16, 20))

    ratio = result.sar_action.norm(dim=-1) / result.etl_action.norm(dim=-1).clamp_min(1e-8)
    assert torch.all(ratio <= 0.200001)
    torch.testing.assert_close(result.contribution_ratio, ratio)


def test_lambda_zero_is_exactly_pure_etl() -> None:
    model = make_action_model(enabled_scale=0.0)
    latent = torch.randn(4, 20)

    result = model(latent)

    torch.testing.assert_close(result.sar_action, torch.zeros_like(result.sar_action))
    torch.testing.assert_close(result.full_action, result.etl_action)


def test_synergy_basis_is_frozen_and_action_is_bounded() -> None:
    model = make_action_model()

    result = model(torch.full((8, 20), 100.0))
    result.full_action.sum().backward()

    assert "synergy_basis" in dict(model.named_buffers())
    assert model.synergy_basis.grad is None
    assert torch.all(result.full_action <= 1.0)
    assert torch.all(result.full_action >= -1.0)
