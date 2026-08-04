from __future__ import annotations

import torch

from etl_sar.gmvae import GMVAE, gmvae_loss


def test_gmvae_shapes_match_etl_protocol() -> None:
    model = GMVAE(action_dim=39, latent_dim=20, components=20, hidden_dims=(64, 64))
    actions = torch.rand(8, 39) * 2 - 1

    output = model(actions)

    assert output.reconstruction.shape == actions.shape
    assert output.component_probs.shape == (8, 20)
    assert output.q_mean.shape == (8, 20, 20)
    assert output.q_logvar.shape == (8, 20, 20)
    assert output.latent.shape == (8, 20)


def test_gmvae_loss_updates_all_trainable_paths() -> None:
    torch.manual_seed(3)
    model = GMVAE(action_dim=39, latent_dim=20, components=20, hidden_dims=(32,))
    actions = torch.rand(8, 39) * 2 - 1

    output = model(actions)
    loss = gmvae_loss(output, actions)
    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert loss.reconstruction.item() >= 0
    assert loss.latent_kl.item() >= 0
    assert loss.categorical_kl.item() >= 0
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert missing == []


def test_decoder_accepts_twenty_dimensional_latent_actions() -> None:
    model = GMVAE(action_dim=80, latent_dim=20, components=20, hidden_dims=(32,))

    decoded = model.decode(torch.zeros(4, 20))

    assert decoded.shape == (4, 80)
    assert torch.all(decoded <= 1.0)
    assert torch.all(decoded >= -1.0)


def test_seeded_forward_is_reproducible() -> None:
    model = GMVAE(action_dim=8, latent_dim=20, components=20, hidden_dims=(16,))
    actions = torch.linspace(-1, 1, 32).reshape(4, 8)

    torch.manual_seed(11)
    first = model(actions).reconstruction
    torch.manual_seed(11)
    second = model(actions).reconstruction

    torch.testing.assert_close(first, second)
