from __future__ import annotations

import numpy as np
import torch
from torch import nn

from etl_sar.action_model import ETLSARActionModel
from etl_sar.data import TrajectoryEpisode, TrajectoryStore
from etl_sar.gmvae import GMVAE
from etl_sar.representation import RepresentationTrainer
from etl_sar.types import Limb


def make_episode(length: int = 6) -> TrajectoryEpisode:
    rng = np.random.default_rng(9)
    actions = rng.uniform(-1, 1, size=(length, 6)).astype(np.float32)
    return TrajectoryEpisode(
        limb=Limb.HAND,
        source_task="reorient8",
        observations=np.zeros((length, 4), dtype=np.float32),
        sampled_actions=actions,
        executed_actions=actions,
        rewards=np.ones(length, dtype=np.float32),
        next_observations=np.zeros((length, 4), dtype=np.float32),
        terminated=np.zeros(length, dtype=np.bool_),
        truncated=np.zeros(length, dtype=np.bool_),
        behaviors=np.zeros((length, 2), dtype=np.float32),
        success=True,
    )


def make_trainer() -> RepresentationTrainer:
    gmvae = GMVAE(action_dim=6, latent_dim=20, components=20, hidden_dims=(16,))
    action_model = ETLSARActionModel(
        decoder=gmvae.decoder,
        synergy_basis=torch.randn(6, 20),
        latent_dim=20,
        action_low=-torch.ones(6),
        action_high=torch.ones(6),
    )
    return RepresentationTrainer(gmvae=gmvae, action_model=action_model, learning_rate=1e-3)


def test_gmvae_update_consumes_current_action_pool(tmp_path) -> None:
    trainer = make_trainer()
    store = TrajectoryStore(tmp_path, limb=Limb.HAND, source_task="reorient8")
    store.append_episode(make_episode(length=6))

    stats = trainer.update_gmvae(store, steps=2, batch_size=4)

    assert stats.samples == 6
    assert stats.steps == 2
    assert np.isfinite(stats.mean_loss)


def test_decoder_finetune_does_not_update_synergy_basis() -> None:
    trainer = make_trainer()
    before_basis = trainer.action_model.synergy_basis.detach().clone()
    before_decoder = [parameter.detach().clone() for parameter in trainer.action_model.decoder.parameters()]

    loss = trainer.fine_tune_decoder(
        latent=torch.randn(8, 20),
        executed_actions=torch.rand(8, 6) * 2 - 1,
    )

    assert loss >= 0
    assert torch.equal(before_basis, trainer.action_model.synergy_basis)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_decoder, trainer.action_model.decoder.parameters())
    )


def test_optimizer_parameter_sets_are_disjoint() -> None:
    trainer = make_trainer()

    gmvae_ids = trainer.optimizer_parameter_ids(trainer.gmvae_optimizer)
    decoder_ids = trainer.optimizer_parameter_ids(trainer.decoder_optimizer)
    sar_ids = trainer.optimizer_parameter_ids(trainer.sar_optimizer)

    assert gmvae_ids.isdisjoint(decoder_ids)
    assert gmvae_ids.isdisjoint(sar_ids)
    assert decoder_ids.isdisjoint(sar_ids)


def test_sar_head_update_respects_hard_budget() -> None:
    trainer = make_trainer()
    latent = torch.randn(12, 20)
    targets = torch.rand(12, 6) * 2 - 1

    trainer.train_sar_head(latent, targets, steps=2)
    output = trainer.action_model(latent)

    assert torch.all(output.contribution_ratio <= 0.200001)
    assert trainer.action_model.synergy_basis.grad is None
