from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from etl_sar.action_model import ETLSARActionModel
from etl_sar.data import TrajectoryStore
from etl_sar.gmvae import GMVAE, gmvae_loss


@dataclass(frozen=True)
class UpdateStats:
    samples: int
    steps: int
    mean_loss: float


class RepresentationTrainer:
    def __init__(
        self,
        *,
        gmvae: GMVAE,
        action_model: ETLSARActionModel,
        learning_rate: float = 3e-4,
        decoder_learning_rate: float | None = None,
        anchor_weight: float = 1e-4,
        budget_weight: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gmvae = gmvae.to(self.device)
        self.action_model = action_model.to(self.device)
        self.anchor_weight = anchor_weight
        self.budget_weight = budget_weight
        decoder_parameters = list(self.action_model.decoder.parameters())
        decoder_ids = {id(parameter) for parameter in decoder_parameters}
        gmvae_parameters = [
            parameter for parameter in self.gmvae.parameters() if id(parameter) not in decoder_ids
        ]
        self.gmvae_optimizer = torch.optim.Adam(gmvae_parameters, lr=learning_rate)
        self.decoder_optimizer = torch.optim.Adam(
            decoder_parameters,
            lr=decoder_learning_rate or learning_rate,
        )
        self.sar_optimizer = torch.optim.Adam(
            self.action_model.synergy_head.parameters(),
            lr=learning_rate,
        )
        self.decoder_anchor = [
            parameter.detach().clone() for parameter in self.action_model.decoder.parameters()
        ]

    @staticmethod
    def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    def update_gmvae(
        self,
        store: TrajectoryStore,
        *,
        steps: int,
        batch_size: int,
    ) -> UpdateStats:
        actions = store.action_pool()
        if actions.shape[0] == 0:
            raise ValueError("cannot update GMVAE from an empty ETL action pool")
        tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        losses: list[float] = []
        self.gmvae.train()
        for _ in range(steps):
            indices = torch.randint(tensor.shape[0], (min(batch_size, tensor.shape[0]),))
            batch = tensor[indices]
            output = self.gmvae(batch)
            loss = gmvae_loss(output, batch).total
            self.gmvae_optimizer.zero_grad(set_to_none=True)
            self.decoder_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.gmvae_optimizer.step()
            self.decoder_optimizer.step()
            losses.append(float(loss.detach()))
        return UpdateStats(
            samples=int(actions.shape[0]),
            steps=steps,
            mean_loss=float(np.mean(losses)),
        )

    def fine_tune_decoder(self, *, latent: Tensor, executed_actions: Tensor) -> float:
        latent = latent.to(self.device)
        executed_actions = executed_actions.to(self.device)
        prediction = self.action_model.decoder(latent)
        reconstruction = F.mse_loss(prediction, executed_actions)
        anchor = sum(
            (parameter - initial).square().sum()
            for parameter, initial in zip(
                self.action_model.decoder.parameters(),
                self.decoder_anchor,
            )
        )
        loss = reconstruction + self.anchor_weight * anchor
        self.decoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.decoder_optimizer.step()
        return float(loss.detach())

    def train_sar_head(self, latent: Tensor, targets: Tensor, *, steps: int) -> float:
        latent = latent.to(self.device)
        targets = targets.to(self.device)
        basis = self.action_model.synergy_basis
        projection = basis @ torch.linalg.pinv(basis)
        losses: list[float] = []
        for _ in range(steps):
            with torch.no_grad():
                etl_action = self.action_model.decoder(latent)
            coefficients = self.action_model.synergy_head(latent)
            raw = (
                self.action_model.enabled_scale
                * coefficients
                @ basis.transpose(0, 1)
            )
            etl_norm = etl_action.norm(dim=-1, keepdim=True)
            raw_norm = raw.norm(dim=-1, keepdim=True).clamp_min(
                self.action_model.epsilon
            )
            gate = torch.minimum(
                torch.ones_like(raw_norm),
                self.action_model.rho * etl_norm / raw_norm,
            )
            bounded = raw * gate
            target_residual = (targets - etl_action) @ projection.transpose(0, 1)
            reconstruction = F.mse_loss(bounded, target_residual)
            raw_ratio = raw_norm / etl_norm.clamp_min(self.action_model.epsilon)
            budget = torch.relu(raw_ratio - self.action_model.rho).square().mean()
            loss = reconstruction + self.budget_weight * budget
            self.sar_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.sar_optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))
