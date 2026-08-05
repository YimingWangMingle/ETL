from __future__ import annotations

from typing import Any

import torch
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, SACPolicy
from torch import nn

from etl_sar.lattice.distributions import (
    LatticeNoiseDistribution,
    LatticeStateDependentNoiseDistribution,
    SquashedLatticeNoiseDistribution,
)

try:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
except ImportError as error:  # pragma: no cover - dependency message
    raise ImportError(
        "Lattice Hand training requires sb3-contrib; install project dependencies"
    ) from error


class LatticeRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """SB3 2.x compatibility port of the official recurrent Lattice policy."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_lattice: bool = True,
        std_clip: tuple[float, float] = (1e-3, 10),
        expln_eps: float = 1e-6,
        std_reg: float = 0,
        alpha: float = 1,
        **kwargs: Any,
    ) -> None:
        self.use_lattice = use_lattice
        self.std_clip = std_clip
        self.expln_eps = expln_eps
        self.std_reg = std_reg
        self.alpha = alpha
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if use_lattice:
            if self.use_sde:
                self.dist_kwargs.update(
                    {
                        "epsilon": expln_eps,
                        "std_clip": std_clip,
                        "std_reg": std_reg,
                        "alpha": alpha,
                    }
                )
                self.action_dist = LatticeStateDependentNoiseDistribution(
                    get_action_dim(self.action_space), **self.dist_kwargs
                )
            else:
                self.action_dist = LatticeNoiseDistribution(
                    get_action_dim(self.action_space)
                )
            self._build(lr_schedule)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            use_lattice=self.use_lattice,
            std_clip=self.std_clip,
            expln_eps=self.expln_eps,
            std_reg=self.std_reg,
            alpha=self.alpha,
        )
        return data


class LatticeActor(Actor):
    """SB3 2.x compatibility port of the official SAC Lattice actor."""

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        use_lattice: bool = False,
        std_clip: tuple[float, float] = (1e-3, 10),
        expln_eps: float = 1e-6,
        std_reg: float = 0,
        alpha: float = 1,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )
        self.use_lattice = use_lattice
        self.std_clip = std_clip
        self.expln_eps = expln_eps
        self.std_reg = std_reg
        self.alpha = alpha
        if use_lattice:
            last_layer_dim = net_arch[-1] if net_arch else features_dim
            action_dim = get_action_dim(self.action_space)
            if self.use_sde:
                self.action_dist = LatticeStateDependentNoiseDistribution(
                    action_dim,
                    full_std=full_std,
                    use_expln=use_expln,
                    squash_output=True,
                    learn_features=True,
                    epsilon=expln_eps,
                    std_clip=std_clip,
                    std_reg=std_reg,
                    alpha=alpha,
                )
                self.mu, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=last_layer_dim,
                    latent_sde_dim=last_layer_dim,
                    log_std_init=log_std_init,
                    clip_mean=clip_mean,
                )
            else:
                self.action_dist = SquashedLatticeNoiseDistribution(action_dim)
                self.mu, self.log_std = self.action_dist.proba_distribution_net(
                    last_layer_dim, log_std_init, state_dependent=True
                )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            use_lattice=self.use_lattice,
            std_clip=self.std_clip,
            expln_eps=self.expln_eps,
            std_reg=self.std_reg,
            alpha=self.alpha,
        )
        return data


class LatticeSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_lattice: bool = False,
        std_clip: tuple[float, float] = (1e-3, 10),
        expln_eps: float = 1e-6,
        std_reg: float = 0,
        use_sde: bool = False,
        alpha: float = 1,
        **kwargs: Any,
    ) -> None:
        self.lattice_kwargs = {
            "use_lattice": use_lattice,
            "expln_eps": expln_eps,
            "std_clip": std_clip,
            "std_reg": std_reg,
            "alpha": alpha,
        }
        super().__init__(
            observation_space, action_space, lr_schedule, use_sde=use_sde, **kwargs
        )
        self.actor_kwargs.update(self.lattice_kwargs)
        if use_lattice:
            self._build(lr_schedule)

    def make_actor(
        self, features_extractor: BaseFeaturesExtractor | None = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return LatticeActor(**actor_kwargs).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(self.lattice_kwargs)
        return data
