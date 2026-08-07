from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from etl_sar.action_model import ETLSARActionModel
from etl_sar.bdr import StateEncoder, behavior_metric_loss
from etl_sar.dmc.config import DMCTransferConfig
from etl_sar.dmc.env import make_dmc_env
from etl_sar.dmc.matrix import DMCMethod
from etl_sar.envs import LatentActionWrapper, validate_environment
from etl_sar.exploration import DirectionalExplorationWrapper
from etl_sar.gmvae import GMVAE, gmvae_loss
from etl_sar.lattice.policies import LatticeSACPolicy
from etl_sar.representation import RepresentationTrainer


def common_sac_kwargs(*, seed: int, device: str = "auto") -> dict[str, Any]:
    """Return the shared SAC backbone used by every DMC target method."""
    return {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "use_sde": True,
        "sde_sample_freq": 1,
        "seed": seed,
        "device": device,
        "policy_kwargs": {
            "net_arch": {"pi": [256, 256], "qf": [256, 256]},
        },
    }


def lattice_sac_kwargs(*, seed: int, device: str = "auto") -> dict[str, Any]:
    kwargs = common_sac_kwargs(seed=seed, device=device)
    kwargs["policy_kwargs"] = {
        **kwargs["policy_kwargs"],
        "use_lattice": True,
        "use_expln": True,
        "log_std_init": 0.0,
        "std_clip": (1e-3, 10.0),
        "expln_eps": 1e-6,
        "std_reg": 0.0,
        "alpha": 1.0,
    }
    return kwargs


class SourceEpisodeStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
    ) -> Path:
        index = len(self.paths())
        destination = self.root / f"episode_{index:06d}.npz"
        np.savez_compressed(
            destination,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_observations=np.asarray(next_observations, dtype=np.float32),
        )
        return destination

    def paths(self) -> list[Path]:
        return sorted(self.root.glob("episode_*.npz"))

    def _episodes(self) -> list[tuple[float, np.ndarray]]:
        episodes: list[tuple[float, np.ndarray]] = []
        for path in self.paths():
            with np.load(path, allow_pickle=False) as data:
                episodes.append(
                    (float(np.sum(data["rewards"])), data["actions"].copy())
                )
        return episodes

    def action_pool(self) -> np.ndarray:
        episodes = self._episodes()
        if not episodes:
            raise ValueError("source episode store is empty")
        return np.concatenate([actions for _, actions in episodes], axis=0)

    def top_return_action_pool(self, fraction: float) -> np.ndarray:
        episodes = sorted(self._episodes(), key=lambda item: item[0], reverse=True)
        if not episodes:
            raise ValueError("source episode store is empty")
        count = max(1, int(math.ceil(len(episodes) * fraction)))
        return np.concatenate([actions for _, actions in episodes[:count]], axis=0)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in self.paths():
            digest.update(path.name.encode("ascii"))
            digest.update(path.read_bytes())
        return digest.hexdigest()


class SourceCollectionCallback(BaseCallback):
    def __init__(
        self,
        *,
        store: SourceEpisodeStore,
        state_encoder: StateEncoder,
    ) -> None:
        super().__init__(verbose=0)
        self.store = store
        self.state_encoder = state_encoder
        self.encoder_optimizer = torch.optim.Adam(state_encoder.parameters(), lr=1e-3)
        self.episode: list[dict[str, Any]] = []
        self.bdr_buffer: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))
        for info, done in zip(infos, dones):
            if "etl/current_observation" not in info:
                continue
            item = {
                "observation": np.asarray(
                    info["etl/current_observation"], dtype=np.float32
                ),
                "action": np.asarray(info["etl/executed_action"], dtype=np.float32),
                "reward": float(info["etl/extrinsic_reward"]),
                "next_observation": np.asarray(
                    info["etl/next_observation"], dtype=np.float32
                ),
            }
            self.episode.append(item)
            self.bdr_buffer.append(item)
            if len(self.bdr_buffer) >= 4:
                self._update_bdr()
            if done:
                self._flush_episode()
        return True

    def _update_bdr(self) -> None:
        batch = self.bdr_buffer[-4:]
        observations = torch.as_tensor(
            np.stack([item["observation"] for item in batch]), dtype=torch.float32
        )
        next_observations = torch.as_tensor(
            np.stack([item["next_observation"] for item in batch]), dtype=torch.float32
        )
        rewards = torch.as_tensor(
            [item["reward"] for item in batch], dtype=torch.float32
        )
        loss = behavior_metric_loss(
            self.state_encoder(observations),
            self.state_encoder(next_observations),
            rewards,
            gamma=0.99,
            bdr_weight=0.1,
            epsilon=0.05,
            margin=0.5,
        ).total
        self.encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_optimizer.step()
        self.bdr_buffer.clear()

    def _flush_episode(self) -> None:
        if not self.episode:
            return
        self.store.append(
            observations=np.stack([item["observation"] for item in self.episode]),
            actions=np.stack([item["action"] for item in self.episode]),
            rewards=np.asarray(
                [item["reward"] for item in self.episode], dtype=np.float32
            ),
            next_observations=np.stack(
                [item["next_observation"] for item in self.episode]
            ),
        )
        self.episode.clear()


def _fit_synergy_basis(actions: np.ndarray, components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if actions.ndim != 2 or min(actions.shape) < components:
        raise ValueError("not enough source actions for configured SAR components")
    scaler = StandardScaler().fit(actions)
    standardized = scaler.transform(actions)
    pca = PCA(n_components=components, random_state=seed).fit(standardized)
    pca_codes = pca.transform(standardized)
    ica = FastICA(
        n_components=components,
        whiten="unit-variance",
        random_state=seed,
        max_iter=2000,
        tol=1e-4,
    ).fit(pca_codes)
    raw_codes = ica.transform(pca_codes)
    scale = np.maximum(np.max(np.abs(raw_codes), axis=0), 1e-8)

    def inverse(codes: np.ndarray) -> np.ndarray:
        pca_values = ica.inverse_transform(codes * scale)
        return scaler.inverse_transform(pca.inverse_transform(pca_values))

    origin = inverse(np.zeros((1, components), dtype=np.float64))[0]
    basis = (inverse(np.eye(components, dtype=np.float64)) - origin).T
    return basis.astype(np.float32), pca.explained_variance_ratio_.copy()


def _fit_source_bundle(
    *,
    config: DMCTransferConfig,
    store: SourceEpisodeStore,
    state_encoder: StateEncoder,
    action_low: np.ndarray,
    action_high: np.ndarray,
    output: Path,
    device: str,
) -> Path:
    actions = store.action_pool().astype(np.float32)
    selected = store.top_return_action_pool(config.source_top_fraction).astype(
        np.float32
    )
    resolved_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device == "auto" else device
    torch_device = torch.device(resolved_device)
    torch.manual_seed(config.seed)
    gmvae = GMVAE(
        action_dim=actions.shape[1],
        latent_dim=config.etl_latent_dim,
        components=config.etl_components,
        hidden_dims=(256, 256),
    ).to(torch_device)
    optimizer = torch.optim.Adam(gmvae.parameters(), lr=3e-4)
    action_tensor = torch.as_tensor(actions, device=torch_device)
    for _ in range(config.representation_steps):
        indices = torch.randint(
            action_tensor.shape[0],
            (min(256, action_tensor.shape[0]),),
            device=torch_device,
        )
        output_value = gmvae(action_tensor[indices])
        loss = gmvae_loss(output_value, action_tensor[indices]).total
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    basis, explained_variance = _fit_synergy_basis(
        selected, config.sar_components, config.seed
    )
    action_model = ETLSARActionModel(
        decoder=gmvae.decoder,
        synergy_basis=torch.as_tensor(basis, device=torch_device),
        latent_dim=config.etl_latent_dim,
        action_low=torch.as_tensor(action_low, device=torch_device),
        action_high=torch.as_tensor(action_high, device=torch_device),
        rho=config.sar_rho,
        enabled_scale=1.0,
    )
    representation = RepresentationTrainer(
        gmvae=gmvae,
        action_model=action_model,
        learning_rate=3e-4,
        decoder_learning_rate=3e-5,
        device=torch_device,
    )
    rng = np.random.default_rng(config.seed)
    subset = selected[
        rng.choice(len(selected), size=min(len(selected), 4096), replace=False)
    ]
    subset_tensor = torch.as_tensor(subset, dtype=torch.float32, device=torch_device)
    with torch.no_grad():
        torch.manual_seed(config.seed)
        latent = gmvae(subset_tensor).latent
    representation.train_sar_head(latent, subset_tensor, steps=config.sar_steps)
    bundle = output / "representation_bundle.pt"
    torch.save(
        {
            "schema_version": 2,
            "domain": config.domain,
            "source_task": config.source_task,
            "action_dim": int(actions.shape[1]),
            "latent_dim": config.etl_latent_dim,
            "mixture_components": config.etl_components,
            "sar_components": config.sar_components,
            "rho": config.sar_rho,
            "hidden_dims": (256, 256),
            "gmvae": gmvae.state_dict(),
            "action_model": action_model.state_dict(),
            "state_encoder": state_encoder.state_dict(),
            "synergy_basis": basis,
            "synergy_explained_variance_ratio": explained_variance,
            "source_data_fingerprint": store.fingerprint(),
            "source_episode_count": len(store.paths()),
            "source_action_count": int(actions.shape[0]),
            "selected_source_action_count": int(selected.shape[0]),
        },
        bundle,
    )
    return bundle


def _write_complete(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_source_stage(
    config: DMCTransferConfig,
    *,
    run_dir: str | Path,
    device: str = "auto",
) -> Path:
    output = Path(run_dir)
    bundle = output / "representation_bundle.pt"
    complete = output / "complete.json"
    if complete.is_file() and bundle.is_file():
        return bundle
    output.mkdir(parents=True, exist_ok=True)
    probe = make_dmc_env(
        config.domain,
        config.source_task,
        seed=config.seed,
        action_repeat=config.action_repeat,
    )
    contract = validate_environment(probe)
    probe.close()
    state_encoder = StateEncoder(contract.observation_dim, config.etl_latent_dim)
    store = SourceEpisodeStore(output / "episodes")

    def build() -> gym.Env:
        base = make_dmc_env(
            config.domain,
            config.source_task,
            seed=config.seed,
            action_repeat=config.action_repeat,
        )
        return Monitor(
            DirectionalExplorationWrapper(
                base,
                state_encoder=state_encoder,
                direction_count=config.etl_components,
                bonus_scale=1.0,
                seed=config.seed,
            )
        )

    vector = DummyVecEnv([build])
    normalized = VecNormalize(vector, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = SAC(
        "MlpPolicy",
        normalized,
        tensorboard_log=str(output / "tensorboard"),
        verbose=1,
        **common_sac_kwargs(seed=config.seed, device=device),
    )
    model.learn(
        total_timesteps=config.source_budget,
        callback=SourceCollectionCallback(store=store, state_encoder=state_encoder),
    )
    model.save(output / "source_policy")
    model.save_replay_buffer(output / "source_replay_buffer.pkl")
    normalized.save(output / "source_vecnormalize.pkl")
    normalized.close()
    bundle = _fit_source_bundle(
        config=config,
        store=store,
        state_encoder=state_encoder,
        action_low=contract.action_low,
        action_high=contract.action_high,
        output=output,
        device=device,
    )
    _write_complete(
        complete,
        {
            "stage": "source",
            "domain": config.domain,
            "seed": config.seed,
            "transitions": config.source_budget,
            "bundle": bundle.name,
        },
    )
    return bundle


def load_action_model(
    bundle_path: str | Path,
    *,
    env: gym.Env,
    sar_scale: float,
    device: str = "auto",
) -> tuple[ETLSARActionModel, RepresentationTrainer]:
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    contract = validate_environment(env, expected_action_dim=int(payload["action_dim"]))
    gmvae = GMVAE(
        action_dim=contract.action_dim,
        latent_dim=int(payload["latent_dim"]),
        components=int(payload["mixture_components"]),
        hidden_dims=tuple(payload["hidden_dims"]),
    )
    gmvae.load_state_dict(payload["gmvae"])
    action_model = ETLSARActionModel(
        decoder=gmvae.decoder,
        synergy_basis=torch.as_tensor(payload["synergy_basis"]),
        latent_dim=int(payload["latent_dim"]),
        action_low=torch.as_tensor(contract.action_low),
        action_high=torch.as_tensor(contract.action_high),
        rho=float(payload["rho"]),
        enabled_scale=sar_scale,
    )
    action_model.load_state_dict(payload["action_model"])
    action_model.enabled_scale = float(sar_scale)
    representation = RepresentationTrainer(
        gmvae=gmvae,
        action_model=action_model,
        learning_rate=3e-4,
        decoder_learning_rate=3e-5,
        device="cpu",
    )
    return action_model, representation


@dataclass(frozen=True)
class ReturnSummary:
    environment_steps: int
    episodes: int
    mean_return: float
    median_return: float
    mean_episode_length: float


def evaluate_returns(
    model: SAC,
    training_normalizer: VecNormalize,
    *,
    env_factory: Callable[[int], gym.Env],
    seeds: tuple[int, ...],
    output_dir: str | Path,
    environment_steps: int,
) -> ReturnSummary:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode, seed in enumerate(seeds):
        env = env_factory(seed)
        try:
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            episode_return = 0.0
            length = 0
            while not (terminated or truncated):
                normalized_observation = training_normalizer.normalize_obs(
                    np.asarray(observation, dtype=np.float32).copy()
                )
                action, _ = model.predict(normalized_observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                length += 1
            rows.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "return": episode_return,
                    "length": length,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
        finally:
            env.close()
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    returns = np.asarray([row["return"] for row in rows], dtype=float)
    summary = ReturnSummary(
        environment_steps=environment_steps,
        episodes=len(rows),
        mean_return=float(np.mean(returns)),
        median_return=float(np.median(returns)),
        mean_episode_length=float(np.mean([row["length"] for row in rows])),
    )
    (output / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    return summary


def evaluation_seeds(config: DMCTransferConfig, episodes: int) -> tuple[int, ...]:
    domain_offset = 0 if config.domain == "humanoid" else 1_000_000
    start = 20_000_000 + domain_offset + config.seed * 10_000
    return tuple(range(start, start + episodes))


def save_target_checkpoint(
    *,
    model: SAC,
    normalizer: VecNormalize,
    run_dir: Path,
    action_model: ETLSARActionModel | None,
    name: str = "latest",
    include_replay: bool = True,
) -> None:
    model.save(run_dir / f"{name}_policy")
    normalizer.save(run_dir / f"{name}_vecnormalize.pkl")
    if include_replay:
        model.save_replay_buffer(run_dir / "latest_replay_buffer.pkl")
    if action_model is not None:
        torch.save(action_model.state_dict(), run_dir / f"{name}_action_model.pt")
    if name == "latest":
        (run_dir / "transition_count.json").write_text(
            json.dumps({"transitions": int(model.num_timesteps)}, indent=2),
            encoding="utf-8",
        )


class TargetFineTuneCallback(BaseCallback):
    def __init__(self, representation: RepresentationTrainer, *, freeze_steps: int) -> None:
        super().__init__(verbose=0)
        self.representation = representation
        self.freeze_steps = freeze_steps
        self.latent: list[np.ndarray] = []
        self.executed: list[np.ndarray] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "etl_sar/latent_action" in info:
                self.latent.append(
                    np.asarray(info["etl_sar/latent_action"], dtype=np.float32)
                )
                self.executed.append(
                    np.asarray(info["etl_sar/executed_action"], dtype=np.float32)
                )
        if self.num_timesteps >= self.freeze_steps and len(self.latent) >= 8:
            self.representation.fine_tune_decoder(
                latent=torch.as_tensor(np.stack(self.latent)),
                executed_actions=torch.as_tensor(np.stack(self.executed)),
            )
            self.latent.clear()
            self.executed.clear()
        return True


class DMCCheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        config: DMCTransferConfig,
        run_dir: Path,
        normalizer: VecNormalize,
        env_factory: Callable[[int], gym.Env],
        source_offset: int,
        action_model: ETLSARActionModel | None,
    ) -> None:
        super().__init__(verbose=0)
        self.config = config
        self.run_dir = run_dir
        self.normalizer = normalizer
        self.env_factory = env_factory
        self.source_offset = source_offset
        self.action_model = action_model
        self.next_checkpoint = config.checkpoint_interval

    def _on_training_start(self) -> None:
        self.next_checkpoint = (
            (self.model.num_timesteps // self.config.checkpoint_interval) + 1
        ) * self.config.checkpoint_interval

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_checkpoint:
            return True
        charged_steps = self.source_offset + self.num_timesteps
        save_target_checkpoint(
            model=self.model,
            normalizer=self.normalizer,
            run_dir=self.run_dir,
            action_model=self.action_model,
            name=f"checkpoint_{charged_steps}",
            include_replay=False,
        )
        save_target_checkpoint(
            model=self.model,
            normalizer=self.normalizer,
            run_dir=self.run_dir,
            action_model=self.action_model,
        )
        evaluate_returns(
            self.model,
            self.normalizer,
            env_factory=self.env_factory,
            seeds=evaluation_seeds(
                self.config, self.config.intermediate_episodes
            ),
            output_dir=self.run_dir / "evaluation" / f"checkpoint_{charged_steps}",
            environment_steps=charged_steps,
        )
        self.next_checkpoint += self.config.checkpoint_interval
        return True


def train_target_job(
    config: DMCTransferConfig,
    *,
    method: DMCMethod,
    run_dir: str | Path,
    bundle_path: str | Path | None = None,
    device: str = "auto",
) -> Path:
    output = Path(run_dir)
    complete = output / "complete.json"
    if complete.is_file() and (output / "evaluation" / "final" / "summary.json").is_file():
        return complete
    output.mkdir(parents=True, exist_ok=True)
    is_lattice = method is DMCMethod.LATTICE
    action_model: ETLSARActionModel | None = None
    representation: RepresentationTrainer | None = None

    def raw_env(seed: int) -> gym.Env:
        return make_dmc_env(
            config.domain,
            config.target_task,
            seed=seed,
            action_repeat=config.action_repeat,
        )

    if is_lattice:
        def policy_env(seed: int) -> gym.Env:
            return Monitor(raw_env(seed))
    else:
        if bundle_path is None:
            raise ValueError("ETL target job requires a source representation bundle")
        probe = raw_env(config.seed)
        sar_scale = 1.0 if method is DMCMethod.ETL_SAR else 0.0
        action_model, representation = load_action_model(
            bundle_path, env=probe, sar_scale=sar_scale, device=device
        )
        probe.close()

        def policy_env(seed: int) -> gym.Env:
            return Monitor(LatentActionWrapper(raw_env(seed), action_model))

    base_vector = DummyVecEnv([lambda: policy_env(config.seed)])
    count_path = output / "transition_count.json"
    if count_path.is_file():
        if action_model is not None:
            state_path = output / "latest_action_model.pt"
            if not state_path.is_file():
                raise ValueError("incomplete ETL target checkpoint")
            action_model.load_state_dict(
                torch.load(state_path, map_location="cpu", weights_only=True)
            )
        normalizer = VecNormalize.load(output / "latest_vecnormalize.pkl", base_vector)
        normalizer.training = True
        model = SAC.load(
            output / "latest_policy.zip", env=normalizer, device=device
        )
        replay = output / "latest_replay_buffer.pkl"
        if not replay.is_file():
            raise ValueError("incomplete SAC target checkpoint")
        model.load_replay_buffer(replay)
        completed = int(json.loads(count_path.read_text(encoding="utf-8"))["transitions"])
        if completed != int(model.num_timesteps):
            raise ValueError("SAC policy transition count does not match checkpoint")
    else:
        normalizer = VecNormalize(
            base_vector, norm_obs=True, norm_reward=False, clip_obs=10.0
        )
        kwargs = (
            lattice_sac_kwargs(seed=config.seed, device=device)
            if is_lattice
            else common_sac_kwargs(seed=config.seed, device=device)
        )
        policy: Any = LatticeSACPolicy if is_lattice else "MlpPolicy"
        model = SAC(
            policy,
            normalizer,
            tensorboard_log=str(output / "tensorboard"),
            verbose=1,
            **kwargs,
        )
        completed = 0
    budget = config.lattice_budget if is_lattice else config.target_budget
    if completed > budget:
        raise ValueError("resume checkpoint exceeds the target budget")
    callbacks: list[BaseCallback] = []
    if representation is not None:
        callbacks.append(
            TargetFineTuneCallback(
                representation,
                freeze_steps=min(100_000, config.target_budget // 4),
            )
        )
    source_offset = 0 if is_lattice else config.source_budget
    callbacks.append(
        DMCCheckpointCallback(
            config=config,
            run_dir=output,
            normalizer=normalizer,
            env_factory=policy_env,
            source_offset=source_offset,
            action_model=action_model,
        )
    )
    if completed < budget:
        model.learn(
            total_timesteps=budget - completed,
            callback=callbacks,
            reset_num_timesteps=completed == 0,
        )
    save_target_checkpoint(
        model=model,
        normalizer=normalizer,
        run_dir=output,
        action_model=action_model,
    )
    evaluate_returns(
        model,
        normalizer,
        env_factory=policy_env,
        seeds=evaluation_seeds(config, config.final_episodes),
        output_dir=output / "evaluation" / "final",
        environment_steps=config.total_budget,
    )
    normalizer.close()
    _write_complete(
        complete,
        {
            "stage": "target",
            "domain": config.domain,
            "method": method.value,
            "seed": config.seed,
            "source_transitions": source_offset,
            "target_transitions": budget,
            "total_transitions": config.total_budget,
        },
    )
    return complete
