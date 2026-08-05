from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import typer
from stable_baselines3 import PPO

from etl_sar.action_model import ETLSARActionModel
from etl_sar.bdr import StateEncoder
from etl_sar.checkpoints import load_checkpoint_pair
from etl_sar.config import ExperimentConfig
from etl_sar.formal.config import FormalDomainConfig
from etl_sar.formal.evaluate import evaluate_formal
from etl_sar.formal.matrix import ExperimentMatrix
from etl_sar.data import TrajectoryStore
from etl_sar.envs import LatentActionWrapper, validate_environment
from etl_sar.lattice.evaluation import evaluate_lattice_model
from etl_sar.lattice.runtime import train_lattice_job
from etl_sar.evaluation import EvaluationSummary, compare_runs, evaluate_checkpoint
from etl_sar.exploration import ExploreTrainer, InsufficientSourceSuccessError
from etl_sar.gmvae import GMVAE
from etl_sar.pilot import build_pilot_summary
from etl_sar.representation import RepresentationTrainer
from etl_sar.synergy import SynergyArtifact
from etl_sar.trainers import TransferTrainer
from etl_sar.types import Limb


app = typer.Typer(no_args_is_help=True, help="ETL core with SAR transfer and SB3 engineering")


def _resolve_sar_scale(bundle_scale: float, override: float | None) -> float:
    scale = float(bundle_scale if override is None else override)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("SAR scale must be between 0 and 1")
    return scale


def _require_myosuite() -> None:
    try:
        import myosuite  # noqa: F401
    except ImportError as error:
        raise typer.BadParameter(
            "MyoSuite is required for environment commands; install the 'myosuite' extra"
        ) from error


def _make_env(env_id: str) -> gym.Env:
    _require_myosuite()
    try:
        return gym.make(env_id)
    except gym.error.Error as error:
        raise typer.BadParameter(f"environment {env_id!r} is not registered") from error


def _load_bundle(
    config: ExperimentConfig,
    bundle_path: Path,
    env: gym.Env,
    sar_scale: float | None = None,
) -> tuple[ETLSARActionModel, RepresentationTrainer]:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if bundle["limb"] != config.limb.value:
        raise typer.BadParameter("representation bundle limb does not match config")
    if bundle["source_task"] != config.source.env_id:
        raise typer.BadParameter("representation bundle source task does not match config")
    contract = validate_environment(env, expected_action_dim=int(bundle["action_dim"]))
    artifact = SynergyArtifact.load(
        bundle_path.parent / bundle["synergy_file"],
        expected_limb=config.limb,
        expected_source_task=config.source.env_id,
    )
    gmvae = GMVAE(
        action_dim=contract.action_dim,
        latent_dim=20,
        components=20,
        hidden_dims=tuple(bundle["hidden_dims"]),
    )
    gmvae.load_state_dict(bundle["gmvae"])
    action_model = ETLSARActionModel(
        decoder=gmvae.decoder,
        synergy_basis=torch.as_tensor(artifact.basis),
        latent_dim=20,
        action_low=torch.as_tensor(contract.action_low),
        action_high=torch.as_tensor(contract.action_high),
        rho=float(bundle["rho"]),
        enabled_scale=_resolve_sar_scale(bundle["enabled_scale"], sar_scale),
    )
    action_model.load_state_dict(bundle["action_model"])
    representation = RepresentationTrainer(
        gmvae=gmvae,
        action_model=action_model,
        learning_rate=3e-4,
        decoder_learning_rate=3e-5,
    )
    return action_model, representation


@app.command()
def inspect(config: Path = typer.Option(..., exists=True, dir_okay=False)) -> None:
    cfg = ExperimentConfig.from_yaml(config)
    typer.echo("ETL core: directional exploration, BDR, GMVAE, latent PPO, decoder Eq. 10")
    typer.echo("SAR transfer: source-task 20-component ICAPCA frozen in target tasks")
    typer.echo("SB3 engineering: PPO trainers, callbacks, checkpoints, TensorBoard")
    typer.echo(f"source={cfg.source.env_id}")
    typer.echo(f"target={cfg.target.env_id}")


@app.command("formal-dry-run")
def formal_dry_run(
    hand_config: Path = typer.Option(..., exists=True, dir_okay=False),
    leg_config: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
) -> None:
    matrix = ExperimentMatrix.from_configs(
        [
            FormalDomainConfig.from_yaml(hand_config),
            FormalDomainConfig.from_yaml(leg_config),
        ]
    )
    payload = matrix.to_mapping()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(json.dumps(payload["summary"], indent=2))


@app.command("formal-lattice-train")
def formal_lattice_train(
    domain: str = typer.Option(...),
    env_id: str = typer.Option(...),
    seed: int = typer.Option(..., min=0),
    timesteps: int = typer.Option(..., min=1),
    checkpoint_interval: int = typer.Option(250_000, min=1),
    evaluation_episodes: int = typer.Option(20, min=1),
    final_episodes: int = typer.Option(..., min=1),
    num_envs: int = typer.Option(16, min=1),
    run_dir: Path = typer.Option(..., file_okay=False),
    device: str = typer.Option("auto"),
) -> None:
    """Train the pinned official Lattice configuration on a shared MyoSuite task."""
    if domain not in {"hand", "leg"}:
        raise typer.BadParameter("domain must be hand or leg")

    def env_factory() -> gym.Env:
        return _make_env(env_id)

    def evaluate_at(model, vecnormalize, transitions: int) -> None:
        seeds = ExperimentMatrix.seed_bank(
            domain=domain, seed=seed, episodes=evaluation_episodes
        )
        evaluate_lattice_model(
            model=model,
            training_vecnormalize=vecnormalize,
            env_factory=env_factory,
            domain=domain,
            seeds=seeds,
            output_dir=run_dir / "evaluation" / f"checkpoint_{transitions}",
            environment_steps=transitions,
            recurrent=domain == "hand",
        )

    model = train_lattice_job(
        domain=domain,
        env_id=env_id,
        seed=seed,
        total_transitions=timesteps,
        checkpoint_interval=checkpoint_interval,
        run_dir=run_dir,
        num_envs=num_envs,
        device=device,
        evaluate=evaluate_at,
    )
    evaluate_lattice_model(
        model=model,
        training_vecnormalize=model.get_env(),
        env_factory=env_factory,
        domain=domain,
        seeds=ExperimentMatrix.seed_bank(
            domain=domain, seed=seed, episodes=final_episodes
        ),
        output_dir=run_dir / "evaluation" / "final",
        environment_steps=timesteps,
        recurrent=domain == "hand",
    )
    typer.echo(str(run_dir / "latest_policy.zip"))


@app.command()
def explore(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    run_dir: Path = typer.Option(..., file_okay=False),
    timesteps: int = typer.Option(100_000, min=32),
    min_timesteps: int = typer.Option(0, min=0),
    min_success_actions: int = typer.Option(0, min=0),
) -> None:
    if min_timesteps > timesteps:
        raise typer.BadParameter("min_timesteps cannot exceed timesteps")
    cfg = ExperimentConfig.from_yaml(config)
    probe = _make_env(cfg.source.env_id)
    contract = validate_environment(probe)
    probe.close()
    hidden_dims = (256, 256)
    gmvae = GMVAE(
        action_dim=contract.action_dim,
        latent_dim=20,
        components=20,
        hidden_dims=hidden_dims,
    )
    action_model = ETLSARActionModel(
        decoder=gmvae.decoder,
        synergy_basis=torch.zeros(contract.action_dim, 20),
        latent_dim=20,
        action_low=torch.as_tensor(contract.action_low),
        action_high=torch.as_tensor(contract.action_high),
        enabled_scale=0.0,
    )
    representation = RepresentationTrainer(
        gmvae=gmvae,
        action_model=action_model,
        learning_rate=3e-4,
    )
    store = TrajectoryStore(
        run_dir / "data",
        limb=cfg.limb,
        source_task=cfg.source.env_id,
        action_dim=contract.action_dim,
    )
    trainer = ExploreTrainer(
        env_factory=lambda: _make_env(cfg.source.env_id),
        state_encoder=StateEncoder(contract.observation_dim, 20),
        representation=representation,
        trajectory_store=store,
        limb=cfg.limb,
        source_task=cfg.source.env_id,
        run_dir=run_dir,
        total_timesteps=timesteps,
        n_steps=256,
        batch_size=64,
        representation_update_interval=1024,
        seed=cfg.seed,
        min_timesteps=min_timesteps,
        min_success_actions=min_success_actions,
    )
    try:
        artifacts = trainer.run()
    except InsufficientSourceSuccessError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "representation_checkpoint": str(
                    artifacts.representation_checkpoint
                ),
                "environment_steps": artifacts.environment_steps,
                "successful_actions": artifacts.successful_actions,
            },
            indent=2,
        )
    )


@app.command("fit-representation")
def fit_representation(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    data_dir: Path = typer.Option(..., exists=True, file_okay=False),
    explore_checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(..., file_okay=False),
    sar_steps: int = typer.Option(500, min=1),
) -> None:
    cfg = ExperimentConfig.from_yaml(config)
    store = TrajectoryStore(
        data_dir,
        limb=cfg.limb,
        source_task=cfg.source.env_id,
    )
    success_actions = store.success_pool()
    if success_actions.shape[0] < 20:
        raise typer.BadParameter("SAR requires at least 20 successful source-task actions")
    artifact = SynergyArtifact.fit(
        success_actions,
        components=20,
        limb=cfg.limb,
        source_task=cfg.source.env_id,
        data_fingerprint=store.fingerprint(),
    )
    checkpoint = torch.load(explore_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["limb"] != cfg.limb.value or checkpoint["source_task"] != cfg.source.env_id:
        raise typer.BadParameter("explore checkpoint metadata does not match config")
    action_dim = int(success_actions.shape[1])
    hidden_dims = (256, 256)
    gmvae = GMVAE(action_dim=action_dim, hidden_dims=hidden_dims)
    gmvae.load_state_dict(checkpoint["gmvae"])
    action_model = ETLSARActionModel(
        decoder=gmvae.decoder,
        synergy_basis=torch.as_tensor(artifact.basis),
        latent_dim=20,
        action_low=-torch.ones(action_dim),
        action_high=torch.ones(action_dim),
        rho=cfg.synergy.rho,
        enabled_scale=cfg.synergy.enabled_scale,
    )
    representation = RepresentationTrainer(
        gmvae=gmvae,
        action_model=action_model,
        learning_rate=3e-4,
    )
    action_tensor = torch.as_tensor(success_actions, dtype=torch.float32)
    with torch.no_grad():
        torch.manual_seed(cfg.seed)
        latent = gmvae(action_tensor).latent
    representation.train_sar_head(latent, action_tensor, steps=sar_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    synergy_path = output_dir / "synergy.joblib"
    artifact.save(synergy_path)
    bundle_path = output_dir / "representation_bundle.pt"
    torch.save(
        {
            "schema_version": 1,
            "limb": cfg.limb.value,
            "source_task": cfg.source.env_id,
            "action_dim": action_dim,
            "latent_dim": 20,
            "hidden_dims": hidden_dims,
            "rho": cfg.synergy.rho,
            "enabled_scale": cfg.synergy.enabled_scale,
            "synergy_file": synergy_path.name,
            "gmvae": gmvae.state_dict(),
            "action_model": action_model.state_dict(),
            "data_fingerprint": store.fingerprint(),
        },
        bundle_path,
    )
    typer.echo(str(bundle_path))


@app.command()
def transfer(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    bundle: Path = typer.Option(..., exists=True, dir_okay=False),
    run_dir: Path = typer.Option(..., file_okay=False),
    timesteps: int = typer.Option(100_000, min=32),
    decoder_freeze_steps: int = typer.Option(10_000, min=0),
    sar_scale: float | None = typer.Option(None),
    eval_freq: int | None = typer.Option(None, min=1),
    evaluation_episodes: int = typer.Option(1, min=1),
    evaluation_seed: int = typer.Option(10_000, min=0),
    resume_manifest: Path | None = typer.Option(
        None, exists=True, dir_okay=False
    ),
) -> None:
    """Train the target policy and return its verified paired checkpoint manifest."""
    cfg = ExperimentConfig.from_yaml(config)
    probe = _make_env(cfg.target.env_id)
    action_model, representation = _load_bundle(
        cfg, bundle, probe, sar_scale=sar_scale
    )
    probe.close()
    trainer = TransferTrainer(
        env_factory=lambda: _make_env(cfg.target.env_id),
        action_model=action_model,
        representation=representation,
        run_dir=run_dir,
        total_timesteps=timesteps,
        decoder_freeze_steps=decoder_freeze_steps,
        eval_freq=eval_freq,
        evaluation_episodes=evaluation_episodes,
        evaluation_seed=evaluation_seed,
        resume_manifest=resume_manifest,
        formal_domain=cfg.limb.value,
        seed=cfg.seed,
    )
    typer.echo(str(trainer.run().best_manifest))


@app.command()
def evaluate(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    bundle: Path = typer.Option(..., exists=True, dir_okay=False),
    pair_manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(..., file_okay=False),
    episodes: int = typer.Option(20, min=1),
    environment_steps: int = typer.Option(..., min=1),
    sar_scale: float | None = typer.Option(None),
) -> None:
    cfg = ExperimentConfig.from_yaml(config)
    base_env = _make_env(cfg.target.env_id)
    action_model, _ = _load_bundle(
        cfg, bundle, base_env, sar_scale=sar_scale
    )
    env = LatentActionWrapper(base_env, action_model)
    try:
        model, _ = load_checkpoint_pair(
            pair_manifest, policy_loader=PPO.load, action_model=action_model
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    summary = evaluate_checkpoint(
        model,
        env,
        episodes=episodes,
        output_dir=output_dir,
        environment_steps=environment_steps,
        seed=cfg.seed + 10_000,
        environment_id=cfg.target.env_id,
        sar_scale=action_model.enabled_scale,
    )
    typer.echo(json.dumps(summary.__dict__, indent=2))


@app.command("formal-etl-evaluate")
def formal_etl_evaluate(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    bundle: Path = typer.Option(..., exists=True, dir_okay=False),
    pair_manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(..., file_okay=False),
    episodes: int = typer.Option(..., min=1),
    environment_steps: int = typer.Option(..., min=1),
) -> None:
    """Evaluate an ETL pair with the shared formal metrics and seed bank."""
    cfg = ExperimentConfig.from_yaml(config)
    base_env = _make_env(cfg.target.env_id)
    action_model, _ = _load_bundle(cfg, bundle, base_env)
    env = LatentActionWrapper(base_env, action_model)
    try:
        model, _ = load_checkpoint_pair(
            pair_manifest, policy_loader=PPO.load, action_model=action_model
        )
    except ValueError as error:
        env.close()
        raise typer.BadParameter(str(error)) from error
    seeds = ExperimentMatrix.seed_bank(
        domain=cfg.limb.value,
        seed=cfg.seed,
        episodes=episodes,
    )
    summary = evaluate_formal(
        model,
        env,
        domain=cfg.limb.value,
        seeds=seeds,
        output_dir=output_dir,
        environment_steps=environment_steps,
    )
    typer.echo(json.dumps(summary.__dict__, indent=2))


@app.command()
def compare(
    baseline: Path = typer.Option(..., exists=True, dir_okay=False),
    extension: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
) -> None:
    result = compare_runs(
        EvaluationSummary.load(baseline),
        EvaluationSummary.load(extension),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    typer.echo(json.dumps(result, indent=2))


@app.command("pilot-summary")
def pilot_summary(
    hand: Path = typer.Option(..., exists=True, dir_okay=False),
    leg: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    legacy_reference: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
    ),
) -> None:
    hand_comparison = json.loads(hand.read_text(encoding="utf-8"))
    leg_comparison = json.loads(leg.read_text(encoding="utf-8"))
    legacy = (
        json.loads(legacy_reference.read_text(encoding="utf-8"))
        if legacy_reference is not None
        else None
    )
    result = build_pilot_summary(hand_comparison, leg_comparison, legacy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    typer.echo(json.dumps(result, indent=2))
