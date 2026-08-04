# ETL-SAR Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, tested SAR-guided ETL pipeline in which ETL remains the exploration, representation, latent-control, and decoder-fine-tuning core; SAR transfers 20 ICAPCA synergies from simple to complex MyoSuite tasks; SB3 replaces Ray as the training framework.

**Architecture:** Source-task PPO exploration records full actuator trajectories while a behavioral encoder supplies ETL directional bonuses and BDR. A GMVAE is updated from the growing action buffer, SAR fits a fixed 20-component PCA+ICA mapping on successful source trajectories, and target-task PPO operates only in the GMVAE latent action space through a decoding Gymnasium wrapper. Decoder fine-tuning is a separate supervised ETL Eq. 10 update over interaction pairs, never a PPO gradient.

**Tech Stack:** Python 3.11, MyoSuite 2.12.x, Gymnasium, Stable-Baselines3 2.x, PyTorch, NumPy, scikit-learn, PyYAML, Typer, pytest, TensorBoard.

## Global Constraints

- ETL owns BDR, directional exploration, GMVAE, latent control, and supervised decoder fine-tuning.
- SAR owns only source-task ICAPCA extraction and the frozen target-task synergy basis.
- Lattice contributes only SB3-style trainers, vector environments, callbacks, checkpoints, and logs.
- Hand and Leg data, statistics, models, and checkpoints are never interchangeable.
- Default synergy count, GMVAE latent dimension, and GMVAE mixture count are all exactly 20.
- SAR outputs are normalized to `[-1, 1]`; target-task `W_SAR` is frozen.
- SAR correction is hard-capped at `rho=0.20` of the ETL decoder output norm.
- Complex-task data never enters PCA/ICA fitting.
- Missing task registrations and incompatible shapes fail loudly; no environment substitution is allowed.

## File Map

- `pyproject.toml`: packaging, dependencies, CLI, and pytest configuration.
- `src/etl_sar/config.py`: typed YAML configuration and cross-field validation.
- `src/etl_sar/types.py`: limb/task enums and trajectory/checkpoint metadata.
- `src/etl_sar/data.py`: append-only trajectory buffer, two data views, and fingerprints.
- `src/etl_sar/bdr.py`: ETL state encoder, directional bonus, and BDR loss.
- `src/etl_sar/gmvae.py`: GMVAE encoder, mixture prior, decoder, and negative ELBO.
- `src/etl_sar/synergy.py`: 20-component ICAPCA fitting, artifacts, projection, and validation.
- `src/etl_sar/action_model.py`: ETL decoder plus bounded SAR residual.
- `src/etl_sar/envs.py`: strict task registry, metadata wrapper, and latent-action wrapper.
- `src/etl_sar/representation.py`: interleaved GMVAE updates, SAR-head training, and Eq. 10 decoder updates.
- `src/etl_sar/trainers.py`: ExploreTrainer, TransferTrainer, callbacks, and checkpoint lifecycle.
- `src/etl_sar/evaluation.py`: deterministic evaluation and CSV/JSON summaries.
- `src/etl_sar/cli.py`: inspect, explore, fit-representation, transfer, evaluate, and compare commands.
- `configs/hand_quick.yaml`, `configs/leg_quick.yaml`: executable limited-budget protocols.
- `tests/`: unit, integration-with-dummy-env, and optional MyoSuite smoke tests.

---

### Task 1: Package Skeleton and Validated Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/etl_sar/__init__.py`
- Create: `src/etl_sar/config.py`
- Create: `src/etl_sar/types.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Limb`, `TaskRole`, `TaskMetadata`, `ExperimentConfig.from_yaml(path)`, and `ExperimentConfig.validate()`.
- Consumes: no project interfaces.

- [ ] **Step 1: Write failing validation tests**

```python
def test_default_representation_is_sar_protocol():
    cfg = ExperimentConfig.minimal(limb=Limb.HAND, source_env="source", target_env="target")
    assert cfg.representation.latent_dim == 20
    assert cfg.representation.mixture_components == 20
    assert cfg.synergy.components == 20
    assert cfg.synergy.rho == 0.20

def test_rejects_cross_limb_task_metadata():
    cfg = ExperimentConfig.minimal(limb=Limb.HAND, source_env="source", target_env="target")
    cfg.target.limb = Limb.LEG
    with pytest.raises(ValueError, match="limb"):
        cfg.validate()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_config.py -v`

Expected: collection fails because `etl_sar.config` does not exist.

- [ ] **Step 3: Implement exact typed contracts**

```python
class Limb(str, Enum):
    HAND = "hand"
    LEG = "leg"

@dataclass
class RepresentationConfig:
    latent_dim: int = 20
    mixture_components: int = 20

@dataclass
class SynergyConfig:
    components: int = 20
    rho: float = 0.20
    enabled_scale: float = 1.0

def validate(self) -> None:
    if self.source.limb != self.limb or self.target.limb != self.limb:
        raise ValueError("source and target limb must match experiment limb")
    if not (self.representation.latent_dim == self.synergy.components == 20):
        raise ValueError("default ETL/SAR protocol requires 20 aligned components")
```

- [ ] **Step 4: Run tests and package metadata checks**

Run: `python -m pytest tests/test_config.py -v`

Expected: all tests pass.

Run: `python -m pip install -e . --no-deps`

Expected: editable package installs and `python -c "import etl_sar"` exits 0.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/etl_sar/__init__.py src/etl_sar/config.py src/etl_sar/types.py tests/test_config.py
git commit -m "feat: add validated ETL-SAR configuration"
```

### Task 2: Trajectory Data Contracts and Isolation

**Files:**
- Create: `src/etl_sar/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `Limb`, `TaskMetadata` from Task 1.
- Produces: `Transition`, `Trajectory`, `TrajectoryStore.append_episode()`, `action_pool()`, `success_pool()`, `fingerprint()`.

- [ ] **Step 1: Write failing data-view tests**

```python
def test_action_and_success_views_are_distinct(tmp_path):
    store = TrajectoryStore(tmp_path, limb=Limb.HAND, source_task="reorient8")
    store.append_episode(make_episode(success=False, action=0.1))
    store.append_episode(make_episode(success=True, action=0.8))
    assert store.action_pool().shape[0] == 2
    assert store.success_pool().shape[0] == 1

def test_store_rejects_wrong_limb(tmp_path):
    store = TrajectoryStore(tmp_path, limb=Limb.HAND, source_task="reorient8")
    with pytest.raises(ValueError, match="limb"):
        store.append_episode(make_episode(limb=Limb.LEG))
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_data.py -v`

Expected: FAIL because `TrajectoryStore` is undefined.

- [ ] **Step 3: Implement append-only NPZ episodes and SHA-256 metadata**

```python
@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    sampled_action: np.ndarray
    executed_action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    behavior: np.ndarray
    success: bool

class TrajectoryStore:
    def success_pool(self) -> np.ndarray:
        episodes = [ep for ep in self.iter_episodes() if ep.success]
        return np.concatenate([ep.executed_actions for ep in episodes], axis=0)
```

- [ ] **Step 4: Run data tests**

Run: `python -m pytest tests/test_data.py -v`

Expected: all tests pass, and two stores with identical episodes have identical fingerprints.

- [ ] **Step 5: Commit**

```bash
git add src/etl_sar/data.py tests/test_data.py
git commit -m "feat: add isolated trajectory data views"
```

### Task 3: ETL Directional Bonus and BDR Encoder

**Files:**
- Create: `src/etl_sar/bdr.py`
- Test: `tests/test_bdr.py`

**Interfaces:**
- Produces: `StateEncoder`, `sample_unit_directions()`, `directional_bonus()`, `behavior_metric_loss()`.
- Consumes: PyTorch tensors shaped `[batch, observation_dim]`.

- [ ] **Step 1: Write failing mathematical tests**

```python
def test_directional_bonus_matches_etl_equation_one():
    phi_s = torch.tensor([[0.0, 1.0]])
    phi_next = torch.tensor([[2.0, 1.0]])
    direction = torch.tensor([[1.0, 0.0]])
    assert directional_bonus(phi_s, phi_next, direction).item() == pytest.approx(2.0)

def test_bdr_hinge_penalizes_collapsed_low_reward_pair():
    collapsed = torch.zeros(2, 3)
    reward = torch.zeros(2)
    loss = discriminability_hinge(collapsed, reward, epsilon=0.1, margin=1.0)
    assert loss.item() > 0
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_bdr.py -v`

Expected: FAIL because `etl_sar.bdr` does not exist.

- [ ] **Step 3: Implement ETL Eq. 1 and stable Eq. 5 semantics**

```python
def directional_bonus(phi_s, phi_next, direction):
    return ((phi_next - phi_s) * direction).sum(dim=-1)

def discriminability_hinge(embedding, reward, epsilon, margin):
    distance = torch.linalg.vector_norm(embedding[0::2] - embedding[1::2], dim=-1)
    reward_gap = (reward[0::2] - reward[1::2]).abs()
    active = (reward_gap < epsilon).to(distance.dtype)
    return (active * torch.relu(margin - distance).square()).mean()
```

The hinge is the bounded implementation of Eq. 5's stated anti-collapse intent; README must disclose this numerical interpretation.

- [ ] **Step 4: Run BDR tests**

Run: `python -m pytest tests/test_bdr.py -v`

Expected: all tests pass; sampled directions have unit L2 norm.

- [ ] **Step 5: Commit**

```bash
git add src/etl_sar/bdr.py tests/test_bdr.py
git commit -m "feat: add ETL directional exploration and BDR"
```

### Task 4: ETL GMVAE Representation

**Files:**
- Create: `src/etl_sar/gmvae.py`
- Test: `tests/test_gmvae.py`

**Interfaces:**
- Produces: `GMVAE`, `GMVAEOutput`, `gmvae_loss(output, actions)` and `GMVAE.decode(z)`.
- Consumes: action tensors `[batch, action_dim]`, latent dimension 20, mixture count 20.

- [ ] **Step 1: Write failing shape and gradient tests**

```python
def test_gmvae_loss_updates_all_paths():
    model = GMVAE(action_dim=39, latent_dim=20, components=20, hidden_dims=(64, 64))
    actions = torch.rand(8, 39)
    output = model(actions)
    loss = gmvae_loss(output, actions).total
    loss.backward()
    assert output.reconstruction.shape == actions.shape
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_gmvae.py -v`

Expected: FAIL because `GMVAE` is undefined.

- [ ] **Step 3: Implement mixture posterior and negative ELBO**

```python
@dataclass
class GMVAELoss:
    total: Tensor
    reconstruction: Tensor
    latent_kl: Tensor
    categorical_kl: Tensor

def gmvae_loss(output, target):
    rec = F.mse_loss(output.reconstruction, target)
    latent_kl = gaussian_kl(output.q_mean, output.q_logvar,
                            output.prior_mean, output.prior_logvar).mean()
    uniform_log_prob = -math.log(output.component_probs.shape[-1])
    cat_kl = (output.component_probs *
              (output.component_log_probs - uniform_log_prob)).sum(-1).mean()
    return GMVAELoss(rec + latent_kl + cat_kl, rec, latent_kl, cat_kl)
```

- [ ] **Step 4: Run GMVAE tests**

Run: `python -m pytest tests/test_gmvae.py -v`

Expected: all tests pass, loss is finite, and seeded forward passes are reproducible.

- [ ] **Step 5: Commit**

```bash
git add src/etl_sar/gmvae.py tests/test_gmvae.py
git commit -m "feat: add ETL Gaussian-mixture VAE"
```

### Task 5: SAR ICAPCA Artifact and Bounded Residual

**Files:**
- Create: `src/etl_sar/synergy.py`
- Create: `src/etl_sar/action_model.py`
- Test: `tests/test_synergy.py`
- Test: `tests/test_action_model.py`

**Interfaces:**
- Produces: `SynergyArtifact.fit(actions, metadata)`, `transform()`, `inverse_transform()`, `save()`, `load()`; `SARResidual`; `ETLSARActionModel`.
- Consumes: successful source actions, frozen `GMVAE.decoder`, `Limb`, source-task fingerprint.

- [ ] **Step 1: Write failing ICAPCA and hard-budget tests**

```python
def test_icapca_uses_twenty_components_and_bounded_codes():
    artifact = SynergyArtifact.fit(actions=np.random.randn(128, 39), components=20,
                                   limb=Limb.HAND, source_task="reorient8")
    codes = artifact.transform(np.random.randn(4, 39))
    assert codes.shape == (4, 20)
    assert np.max(np.abs(codes)) <= 1.0 + 1e-6

def test_sar_residual_never_exceeds_twenty_percent():
    model = make_action_model(rho=0.20)
    result = model(torch.randn(16, 20))
    ratio = result.sar_action.norm(dim=-1) / result.etl_action.norm(dim=-1).clamp_min(1e-8)
    assert torch.all(ratio <= 0.200001)
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_synergy.py tests/test_action_model.py -v`

Expected: FAIL because both modules are absent.

- [ ] **Step 3: Implement serializable PCA+FastICA and frozen basis**

```python
class SynergyArtifact:
    @classmethod
    def fit(cls, actions, components, limb, source_task):
        scaler = StandardScaler().fit(actions)
        pca = PCA(n_components=components, random_state=0).fit(scaler.transform(actions))
        ica = FastICA(n_components=components, whiten="unit-variance", random_state=0,
                      max_iter=2000).fit(pca.transform(scaler.transform(actions)))
        return cls(scaler=scaler, pca=pca, ica=ica, limb=limb,
                   source_task=source_task, components=components)
```

- [ ] **Step 4: Implement exact hard gate and lambda-zero identity**

```python
raw = self.enabled_scale * self.synergy_head(z) @ self.synergy_basis.T
etl_norm = etl_action.norm(dim=-1, keepdim=True)
raw_norm = raw.norm(dim=-1, keepdim=True).clamp_min(self.eps)
scale = torch.minimum(torch.ones_like(raw_norm), self.rho * etl_norm / raw_norm)
sar_action = raw * scale
full_action = torch.clamp(etl_action + sar_action, self.action_low, self.action_high)
```

- [ ] **Step 5: Run tests and artifact round trip**

Run: `python -m pytest tests/test_synergy.py tests/test_action_model.py -v`

Expected: all tests pass; a Hand artifact loaded as Leg raises `ValueError`; `enabled_scale=0` is exactly pure ETL.

- [ ] **Step 6: Commit**

```bash
git add src/etl_sar/synergy.py src/etl_sar/action_model.py tests/test_synergy.py tests/test_action_model.py
git commit -m "feat: add SAR extraction and bounded ETL residual"
```

### Task 6: Representation and Decoder Fine-Tuning Services

**Files:**
- Create: `src/etl_sar/representation.py`
- Test: `tests/test_representation.py`

**Interfaces:**
- Consumes: `TrajectoryStore`, `GMVAE`, `SynergyArtifact`, `ETLSARActionModel`.
- Produces: `RepresentationTrainer.update_gmvae()`, `fit_synergy()`, `train_sar_head()`, `fine_tune_decoder()` and versioned checkpoint bundles.

- [ ] **Step 1: Write failing interleaving and freeze tests**

```python
def test_decoder_finetune_does_not_update_synergy_basis():
    trainer = make_representation_trainer()
    before = trainer.action_model.synergy_basis.detach().clone()
    trainer.fine_tune_decoder(z=torch.randn(8, 20), executed_actions=torch.rand(8, 39))
    assert torch.equal(before, trainer.action_model.synergy_basis)

def test_gmvae_update_consumes_current_action_pool(tmp_path):
    trainer, store = make_trainer_and_store(tmp_path)
    store.append_episode(make_episode(length=4))
    assert trainer.update_gmvae(store, steps=1).samples == 4
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_representation.py -v`

Expected: FAIL because `RepresentationTrainer` does not exist.

- [ ] **Step 3: Implement separate optimizers and ETL Eq. 10**

```python
def fine_tune_decoder(self, z, executed_actions):
    prediction = self.action_model.decoder(z)
    reconstruction = F.mse_loss(prediction, executed_actions)
    anchor = sum((p - p0).square().sum()
                 for p, p0 in zip(self.action_model.decoder.parameters(), self.decoder_anchor))
    loss = reconstruction + self.anchor_weight * anchor
    self.decoder_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    self.decoder_optimizer.step()
    return float(loss.detach())
```

- [ ] **Step 4: Run representation tests**

Run: `python -m pytest tests/test_representation.py -v`

Expected: all tests pass; GMVAE, SAR head, and decoder optimizer parameter IDs are disjoint.

- [ ] **Step 5: Commit**

```bash
git add src/etl_sar/representation.py tests/test_representation.py
git commit -m "feat: add ETL representation training services"
```

### Task 7: Strict Task Registry and Latent Action Environment

**Files:**
- Create: `src/etl_sar/envs.py`
- Test: `tests/test_envs.py`

**Interfaces:**
- Consumes: `TaskMetadata`, `ETLSARActionModel`.
- Produces: `validate_environment()`, `LatentActionWrapper`, `TaskRegistry.resolve_exact()`.

- [ ] **Step 1: Write failing wrapper tests with a dummy Gymnasium env**

```python
def test_latent_wrapper_exposes_twenty_dimensional_action_space():
    base = DummyMuscleEnv(action_dim=39)
    wrapped = LatentActionWrapper(base, make_action_model(action_dim=39))
    assert wrapped.action_space.shape == (20,)
    wrapped.reset(seed=7)
    _, _, _, _, info = wrapped.step(np.zeros(20, dtype=np.float32))
    assert info["etl_sar/executed_action"].shape == (39,)

def test_registry_never_substitutes_missing_environment():
    with pytest.raises(KeyError, match="not registered"):
        TaskRegistry({}).resolve_exact("leg_uneven")
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_envs.py -v`

Expected: FAIL because `LatentActionWrapper` is undefined.

- [ ] **Step 3: Implement decode boundary and metadata checks**

```python
class LatentActionWrapper(gym.ActionWrapper):
    def __init__(self, env, action_model):
        super().__init__(env)
        self.action_model = action_model.eval()
        self.action_space = gym.spaces.Box(-1.0, 1.0,
                                           shape=(action_model.latent_dim,), dtype=np.float32)

    def action(self, z):
        with torch.no_grad():
            output = self.action_model(torch.as_tensor(z).unsqueeze(0))
        self.last_decoded = output.full_action[0].cpu().numpy()
        return self.last_decoded
```

- [ ] **Step 4: Run environment tests**

Run: `python -m pytest tests/test_envs.py -v`

Expected: all dummy-environment tests pass; shape mismatch fails before first rollout.

- [ ] **Step 5: Commit**

```bash
git add src/etl_sar/envs.py tests/test_envs.py
git commit -m "feat: add strict latent action environments"
```

### Task 8: SB3 Trainers, Callbacks, and Checkpoints

**Files:**
- Create: `src/etl_sar/trainers.py`
- Test: `tests/test_trainers.py`

**Interfaces:**
- Consumes: validated config, `TrajectoryStore`, BDR encoder, `RepresentationTrainer`, `LatentActionWrapper`.
- Produces: `ExploreTrainer.run()`, `TransferTrainer.run()`, `TrainingArtifacts`, and resumable checkpoints.

- [ ] **Step 1: Write failing short-training lifecycle tests**

```python
def test_transfer_trainer_runs_latent_ppo_and_writes_checkpoints(tmp_path):
    trainer = make_transfer_trainer(tmp_path, total_timesteps=64)
    result = trainer.run()
    assert result.latest_checkpoint.exists()
    assert result.best_checkpoint.exists()
    assert trainer.model.action_space.shape == (20,)

def test_decoder_updates_start_after_freeze_steps(tmp_path):
    trainer = make_transfer_trainer(tmp_path, total_timesteps=64, decoder_freeze_steps=32)
    trainer.run()
    assert trainer.decoder_update_steps
    assert min(trainer.decoder_update_steps) >= 32
```

```python
def test_checkpoint_metadata_mismatch_and_nan_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="checkpoint.*limb"):
        make_transfer_trainer(tmp_path, limb=Limb.HAND).resume(make_leg_checkpoint(tmp_path))
    with pytest.raises(FloatingPointError, match="NaN|Inf"):
        FiniteTrainingCallback().check({"loss": float("nan")})
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_trainers.py -v`

Expected: FAIL because trainers are absent.

- [ ] **Step 3: Implement Lattice-style SB3 lifecycle**

```python
class TransferTrainer:
    def run(self):
        model = PPO("MlpPolicy", self.vec_env,
                    n_steps=self.config.ppo.n_steps,
                    batch_size=self.config.ppo.batch_size,
                    learning_rate=self.config.ppo.learning_rate,
                    seed=self.config.seed,
                    tensorboard_log=str(self.run_dir / "tensorboard"))
        callbacks = CallbackList([self.eval_callback, self.interaction_callback,
                                  self.decoder_finetune_callback])
        model.learn(self.config.total_timesteps, callback=callbacks,
                    reset_num_timesteps=not self.config.resume)
        return self.artifacts()
```

`load_checkpoint_bundle()` must compare schema version, limb, source task, target task, action dimension, latent dimension, and data fingerprint before restoring. `FiniteTrainingCallback` checks rollout values, losses, actions, and gradients and raises `FloatingPointError` before writing `latest`.

- [ ] **Step 4: Run trainer tests**

Run: `python -m pytest tests/test_trainers.py -v`

Expected: tests complete in under 60 seconds on dummy environments and produce latest/best checkpoints plus CSV logs.

- [ ] **Step 5: Commit**

```bash
git add src/etl_sar/trainers.py tests/test_trainers.py
git commit -m "feat: add SB3 ETL exploration and transfer trainers"
```

### Task 9: CLI, Quick Protocols, Evaluation, and Documentation

**Files:**
- Create: `src/etl_sar/evaluation.py`
- Create: `src/etl_sar/cli.py`
- Create: `configs/hand_quick.yaml`
- Create: `configs/leg_quick.yaml`
- Create: `tests/test_cli.py`
- Create: `tests/test_evaluation.py`
- Create: `tests/test_myo_smoke.py`
- Create: `README.md`
- Create: `.gitignore`

**Interfaces:**
- Consumes: all previous public interfaces.
- Produces: `etl-sar inspect|explore|fit-representation|transfer|evaluate|compare` and `evaluate_checkpoint()`.

- [ ] **Step 1: Write failing CLI and deterministic evaluation tests**

```python
def test_inspect_prints_etl_sar_lattice_roles(runner, tmp_path):
    result = runner.invoke(app, ["inspect", "--config", str(write_config(tmp_path))])
    assert result.exit_code == 0
    assert "ETL core" in result.stdout
    assert "SAR transfer" in result.stdout
    assert "SB3 engineering" in result.stdout

def test_compare_reports_equal_budget_delta():
    table = compare_runs(make_eval("sb3_etl", 0.4), make_eval("sar_guided_etl", 0.6))
    assert table["success_rate_delta"] == pytest.approx(0.2)
```

```python
def test_evaluation_never_changes_parameters(tmp_path):
    model = make_evaluation_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    evaluate_checkpoint(model, make_eval_env(), episodes=2, output_dir=tmp_path)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
```

- [ ] **Step 2: Run and verify failure**


Run: `python -m pytest tests/test_cli.py tests/test_evaluation.py -v`

Expected: FAIL because CLI and evaluation modules do not exist.

- [ ] **Step 3: Implement commands and quick configs**

```python
@app.command()
def compare(baseline: Path, extension: Path, output: Path):
    summary = compare_runs(load_summary(baseline), load_summary(extension))
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

@app.command()
def inspect(config: Path):
    cfg = ExperimentConfig.from_yaml(config)
    cfg.validate()
    typer.echo("ETL core | SAR transfer | SB3 engineering")
```

Hand config uses exact `myoHandReorient8-v0` and `myoHandReorient100-v0`. Leg config uses explicit registered IDs for Flat Walk and Uneven; `inspect` must fail if either is unavailable rather than substitute another task.

- [ ] **Step 4: Add optional MyoSuite smoke tests**

```python
myosuite = pytest.importorskip("myosuite")

@pytest.mark.myo
def test_hand_source_environment_contract():
    env = gym.make("myoHandReorient8-v0")
    validate_environment(env, expected_limb=Limb.HAND)
```

- [ ] **Step 5: Run complete verification**

Run: `python -m pytest -m "not myo" -v`

Expected: all pure and dummy-environment tests pass.

Run: `python -m pytest -m myo -v`

Expected: tests pass when MyoSuite task IDs are installed; otherwise tests skip with a clear dependency reason, never silently pass as another environment.

Run: `etl-sar inspect --config configs/hand_quick.yaml`

Expected: validated role summary and exact task IDs.

- [ ] **Step 6: Commit**

```bash
git add README.md .gitignore configs src/etl_sar/cli.py src/etl_sar/evaluation.py tests/test_cli.py tests/test_evaluation.py tests/test_myo_smoke.py
git commit -m "feat: add runnable ETL-SAR experiment workflow"
```

## Final Verification

- [ ] Run `python -m pytest -m "not myo" -v` and require zero failures.
- [ ] Run `python -m compileall src tests` and require exit code 0.
- [ ] Run `etl-sar inspect` for both quick configs.
- [ ] Run the dummy-environment explore, representation, and transfer smoke workflow.
- [ ] Run `git status --short` and confirm only intentionally untracked runtime outputs remain.
- [ ] Record any unavailable MyoSuite/SAR custom environment IDs as explicit external prerequisites, not successful tests.
