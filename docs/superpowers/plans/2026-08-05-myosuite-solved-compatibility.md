# MyoSuite Success Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve successful MyoSuite source trajectories so SAR representation fitting can consume them.

**Architecture:** Normalize the two supported success-info conventions at the exploration callback, immediately before trajectory records are built. Keep the trajectory schema and all ETL/SAR training behavior unchanged.

**Tech Stack:** Python 3.11, Gymnasium, MyoSuite 2.12.2, Stable-Baselines3, pytest

## Global Constraints

- ETL remains the dominant algorithm; this bug fix must not alter rewards or training objectives.
- `success` takes precedence when present; `solved` is a fallback only.
- A missing success indicator maps to `False`.
- Hand and Leg must use the same normalization rule.

---

### Task 1: Normalize MyoSuite Success Metadata

**Files:**
- Modify: `tests/test_explore_trainer.py`
- Modify: `src/etl_sar/exploration.py:139`

**Interfaces:**
- Consumes: Gymnasium step `info: dict[str, Any]`
- Produces: `TrajectoryEpisode.success: bool` through the existing callback record

- [ ] **Step 1: Write the failing test**

Add a source environment that reports only MyoSuite's `solved` field and assert
that the real exploration lifecycle populates the successful-action pool:

```python
class TinySolvedSourceEnv(TinySourceEnv):
    def reset(self, *, seed=None, options=None):
        observation, _ = super().reset(seed=seed, options=options)
        return observation, {"solved": False}

    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, {"solved": terminated}


def test_explore_trainer_accepts_myosuite_solved_flag(tmp_path) -> None:
    store = TrajectoryStore(
        tmp_path / "data", limb=Limb.HAND, source_task="reorient8", action_dim=6
    )
    trainer = ExploreTrainer(
        env_factory=TinySolvedSourceEnv,
        state_encoder=StateEncoder(4, 20, hidden_dims=(16,)),
        representation=make_representation(),
        trajectory_store=store,
        limb=Limb.HAND,
        source_task="reorient8",
        run_dir=tmp_path / "run",
        total_timesteps=16,
        n_steps=16,
        batch_size=8,
        representation_update_interval=12,
        seed=3,
    )
    trainer.run()
    assert store.success_pool().shape[0] > 0
```

- [ ] **Step 2: Verify the regression test fails**

Run: `python -m pytest tests/test_explore_trainer.py::test_explore_trainer_accepts_myosuite_solved_flag -q`

Expected: FAIL because `success_pool()` has zero rows.

- [ ] **Step 3: Implement the minimal compatibility mapping**

Replace the callback record's success expression with:

