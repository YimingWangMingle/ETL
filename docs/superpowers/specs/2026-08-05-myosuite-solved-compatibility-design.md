# MyoSuite Success Compatibility Design

## Problem

The exploration callback currently reads only `info["success"]`. MyoSuite 2.12.2
reports task completion as the top-level `info["solved"]` field for both the Hand
and Leg source tasks. Consequently every collected MyoSuite episode is stored as
unsuccessful and SAR representation fitting cannot obtain its required pool of at
least 20 successful source-task actions.

## Selected Design

Normalize the environment success protocol at the trajectory collection boundary:

```python
bool(info.get("success", info.get("solved", False)))
```

`success` remains authoritative when explicitly present. `solved` is used only
when `success` is absent. Missing fields remain unsuccessful. The change does not
alter rewards, episode termination, SAR thresholds, or stored trajectory schema.

## Alternatives Considered

1. Read only `solved`. This matches MyoSuite but breaks existing Gym environments
   and the project's current test environment, which report `success`.
2. Prefer `success` and fall back to `solved`. This preserves existing behavior
   and adds the required MyoSuite compatibility with one localized change.
3. Add a configurable success-field name. This supports more protocols but adds
   configuration and validation that are unnecessary for the Hand/Leg scope.

Option 2 is selected.

## Testing

A regression environment will expose `solved=True` on terminal transitions and no
`success` field. A short `ExploreTrainer` run must store those episode actions in
`TrajectoryStore.success_pool()`. The existing `success`-based test must continue
to pass.

The full non-MyoSuite suite and real MyoSuite smoke suite must pass. A read-only
