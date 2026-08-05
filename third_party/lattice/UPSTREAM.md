# Lattice Upstream Provenance

- Repository: <https://github.com/amathislab/lattice>
- Commit: `846d02fa993b9b80ce5ecb806463e0a05711bad3`
- Paper: *Latent Exploration for Reinforcement Learning*
  (<https://arxiv.org/abs/2305.20065>)
- License: MIT, preserved in `LICENSE`

## Vendored Files

The following files are unmodified copies from the pinned commit. Tests verify
their SHA-256 values before they can be treated as the official baseline source.

| Vendored file | Upstream file | SHA-256 |
| --- | --- | --- |
| `distributions.py` | `src/models/distributions.py` | `D6CC339D73765D588E00E4A07E23265F1333163E866A87BC7A3644061A2A230D` |
| `ppo_policies.py` | `src/models/ppo_policies.py` | `46C826E616BB1614D18C2748F832D194A5251692EBCFAB54BBB94EF66D2C6416` |
| `sac_policies.py` | `src/models/sac_policies.py` | `D046C8AAB013B32DFC1F431F3C49BEF9B0372751464410AC99F10D813A2FA59B` |
| `main_reorient.py` | `src/main_reorient.py` | `A9D4A992B360828E5ED8A68A37B1961581717C0292919670E96090637CFE4CF1` |
| `main_walker.py` | `src/main_walker.py` | `B043C22B1E6B882CF97F4B48D08A5F4DD6E01F7F63BD383AE72F5B69D6990986` |

## Compatibility Port

The runtime distribution equations in
`src/etl_sar/lattice/distributions.py` are byte-identical to the official
vendored source and covered by fixed-tensor parity tests. The policy wrappers in
`src/etl_sar/lattice/policies.py` preserve the official distribution selection
and parameters while adapting constructor and serialization hooks to SB3 2.x and
`sb3-contrib` 2.x.

The training runtime adds only experiment infrastructure absent upstream:
Gymnasium/MyoSuite vector environments, `VecNormalize`, exact transition-budget
callbacks, resumable checkpoints, periodic shared-protocol evaluation, and
server job orchestration. It does not change the Lattice covariance equations.

Environment adapters use MyoSuite 2.12.x and Gymnasium so all compared methods
see the same task implementation. They do not import upstream custom reward
shaping or PyBullet environments. Hand retains the upstream RecurrentPPO
Reorient configuration; Leg retains the upstream SAC Walker configuration with
only the environment replaced by `myoLegRoughTerrainWalk-v0`. These environment
substitutions are necessary because the official repository does not include the
MyoSuite Hand/Leg target pair used by this comparison.
