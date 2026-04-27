# RL API Specs

This document describes the currently available RL observation and action specs.
The Python config API uses pydantic discriminator fields so future specs can add
different options without changing the outer `VectorizedEnv` constructor shape.

```python
from owl.rl import ActionPureConfig, ObsV1Config, VectorizedEnv

env = VectorizedEnv(
    n_envs=128,
    obs_spec=ObsV1Config(max_entities=512),
    action_spec=ActionPureConfig(max_per_planet_launches=1),
)
```

## Shared Constants

- `MAX_PLANETS = 40`
- `MAX_COMETS = 4`
- `MAX_COMET_PATH_LENGTH = 40`
- `DEFAULT_MAX_ENTITIES = 512`
- `ACTION_ENTITY_SLOTS = MAX_PLANETS + MAX_COMETS = 44`
- `OUTER_PLAYER_SLOTS = 4`

`max_entities` controls total non-global entity capacity. Fleet capacity is:

```text
max_fleets = max_entities - (MAX_PLANETS + MAX_COMETS)
```

The default `max_entities=512` gives `max_fleets=468`.

`MAX_PLANETS` matches the current generated-map upper bound:
`MAX_PLANET_GROUPS * 4 = 10 * 4`. `MAX_COMET_PATH_LENGTH` matches the comet
generator's maximum accepted visible path length; generated comet paths outside
the range `5..=40` are rejected.

## ObsV1

Config:

```python
{"obs_spec": "obs_v1", "max_entities": 512}
```

`ObsV1` writes observations into reusable caller-owned buffers. The vectorized
environment returns an `ObsBatch` with these tensors:

| Tensor | dtype | Shape |
| --- | --- | --- |
| `planets` | `float32` | `(n_envs, MAX_PLANETS, 16)` |
| `fleets` | `float32` | `(n_envs, max_fleets, 10)` |
| `comets` | `float32` | `(n_envs, MAX_COMETS, 88)` |
| `planet_mask` | `bool` | `(n_envs, MAX_PLANETS)` |
| `fleet_mask` | `bool` | `(n_envs, max_fleets)` |
| `comet_mask` | `bool` | `(n_envs, MAX_COMETS)` |
| `still_playing` | `bool` | `(n_envs, 4)` |
| `global_features` | `float32` | `(n_envs, 3)` |
| `can_act` | `bool` | `(n_envs, 4, ACTION_ENTITY_SLOTS)` |
| `max_launch` | `int64` | `(n_envs, 4, ACTION_ENTITY_SLOTS)` |

All reused buffers are fully overwritten on each observation write. Inactive
rows are zero-filled and their masks are set to `False`.

`still_playing` is true for player slots that are still active in the episode.
The Python wrapper currently initializes it to `True` on reset and mirrors
`~dones` after `step`; the Rust observation writer will own this tensor once it
is wired into the low-level API.

### Normalization

- Positions use `x_norm = (x / BOARD_SIZE) * 2 - 1`, with `BOARD_SIZE = 100`.
  The four map corners are `(-1, -1)`, `(1, -1)`, `(-1, 1)`, and `(1, 1)`.
- Radius uses `radius / 3`.
- Ships use `ships / 250`.
- Log ships use `ln(max(ships, 0) + 1) / ln(100)`. Planets can reach zero
  ships after combat, so this avoids `ln(0)`.
- Angular velocity uses `(angular_velocity - 0.025) / 0.025`. Generated games
  currently map the expected range `[0.025, 0.05]` to `[0, 1]`. The value is
  not clamped.
- `steps_until_next_comet_spawn` is divided by `100`.

### Planet Tensor

Shape per env: `(MAX_PLANETS, 16)`.

Only non-comet planets are included. If more than `MAX_PLANETS` non-comet
planets exist, the encoder panics. Generated games currently produce up to
`MAX_PLANET_GROUPS * 4 = 40` planets. Rows are written in the simulator planet
order after excluding comet planets.

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | neutral owner |
| `5` | normalized `x` |
| `6` | normalized `y` |
| `7..11` | production one-hot for production values `1..5` |
| `12` | normalized radius |
| `13` | normalized ships |
| `14` | normalized log ships |
| `15` | `1.0` if orbiting, else `0.0` |

`planet_mask[i]` is `True` only for active planet rows.

### Fleet Tensor

Shape per env: `(max_fleets, 10)`.

Fleets are sorted by descending ship count, with fleet id as the tie-breaker.
If there are more active fleets than `max_fleets`, the largest fleets are kept
and the rest are ignored. Each overflow logs to stderr:

```text
max_entities exceeded: N fleets ignored
```

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | normalized `x` |
| `5` | normalized `y` |
| `6` | normalized `vx`, divided by `shipSpeed` |
| `7` | normalized `vy`, divided by `shipSpeed` |
| `8` | normalized ships |
| `9` | normalized log ships |

`fleet_mask[i]` is `True` only for active fleet rows.

### Comet Tensor

Shape per env: `(MAX_COMETS, 88)`.

Comets are encoded separately from normal planets. They are emitted in comet
group order, then `planet_ids` order within each comet group, up to
`MAX_COMETS`.

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | neutral owner |
| `5` | normalized ships |
| `6` | normalized log ships |
| `7` | remaining path steps divided by `MAX_COMET_PATH_LENGTH` |
| `8..87` | future path positions as `MAX_COMET_PATH_LENGTH` `(x, y)` pairs |

The future path starts at the comet group's current `path_index`. In normal
post-step observations for an active comet, the first `(x, y)` pair is the
comet's current position, followed by future positions. If `path_index < 0`,
the encoder starts at path index `0`. Each path position is normalized with the
same `[-1, 1]` map transform as planets and fleets. Unused path slots are
zero-filled.

`comet_mask[i]` is `True` only for active comet rows.

### Global Tensor

Shape per env: `(3,)`.

| Index | Feature |
| --- | --- |
| `0` | `step / episode_steps` |
| `1` | `steps_until_next_comet_spawn / 100` |
| `2` | normalized angular velocity |

Comet spawn steps currently come from the rules engine constant:

```text
[50, 150, 250, 350, 450]
```

If no future comet spawn remains, `steps_until_next_comet_spawn` is `0`.

## Pure Action Spec

Config:

```python
{"action_spec": "pure", "max_per_planet_launches": 1}
```

The pure action spec exposes all launch decisions in direct tensor form. The
same entity axis is used for action masks and submitted actions.
`max_per_planet_launches` is validated in Python and Rust and must be between
`1` and `4`, inclusive.

Sharp edge: action entity slots are ordered as all `MAX_PLANETS` planet tokens
first, followed by `MAX_COMETS` comet tokens. This assumes the model appends
comet hidden states after planet hidden states before producing actions.

```text
0..39  -> planet slots
40..43 -> comet slots
```

Unused planet slots, unused comet slots, and inactive player slots are explicitly
filled with `False` / `0` in the action-spec output tensors.

### Action-Spec Output Tensors

These are written alongside the observation tensors:

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `can_act` | `bool` | `(n_envs, 4, 44)` | whether a player can launch from an entity slot |
| `max_launch` | `int64` | `(n_envs, 4, 44)` | maximum launchable ship count for that slot |

`can_act[player, entity]` is true when the entity is owned by that player and
has at least one ship. In 2-player games, player slots `2` and `3` are always
inactive.

### Submitted Action Tensors

Call:

```python
obs, rewards, dones = env.step(launch, angle, ships)
```

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `launch` | `bool` | `(n_envs, 4, 44, max_per_planet_launches)` | `True` means execute a launch |
| `angle` | `float32` | `(n_envs, 4, 44, max_per_planet_launches)` | launch angle in radians |
| `ships` | `int64` | `(n_envs, 4, 44, max_per_planet_launches)` | requested ship count |

If `launch` is `False`, that slot is a no-op and `angle` / `ships` are ignored.
If `launch` is `True`, `ships >= 1` is required and the Rust API panics if the
invariant is violated.
For each player and source entity, decoding stops at the first `False` launch
slot, so later slots for that source are ignored.

The pure action decoder currently performs only the explicit `ships >= 1`
validation itself. Ownership, source validity, and overspending are still
checked by the rules engine's normal fail-fast launch validation.
