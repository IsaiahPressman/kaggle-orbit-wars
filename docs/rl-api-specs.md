# RL API Specs

This document describes the currently available RL observation and action specs.
The Python config API uses pydantic discriminator fields so future specs can add
different options without changing the outer `VectorizedEnv` constructor shape.
`EnvConfig.n_envs` defaults to `2` and must be even so built-in checkpoint
evaluation can split evaluation games across 2-player and 4-player batches.
All tensor shapes in this document are local to one `VectorizedEnv` instance.
Distributed PPO creates one vectorized environment per rank, so global rollout
width is a training-layer concern rather than an RL API shape change.

```python
from owl.rl import ActionPureConfig, EntityBasedConfig, VectorizedEnv

env = VectorizedEnv(
    n_envs=128,
    obs_spec=EntityBasedConfig(max_entities=512),
    action_spec=ActionPureConfig(max_per_planet_launches=3),
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

## EntityBased

Config:

```python
{"obs_spec": "entity_based", "max_entities": 512}
```

`EntityBased` writes observations into reusable caller-owned buffers. The vectorized
environment returns an `ObsBatch` with these tensors:

| Tensor | dtype | Shape |
| --- | --- | --- |
| `planets` | `float32` | `(n_envs, MAX_PLANETS, 107)` |
| `orbiting_planets` | `bool` | `(n_envs, MAX_PLANETS)` |
| `fleets` | `float32` | `(n_envs, max_fleets, 79)` |
| `comets` | `float32` | `(n_envs, MAX_COMETS, 330)` |
| `entity_mask` | `bool` | `(n_envs, max_entities)` |
| `still_playing` | `bool` | `(n_envs, 4)` |
| `global_features` | `float32` | `(n_envs, 3)` |
| `can_act` | `bool` | action-spec dependent |
| `max_launch` | `int64` | `(n_envs, 4, ACTION_ENTITY_SLOTS)` |

All reused buffers are fully overwritten on each observation write. Inactive
rows are zero-filled and their `entity_mask` slots are set to `False`.
`entity_mask` is ordered as all planet slots, then comet slots, then fleet
slots. This matches the model token order and keeps the action entity axis in
the first `ACTION_ENTITY_SLOTS` positions.

`still_playing` is true for outer player slots that are active in the current
observation's episode and have not finished. The Rust vectorized environment
samples a fresh random internal-to-outer player-slot mapping for each sub-env
reset. In 4-player episodes all four outer slots are active; in 2-player
episodes any two of the four outer slots may be active. Inactive outer player
slots are `False`. This mapping also randomizes starting-seat assignment for
fixed outer-slot policies in evaluation and benchmarking code, so callers do not
need to rotate policy-to-slot assignments themselves. After a terminal
auto-reset, `still_playing` describes the returned reset observation, while
`dones` still describes the transition that just finished.

### Normalization

- Positions use `x_norm = (x / BOARD_SIZE) * 2 - 1`, with `BOARD_SIZE = 100`.
  The four map corners are `(-1, -1)`, `(1, -1)`, `(-1, 1)`, and `(1, 1)`.
- Radius uses `radius / 3`.
- Neutral planet linear ships use `ships / 100`. Player-owned planet, fleet,
  and comet linear ships use `ships / 500`.
- Log ships use `ln(ships + 1) / ln(100)`.
- Ship-count basis channels append linear two-hot, ln-space two-hot, and
  overflow features. Zero ships are a special-case bucket when the bucket grid
  includes zero. Neutral planet and comet buckets are
  `[0, 1, 2, 4, 8, 16, 32, 64, 99]`; owned planet buckets are
  `[0, 1, 2, 4, ..., 1024]`; owned comet buckets are
  `[0, 1, 2, 4, ..., 512]`. Fleet buckets omit the zero bucket, start at `1`,
  then continue with the next power of two strictly above `min_fleet_size`, up
  to `512`. The two overflow channels are `ships > max_bucket` and
  `ln(max(ships - max_bucket, 1))`.
- Angular velocity uses `(angular_velocity - 0.025) / 0.025`. Generated games
  currently map the expected range `[0.025, 0.05]` to `[0, 1]`. The value is
  not clamped.
- `steps_until_next_comet_spawn` is divided by `100`.
- Appended spatial channels use the already-normalized `x` and `y` values.
  Cartesian Fourier features use frequencies `[1, 2, 4, 8, 16, 32]`. Radial
  Fourier features use frequencies `[1, 2, 4, 8]`.

### Planet Tensor

Shape per env: `(MAX_PLANETS, 107)`.

Only non-comet planets are included. If more than `MAX_PLANETS` non-comet
planets exist, the encoder panics. Generated games currently produce up to
`MAX_PLANET_GROUPS * 4 = 40` planets. Rows are written in increasing planet ID
order after excluding comet planets. This matches generated-map order because
generated planet IDs are unique and contiguous before comet insertion.

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | neutral owner |
| `5` | normalized `x` |
| `6` | normalized `y` |
| `7..11` | production one-hot for production values `1..5` |
| `12` | normalized radius |
| `13` | neutral normalized ships, else `0` |
| `14` | neutral normalized log ships, else `0` |
| `15` | player-owned normalized ships, else `0` |
| `16` | player-owned normalized log ships, else `0` |
| `17..37` | neutral planet ship-count basis, else `0` |
| `37..63` | player-owned planet ship-count basis, else `0` |
| `63..87` | Cartesian Fourier position features for normalized `(x, y)` |
| `87` | sun-centered radius `r = sqrt(x^2 + y^2)` |
| `88` | `log1p(r)` |
| `89` | `sin(theta)` for `theta = atan2(y, x)` |
| `90` | `cos(theta)` |
| `91..97` | angular harmonics `sin(k theta), cos(k theta)` for `k = 2..4` |
| `97..105` | radial Fourier features `sin(pi f r), cos(pi f r)` |
| `105` | orbiting planet `vx = -angular_velocity * y`, else `0` |
| `106` | orbiting planet `vy = angular_velocity * x`, else `0` |

### Orbiting Planet Tensor

Shape per env: `(MAX_PLANETS,)`.

Rows are aligned with the planet tensor. A row is `True` if the matching planet
row is orbiting, else `False`. Inactive rows are `False`.

`entity_mask[i]` is `True` only for active planet rows for
`i < MAX_PLANETS`.

### Fleet Tensor

Shape per env: `(max_fleets, 79)`.

When all active fleets fit in `max_fleets`, fleets are emitted in simulator
fleet order. If there are more active fleets than `max_fleets`, fleets are
sorted by descending ship count, with fleet id as the tie-breaker, so the
largest fleets are kept and the rest are ignored. Each overflow logs to stderr:

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
| `10..32` | fleet ship-count basis |
| `32..56` | Cartesian Fourier position features for normalized `(x, y)` |
| `56` | sun-centered radius `r = sqrt(x^2 + y^2)` |
| `57` | `log1p(r)` |
| `58` | `sin(theta)` for `theta = atan2(y, x)` |
| `59` | `cos(theta)` |
| `60..66` | angular harmonics `sin(k theta), cos(k theta)` for `k = 2..4` |
| `66..74` | radial Fourier features `sin(pi f r), cos(pi f r)` |
| `74` | normalized speed `sqrt(vx^2 + vy^2)` |
| `75` | heading `x` component, or `0` for zero-speed fleets |
| `76` | heading `y` component, or `0` for zero-speed fleets |
| `77` | radial velocity in the sun-centered radial basis |
| `78` | tangential velocity in the counterclockwise tangent basis |

`entity_mask[ACTION_ENTITY_SLOTS + i]` is `True` only for active fleet rows.

### Comet Tensor

Shape per env: `(MAX_COMETS, 330)`.

Comets are encoded separately from normal planets. Active comet planet IDs are
sorted in ascending ID order, deduplicated, and emitted up to `MAX_COMETS`.
This matches the comet portion of the action entity axis.

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | neutral owner |
| `5` | normalized ships |
| `6` | normalized log ships |
| `7..27` | neutral comet ship-count basis, else `0` |
| `27..51` | player-owned comet ship-count basis, else `0` |
| `51` | remaining path steps divided by `MAX_COMET_PATH_LENGTH` |
| `52` | current normalized `x` from the path |
| `53` | current normalized `y` from the path |
| `54..96` | current Cartesian Fourier, polar, angular-harmonic, and radial Fourier spatial features |
| `96` | normalized `vx` from the next path point minus the current path point |
| `97` | normalized `vy` from the next path point minus the current path point |
| `98` | speed `sqrt(vx^2 + vy^2)` |
| `99` | heading `x` component, or `0` if no next path point exists |
| `100` | heading `y` component, or `0` if no next path point exists |
| `101` | radial velocity in the sun-centered radial basis |
| `102` | tangential velocity in the counterclockwise tangent basis |
| `103..108` | future-valid flags for offsets `[1, 2, 4, 8, 16]`, encoded as `0.0` or `1.0` |
| `108..118` | selected future normalized `(x, y)` pairs for offsets `[1, 2, 4, 8, 16]` |
| `118..328` | spatial features for each selected future position |
| `328` | normalized final path `x` minus current normalized `x` |
| `329` | normalized final path `y` minus current normalized `y` |

The current path point starts at the comet group's current `path_index`. If
`path_index < 0`, the encoder starts at path index `0`. If a selected future
offset is outside the remaining known path, its valid flag and feature slots are
zero-filled. If no next path point exists, comet velocity, speed, heading,
radial velocity, and tangential velocity are zero-filled. The displacement to
the final path point uses the path's last known point.

`entity_mask[MAX_PLANETS + i]` is `True` only for active comet rows.

### Global Tensor

Shape per env: `(3,)`.

| Index | Feature |
| --- | --- |
| `0` | `step / episode_steps` |
| `1` | `steps_until_next_comet_spawn / 100` |
| `2` | normalized angular velocity |

Standalone observation encoding uses strict application-boundary parsing:
required observation keys are read directly, `step` and `episode_steps` must be
integers rather than coerced strings or floats, comet groups must provide
matching `planet_ids`, `paths`, and `path_index`, and comet/path overflow is
rejected rather than truncated. It also rejects non-finite `angular_velocity`
and requires `episode_steps > 0` before computing these values.
`encode_python_observation()` returns a single-env `ObsBatch` with tensors
shaped as `(1, ...)`, matching model input shape directly.

Comet spawn steps currently come from the rules engine constant:

```text
[50, 150, 250, 350, 450]
```

If no future comet spawn remains, `steps_until_next_comet_spawn` is `0`.

## Action Entity Slots

Action entity slots are ordered as all `MAX_PLANETS` planet tokens first,
followed by `MAX_COMETS` comet tokens:

```text
0..39  -> non-comet planet slots in ascending planet ID order
40..43 -> comet slots in ascending comet planet ID order
```

Unused planet slots, unused comet slots, and inactive player slots are explicitly
filled with `False` / `0` in action-spec output tensors.

## Pure Action Spec

Config:

```python
{"action_spec": "pure", "max_per_planet_launches": 3, "min_fleet_size": 1}
```

The pure action spec exposes all launch decisions in direct tensor form. The
same entity axis is used for action masks and submitted actions.
`max_per_planet_launches` is validated in Python and Rust and must be between
`1` and `4`, inclusive. `min_fleet_size` is validated in Python and Rust and
must fit in the positive `i32` ship-count range. `ActionPureConfig()` defaults
to `max_per_planet_launches=3` and `min_fleet_size=1` so callers use the
existing multi-launch autoregressive action space unless they explicitly opt
into a smaller action shape or larger minimum fleet.

### Pure Output Tensors

These are written alongside the observation tensors:

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `can_act` | `bool` | `(n_envs, 4, 44)` | whether a player can launch from an entity slot |
| `max_launch` | `int64` | `(n_envs, 4, 44)` | maximum launchable ship count for that slot |

`can_act[player, entity]` is true when the entity is owned by that outer player
slot and has at least `min_fleet_size` ships. In 2-player games, two random
outer player slots are active for the episode and the other two are inactive.

### Pure Submitted Actions

Call:

```python
obs, rewards, dones, episode_metrics = env.step(launch, angle, ships)
```

`rewards` and `dones` have shape `(n_envs, 4)`. Inactive player slots are
always `done=True` with reward `0`. Active players receive reward `0` with
`done=False`. A player that newly loses receives reward `-1` with `done=True`;
later steps for that already-finished player stay `done=True` with reward `0`.
A sole winner receives reward `1` with `done=True`. If multiple players tie as
winners, each tied winner receives the average of one winner reward and the
remaining tied loser rewards: `(1 - (winner_count - 1)) / winner_count`.

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `launch` | `bool` | `(n_envs, 4, 44, max_per_planet_launches)` | `True` means execute a launch |
| `angle` | `float32` | `(n_envs, 4, 44, max_per_planet_launches)` | launch angle in radians |
| `ships` | `int64` | `(n_envs, 4, 44, max_per_planet_launches)` | requested ship count |

Python requires exact submitted action dtypes at the boundary; wrong dtypes are
rejected instead of cast.
If `launch` is `False`, that slot is a no-op and `angle` / `ships` are ignored.
If `launch` is `True`, `ships >= min_fleet_size`, `ships <= i32::MAX`, a finite
`angle`, a valid source entity, source ownership by the acting player, and
enough remaining ships on that source are required. Invalid submitted actions
raise `ValueError`. Invalid-action errors are not atomic across sub-envs:
because decoding, stepping, and observation writing run in one parallel pass per
env, other sub-envs may have advanced before the error is returned.
For each player and source entity, decoding stops at the first `False` launch
slot, so later slots for that source are ignored.

For Kaggle submissions, `actions_to_kaggle(obs, player, launch, angle, ships,
action_spec=...)` accepts a single batched model output with shape
`(1, 4, 44, max_per_planet_launches)` and returns the selected player's
`list[list[float]]` action triples. Pure triples are
`[from_planet_id, angle, ships]` and use the same Rust validation path as
environment stepping.

## Discrete Targets Action Spec

Config:

```python
{"action_spec": "discrete_targets", "max_per_planet_launches": 3, "min_fleet_size": 1}
```

`ActionDiscreteTargetsConfig` uses the same launch-count and minimum-fleet
validation as `ActionPureConfig`, but submitted actions choose integer target
entity slots instead of raw launch angles. `StatelessTransformerV1` and PPO can
train against this action spec when the model actor config also uses
`"discrete_targets"` and `max_per_planet_launches=1`.

### Discrete Targets Output Tensors

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `can_act` | `bool` | `(n_envs, 4, 44, 44)` | whether a player can launch from a source slot to a target slot |
| `max_launch` | `int64` | `(n_envs, 4, 44)` | maximum launchable ship count for that source slot |

`can_act[player, source, target]` is true when the source entity is owned by
that outer player slot, has at least `min_fleet_size` ships, the target slot
exists, and `source != target`. Neutral, enemy, owned, planet, and comet targets
are all legal if they exist.

### Discrete Targets Submitted Actions

Call:

```python
obs, rewards, dones, episode_metrics = env.step(launch, target, ships)
```

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `launch` | `bool` | `(n_envs, 4, 44, max_per_planet_launches)` | `True` means execute a launch |
| `target` | `int64` | `(n_envs, 4, 44, max_per_planet_launches)` | target action entity slot index in `[0, 44)` |
| `ships` | `int64` | `(n_envs, 4, 44, max_per_planet_launches)` | requested ship count |

Validation matches `pure` for inactive players, source ownership, source budget,
ship count, missing source slots, and stale source slots. Launched target slots
must be in range, present, and different from the source slot.
The Kaggle conversion helper accepts the same batched model output shape and
uses the Rust target decoder to convert discrete target slots into submitted
`[from_planet_id, angle, ships]` triples for one selected player.

### Targeting Rules

For static planets, decoding first tries the centerline. If that path is blocked
by the sun or another static planet, it tries both edge paths using
`target_radius - eps`, then prefers an unobstructed path, then a path that avoids
the sun, and finally the centerline if every candidate hits the sun.

For orbiting non-comet planets, decoding computes a single centerline
time-of-impact trajectory against the analytic orbit curve. It first uses the
orbit radius to bound the possible impact interval, then uses monotonic bisection
when the fleet is faster than the target's tangential speed, otherwise a capped
coarse scan followed by bisection. This fast orbiting path does not validate sun
or planet obstruction before emitting the launch angle. For comets, decoding
solves analytic centerline intercepts against each stored linear path segment,
then applies the existing sun and static-planet blocker preference over the
resulting candidates. If no comet intercept exists within the known future path,
the submitted launch is treated as a no-op and counted in
`comet_launch_failures_per_game`.

## Replay Snapshots

`VectorizedEnv.state_snapshot(env_index)` returns a JSON-serializable snapshot
of one current Rust sub-env. `VectorizedEnv.terminal_snapshot(env_index)` returns
the terminal snapshot captured during the most recent `step(...)`, or `None` if
that sub-env did not terminate on that step. Terminal snapshots are captured
after the terminal transition and before the vectorized env auto-resets the
sub-env.

Snapshots are intended for replay rendering and debugging, not model input. They
include raw board-space values: board constants, step/config fields, player
count, owner IDs remapped into outer player slots, the internal/outer player map,
outer-slot `player_finished`, action entity slot planet IDs, planets, fleets,
comet groups, and comet paths. Neutral ownership remains `-1`, and each planet
or fleet also includes `internal_owner` when the engine-owned ID is needed.

## Episode Metrics

`episode_metrics` is a `dict[str, list[float]]` populated only for sub-envs that
terminated during this step. Empty steps return `{}`. Each list contains one
value per terminal episode for that metric, so Python training can aggregate
across the rollout before logging W&B scalars under `train/`.

Terminal episode metrics:

| Key | Meaning |
| --- | --- |
| `max_entities_exceeded_per_game` | Count of post-step turns where active fleets exceeded `max_fleets`. |
| `game_length_mean` | Terminal game step count. |
| `full_length_rate` | `1.0` when a game reaches the configured episode horizon, otherwise `0.0`. |
| `terminal_ship_count` | Total ships on planets and in active fleets at terminal. |
| `planets_captured_per_game` | Total planet captures over the episode, counting repeat captures. |
| `comets_captured_per_game` | Total comet planet captures over the episode, counting repeat captures. |
| `comet_launch_failures_per_game` | Submitted discrete-target comet launches skipped because no intercept exists before the comet leaves its known path. Python training logs this as `train/comet_launch_failures_per_game`. |
| `launches_per_turn` | Mean launches per player per turn. |
| `fleet_size_max` | Largest fleet launched during the episode. |
| `fleet_size_min` | Smallest fleet launched during the episode, or `0.0` when no fleets launched. |
| `fleet_size_std` | Population standard deviation of launched fleet sizes during the episode. |
| `win_rate_player_0`..`win_rate_player_3` | `1.0` for a winning model-visible outer player slot, `0.0` otherwise; inactive outer slots have no value in 2-player games. |
| `launches_per_planet_mean` | Per-game mean launches per occupied non-comet planet per turn. |
| `launches_per_launch_mean` | Mean launches from a planet on planet-turns where that planet launched at least once. |
| `ships_per_launch_mean` | Mean submitted ship count per launch action. |
| `ships_lost_in_combat_per_game` | Ships destroyed during fleet-vs-fleet and fleet-vs-planet combat resolution. |
| `ships_lost_per_game_mean` | Ships removed by combat, sun, or out-of-bounds fleet loss. |
| `ships_lost_in_sun_per_game_mean` | Ships removed by sun fleet loss. |
| `ships_lost_out_of_bounds_per_game_mean` | Ships removed by out-of-bounds fleet loss. |
| `ships_lost_to_sun_or_oob_rate` | `(ships_lost_in_sun_per_game_mean + ships_lost_out_of_bounds_per_game_mean) / ships_lost_per_game_mean`. Omitted when no ships were lost. |
| `fleets_lost_in_combat_per_game` | Fleets removed during planet/combat resolution. |
| `fleets_lost_per_game_mean` | Fleets removed by combat, sun, or out-of-bounds loss. |
| `fleets_lost_in_sun_per_game_mean` | Fleets removed by sun loss. |
| `fleets_lost_out_of_bounds_per_game_mean` | Fleets removed by out-of-bounds loss. |
| `fleets_lost_to_sun_or_oob_rate` | `(fleets_lost_in_sun_per_game_mean + fleets_lost_out_of_bounds_per_game_mean) / fleets_lost_per_game_mean`. Omitted when no fleets were lost. |
| `terminal_planet_occupancy_rate_2p` | Occupied non-comet planet fraction at terminal for 2-player games. |
| `terminal_planet_occupancy_rate_4p` | Occupied non-comet planet fraction at terminal for 4-player games. |
