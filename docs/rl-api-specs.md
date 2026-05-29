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
    obs_spec=EntityBasedConfig(max_entities=256),
    action_spec=ActionPureConfig(max_per_planet_launches=1),
)
```

## Shared Constants

- `MAX_PLANETS = 40`
- `MAX_COMETS = 4`
- `MAX_COMET_PATH_LENGTH = 40`
- `DEFAULT_MAX_ENTITIES = 256`
- `ACTION_ENTITY_SLOTS = MAX_PLANETS + MAX_COMETS = 44`
- `OUTER_PLAYER_SLOTS = 4`
- `GLOBAL_EXT_V2_CHANNELS = 14`
- `PLAYER_FEATURE_CHANNELS = 14`
- `CROSS_ATTENTION_FLEET_CHANNELS = 46`
- `TARGET_INCOMING_CHANNELS = 48`

`max_entities` controls total non-global entity capacity. Fleet capacity is:

```text
max_fleets = max_entities - (MAX_PLANETS + MAX_COMETS)
```

The default `max_entities=256` gives `max_fleets=212`.

`MAX_PLANETS` matches the current generated-map upper bound:
`MAX_PLANET_GROUPS * 4 = 10 * 4`. `MAX_COMET_PATH_LENGTH` matches the comet
generator's maximum accepted visible path length; generated comet paths outside
the range `5..=40` are rejected.

## EntityBased

Config:

```python
{"obs_spec": "entity_based", "max_entities": 256}
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
| `action_mask.can_act` | `bool` | action-spec dependent |
| `action_mask.max_launch` | `int64` | pure and discrete-target masks only; `(n_envs, 4, ACTION_ENTITY_SLOTS)` |

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
For target-bin observations, `DiscreteTargetBinActionMask` has no `max_launch`
member.

## EntityBasedExtV1

Config:

```python
{"obs_spec": "entity_based_ext_v1", "max_entities": 256, "ship_count_one_hot_max": 50}
```

`EntityBasedExtV1` includes every `EntityBased` feature and appends configurable
ship-count one-hot vectors to planet and fleet rows only. Comet rows are
unchanged.

With `ship_count_one_hot_max=N`, planet rows append `N + 1` channels: bin `0`
is exact zero ships, bins `1..N-1` are exact ship counts, and bin `N` is the
overflow bin for `ships >= N`. Fleet rows append `N` channels because fleets
cannot have zero ships: bins `0..N-2` represent exact ship counts `1..N-1`, and
bin `N-1` is the overflow bin for `ships >= N`. The default is `N=50`, so
planet rows have `158` channels and fleet rows have `129` channels by default.

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

The low-level `encode_entity_based` Rust API filters fleets smaller than its
fleet-filter threshold before writing fleet rows. The default threshold is
`min_fleet_size`; callers that need agent-only filtering without changing action
masks can pass a separate `fleet_filter_min_size`. Fleets at or above the
threshold are kept. If a player owns no current planet and none of their fleets
meet the threshold, the encoder keeps that player's largest below-threshold
fleet, using the lower fleet id as the tie-breaker. The API also returns the
number of fleets dropped by this filter.

When all remaining active fleets fit in `max_fleets`, fleets are emitted in
simulator fleet order. If there are more remaining active fleets than
`max_fleets`, fleets are sorted by descending ship count, with fleet id as the
tie-breaker, so the largest fleets are kept and the rest are ignored. Overflow
is silent; vector-env training metrics report
`max_entities_exceeded_per_game` for terminal episodes.

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

## EntityBasedExtV2

Config:

```python
{"obs_spec": "entity_based_ext_v2", "max_entities": 256}
```

`EntityBasedExtV2` extends the base `EntityBased` observation directly. It does
not include the `EntityBasedExtV1` ship-count one-hot appendices, so planet and
fleet row widths remain `107` and `79`.

The spec adds:

| Tensor | dtype | Shape |
| --- | --- | --- |
| `global_features` | `float32` | `(n_envs, 17)` |
| `player_features` | `float32` | `(n_envs, 4, 14)` |

For non-v2 specs, `ObsBatch.player_features` is `None`.

V2 appends fourteen channels after the three base global channels. The
neutral features are for planets that are still neutral in the current
observation; if every planet has been colonized, all neutral production, ship,
and count channels are `0`.

| Channel | Feature |
| --- | --- |
| `3` | neutral total production, including comet planets, divided by `100` |
| `4` | neutral comet-planet production divided by `100` |
| `5` | neutral non-comet planet production divided by `100` |
| `6` | neutral total ships, including comet planets, divided by `5000` |
| `7` | `log1p` of neutral total ships, divided by `ln(1000)` |
| `8` | neutral comet-planet ships divided by `5000` |
| `9` | `log1p` of neutral comet-planet ships, divided by `ln(1000)` |
| `10` | neutral non-comet planet ships divided by `5000` |
| `11` | `log1p` of neutral non-comet planet ships, divided by `ln(1000)` |
| `12` | neutral comet count divided by `MAX_COMETS` |
| `13` | neutral non-comet planet count divided by `MAX_PLANETS` |
| `14` | one-hot for exactly `2` alive players |
| `15` | one-hot for exactly `3` alive players |
| `16` | one-hot for exactly `4` alive players |

Each outer player slot receives fourteen absolute summary channels:

| Channel | Feature |
| --- | --- |
| `0` | total production, including owned comet planets, divided by `100` |
| `1` | comet-planet production divided by `100` |
| `2` | non-comet planet production divided by `100` |
| `3` | total ships, including planets, comet planets, and fleets, divided by `5000` |
| `4` | `log1p` of total ships, divided by `ln(1000)` |
| `5` | comet-planet ships divided by `5000` |
| `6` | `log1p` of comet-planet ships, divided by `ln(1000)` |
| `7` | non-comet planet ships divided by `5000` |
| `8` | `log1p` of non-comet planet ships, divided by `ln(1000)` |
| `9` | fleet ships divided by `5000` |
| `10` | `log1p` of fleet ships, divided by `ln(1000)` |
| `11` | non-comet planet count divided by `MAX_PLANETS` |
| `12` | comet count divided by `MAX_COMETS` |
| `13` | fleet count divided by `100` |

Component production and linear ship channels use the same normalizer as their
total, so comet plus non-comet production equals total production, and comet
plus non-comet planet plus fleet linear ships equals total linear ships.
Inactive outer player slots are zero-filled.

## EntityBasedCrossAttnV1

Config:

```python
{"obs_spec": "entity_based_cross_attn_v1", "max_entities": 256}
```

`EntityBasedCrossAttnV1` keeps the same planet, comet, global, and
per-player features as `EntityBasedExtV2`, but fleet rows are no longer
self-attention entity tokens. The vectorized environment and standalone encoder
add two tensors:

| Tensor | dtype | Shape |
| --- | --- | --- |
| `fleets` | `float32` | `(n_envs, max_fleets, 46)` |
| `fleet_target` | `int64` | `(n_envs, max_fleets)` |
| `target_incoming_features` | `float32` | `(n_envs, ACTION_ENTITY_SLOTS, 48)` |
| `global_features` | `float32` | `(n_envs, 17)` |
| `player_features` | `float32` | `(n_envs, 4, 14)` |

Rust routes current fleets by forward-simulating them until they collide with a
current planet or comet planet, leave the board, hit the sun, or the episode
ends. Fleets that leave the board, hit the sun, or collide only with an
expiring comet planet are omitted. Existing comet paths are simulated; future
comet spawns are intentionally ignored. Routed fleet rows are sorted by
descending ship count only when there are more routed fleets than `max_fleets`.

`entity_mask` keeps the same planet, comet, fleet-tail order. For this spec,
the model treats only the first `ACTION_ENTITY_SLOTS` mask entries as trunk
entity tokens and uses the fleet tail as the cross-attention memory mask.
`fleet_target[i]` is the action-entity slot that fleet row `i` will collide
with, or `-1` for inactive rows. Kaggle serving may compact inactive action
entities before inference; in that runtime path `target_incoming_features` is
sliced to the compacted action-entity axis and `fleet_target` is remapped to the
same compacted indices.

Fleet channels:

| Channels | Feature |
| --- | --- |
| `0..3` | owner one-hot for players `0..3` |
| `4` | normalized ships |
| `5` | normalized log ships |
| `6..27` | fleet ship-count basis |
| `28..43` | ETA one-hot buckets for turns `1..15` and `16+` |
| `44` | ETA overflow `(eta - 15) / 10` for `eta >= 16`, else `0` |
| `45` | post-comet flag; `1` when arrival is at or after the next scheduled comet spawn |

`target_incoming_features` has three 16-bucket groups over ETA buckets
`1..15` and `16+`:

| Channels | Feature |
| --- | --- |
| `0..15` | incoming fleet count divided by `100` |
| `16..31` | incoming ships divided by `5000` |
| `32..47` | `log1p` incoming ships divided by `ln(1000)` |

These per-target aggregates include all routed fleets before fleet-memory
truncation, so counts remain accurate even when more than `max_fleets` fleets
are inbound.

Standalone observation encoding uses strict application-boundary parsing:
required observation keys are read directly, `step` and `episode_steps` must be
integers rather than coerced strings or floats, comet groups must provide
matching `planet_ids`, `paths`, and `path_index`, and comet/path overflow is
rejected rather than truncated. It also rejects non-finite `angular_velocity`
and requires `episode_steps > 0` before computing these values.
The parser is still an internal high-throughput bridge into typed Rust state:
impossible ID invariants such as duplicate planet IDs or IDs outside the fixed
rules-engine limit may trip Rust assertions instead of being converted into
Python `ValueError`s.
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
{"action_spec": "pure", "max_per_planet_launches": 1, "min_fleet_size": 6}
```

The pure action spec exposes all launch decisions in direct tensor form. The
same entity axis is used for action masks and submitted actions.
`max_per_planet_launches` is validated in Python and Rust and must equal `1`.
`min_fleet_size` is validated in Python and Rust and must fit in the positive
`i32` ship-count range. `ActionPureConfig()` defaults to
`max_per_planet_launches=1` and `min_fleet_size=6`.

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
actions = PureActions(launch=launch, angle=angle, ships=ships)
obs, rewards, dones, episode_metrics = env.step(actions)
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
Each player/source entity has one launch slot.

For Kaggle submissions, `actions_to_kaggle(obs, player, actions, action_spec=...)`
accepts a `PureActions` bundle with single batched tensors shaped
`(1, 4, 44, max_per_planet_launches)` and returns the selected player's
`list[list[float]]` action triples. Pure triples are
`[from_planet_id, angle, ships]` and use the same Rust validation path as
environment stepping.

## Discrete Targets Action Spec

Config:

```python
{
    "action_spec": "discrete_targets",
    "max_per_planet_launches": 1,
    "min_fleet_size": 6,
    "targeting_mode": "full_mask",
}
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

`targeting_mode` controls target masking and selected bad-launch handling:

| Mode | Target mask | Bad selected target launch |
| --- | --- | --- |
| `"full_mask"` | Existing targets except self, plus the simulator's full static-target eligibility filter. | Selected sun-blocked static targets are replaced with no-op. Selected planet-blocked static targets can still fall back to a sun-safe ray. Selected dynamic targets are replaced with no-op when no allowed or fallback ray exists. |
| `"stop_bad_launch"` | Existing targets except self; static obstruction, sun crossing, and dynamic feasibility are not masked. | Falls back through the target cone for a sun-avoiding ray and is replaced with no-op only when no sun-avoiding target ray exists. |
| `"anything_goes"` | Existing targets except self; static obstruction, sun crossing, and dynamic feasibility are not masked. | Submitted even when the computed ray crosses the sun. Dynamic targets with no target-hit window still become no-ops because no launch angle is defined. |

In `"full_mask"`, static-source to static-target pairs use the reset-time
cached blocker-safe static target-cone result for masking. Fully
blocker-covered static targets are masked out. Selected static-source to
static-target launches also reuse the cached static-safe target arcs, then only
check dynamic blockers at launch time. Dynamic-source to static-target pairs
recompute the same static target-cone sun/static-blocker check for the current
step while masking.
Dynamic targets remain eligible in the mask because full obstruction checks are
deferred until the selected launch is decoded. In the loose modes,
`can_act[player, source, target]` is true when the source entity is owned by
that outer player slot, has at least `min_fleet_size` ships, the target slot
exists, and `source != target`.

### Discrete Targets Submitted Actions

Call:

```python
actions = DiscreteTargetActions(launch=launch, target=target, ships=ships)
obs, rewards, dones, episode_metrics = env.step(actions)
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

For static planets, decoding uses the fixed target angular cone with
`target_radius - eps`, subtracts sun plus static and dynamic blocker forbidden
arcs inflated by small avoidance epsilons, then chooses the feasible angle
closest to the target centerline. Static-source/static-target selected launches
start from the cached static-safe arc and only check dynamic blockers at the
shot horizon. Other static-target selected launches recompute sun/static arcs
live, with cached static blocker geometry reused per decode. Dynamic blockers
use their cached orbit paths or comet path segments up to the selected shot's
impact horizon. If the sun removes the whole eligible arc, decoding skips
blocker search and falls back immediately. If blockers cover the whole static
target cone, selected launches fall back to the closest sun-avoiding angle.
This selected-launch fallback is separate from the `full_mask` target mask:
fully planet-blocked static targets are masked out as legal targets, but if one
is selected anyway it may still fire. If no sun-avoiding angle exists,
`"full_mask"` and `"stop_bad_launch"` decode the launch as a no-op, while
`"anything_goes"` uses the centerline.

For orbiting non-comet planets, reset caches future tick positions across the
episode horizon, and decoding solves target-hit time windows against the cached
linear segments for the selected target. Manually reconstructed states without
that cache build the selected target path lazily during decode. For comets,
decoding solves bounded target-hit time windows against each stored linear path
segment. Windows are processed in increasing impact time and capped to avoid
searching a hopeless long tail. Within each window, the decoder first tries the
window midpoint angle against sun, bounds, and cached blocker metadata. Only if
that optimistic path fails does it compute the target's eligible angular arc,
subtract sun and static/dynamic blocker forbidden arcs up to the window end
with the same avoidance epsilons, and choose the closest feasible angle for
that window. Dynamic blockers use the same orbit/comet path sources as dynamic
targets, sampled at fixed horizon fractions plus radial crossing times. If no
window has a collision-avoidance angle, decoding falls back to the first
sun-avoiding, in-bounds target arc. If no such fallback exists,
`"anything_goes"` fires along the first window midpoint and
`"full_mask"`/`"stop_bad_launch"` decode the launch as a no-op. Submitted
discrete-target launches that cannot produce an allowed or defined ray are
counted in `launch_failures_per_game`.

## Discrete Target Bins Action Spec

Config:

```python
{
    "action_spec": "discrete_target_bins",
    "min_fleet_size": 6,
    "n_bins": 11,
    "targeting_mode": "full_mask",
}
```

`ActionDiscreteTargetBinsConfig` uses the same `targeting_mode` target-slot
eligibility and bad-launch handling as `discrete_targets`, but fleet size is
selected as a categorical bin instead of an integer ship count. `n_bins` is the
total action count, including no-op, and must be at least `2`.

### Discrete Target Bins Output Tensors

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `can_act` | `bool` | `(n_envs, 4, 44, 44, n_bins)` | whether a player can choose a source, target, and fleet-size bin |
| `max_launch` | omitted/`None` | n/a | not used by this action spec |

For valid source-target pairs, bin `0` is always available and decodes as
no-op. Bins `1..n_bins-1` map to:

```text
round_half_up(bin * available_ships / (n_bins - 1))
```

Bin `n_bins - 1` maps to all available source ships. Launch bins that would
produce fewer than `min_fleet_size` ships are masked. If multiple bins round to
the same ship count, only the highest bin for that ship count remains
available. For example, with `available_ships=5`, `n_bins=11`, and
`min_fleet_size=1`, the available bins are `0, 2, 4, 6, 8, 10`.

### Discrete Target Bins Submitted Actions

Call:

```python
actions = DiscreteTargetBinActions(target=target, fleet_bin=fleet_bin)
obs, rewards, dones, episode_metrics = env.step(actions)
```

where `actions` carries:

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `target` | `int64` | `(n_envs, 4, 44)` | target action entity slot index in `[0, 44)` |
| `fleet_bin` | `int64` | `(n_envs, 4, 44)` | fleet-size action bin in `[0, n_bins)` |

Bin `0` ignores the target value and emits no launch. Nonzero bins validate the
selected source-target-bin tuple against `can_act`, decode the bin to a ship
count, then use the same target-to-angle decoder and launch-failure accounting
as `discrete_targets`.

## Decoded Launch Actions

`VectorizedEnv` can expose action masks for action specs other than the env's
construction spec, expose observations for observation specs other than the
env's construction spec, decode model actions from those specs, and step the
simulator with a common launch representation. This is intended for evaluation
code that benchmarks checkpoints trained with different observation or action
specs against each other.

```python
obs_a = env.observation_for_spec(
    checkpoint_a.config.env.obs_spec,
    checkpoint_a.config.env.action_spec,
)
obs_b = env.observation_for_spec(
    checkpoint_b.config.env.obs_spec,
    checkpoint_b.config.env.action_spec,
)
decoded_a = env.decode_actions(output_a.actions, action_spec=checkpoint_a.config.env.action_spec)
decoded_b = env.decode_actions(output_b.actions, action_spec=checkpoint_b.config.env.action_spec)
obs, rewards, dones, episode_metrics = env.step_decoded_actions(selected_decoded)
```

`DecodedLaunchActions` has fixed-rank tensors:

| Tensor | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `valid` | `bool` | `(n_envs, 4, max_actions)` | whether this decoded launch slot should execute |
| `from_planet_id` | `int64` | `(n_envs, 4, max_actions)` | source planet ID for valid launches |
| `angle` | `float32` | `(n_envs, 4, max_actions)` | launch angle for valid launches |
| `ships` | `int64` | `(n_envs, 4, max_actions)` | ship count for valid launches |

`decode_actions(...)` chooses `max_actions` from the source action spec:
`44 * max_per_planet_launches` for `pure` and `discrete_targets`, and `44` for
`discrete_target_bins`. Callers that merge decoded actions from different specs
should pad to the larger `max_actions` and select per outer player slot.
`step_decoded_actions(...)` ignores payload fields where `valid` is false,
rejects launches from inactive outer slots, validates source ownership, finite
angles, positive `i32` ship counts, and per-source ship budgets, then uses the
normal Rust rules-engine step path. Spec-specific constraints such as
`min_fleet_size`, target eligibility, target-to-angle conversion, and target-bin
rounding are enforced during `decode_actions(...)`.

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
| `total_games_played` | Count marker emitted once per terminal episode. Python training sums this as `train/total_games_played` across the rollout instead of averaging it. |
| `max_entities_exceeded_per_game` | Count of post-step turns where active fleets exceeded `max_fleets`. |
| `game_length_mean` | Terminal game step count. |
| `full_length_rate` | `1.0` when a game reaches the configured episode horizon, otherwise `0.0`. |
| `terminal_ship_count` | Total ships on planets and in active fleets at terminal. |
| `planets_captured_per_game` | Total planet captures over the episode, counting repeat captures. |
| `comets_captured_per_game` | Total comet planet captures over the episode, counting repeat captures. |
| `_neutral_planets_captured_per_game` | Hidden aggregate input: successful neutral non-comet planet captures in the terminal episode. Used to compute neutral undershot rates; not logged directly. |
| `_neutral_comets_captured_per_game` | Hidden aggregate input: successful neutral comet captures in the terminal episode. Used to compute neutral undershot rates; not logged directly. |
| `_neutral_planet_undershots_per_game` | Hidden aggregate input: neutral non-comet planet capture undershots in the terminal episode. Used to compute neutral undershot rates; not logged directly. |
| `_neutral_comet_undershots_per_game` | Hidden aggregate input: neutral comet capture undershots in the terminal episode. Used to compute neutral undershot rates; not logged directly. |
| `neutral_planet_undershot_rate` | Neutral non-comet planet capture undershots divided by successful neutral non-comet planet captures plus those undershots. An undershot is a neutral arrival whose surviving incoming ships are less than or equal to the neutral planet ship count. Omitted when no neutral non-comet planet capture or undershot occurred. |
| `neutral_comet_undershot_rate` | Neutral comet capture undershots divided by successful neutral comet captures plus those undershots. An undershot is a neutral arrival whose surviving incoming ships are less than or equal to the neutral comet ship count. Omitted when no neutral comet capture or undershot occurred. |
| `launch_failures_per_game` | Submitted discrete-target launches skipped because no valid selected ray exists, including static targets with no sun-avoiding ray and dynamic targets with no intercept, no allowed fallback, or out-of-bounds impact points. Python training logs this as `train/launch_failures_per_game`. |
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
