# Model Architecture

This document summarizes the current trainable actor-critic model for the Orbit
Wars RL API.

## Tagged Config

The stateless model config is `StatelessTransformerV1Config` with discriminator:

```python
{"model_arch": "stateless_transformer_v1"}
```

The recurrent model config is `RecurrentTransformerV1Config` with discriminator:

```python
{"model_arch": "recurrent_transformer_v1"}
```

The exported `ModelConfig` type is a pydantic discriminated union alias over
both configs. Callers should construct models through the shared model factory
instead of instantiating `StatelessTransformerV1` directly when checkpoint
configs may contain either architecture.

## Config Reference

`StatelessTransformerV1Config` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `model_arch` | `"stateless_transformer_v1"` | Pydantic discriminator tag. |
| `embed_dim` | `128` | Hidden width for all projected tokens and transformer blocks. |
| `depth` | `4` | Number of transformer blocks. |
| `n_heads` | `8` | Attention heads; must evenly divide `embed_dim`. |
| `mlp_ratio` | `4.0` | FFN hidden width multiplier. |
| `player_count_adapters_enabled` | `False` | Enable per-still-playing-player-count actor/critic heads and optional trunk adapter blocks. |
| `player_count_adapter_blocks` | `0` | Number of final transformer blocks to move from the shared trunk into each per-player-count branch; requires `player_count_adapters_enabled=True`. |
| `activation` | `"gelu"` | FFN activation: `"gelu"`, `"silu"`, or `"swiglu"`. |
| `force_flash_attn` | `False` | Require packed varlen flash-attn; raise an error instead of falling back when tensors are not flash-compatible. |
| `use_learned_pairwise_bias` | `False` | Enable an auxiliary source-target feature MLP for discrete target selection. Only valid with `"discrete_targets"` and `"discrete_target_bins"` actors. |
| `n_scratch_tokens` | `4` | Learned shared scratch tokens appended to the trunk sequence. |
| `actor` | `{"action_spec": "pure"}` | Discriminated actor-head config. Supported actor specs are `"pure"`, `"discrete_targets"`, and `"discrete_target_bins"`. |

Actor-specific fields live inside the actor config. `ActorPureConfig` owns
pure-head fields such as `n_angle_mixtures`, `n_fleet_size_mixtures`,
`kappa_min=1e-3`, `kappa_max=1e6`, `dir_eps`,
`entropy_ship_quantiles=16`, and the logistic-mixture scale parameters
`scale_min=0.10`, `scale_max_frac=0.5`, and `scale_max_abs_floor=8.0`.
`ActorDiscreteTargetsConfig` owns
`launch_mode="binary"`, `n_action_mixtures`,
`entropy_ship_quantiles=16`, and the same logistic-mixture scale parameters.
Set `launch_mode="target_token"` to replace the separate Bernoulli
launch/stop choice with a learned no-launch target candidate in the target
categorical. Set `launch_mode="binary_after"` to keep a Bernoulli launch/stop
choice but project its logits from the selected target-conditioned hidden
representation used by the fleet-size heads.
`ActorDiscreteTargetBinsConfig` owns `n_bins`, which must match the
environment's `ActionDiscreteTargetBinsConfig.n_bins`.
`kappa_min` must be less than or equal to `kappa_max`.
Model YAML files can reference actor presets by name through adjacent
`configs/model/actor/*.yaml` files, for example `actor: discrete_targets`, or
can inline an actor config to override preset fields such as mixture count.

`force_flash_attn=True` is ignored for CPU tensors; CPU execution always uses
the regular SDPA fallback path.

`FullConfig` validates that `env.action_spec.action_spec` matches
`model.actor.action_spec`. Direct model construction performs the same check
against the supplied environment action spec.

`RecurrentTransformerV1Config` intentionally supports only the
`"discrete_targets"` actor with `launch_mode="binary"`. The policy first samples
whether to launch, then samples the target, then samples fleet size. Pure,
target-bin, `binary_after`, and `target_token` actor modes are rejected for this
architecture. The stateless per-player-count adapter option is disabled and
fixed at `0` blocks for this architecture. Its recurrent-token scope is
controlled by
`recurrence_mode`, which defaults to `"global_only"` and can be set to
`"include_planets"`.

Observation and action specs are owned by `EnvConfig`. `StatelessTransformerV1`
receives `env.obs_spec` and `env.action_spec` when it is instantiated, so model
config presets cannot silently diverge from the environment tensor shapes.

## Input Encoding

`StatelessTransformerV1` consumes an `ObsBatch` containing on-device torch tensors
from `docs/rl-api-specs.md`.

Each observation tensor receives a small MLP stem from raw channels to
`int(embed_dim * mlp_ratio)` and then to `embed_dim`:

- static planets: `(batch, MAX_PLANETS, 107) -> (batch, MAX_PLANETS, embed_dim)`
- orbiting planets: `(batch, MAX_PLANETS, 107) -> (batch, MAX_PLANETS, embed_dim)`
- fleets: `(batch, max_fleets, 79) -> (batch, max_fleets, embed_dim)`
- comets: `(batch, MAX_COMETS, 330) -> (batch, MAX_COMETS, embed_dim)`
- globals: `(batch, 3) -> (batch, embed_dim)`
- v2 player features, when present:
  `(batch, OUTER_PLAYER_SLOTS, 10) -> (batch, OUTER_PLAYER_SLOTS, embed_dim)`

For `EntityBasedExtV1`, planet input widths append
`ship_count_one_hot_max + 1` channels and fleet input widths append
`ship_count_one_hot_max` channels. With the default `ship_count_one_hot_max=50`,
the planet width is `158` and the fleet width is `129`; comet and global widths
are unchanged.

For `EntityBasedExtV2`, planet, fleet, and comet widths stay at the base
`EntityBased` sizes. The global input width increases from `3` to `11`, and
`ObsBatch.player_features` supplies a ten-channel per-outer-player summary. The
model creates a player-feature projection only when
`obs_spec.player_feature_channels > 0`; old `entity_based` and
`entity_based_ext_v1` stateless checkpoints therefore do not gain
`player_feature_proj` parameters.

The boolean `orbiting_planets` mask selects the orbiting-planet projection for
orbiting rows and the static-planet projection for all other planet rows.
Planet, comet, and fleet tokens are concatenated on the entity axis in that
order. This keeps the action-origin hidden states contiguous as the first
`ACTION_ENTITY_SLOTS` tokens. The global projection is appended as its own
global-feature token. For v2 observations, the projected per-player summary is
added directly to the learned player token for the matching outer player slot
before the transformer trunk. The full trunk sequence is:

```text
[planet tokens]
[comet tokens]
[fleet tokens]
[player tokens]
[global-feature token]
[board scratch tokens]
[actor plan tokens]
[critic value tokens]
```

With the default four board scratch tokens this gives
`(batch, max_entities + 17, embed_dim)`. The `entity_mask` uses the same
planet, comet, fleet order and is concatenated with masks for the learned
tokens. Player, actor-plan, and critic-value tokens use `still_playing`; the
global-feature and board scratch tokens are always unmasked. Masked tokens are
excluded from attention keys and are zeroed in the returned hidden states.
Downstream code must consume the named `EncodedObservations` fields rather than
assuming output meaning from positional slices.

Kaggle serving may compact inactive planet, comet, and fleet rows before model
inference. Recurrent checkpoints with `recurrence_mode="include_planets"` keep
the fixed planet prefix and compact only comets/fleets, because their recurrent
layout indexes the planet tokens directly. In the compacted path the
actor-visible action slot count is the runtime
`ObsBatch.action_mask.can_act.shape[2]` instead of the fixed
`ACTION_ENTITY_SLOTS` API width. The compacted action slots still appear first
in planet-then-comet order, and `Agent` expands sampled actions back to the full
44-slot Rust/Kaggle contract before conversion.

## Transformer Trunk

The shared trunk is a stack of pre-norm transformer blocks configured by:

- `depth`
- `n_heads`
- `mlp_ratio`
- `embed_dim`

`n_heads` must evenly divide `embed_dim`, and `int(embed_dim * mlp_ratio)` must
be at least 1. The default activation is GELU. LayerNorm is used for
normalization, and no dropout is applied.

When `player_count_adapters_enabled=True`, the stateless model creates one
adapter branch for each still-playing player count from two through four. If
`player_count_adapter_blocks > 0`, that count is subtracted from the shared
trunk depth and moved into each branch. For example, `depth=16` and
`player_count_adapter_blocks=4` builds 12 shared blocks followed by four
per-count blocks. If `player_count_adapter_blocks=0`, the full transformer trunk
remains shared and only the actor/critic heads are per-count. The per-count
branch also owns the actor input projections, actor module, learned pairwise-bias
MLP if enabled, and critic head. Rows are selected by
`obs.still_playing.sum(dim=1)` and scattered back to the original batch order;
one-player terminal-like rows are routed through the two-player branch while
keeping their original `still_playing` mask. With packed flash attention, each
branch with trunk adapter blocks receives a packed subsequence for its selected
batch rows and keeps the original maximum sequence length.

CPU execution uses torch scaled-dot-product attention over regular
`(batch, seq, dim)` tensors with the token mask passed as the attention key
mask. CUDA execution uses packed varlen `flash-attn` when it is installed and
the attention tensors are fp16/bf16; otherwise it uses the same regular-shaped
scaled-dot-product attention path without packing and unpacking activations.
Set `force_flash_attn=True` to require packed varlen flash-attn and fail fast
when the backend, device, or dtype is not compatible on CUDA. CPU execution
ignores this flag and uses the SDPA fallback.

Attention uses separate `q`, `k`, and `v` linear layers instead of one packed
QKV projection. SwiGLU also uses separate gate and value projections. This keeps
each weight matrix tied to one projection role, which is a better fit for Muon
optimizer assumptions than packing multiple operations into one parameter.
Training also defaults to compiling each transformer-block MLP in place with
`rl.model_compile="mlp"` and
`rl.model_compile_mode="max-autotune-no-cudagraphs"`. Packing, unpacking, and
flash-attn varlen calls remain eager. Per-player-count adapter block MLPs are
compiled by the same setting.

## Recurrent Transformer V1

`RecurrentTransformerV1` reuses the stateless input stems, token layout,
discrete-target actor, critic, pairwise-bias option, and masked attention
behavior. Its trunk replaces each stateless transformer block with:

```text
transformer block over all current observation tokens
minGRU block over the configured recurrent token set
```

`recurrence_mode` controls the recurrent token set:

- `global_only` (default): global-feature token, board scratch tokens,
  player tokens, actor-plan tokens, and critic-value tokens
- `include_planets`: all `global_only` tokens plus non-comet planet tokens
  `0..MAX_PLANETS-1`

Comet and fleet tokens are not recurrent. Planet recurrence is keyed by the
existing non-comet planet row order, which the RL API defines as ascending
planet ID order. Planet token state is env-level state, so ownership changes do
not reset it.
The recurrent token layout is memoized per runtime entity count, so inference
paths that compact inactive fleet rows can shift later token positions without
changing the hidden-state contract.

The recurrent hidden state has shape:

```text
(depth, batch, recurrent_tokens, embed_dim)
```

Shared token state resets when the whole environment episode resets. Per-player
token state resets when that player slot is done. In `include_planets` mode,
planet token state also resets only when the whole environment episode resets.
During PPO updates, minGRU uses an affine parallel scan over the segment time
dimension, with `dones[:, t]` applied as the reset boundary before processing
observation `t + 1`.

For packed flash-attention execution, the recurrent block builds an inverse map
from padded token coordinates to packed rows, gathers only recurrent token rows,
runs the dense recurrent scan as `(batch, time, recurrent_tokens, dim)`, and
scatters the updated recurrent rows back into the packed tensor. Missing masked
recurrent tokens do not scatter back and their recurrent state is zeroed.

## Initialization

Models expose `reset_parameters()` through `BaseModelAPI`. Fresh training calls
this method explicitly before optimizer construction; checkpoint-loading paths
construct the module and load saved weights without resetting first.

Linear layers use orthogonal initialization with zero biases. Only the first
linear layer in each observation stem is treated as an input projection and
uses unit gain; hidden projections use ReLU-style gain, and transformer
residual output projections, including per-count adapter blocks, are scaled by
`1 / sqrt(2 * depth)`.
Learned token state parameters, including player, board scratch, actor-plan,
critic-value, and discrete source/target role tags, are also classified as
input layers for optimizer grouping so Muon does not update them.

Actor and critic output heads are two-layer MLP projections with hidden width
`embed_dim`, the configured activation in the middle, and output-specific final
widths. Only the second linear layer in each output MLP is treated as an output
layer for optimizer grouping and final-head initialization. Actor final output
layers use small `0.01` gain with zero biases, matching the normal RL
policy-layer initialization. The critic final output layer uses unit gain. When
per-player-count adapters are enabled, each branch has its own actor and critic
heads with the same initialization rules.

## Critic

The critic reads the per-player `critic_value_hidden` field from
`EncodedObservations`. A two-layer MLP head produces one logit per player, then
applies a masked softmax using `obs.still_playing` with shape `(batch, 4)`.
With per-player-count adapters enabled, the branch selected by each row's
still-playing player count owns the critic head for that row.

The resulting winner probabilities are mapped linearly into value targets:

```text
value = 2 * winner_probability - 1
```

This gives `0 -> -1`, `0.5 -> 0`, and `1 -> 1`.

`still_playing` is explicit in `ObsBatch`. It should not be inferred from
`can_act`, since a player can be alive without having a launchable entity on a
specific turn.

## Actor

The actor uses hidden states for the action entity slots:

```text
0..39  -> planet tokens
40..43 -> comet tokens
```

For regular training and vector-env evaluation this is the fixed 44-slot API
layout. For compacted Kaggle serving, the same actor heads operate on a smaller
runtime slot count while preserving the compact planet-then-comet order.

Both actor heads start from the same shared transformer trunk. For each
`(batch, player, action_entity)` position, the actor combines:

- source entity hidden state
- player hidden token
- the actor plan token for that player, for the discrete-target actor

The final action head is selected by `config.actor.action_spec`. The concrete
heads live under `python/owl/model/actor/`: `PureActor` for raw angles and
`DiscreteTargetsActor` for target slots.
With per-player-count adapters enabled, the selected branch owns these actor
input projections and action heads for its rows.

When `use_learned_pairwise_bias=True`, the discrete-target and discrete
target-bin actors receive an auxiliary learned bias after the shared
self-attention trunk. The model builds six raw source-target features over the
runtime action entity slots, applies a two-layer MLP
`6 -> embed_dim -> 1`, and adds the result to the target-selection attention
score before action masking:

- `has_more_ships`: source ship count is greater than target ship count.
- `target_is_neutral`: target owner is neutral.
- `target_is_mine`: source and target share the same non-neutral owner.
- `target_is_enemy`: target has a non-neutral owner different from the source.
- `normalized_distance`: Euclidean source-target distance in normalized board
  coordinates, divided by the normalized board diagonal `sqrt(8)`.
- `sun_proximity`: `1 - d / sqrt(2)`, where `d` is the minimum Euclidean
  distance from the sun center `(0, 0)` to the source-target line segment in
  normalized board coordinates.

The source-target segment distance makes `sun_proximity` mathematically
well-defined: it is the standard point-to-segment distance, with zero-length
segments falling back to the source/target point distance. Feature construction
uses planet slots `0..39` and comet slots `40..43`; comet positions use their
current path position. Neutral planet ships are denormalized from `/100`,
while owned planet and comet ships are denormalized from `/500` before
comparison. Existing model configs default this path off and therefore do not
gain extra parameters.

### Pure Actor

The pure actor supports `ActionPureConfig`.

The pure actor supports `max_per_planet_launches=1`. `ActionPureConfig()`
defaults to `max_per_planet_launches=1` and `min_fleet_size=6`. Python config
validation and model construction both reject larger pure launch counts.

For the launch slot, the policy emits:

- Bernoulli launch/stop logits
- mixture logits for angle components
- von Mises angle parameters
- discretized logistic mixture fleet-size parameters

Each emitted parameter group uses its own two-layer MLP output head.

The actor uses separate source and target streams matching the discrete-target
actor's single-launch structure. The main policy-family difference is that
pure actions sample a von Mises mixture angle instead of a categorical target
slot. Each angle mixture has a learned base direction initialized to evenly
spaced unit vectors around the circle; the direction head predicts residual
Cartesian offsets before normalization. Von Mises concentration uses
log interpolation between `kappa_min` and `kappa_max`, so very tight angles can
be represented with ordinary logits.

After sampling or replaying an angle, the actor projects `(sin(angle),
cos(angle))` into the model width, adds it to the source stream, and uses the
result as a query over target-stream keys for existing action entities other
than the source. The soft attention value is added to the source residual
stream before the fleet-size heads. This mirrors the discrete-target actor's
selected-target value path while keeping the action itself continuous. Fleet
sizes use the same discretized logistic mixture parameterization as
`DiscreteTargetsActor`, but with separately configurable
`n_fleet_size_mixtures`.

The model returns a typed action bundle owned by the active action spec. Pure
and discrete-target actions keep the final launch-slot dimension expected by
their Rust API paths. For regular env/training batches the action-slot width is
44; for compacted Kaggle serving the width is the runtime compact slot count
until `Agent` expands the actions back to 44:

- `launch`: bool, `(batch, 4, action_slots, max_per_planet_launches)`
- `ships`: int64, `(batch, 4, action_slots, max_per_planet_launches)`
- `angle`: float32, same shape, pure actor only
- `target`: int64, same shape, discrete-target actor only
- `target`: int64, `(batch, 4, action_slots)`, discrete target-bin actor only
- `fleet_bin`: int64, `(batch, 4, action_slots)`, discrete target-bin actor only

It also returns decomposed log-prob and entropy tensors for launch gates and
target or angle/size events, plus per-player action-entity totals with shape
`(batch, 4, action_slots)`. PPO stores and submits the typed action bundle
directly, so action-spec-specific payloads such as `fleet_bin` cannot be
silently dropped.
Observation masks are likewise held in typed `ObsBatch.action_mask` bundles:
pure and discrete-target masks carry `can_act` plus `max_launch`, while
discrete target-bin masks carry only `can_act`.
Entropy outputs also carry policy-specific component names for logging, such as
`launch`, `target`, `fleet_size_full`, `fleet_size_mixture`,
`fleet_size_logistic`, or `event`.

Serving callers that only need actions and critic values use
`BaseModelAPI.serve()`, which returns a `ModelServingOutput` without
log-probability or entropy tensors. `StatelessTransformerV1` specializes this
path for the discrete-target and target-bin actors by sampling actions directly
from the policy parameters needed for action selection. `RecurrentTransformerV1`
overrides the same serving path so runtime hidden state is consumed and the next
hidden state is returned without computing PPO-only action statistics. Training
still uses `forward()` to return the full PPO action statistics.

Pure and discrete-target deterministic action selection resolves the launch
gate before computing the exact fleet-size MAP. No-launch rows skip ship-support
enumeration entirely. On CPU, the model enumerates only each launched row's own
`min_fleet_size..max_launch` support. On non-CPU devices, it gathers launched
rows, scores support only for that compacted row set, and scatters the selected
ship counts back into the action tensor. Stochastic sampling never enumerates
integer ship support; it samples one logistic component and uses inverse-CDF
sampling for that component.

The pure actor's angle entropy is an augmented latent-mixture entropy estimate:
mixture-label entropy plus expected von Mises component entropy. Fleet-size
entropy uses the same deterministic truncated-logistic quantile quadrature as
the discrete-target actor, controlled by `entropy_ship_quantiles`; the size
entropy path conditions on the current policy's deterministic angle proxy.

### Discrete Targets Actor

The discrete-target actor supports `ActionDiscreteTargetsConfig` with
`max_per_planet_launches=1`. Model construction fails fast if the environment
uses a larger per-planet launch count for this actor.

The discrete-target actor uses one feedforward action block per source entity.
The model constructs separate
source and target streams for each player/action entity position. Both streams
receive entity hidden state, player hidden state, and the player's actor plan
token. The actor adds learned source/target role embeddings before normalizing
the two streams, then projects source slots to queries and target slots to keys
and values with a single target-selection head independent of the shared
transformer trunk's attention head count. It computes scaled dot-product target
logits for every `(source, target)` pair and masks those logits with the 4-D
discrete `can_act` tensor. The environment action spec's `targeting_mode`
decides whether this mask includes full simulator target eligibility or only
entity existence plus self-targeting. Fully masked source rows are sanitized to
finite zero logits and are suppressed by the launch/source mask.

After sampling or replaying a target from the masked softmax, the actor gathers
only the selected target value vector, adds it to the source residual stream,
and applies a feedforward residual block. This is intentionally close to a
standard attention block, except the value path uses the selected target row
instead of a softmax-weighted average. The launch/stop decision and selected
target-conditioned size decision use separate source projections so the
source-only continue gate is not coupled to the pair-conditioned size head.

With the default `launch_mode="binary"`, each source emits:

- Bernoulli launch/stop logits
- masked categorical target logits over the runtime action entity slots
- mixture parameters for a truncated discretized logistic fleet-size policy

With `launch_mode="binary_after"`, the target categorical remains over the
runtime action entity slots and no no-launch target is added. The actor samples or
replays the selected target first, gathers that target's value, applies the
same residual feedforward and size-pair projection used by the fleet-size
heads, then projects Bernoulli launch/stop logits from that target-conditioned
representation. No-launch actions retain the sampled target in
`DiscreteTargetActions.target` so replay can score
`log P(target) + log P(no-launch | target)`; the environment still ignores the
target when `launch=False`. Launch entropy uses the Bernoulli entropy of the
selected target approximation, matching the fleet-size entropy approximation.

With `launch_mode="target_token"`, the actor appends one learned
no-launch token to the target/key/value stream only; the source/query stream
remains the real runtime action entity slots. Selecting that extra target maps
back to `launch=False` in the external `DiscreteTargetActions` bundle; selecting
any real target maps to `launch=True`. Pairwise source-target bias is still
defined only over the real target slots, and the actor appends a zero-bias
column for the no-launch target before masking. In this mode the binary continue
projection/head is not allocated, and the launch log-probability and launch
entropy tensors are zero while target log-probability and target entropy include
the no-launch candidate. The default binary mode does not allocate the extra
learned token, so existing default model parameter shapes are unchanged.

The fleet-size mixture maps raw means through a sigmoid into the current
`min_fleet_size..max_launch` budget range. Raw scale outputs are passed through
a sigmoid and log-interpolated between `scale_min` and
`max(scale_max_abs_floor, scale_max_frac * support_width)`, where
`support_width = max_launch - min_fleet_size + 1`. This keeps very small scales
available for near-deterministic counts while still allowing broad fractional
exploration for large ship budgets. PPO replay uses the marginal mixture
log-probability of the integer ship count, not the sampled component
log-probability. The discrete-target entropy bonus is an
exploration heuristic rather than the exact joint-action entropy: it sums
launch entropy, target entropy, and a target-conditioned size entropy without
weighting target and size entropy by launch probability. To avoid materializing
all source-target size parameters, the size entropy term uses the current
policy's argmax target as a proxy; replayed action targets are used only for
action log-probability, so no-launch placeholder targets do not affect entropy.
Fleet-size entropy is estimated with deterministic per-component quantile
quadrature under the continuous truncated logistic mixture. This accounts for
component overlap and uses memory proportional to
`n_action_mixtures * entropy_ship_quantiles`, independent of the ship budget.
Because this is a continuous-density estimate rather than exact entropy over
rounded integer ship counts, very narrow scales can produce a negative size
entropy term; the PPO bonus still encourages broader size distributions, but
the logged value should be interpreted as an exploration heuristic rather than a
non-negative discrete entropy.

### Discrete Target Bins Actor

The discrete target-bin actor supports `ActionDiscreteTargetBinsConfig`. It uses
the same source/target stream construction and target-selection logits as the
discrete-target actor, but the environment supplies a 5-D mask
`(batch, 4, action_slots, action_slots, n_bins)` and no `max_launch` tensor.

For each active source, the actor samples target first from target slots with at
least one valid bin, then samples `fleet_bin` from categorical logits
conditioned on the selected target value. It returns only `target` and
`fleet_bin` action tensors. The environment decodes bin `0` as no-op and
nonzero bins as rounded ship counts, using the same target-to-angle decoder as
`discrete_targets`.

Replay log-probability follows the same factorization:
`log p(target) + log p(fleet_bin | target)`. Entropy logging exposes `target`
and `fleet_bin` components; the shared `event` field carries the
fleet-bin term for compatibility with PPO loss aggregation. Like the
discrete-target size entropy, fleet-bin entropy is computed only for the
current policy's argmax target proxy rather than for every source-target pair.

## Log-Prob Replay

The model exposes `evaluate_actions(obs, actions)` to replay externally supplied
action tensors through the same actor factorization and return both new-policy
log-probs, entropies, and critic values from one encode.

Inactive and stopped slots are given finite dummy event inputs before masking so
their zeroed log-prob contributions do not introduce NaN gradients.

## Teacher Distillation

Training can add a frozen teacher model through `rl.teacher_mode`. The action
term uses `KL(teacher || student)` for each replayed state and sums the relevant
action-head divergences back to the player-step before applying
`rl.teacher_kl_coef`. The value term uses cross-entropy from the teacher winner
distribution to the student winner distribution over active player slots, then
averages that per-state value over active states before applying
`rl.teacher_value_coef`. Teacher inference is performed only during PPO update
minibatches from stored rollout segments. Teacher models must be stateless;
trainers reject teachers that require recurrent hidden state.

When a teacher is active, PPO calls `evaluate_actions_with_teacher(...)` instead
of separate student replay and KL passes. The method encodes the student once,
returns the normal PPO replay log-probs, entropy, and values from that encoding,
and encodes the teacher once under `torch.no_grad()` to produce teacher value
targets and KL reference distributions. `evaluate_action_kl(...)` remains as a
compatibility path, but PPO updates do not use it. The KL comparison uses the
same masks and factorization gates used by PPO replay. Non-acting source rows
contribute zero KL. For binary discrete-target launch mode, no-launch replay
rows include only the Bernoulli launch KL; target and fleet-size KL are computed
only for rows where the replayed action launched. Discrete target-bin KL
compares the target categorical and the selected target's fleet-bin categorical.
Pure-action KL compares launch, angle, and selected fleet-size distributions for
launched rows.

Fleet-size KL is computed on the selected truncated logistic mixture. Matching
mixture counts use aligned component terms; incompatible mixture counts fall
back to the full marginal mixture where supported. Per-action portions are
logged with the same component naming style as entropy logging, for example
`launch`, `target`, `angle`, `fleet_size_mixture`, `fleet_size_logistic`, and
`fleet_size_full`.
