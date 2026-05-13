# Model Architecture

This document summarizes the current trainable actor-critic model for the Orbit
Wars RL API.

## Tagged Config

The current model config is `StatelessTransformerV1Config` with discriminator:

```python
{"model_arch": "stateless_transformer_v1"}
```

The exported `ModelConfig` type is a pydantic discriminated union alias over the
current config. It is intentionally shaped so future model configs can be added
without changing callers that validate config dictionaries through the union.

## Config Reference

`StatelessTransformerV1Config` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `model_arch` | `"stateless_transformer_v1"` | Pydantic discriminator tag. |
| `embed_dim` | `128` | Hidden width for all projected tokens and transformer blocks. |
| `depth` | `4` | Number of transformer blocks. |
| `n_heads` | `8` | Attention heads; must evenly divide `embed_dim`. |
| `mlp_ratio` | `4.0` | FFN hidden width multiplier. |
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
`n_action_mixtures`, `entropy_ship_quantiles=16`, and the same logistic-mixture
scale parameters.
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

The boolean `orbiting_planets` mask selects the orbiting-planet projection for
orbiting rows and the static-planet projection for all other planet rows.
Planet, comet, and fleet tokens are concatenated on the entity axis in that
order. This keeps the action-origin hidden states contiguous as the first
`ACTION_ENTITY_SLOTS` tokens. The global projection is added to every entity
token and is also appended as its own global-feature token. The full trunk
sequence is:

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

## Transformer Trunk

The shared trunk is a stack of pre-norm transformer blocks configured by:

- `depth`
- `n_heads`
- `mlp_ratio`
- `embed_dim`

`n_heads` must evenly divide `embed_dim`, and `int(embed_dim * mlp_ratio)` must
be at least 1. The default activation is GELU. LayerNorm is used for
normalization, and no dropout is applied.

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

## Initialization

Models expose `reset_parameters()` through `BaseModelAPI`. Fresh training calls
this method explicitly before optimizer construction; checkpoint-loading paths
construct the module and load saved weights without resetting first.

Linear layers use orthogonal initialization with zero biases. Only the first
linear layer in each observation stem is treated as an input projection and
uses unit gain; hidden projections use ReLU-style gain, and transformer
residual output projections are scaled by `1 / sqrt(2 * depth)`.
Learned token state parameters, including player, board scratch, actor-plan,
critic-value, and discrete source/target role tags, are also classified as
input layers for optimizer grouping so Muon does not update them.

Actor and critic output heads are two-layer MLP projections with hidden width
`embed_dim`, the configured activation in the middle, and output-specific final
widths. Only the second linear layer in each output MLP is treated as an output
layer for optimizer grouping and final-head initialization. Actor final output
layers use small `0.01` gain with zero biases, matching the normal RL
policy-layer initialization. The critic final output layer uses unit gain.

## Critic

The critic reads the per-player `critic_value_hidden` field from
`EncodedObservations`. A two-layer MLP head produces one logit per player, then
applies a masked softmax using `obs.still_playing` with shape `(batch, 4)`.

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

Both actor heads start from the same shared transformer trunk. For each
`(batch, player, action_entity)` position, the actor combines:

- source entity hidden state
- player hidden token
- the actor plan token for that player, for the discrete-target actor

The final action head is selected by `config.actor.action_spec`. The concrete
heads live under `python/owl/model/actor/`: `PureActor` for raw angles and
`DiscreteTargetsActor` for target slots.

When `use_learned_pairwise_bias=True`, the discrete-target and discrete
target-bin actors receive an auxiliary learned bias after the shared
self-attention trunk. The model builds six raw source-target features over the
44 action entity slots, applies a two-layer MLP
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
defaults to `max_per_planet_launches=1` and `min_fleet_size=1`; Python config
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
their Rust API paths:

- `launch`: bool, `(batch, 4, 44, max_per_planet_launches)`
- `ships`: int64, `(batch, 4, 44, max_per_planet_launches)`
- `angle`: float32, same shape, pure actor only
- `target`: int64, same shape, discrete-target actor only
- `target`: int64, `(batch, 4, 44)`, discrete target-bin actor only
- `fleet_bin`: int64, `(batch, 4, 44)`, discrete target-bin actor only

It also returns decomposed log-prob and entropy tensors for launch gates and
target or angle/size events, plus per-player action-entity totals with shape
`(batch, 4, 44)`. PPO stores and submits the typed action bundle directly, so
action-spec-specific payloads such as `fleet_bin` cannot be silently dropped.
Observation masks are likewise held in typed `ObsBatch.action_mask` bundles:
pure and discrete-target masks carry `can_act` plus `max_launch`, while
discrete target-bin masks carry only `can_act`.
Entropy outputs also carry policy-specific component names for logging, such as
`launch`, `target`, `fleet_size_full`, `fleet_size_mixture`,
`fleet_size_logistic`, or `event`.

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

For each source, the discrete actor emits:

- Bernoulli launch/stop logits
- masked categorical target logits over the 44 action entity slots
- mixture parameters for a truncated discretized logistic fleet-size policy

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
`(batch, 4, 44, 44, n_bins)` and no `max_launch` tensor.

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
