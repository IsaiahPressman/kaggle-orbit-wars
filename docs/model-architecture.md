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
| `actor` | `{"action_spec": "pure"}` | Discriminated actor-head config. Supported actor specs are `"pure"` and `"discrete_targets"`. |

Actor-specific fields live inside the actor config. `ActorPureConfig` owns
pure-head fields such as `n_action_mixtures`, `kappa_min`, `kappa_max`,
`tau_min`, `alpha_beta_eps`, `dir_eps`, `max_ship_normalizer=500.0`, and
`entropy_ship_support_cap`. `ActorDiscreteTargetsConfig` owns
`n_action_mixtures`, `max_ship_normalizer=500.0`, `entropy_ship_quantiles=16`,
and the logistic-mixture scale parameters `scale_min=0.10`,
`scale_max_frac=0.5`, and `scale_max_abs_floor=8.0`.
Model YAML files can reference actor presets by name through adjacent
`configs/model/actor/*.yaml` files, for example `actor: discrete_targets`, or
can inline an actor config to override preset fields such as mixture count.

`FullConfig` validates that `env.action_spec.action_spec` matches
`model.actor.action_spec`. Direct model construction performs the same check
against the supplied environment action spec.

Observation and action specs are owned by `EnvConfig`. `StatelessTransformerV1`
receives `env.obs_spec` and `env.action_spec` when it is instantiated, so model
config presets cannot silently diverge from the environment tensor shapes.

## Input Encoding

`StatelessTransformerV1` consumes an `ObsBatch` containing on-device torch tensors
from `docs/rl-api-specs.md`.

Each observation tensor receives one feedforward projection to `embed_dim`:

- static planets: `(batch, MAX_PLANETS, 61) -> (batch, MAX_PLANETS, embed_dim)`
- orbiting planets: `(batch, MAX_PLANETS, 61) -> (batch, MAX_PLANETS, embed_dim)`
- fleets: `(batch, max_fleets, 57) -> (batch, max_fleets, embed_dim)`
- comets: `(batch, MAX_COMETS, 286) -> (batch, MAX_COMETS, embed_dim)`
- globals: `(batch, 3) -> (batch, 1, embed_dim)`

The boolean `orbiting_planets` mask selects the orbiting-planet projection for
orbiting rows and the static-planet projection for all other planet rows.
Planet, comet, and fleet tokens are concatenated on the entity axis in that
order. This keeps the action-origin hidden states contiguous as the first
`ACTION_ENTITY_SLOTS` tokens. The global projection is added to every entity
token. Four learned per-player embeddings are then appended for the critic,
giving:

```text
(batch, max_entities + 4, embed_dim)
```

The `entity_mask` uses the same planet, comet, fleet order and is concatenated
with `still_playing` into one token mask. Masked tokens are excluded from
attention keys and are zeroed in the returned hidden states.

## Transformer Trunk

The shared trunk is a stack of pre-norm transformer blocks configured by:

- `depth`
- `n_heads`
- `mlp_ratio`
- `embed_dim`

`n_heads` must evenly divide `embed_dim`. The default activation is GELU. LayerNorm
is used for normalization, and no dropout is applied.

CPU execution uses torch scaled-dot-product attention over regular
`(batch, seq, dim)` tensors with the token mask passed as the attention key
mask. CUDA execution uses packed varlen `flash-attn` when it is installed and
the attention tensors are fp16/bf16; otherwise it uses the same regular-shaped
scaled-dot-product attention path without packing and unpacking activations.
Set `force_flash_attn=True` to require packed varlen flash-attn and fail fast
when the backend, device, or dtype is not compatible.

Attention uses separate `q`, `k`, and `v` linear layers instead of one packed
QKV projection. SwiGLU also uses separate gate and value projections. This keeps
each weight matrix tied to one projection role, which is a better fit for Muon
optimizer assumptions than packing multiple operations into one parameter.

## Initialization

Linear layers use orthogonal initialization with zero biases. Input projections
use unit gain, hidden projections use ReLU-style gain, and transformer residual
output projections are scaled by `1 / sqrt(2 * depth)`.

Actor and critic output heads are two-layer MLP projections with hidden width
`embed_dim`, the configured activation in the middle, and output-specific final
widths. Only the second linear layer in each output MLP is treated as an output
layer for optimizer grouping and final-head initialization. Actor final output
layers use small `0.01` gain with zero biases, matching the normal RL
policy-layer initialization. The critic final output layer uses unit gain.

## Critic

The critic reads the final four player tokens. A two-layer MLP head produces one
logit per player, then applies a masked softmax using `obs.still_playing` with
shape `(batch, 4)`.

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
- normalized `max_launch`

The final action head is selected by `config.actor.action_spec`. The concrete
heads live under `python/owl/model/actor/`: `PureActor` for raw angles and
`DiscreteTargetsActor` for target slots.

### Pure Actor

The pure actor supports `ActionPureConfig`.

The repeated launch slots are generated autoregressively with a 2-layer minGRU
stack. Each recurrent step adds a learned launch-slot embedding so the actor has
an explicit first/second/third/etc. slot identity. Each slot also receives
dynamic inputs for current activity, absolute-normalized remaining ship budget,
previous launch decision, the sine and cosine of the previous sampled launch
angle, previous absolute-normalized ship count, remaining fraction of the
initial launch budget, previous ship-count fraction of the initial launch
budget, and normalized slot index.

The first recurrent slot receives the static actor input plus its slot
embedding. Dynamic recurrent features are added starting with the second launch
slot.

The minGRU cell follows the sequential equation from Feng et al., "Were RNNs
All We Needed?":

```text
z_t = sigmoid(linear_z(x_t))
h_tilde = linear_h(x_t)
h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
```

The sequence length is at most `max_per_planet_launches <= 4`, so this
implementation uses the straightforward sequential recurrence rather than the
paper's parallel scan variant.

`ActionPureConfig()` defaults to `max_per_planet_launches=3` and
`min_fleet_size=1`. PPO model construction uses the environment action spec as
the source of truth, so the model launch-slot count and minimum launched fleet
size match the action tensors submitted to the environment.

For every slot, the policy emits:

- Bernoulli launch/stop logits
- mixture logits for angle/size components
- von Mises angle parameters
- shifted beta-binomial size parameters

Each emitted parameter group uses its own two-layer MLP output head.

Sampling stops per `(batch, player, entity)` lane when the previous launch is
false or when the remaining ship budget falls below `min_fleet_size`. The
shifted beta-binomial size distribution emits ships in
`min_fleet_size..remaining`, and the accumulated sampled ships are capped by
`max_launch`.

The model returns decomposed action tensors. The `launch` tensor always keeps
the final launch-slot dimension expected by the Rust API:

- `launch`: bool, `(batch, 4, 44, max_per_planet_launches)`
- `ships`: int64, `(batch, 4, 44, max_per_planet_launches)`
- `angle`: float32, same shape, pure actor only
- `target`: int64, same shape, discrete-target actor only

It also returns decomposed log-prob and entropy tensors for launch gates and
target or angle/size events, plus per-player action-entity totals with shape
`(batch, 4, 44)`. PPO asks the action container for the submitted action-value
tensor, so the trainer does not branch on action-spec-specific tensor names.
Entropy outputs also carry policy-specific component names for logging, such as
`launch`, `target`, `fleet_size_full`, `fleet_size_mixture`,
`fleet_size_logistic`, or `angle_and_size`.

The angle/size entropy is an augmented latent-mixture entropy estimate: mixture
label entropy plus expected component entropy. It is not the exact marginal
entropy of the emitted action when mixture components overlap. Ship-count
entropy for the pure actor enumerates support only up to
`entropy_ship_support_cap`; residual ship budgets above that cap use truncated
support. The discrete-target actor instead uses deterministic truncated-logistic
quantile quadrature controlled by `entropy_ship_quantiles`.

### Discrete Targets Actor

The discrete-target actor supports `ActionDiscreteTargetsConfig` with
`max_per_planet_launches=1`. Model construction fails fast if the environment
uses a larger per-planet launch count for this actor.

Instead of the pure actor's minGRU, the discrete-target actor uses one
feedforward action block per source entity. It projects source slots to query,
key, and value tensors with a single target-selection head independent of the
shared transformer trunk's attention head count. It computes scaled dot-product
target logits for every `(source, target)` pair and masks those logits with the
4-D discrete `can_act` tensor. Fully masked source rows are sanitized to finite
zero logits and are suppressed by the launch/source mask.

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
exploration for large ship budgets. Stochastic sampling selects a mixture
component and samples the truncated discretized logistic with inverse-CDF
sampling, avoiding full ship-support enumeration in the rollout hot path. PPO
replay uses the marginal mixture log-probability of the integer ship count, not
the sampled component log-probability. The discrete-target entropy bonus is an
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

## Log-Prob Replay

The model exposes `evaluate_actions(obs, actions)` to replay externally supplied
action tensors through the same autoregressive state updates and return both
new-policy log-probs, entropies, and critic values from one encode.

Inactive and stopped slots are given finite dummy event inputs before masking so
their zeroed log-prob contributions do not introduce NaN gradients.
