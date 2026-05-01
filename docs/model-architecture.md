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
| `n_action_mixtures` | `4` | Mixture components for launch angle and fleet-size event heads. |
| `kappa_min` | `1e-3` | Lower bound added to von Mises concentration. |
| `kappa_max` | `200.0` | Optional cap for von Mises concentration. |
| `entropy_ship_support_cap` | `256` | Maximum ship-count support enumerated for entropy estimates. |
| `tau_min` | `1e-3` | Lower bound added to beta-binomial concentration. |
| `alpha_beta_eps` | `1e-4` | Epsilon added to beta-binomial alpha and beta. |
| `dir_eps` | `1e-6` | Epsilon for normalizing raw angle direction vectors. |
| `max_ship_normalizer` | `250.0` | Normalizer for ship-budget actor features. |
| `force_flash_attn` | `False` | Require packed varlen flash-attn; raise an error instead of falling back when tensors are not flash-compatible. |

Observation and action specs are owned by `EnvConfig`. `StatelessTransformerV1`
receives `env.obs_spec` and `env.action_spec` when it is instantiated, so model
config presets cannot silently diverge from the environment tensor shapes.

## Input Encoding

`StatelessTransformerV1` consumes an `ObsBatch` containing on-device torch tensors
from `docs/rl-api-specs.md`.

Each observation tensor receives one feedforward projection to `embed_dim`:

- planets: `(batch, MAX_PLANETS, 16) -> (batch, MAX_PLANETS, embed_dim)`
- fleets: `(batch, max_fleets, 10) -> (batch, max_fleets, embed_dim)`
- comets: `(batch, MAX_COMETS, 88) -> (batch, MAX_COMETS, embed_dim)`
- globals: `(batch, 3) -> (batch, 1, embed_dim)`

Planet, comet, and fleet tokens are concatenated on the entity axis in that
order. This keeps the action-origin hidden states contiguous as the first
`ACTION_ENTITY_SLOTS` tokens. The global projection is added to every entity
token. Four learned per-player embeddings are then appended for the critic,
giving:

```text
(batch, max_entities + 4, embed_dim)
```

The planet, comet, fleet, and `still_playing` masks are concatenated into one
token mask. Masked tokens are excluded from attention keys and are zeroed in
the returned hidden states.

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

Actor output heads use small `0.01` gain with zero biases, matching the normal
RL policy-layer initialization. The critic head uses unit gain.

## Critic

The critic reads the final four player tokens. A linear head produces one logit
per player, then applies a masked softmax using `obs.still_playing` with shape
`(batch, 4)`.

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

For each `(batch, player, action_entity)` position, the actor combines:

- source entity hidden state
- player hidden token
- normalized `max_launch`

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

`StatelessTransformerV1` currently supports only `ActionPureConfig`.
`ActionDiscreteTargetsConfig` is available at the environment/API layer, but a
matching target-index actor and PPO rollout path have not been implemented.
PPO fails fast if the model and environment action specs do not match, or if a
non-`pure` action spec is used with the current trainer.

`ActionPureConfig()` defaults to `max_per_planet_launches=3` and
`min_fleet_size=1`. PPO model construction uses the environment action spec as
the source of truth, so the model launch-slot count and minimum launched fleet
size match the action tensors submitted to the environment.

For every slot, the policy emits:

- Bernoulli launch/stop logits
- mixture logits for angle/size components
- von Mises angle parameters
- shifted beta-binomial size parameters

Sampling stops per `(batch, player, entity)` lane when the previous launch is
false or when the remaining ship budget falls below `min_fleet_size`. The
shifted beta-binomial size distribution emits ships in
`min_fleet_size..remaining`, and the accumulated sampled ships are capped by
`max_launch`.

The model returns decomposed action tensors:

- `launch`: bool, `(batch, 4, 44, max_per_planet_launches)`
- `angle`: float32, `(batch, 4, 44, max_per_planet_launches)`
- `ships`: int64, `(batch, 4, 44, max_per_planet_launches)`

It also returns decomposed log-prob and entropy tensors for launch gates and
angle/size events, plus per-player action-entity totals with shape
`(batch, 4, 44)`.

The angle/size entropy is an augmented latent-mixture entropy estimate: mixture
label entropy plus expected component entropy. It is not the exact marginal
entropy of the emitted action when mixture components overlap. Ship-count
entropy enumerates support only up to `entropy_ship_support_cap`; residual ship
budgets above that cap use truncated support.

## Log-Prob Replay

The model exposes `evaluate_actions(obs, actions)` to replay externally supplied
action tensors through the same autoregressive state updates and return both
new-policy log-probs, entropies, and critic values from one encode.

Inactive and stopped slots are given finite dummy event inputs before masking so
their zeroed log-prob contributions do not introduce NaN gradients.
