# Quantization Optimizations

This is a standing log of checkpoint-compression experiments for Kaggle
submission models. Checkpoint quantization is separate from the configured
serving-time inference path documented in the
[README Kaggle submission section](../README.md#kaggle-submission-build):
submission checkpoints are compressed to stay under the file-size limit, then
stream-dequantized into the model tensor by tensor at agent startup.

## Ground Rules

The primary constraint is the compressed checkpoint size. For the current
`20260523-153151` 152M checkpoint, the working target is at or below roughly
96MiB so the full submission has margin under the 100MiB Kaggle limit.

Behavior preservation is measured before expensive downstream self-play. Weight
MSE is useful for debugging codecs, but candidate selection should prioritize
teacher-vs-candidate policy/value drift on realistic observations.

Any custom format that is not supported by `owl.checkpoint_quantization`
must be treated as experimental until the Kaggle agent can load it.
Supported custom payloads are strict artifact schemas: packed fp4 and grouped
normal-float NF3/NF4/NF5 data must have the exact byte length implied by the
tensor shape and format, and extra trailing bytes are rejected as corrupt or
stale checkpoint data. The agent streaming loader avoids materializing the full
dequantized fp32 state dict at once, and grouped normal-float decode uses
in-place scale application to keep temporary memory lower during startup.
Grouped normal-float scale fitting ignores padded group positions and clamps
least-squares scales to the valid group's max absolute value. This keeps
quantized-then-dequantized tensors exact fixed points of the same quantizer, so
a later packaging pass serializes the same packed codes and fp16 scales for
unchanged weights.
LoRA checkpoints may store adapter tensors with a separate quantization format
from the frozen base model. If a base quantization format is requested without an
adapter format, adapters default to fp16. Runtime inference loaders dequantize
both groups, fold LoRA updates into base linear weights, then apply any int8
inference emulation/quantization.

## Measurement Protocol

Local CPU-only measurements in this log used:

- Checkpoint:
  `artifacts/20260523-153151/checkpoint_04_060_020_736.weights.pt`
- Config: `artifacts/20260523-153151/config.yaml`
- Reference fp4 checkpoint:
  `artifacts/20260523-153151/checkpoint_04_060_020_736_fp4_e2m1fn_x2_scaled_block16.pt`
- Agent serving overrides from `python/owl/agent/agent_config.yaml`:
  `max_entities_override=96`, `targeting_mode_override=full_mask`, and
  `int8_quantization=never`. These historical proxy measurements used fp32
  CPU inference after checkpoint dequantization rather than int8 serving
  numerics.
- Replay source: sorted `replays/*.jsonl` benchmark replay files.
- Calibration filter: keep compacted replay frames when any active player's
  fp32 teacher value is in `[-0.8, 0.8]`.
- Calibration set: first 1000 matching frames after scanning 5488 replay
  frames; 15674 active source decisions.
- Batching: compact each single-frame observation, then pad compacted rows into
  batch size 16 for evaluation.
- Teacher outputs: deterministic fp32 actions plus `evaluate_actions` log-probs,
  values, and winner probabilities are precomputed once.

Replay caveat: `ReplayRecorder` snapshots do not store Kaggle
`initial_planets` separately. The experiment reconstructs `initial_planets`
from the first frame of each replay episode. This preserves the broad board
state and action masks but is not a perfect reconstruction of the original
training observation.

Primary metrics:

- `mean_abs_logp_delta`: mean absolute drift in per-player/entity action
  log-probability for the teacher's deterministic action.
- `mean_logp_drop`: mean `teacher_logp - candidate_logp`; lower is better.
- `value_mae`: mean absolute active-player value drift.
- `launch_agreement`: deterministic launch/no-launch agreement over active
  source slots.
- `target_agreement_on_ref_launch` and `ships_mae_on_ref_launch`: deterministic
  action agreement conditional on the teacher launching.

## Experiment Log

| Work tried | Outcome | Behavior notes | Size and metric impact | Verification and follow-up |
| --- | --- | --- | --- | --- |
| Current `fp4_e2m1fn_x2_scaled_block16` reference | Baseline | Already supported by the Kaggle agent. Uses e2m1 fp4 data plus fp16 scale per block of 16 values. | 90.71MiB. On 1000 compacted replay frames: mean abs log-prob drift 0.0849, mean log-prob drop 0.0732, value MAE 0.0906, launch agreement 98.31%, target agreement 92.70%, ship-count MAE 1.456. | Reference file is present in `artifacts/20260523-153151`. This remains the downstream-comparison baseline. |
| fp4 with least-squares block scales (`fp4_lsq`) | Promising, production-compatible | Uses the same supported per-tensor fp4 payload format as the current agent, but chooses each block scale by a couple of least-squares updates instead of fixed `max_abs / 6`. | 90.73MiB. Mean abs log-prob drift improved 0.0849 -> 0.0325; mean log-prob drop 0.0732 -> 0.0240; value MAE 0.0906 -> 0.0471; launch agreement 98.31% -> 99.14%; target agreement 92.70% -> 94.32%; ship-count MAE 1.456 -> 1.152. | Saved as `artifacts/20260523-153151/checkpoint_04_060_020_736_fp4_lsq.pt`. Because tensor payloads still advertise `fp4_e2m1fn_x2_scaled_block16`, this should load through the current agent. Worth a cheap downstream run if custom qpack support is deferred. |
| fp4-LSQ plus fp8 actor/critic/input-projection tensors | Skipped for now | Mixed supported per-tensor fp4/fp8 payload. Upgraded actor, critic, source/target actor input projections, and biases to fp8. | 95.18MiB on the 324-frame scout run. It improved over fp4-LSQ only slightly on log-prob/value drift and was worse on ship-count MAE, so it was not rerun on the 1000-frame set. Scout metrics: mean abs log-prob drift 0.0252 vs 0.0267 for fp4-LSQ; value MAE 0.0868 vs 0.0946; ship-count MAE 1.048 vs 0.874. | Do not prioritize this exact allocation. If spending fp8 budget, use sensitivity-ranked upgrades rather than broad actor/head upgrades. |
| Groupwise signed INT5, group size 128, fp16 scales, least-squares scale refinement | Promising, requires custom loader | Stores 2D floating tensors as packed 5-bit signed integer codes, grouped along the input dimension in chunks of 128. Non-2D floating tensors are stored fp16. | 93.51MiB. Mean abs log-prob drift 0.0178; mean log-prob drop 0.0106; value MAE 0.0355; launch agreement 99.42%; target agreement 95.95%; ship-count MAE 0.756. This is much better than fp4-LSQ at modest extra size. | Saved experiment summary in `artifacts/20260523-153151/quantization_experiment_large_summary.json`. Needs production qpack serialization/dequantization before submission. |
| Groupwise NF5, group size 128, fp16 scales, least-squares scale refinement | Best simple uniform codec | Uses a fixed 32-value normal-float codebook with per-group fp16 scales and the same row/input grouping as INT5. Non-2D floating tensors are fp16. | 93.51MiB. Mean abs log-prob drift 0.0122; mean log-prob drop 0.0060; value MAE 0.0218; launch agreement 99.63%; target agreement 96.99%; ship-count MAE 0.466. This beat INT5 on every tracked metric at the same size. | Strongest simple candidate. Implement qpack support and benchmark downstream before trying more elaborate PTQ. |
| NF5 plus targeted fp8 upgrades for actor/trunk final-block output weights | Best measured proxy, size is close to cap | Uses NF5 for most 2D tensors, fp16 for non-2D floating tensors, and fp8 for source/target actor input projections, selected actor bridge/output tensors, and `attn.out`/`mlp.down` in the last transformer block discovered from the checkpoint state dict. Critic tensors are not upgraded. | 95.95MiB. Mean abs log-prob drift 0.0121; mean log-prob drop 0.0057; value MAE 0.0210; launch agreement 99.62%; target agreement 97.25%; ship-count MAE 0.377. This only slightly improves log-prob/value drift over plain NF5, but noticeably improves ship-count agreement while staying under the 96MiB target. These metrics were collected before the actor/trunk-only dynamic selector, so they should be remeasured before final submission selection. | Implemented as `nf5_g128_lsq_policy_last_fp8` in `owl.checkpoint_quantization`, with extraction and roundtrip script support through the shared quantization format registry. The original experiment file was saved as `artifacts/20260523-153151/checkpoint_04_060_020_736_nf5_g128_lsq_policy_last_fp8.experimental.pt` and summarized in `artifacts/20260523-153151/quantization_experiment_mixed_summary.json`. |
| Low-bit grouped NF3/NF4 formats on the `20260601-062158` checkpoint | Implemented for compression experiments | Added `nf3_g128_lsq`, `nf3_nf4_structured_3p5`, and `nf4_g128_lsq`. All store 2D floating tensors with group size 128, fp16 scales, and least-squares scale refinement; non-2D floating tensors are fp16. The structured 3.5-bit format upgrades whole sensitive actor/trunk 2D tensors to NF4 until roughly half the padded non-critic code budget is upgraded, leaving the rest at NF3. | On the 1000-frame compacted replay benchmark, estimated payloads were 57.58MiB for NF3, 66.70MiB for structured NF3/NF4, and 75.84MiB for NF4. Mean abs log-prob drift was 0.0853, 0.0685, and 0.0186 respectively; value MAE was 0.3284, 0.2279, and 0.0769; launch agreement was 97.80%, 98.31%, and 99.36%. These metrics were collected before critic tensors were excluded from the NF4 upgrade budget. | Implemented in `owl.checkpoint_quantization` with extraction and roundtrip script support through the shared quantization format registry. Summary saved in `artifacts/20260601-062158/lowbit_quantization_summary.json`. |
| Activation-weighted NF5 scale refinement | Skipped | GPTQ-like diagonal-Hessian proxy. Collected input activation second moments for 198 `nn.Linear` weights on the 1000-frame calibration set, then used those weights in the per-group scale least-squares update. This is not full GPTQ error compensation. | 93.51MiB. Mean abs log-prob drift 0.0131; mean log-prob drop 0.0066; value MAE 0.0186; launch agreement 99.60%; target agreement 96.74%; ship-count MAE 0.527. It improved value drift versus plain NF5 but worsened policy/action metrics, so it did not beat the targeted-fp8 candidate. | Summary saved in `artifacts/20260523-153151/quantization_experiment_skipped_ideas_summary.json`. Do not prioritize this diagonal-only variant; full GPTQ would need real layer-output error compensation. |
| AWQ-style activation-selected exact columns | Skipped | Used the same activation statistics as weighted NF5, selected the top 0.75% input channels per linear layer, and stored those columns exactly in fp16 on top of weighted NF5. | 95.90MiB. Mean abs log-prob drift 0.0123; mean log-prob drop 0.0061; value MAE 0.0189; launch agreement 99.63%; target agreement 96.87%; ship-count MAE 0.488. It fit the budget and improved value drift, but it was worse than targeted fp8 on log-prob, target agreement, and ship-count error. | Summary saved in `artifacts/20260523-153151/quantization_experiment_skipped_ideas_summary.json`. A more selective AWQ allocation could still be tried, but this broad column-protection recipe is not the best candidate. |
| Low-rank residuals over NF5 for policy and final four block outputs | Skipped | Stored NF5 everywhere plus fp16 rank-16 residual factors for source/target actor projections, selected bridge tensors, and `attn.out`/`mlp.down` weights in the then-current final four transformer blocks. | 94.37MiB. Mean abs log-prob drift 0.0122; mean log-prob drop 0.0059; value MAE 0.0215; launch agreement 99.63%; target agreement 96.95%; ship-count MAE 0.467. It was effectively flat versus plain NF5 and worse than targeted fp8 despite using extra bytes. | Summary saved in `artifacts/20260523-153151/quantization_experiment_skipped_ideas_summary.json`. This low-rank target/rank choice is not worth keeping; if revisited, choose final blocks dynamically and avoid critic-only tensors. |
| One-pass head distillation after NF5 plus targeted fp8 | Skipped | Started from the best NF5/fp8 candidate, trained actor/source-target projection and critic-head parameters for one pass on the same calibration set against teacher log-probs, values, and winner probabilities, then repacked with the same NF5/fp8 format. This is a small mini-QAT/distillation proxy rather than full AdaRound. | 95.95MiB. Training loss fell from 0.0780 to 0.0005, but proxy metrics regressed: mean abs log-prob drift 0.0138, mean log-prob drop 0.0071, value MAE 0.0213, launch agreement 99.61%, target agreement 97.12%, ship-count MAE 0.492. | Summary saved in `artifacts/20260523-153151/quantization_experiment_distill_summary.json`. Do not use this naive head-only distillation recipe; if revisited, hold out a validation subset, optimize quantization parameters rather than ordinary fp32 head weights, and skip critic-only tensors. |
| Lossless LZMA wrapper | Rejected by constraint | Preset 3/6 LZMA compressed the existing fp4-LSQ checkpoint from 90.73MiB to 86.38/85.87MiB and the NF5/fp8 checkpoint from 95.95MiB to 91.74/91.29MiB, taking roughly 29-41s per file to compress locally. Decompression would add startup cost and likely overlaps with `submission.tar.gz` compression. | Not ranked. A raw 97.76MiB final-four-block fp8 candidate had slightly better metrics than targeted fp8, but it is over the raw ~96MiB checkpoint budget and is not considered because lossless wrapping is intentionally excluded. | Leave lossless checkpoint wrappers out unless startup-time constraints change. |

## Skipped Ideas

These are still not implemented in this pass:

- Full GPTQ with blockwise Hessian inverse and sequential error compensation.
- Full AdaRound with learned hard-rounding variables.
- Longer fake-quant QAT with held-out validation.
- Entropy-coded custom binary blobs beyond `torch.save`, by request.

Given the measured gap, the next practical step is downstream self-play for the
NF5 plus targeted-fp8 candidate before adding full GPTQ/AdaRound complexity.
