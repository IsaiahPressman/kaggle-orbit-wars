# Simulation Optimizations

This document tracks simulator and vector-environment optimization attempts from
`misc/optimization-ideas.md`. Each candidate is handled independently: measure a
release-mode baseline, make one scoped change, verify behavior, remeasure, then
either commit the improvement or record why it was skipped.

## Behavior-Preservation Gate

Behavior preservation is the primary constraint. Candidates with any meaningful
risk of changing simulation behavior must be escalated for human approval before
keeping the implementation. The escalation should include the suspected behavior
risk, verification performed, and any remaining doubt.

Tiny floating-point differences from algebraically equivalent rewrites are still
documented here when they are accepted. Behavior-sensitive approximation
candidates are skipped unless explicitly approved after review.

## Measurement Protocol

- Build and benchmark release mode only.
- Default benchmark:
  `uv run python scripts/benchmark_envs.py --n-envs 256 --steps 1000 --target rust --no-progress`
- If results are close, rerun with longer timing before deciding.
- Run focused tests for the touched behavior, then `just rs-prepare` after Rust
  edits and `just py-prepare` after Python edits.

## Benchmark Coverage Notes

The default benchmark uses a random valid-launch policy. It is useful for broad
RL vector-env throughput, but it can under- or over-represent some workloads:

- Fleet-heavy games: trained policies may create fewer large fleets or many
  coordinated small fleets. Add an always-launch/high-launch-probability run and
  a handcrafted fleet-heavy state benchmark for collision, combat, and fleet
  observation sorting changes.
- Reset-heavy and spawn-heavy paths: random rollouts include resets and comet
  spawns, but do not isolate them. Add reset-only and comet-spawn-step
  benchmarks before judging generation/comet optimizations.
- Low-action policies: no-op or conservative policies stress production,
  planet movement, observation writing, and action masks without much fleet
  churn. Add a no-launch benchmark for changes outside combat and fleet motion.
- Player-count mix: the default run is 4-player. Also run `--players 2` for
  changes involving player result, player mapping, action masks, or outer player
  slots.
- Observation capacity pressure: benchmark states with fleets below, near, and
  above `max_fleets`, because overflow logging and fleet sorting/truncation can
  dominate differently from ordinary random rollouts.

## Baselines

| Label | Command | Result | Notes |
| --- | --- | --- | --- |
| Initial | `uv run python scripts/benchmark_envs.py --n-envs 256 --steps 1000 --target rust --no-progress` | 84,951 steps/sec; 3.014 seconds for 256,000 env steps; 9.418 launches/step | Captured before the first optimization, using an installed release build. |

## Candidates

| Candidate | Status | Behavior Risk | Benchmark Impact | Verification | Commit/Notes |
| --- | --- | --- | --- | --- | --- |
| 1. Remove `state.planets.clone()` from fleet movement | Skipped | Very low: worker and reviewer found the borrow-only implementation behavior-equivalent, preserving planet order, collision order, fleet math, combat contents, and removal behavior. | Initial short run: 84,951 -> 85,468 steps/sec. Longer rerun did not confirm the win: pre-change 81,259 steps/sec, after-change 80,839 steps/sec. | Worker ran `cargo test rules_engine::env` and `just rs-prepare`; reviewer ran `cargo test rules_engine::env --lib` and `cargo test --lib`. | Code change excluded because release-mode measurements showed no reliable performance improvement. |
| 2. Replace combat `HashMap + sort` with fixed owner scan | Accepted | Low for valid simulator states. The optimized path assumes fleet owners are in `0..=3`; invalid/manual states with out-of-range fleet owners now panic instead of being accepted as arbitrary `HashMap` keys. This follows the simulator's valid-owner invariant and was approved before keeping the change. | 82,403 -> 86,941 steps/sec in the default release benchmark. | Worker ran `cargo test rules_engine::env` and `just rs-prepare`; reviewer found no valid-state behavior issues. Added 4-player top-tie and tied-second tests. Final verification: `cargo test rules_engine::env --lib`, `just build-release`, default benchmark, and `just rs-prepare`. | Kept. |
| 3. Stop rebuilding removed-fleet sets inside every sweep | Accepted | Low for valid simulator states. The optimized path assumes live fleet IDs are unique; invalid/manual states with duplicate live fleet IDs can now skip the second same-id fleet within a single sweep, instead of queueing both from the old per-sweep snapshot. This follows the `next_fleet_id` uniqueness invariant and was approved before keeping the change. | 86,941 -> 95,670 steps/sec in the default release benchmark. | Worker ran `cargo test rules_engine::env::tests` and `just rs-prepare`; reviewer found no valid-state behavior issues. Added sweep precedence tests for first sweep target and planet-before-comet ordering. Final verification: `cargo test rules_engine::env --lib`, `just build-release`, default benchmark, and `just rs-prepare`. | Kept. |
| 4. Compact combat accumulators instead of cloned fleet lists | Skipped | Low for valid simulator states under the already approved valid-owner and unique-fleet-id invariants. The attempted implementation also changed invalid overflow timing by summing ships at queue time, even for combats that might later target a missing planet. | 95,670 -> 90,984 steps/sec in the default release benchmark. | Worker ran `cargo fmt --check`, `cargo test rules_engine::env --lib`, and `just rs-prepare`; reviewer found no valid-state correctness issue. Final verification before exclusion: `just build-release` and default benchmark. | Code change excluded because it was slower. Likely cause: removing fleet clones did not offset repeated linear lookup over compact planet accumulators; an index map may be required but would increase complexity and overlap with later scratch-structure ideas. |
| 5. Remove tiny per-step maps/sets | Partially accepted | Low for the accepted subset: direct comet-ID membership preserves `HashSet` membership behavior, including duplicate comet IDs. Skipped direct comet planet lookup because it would change invalid/manual duplicate planet IDs from `HashMap` last-wins to linear-search first-wins. Also skipped ordered `initial_planets` zip to preserve manual states where current and initial planet order differ. | 93,540 -> 96,680 steps/sec in the default release benchmark. | Worker ran focused rules/action/obs tests and `just rs-prepare`; reviewer identified the duplicate planet ID risk, so that subpart was reverted. Final verification: focused rules/action/obs tests, `just build-release`, default benchmark, and `just rs-prepare`. | Kept only direct comet-ID membership in rules movement, observation encoding, and action entity slot filtering. |
| 6. Rewrite player results with fixed arrays | Accepted | Low for valid simulator states. The implementation uses `MAX_PLAYERS = 4` fixed buffers and intentionally panics for invalid 5+ player states. Scores are still computed only after terminal status is known, preserving old nonterminal overflow timing. | Short default run was noisy and looked worse; longer isolated comparison with the `MAX_PLAYERS` constant in both versions improved from 81,959 -> 87,363 steps/sec. | Worker ran `cargo fmt --check`, `cargo test rules_engine::env::tests`, and `just rs-prepare`; reviewer found the initial nonterminal score-overflow issue, which was fixed. Added a regression test proving nonterminal result computation does not sum scores. Final verification: focused env/RL tests, `just rs-prepare`, `just build-release`, and a longer release benchmark. | Kept adjusted implementation plus shared `MAX_PLAYERS` constant. |
| 7. Avoid RNG creation on non-comet steps | Pending | Low: comet-spawn random streams must be unchanged on spawn steps. | Pending | Pending | Pending |
| 8. Cache action entity slots per environment | Pending | Medium: submitted actions must decode against the prior observation's slots. | Pending | Pending | Pending |
| 9. Reuse decoded action buffers | Pending | Low: invalid-action no-mutation behavior must be preserved. | Pending | Pending | Pending |
| 10. Fuse vector-env update and observation writing | Pending | Low: output tensors must keep identical layout and reset semantics. | Pending | Pending | Pending |
| 11. Optimize fleet observation sorting/scratch | Pending | Low to medium: emitted top fleets and stale rows must remain exact. | Pending | Pending | Pending |
| 12. Stream comet path generation | Pending | Low: sampled path points and random consumption must match exactly. | Pending | Pending | Pending |
| 13. Return `[Planet; 4]` from `symmetric_planets` | Pending | Very low: exact same generated group order expected. | Pending | Pending | Pending |
| 14. Cache generation/orbit metadata | Pending | Medium: metadata alignment and floating-point reuse need proof. | Pending | Pending | Pending |
| 15. Math cleanup and broad-phase collision culling | Pending | Medium to high: boundary collision behavior is sensitive. | Pending | Pending | Pending |
