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
- After release builds or long benchmarks on passively cooled laptops, wait for
  CPU temperature to settle before the next timed run.
- Default benchmark:
  `uv run python scripts/benchmark_envs.py --n-envs 256 --steps 1000 --target rust --no-progress`
- If results are close, rerun paired before/after measurements with longer
  timing and cooldown between runs before deciding.
- Use `--repeats` to report mean, standard deviation, min, and max throughput
  when run-to-run or seed-to-seed variance may be material.
- The accepted-change checkpoint should be approximately monotonic over time:
  a post-candidate result should be close to the next candidate's pre-change
  result when measured with the same command and thermal protocol.
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
- Overflow-warning volume: the current random-policy benchmark can emit many
  `max_entities exceeded` warnings during timed runs. Compare default-capacity
  and high-`--max-entities` runs before attributing regressions to simulator
  logic instead of observation-capacity pressure and warning I/O.

## Baselines

| Label | Command | Result | Notes |
| --- | --- | --- | --- |
| Initial | `uv run python scripts/benchmark_envs.py --n-envs 256 --steps 1000 --target rust --no-progress` | 84,951 steps/sec; 3.014 seconds for 256,000 env steps; 9.418 launches/step | Captured before the first optimization, using an installed release build. |

## Cooled Per-Commit Audit

After early short-run measurements produced non-monotonic results, the
code-changing commits were remeasured with `--steps 10000` and cooldown after
release builds on an M4 MacBook Air.

| Commit | Code State | Result |
| --- | --- | --- |
| `1eeb97b` | Main before this branch | 80,617 steps/sec |
| `908f6b5` | Candidate 2 | 80,574 steps/sec |
| `daca6b2` | Candidates 2 + 3 | 85,226 steps/sec |
| `06ca4bc` | Candidates 2 + 3 + 5 | 82,703 steps/sec |
| `7fdf2f8` | Candidates 2 + 3 + 5 + 6 | 85,494 steps/sec |
| Temporary worktree | Candidates 2 + 3 + 6, candidate 5 reverted | 87,725 steps/sec |
| Current worktree | Candidates 3 + 6, candidates 2 and 5 reverted | 85,181 steps/sec |
| Current worktree | Candidates 3 + 6, candidates 2 and 5 reverted, `--steps 5000 --repeats 3 --cooldown-seconds 60` | Mean 84,882 steps/sec; std 3,532; min 81,867; max 88,769 |

The audit found two measurement issues. First, candidate 5 was a real
regression under the cooled protocol and was reverted. Second, candidate 2 was
not independently improved: it measured 80,574 steps/sec against 80,617
steps/sec on main, and the repeated current-worktree comparison is within the
same several-thousand-steps/sec noise band as the candidate-2-present run. Since
candidate 2 also carried an invalid-state behavior discrepancy, it was reverted
instead of kept on ambiguous data.

## Candidates

| Candidate | Status | Behavior Risk | Benchmark Impact | Verification | Commit/Notes |
| --- | --- | --- | --- | --- | --- |
| 1. Remove `state.planets.clone()` from fleet movement | Skipped | Very low: worker and reviewer found the borrow-only implementation behavior-equivalent, preserving planet order, collision order, fleet math, combat contents, and removal behavior. | Initial short run: 84,951 -> 85,468 steps/sec. Longer rerun did not confirm the win: pre-change 81,259 steps/sec, after-change 80,839 steps/sec. | Worker ran `cargo test rules_engine::env` and `just rs-prepare`; reviewer ran `cargo test rules_engine::env --lib` and `cargo test --lib`. | Code change excluded because release-mode measurements showed no reliable performance improvement. |
| 2. Replace combat `HashMap + sort` with fixed owner scan | Reverted | Low for valid simulator states, but the optimized path assumed fleet owners are in `0..=3`; invalid/manual states with out-of-range fleet owners would panic instead of being accepted as arbitrary `HashMap` keys. This was approved as acceptable only if the performance win held. | Early short default run appeared to improve 82,403 -> 86,941 steps/sec, but cooled per-commit testing did not confirm it: main measured 80,617 steps/sec and candidate 2 measured 80,574 steps/sec. Candidate-2-present repeated comparison was also within noise of the reverted branch. | Worker ran `cargo test rules_engine::env` and `just rs-prepare`; reviewer found no valid-state behavior issues. Added 4-player top-tie and tied-second tests, which still pass after reverting the implementation. Final verification after revert: `cargo test rules_engine::env --lib`, `cargo fmt --check`, `just build-release`, and cooled repeated benchmarks. | Code change reverted because performance was not reliably improved and the invalid-state behavior discrepancy was avoidable. Regression tests kept. |
| 3. Stop rebuilding removed-fleet sets inside every sweep | Accepted | Low for valid simulator states. The optimized path assumes live fleet IDs are unique; invalid/manual states with duplicate live fleet IDs can now skip the second same-id fleet within a single sweep, instead of queueing both from the old per-sweep snapshot. This follows the `next_fleet_id` uniqueness invariant and was approved before keeping the change. | Early short default run appeared to improve 86,941 -> 95,670 steps/sec. Cooled per-commit testing attributed the main confirmed gain to this area: main measured 80,617 steps/sec while candidates 2 + 3 measured 85,226 steps/sec, and candidate 2 alone was flat. | Worker ran `cargo test rules_engine::env::tests` and `just rs-prepare`; reviewer found no valid-state behavior issues. Added sweep precedence tests for first sweep target and planet-before-comet ordering. Final verification: `cargo test rules_engine::env --lib`, `just build-release`, default benchmark, and `just rs-prepare`. | Kept. |
| 4. Compact combat accumulators instead of cloned fleet lists | Skipped | Low for valid simulator states under the unique-fleet-id invariant. The attempted implementation also changed invalid overflow timing by summing ships at queue time, even for combats that might later target a missing planet. | 95,670 -> 90,984 steps/sec in the default release benchmark. | Worker ran `cargo fmt --check`, `cargo test rules_engine::env --lib`, and `just rs-prepare`; reviewer found no valid-state correctness issue. Final verification before exclusion: `just build-release` and default benchmark. | Code change excluded because it was slower. Likely cause: removing fleet clones did not offset repeated linear lookup over compact planet accumulators; an index map may be required but would increase complexity and overlap with later scratch-structure ideas. |
| 5. Remove tiny per-step maps/sets | Reverted | Low for the accepted subset, but the benchmark result did not hold under cooled per-commit testing. Skipped direct comet planet lookup because it would change invalid/manual duplicate planet IDs from `HashMap` last-wins to linear-search first-wins. Also skipped ordered `initial_planets` zip to preserve manual states where current and initial planet order differ. | Early short default run appeared to improve 93,540 -> 96,680 steps/sec, but cooled per-commit testing showed candidate 5 regressed 85,226 -> 82,703 steps/sec. | Worker ran focused rules/action/obs tests and `just rs-prepare`; reviewer identified the duplicate planet ID risk, so that subpart was reverted before the initial commit. Per-commit audit later showed the remaining subset was slower. | Code change reverted after cooled per-commit audit. |
| 6. Rewrite player results with fixed arrays | Accepted | Low for valid simulator states. The implementation uses `MAX_PLAYERS = 4` fixed buffers and intentionally panics for invalid 5+ player states. Scores are still computed only after terminal status is known, preserving old nonterminal overflow timing. | Cooled candidate-2-present interaction benchmark with candidate 5 reverted measured 87,725 steps/sec. After reverting candidate 2, the current branch measured 85,181 steps/sec on a 10,000-step run and 84,882 mean steps/sec with 3,532 std over three 5,000-step repeats, which is roughly flat against the candidate 3 checkpoint under the observed variance. | Worker ran `cargo fmt --check`, `cargo test rules_engine::env::tests`, and `just rs-prepare`; reviewer found the initial nonterminal score-overflow issue, which was fixed. Added a regression test proving nonterminal result computation does not sum scores. Final verification: focused env/RL tests, `just rs-prepare`, `just build-release`, and cooled 10,000-step benchmark audit. | Kept adjusted implementation plus shared `MAX_PLAYERS` constant requested for fixed player-shaped buffers. |
| 7. Avoid RNG creation on non-comet steps | Skipped | Low for valid states. Spawn-step random streams were preserved by still constructing the real RNG on comet-spawn candidate steps. Invalid/manual `state.step == u32::MAX` would overflow slightly earlier because the public `step` checked `state.step + 1` before later validation. | Default short run regressed from 92,646 -> 88,106 steps/sec. Longer 10,000-step A/B with cooldown was effectively flat: 85,363 without the change vs 85,844 with it. | Worker ran `cargo test rules_engine::env::tests` and `just rs-prepare`; reviewer found no valid-state issue. Final verification: focused env tests, `just build-release`, and cooled 10,000-step A/B benchmarks. | Code change excluded because performance was not reliably improved. |
| 8. Cache action entity slots per environment | Accepted | Low for valid vector-env states. Action slots are refreshed while writing observations and consumed by the next `step` decode, preserving the intended prior-observation mapping. Manual invalid states with duplicate planet IDs remain ambiguous because actions ultimately carry only `from_planet_id`; a stale/reordered cached slot could validate one duplicate before the rules engine resolves the first matching ID. This is outside generated-state invariants and is documented as the behavior-risk location. | Short cooled run improved 89,831 -> 90,401 mean steps/sec over two 3,000-step repeats. Longer paired worktree run improved 86,678 -> 88,372 mean steps/sec over three 5,000-step repeats with 30-second cooldown; baseline std 2,446, candidate std 655. | Worker ran focused RL action/vec-env/obs tests, `cargo fmt --check`, and `just rs-prepare`; reviewer found no blocking issues and confirmed prior-observation decode and invalid-action no-mutation semantics. Added cached-slot and stale-slot action decode tests. Final verification: `cargo test rl::action_spec --lib`, `cargo test rl::vec_env --lib`, `cargo test rl::obs_spec --lib`, `cargo test --lib`, `just build-release`, and paired cooled benchmarks. | Kept. |
| 9. Reuse decoded action buffers | Skipped | Low. The attempted implementation preserved the two-phase decode-before-mutation contract, but failed decodes could leave partial actions in internal scratch buffers until the next decode cleared them. Reviewer found no external behavior leak; an integration test confirmed invalid actions did not mutate observations and the next no-op step did not replay partial decoded actions. | Short cooled run regressed 91,312 -> 90,673 mean steps/sec over two 3,000-step repeats. Longer paired worktree run was flat: 89,052 -> 89,135 mean steps/sec over three 5,000-step repeats with 30-second cooldown. | Worker ran focused RL tests, `just rs-prepare`, and `git diff --check`; reviewer found no blocking issues. Local verification before exclusion: `cargo test rl::action_spec --lib`, `cargo test rl::vec_env --lib`, `cargo test --lib`, `just build-release`, and paired cooled benchmarks. | Code change excluded because performance was not reliably improved and the extra per-env scratch state did not justify itself. |
| 10. Fuse vector-env update and observation writing | Skipped | Medium. A true fused `step` pass would need observation output slices validated before state mutation so the fused pass can write observations immediately after stepping. Today action decode happens before mutation, then state/reward/done mutation happens, then observation buffer validation/writing happens. Moving output-buffer validation earlier changes validation timing semantics, while avoiding that would require a larger restructuring. | Not benchmarked; no code change was kept. Pre-candidate short baseline was 88,710 mean steps/sec over two 3,000-step repeats with 30-second cooldown, but high variance made it unsuitable for judging without an implementation. | Worker inspected `src/rl/vec_env.rs`, ran focused RL action/obs/vec-env tests, and made no file changes. | Skipped because the safe implementation path was not small enough and carried validation-timing behavior risk. |
| 11. Optimize fleet observation sorting/scratch | Skipped | Low for the attempted partial-sort subset. The implementation used a total comparator `(ships desc, id asc, original index)` so `select_nth_unstable_by` plus prefix sort preserved the previous full stable sort order, including manual invalid duplicate `(ships, id)` fleets. Reusable scratch was not implemented because the exact partial-sort subset was already slower. | Default paired benchmark regressed 88,893 -> 87,737 mean steps/sec over three 5,000-step repeats with 30-second cooldown. High-launch fleet-pressure run also regressed 53,745 -> 52,268 mean steps/sec over two 3,000-step repeats, with warning I/O dominating both runs. | Worker added full-sort parity tests and ran focused obs tests plus `just rs-prepare`; reviewer found no blocking behavior issues. Local verification before exclusion: `cargo test rl::obs_spec --lib`, `cargo test --lib`, `just build-release`, and paired cooled benchmarks. | Code change excluded because performance was worse in both default and fleet-pressure benchmarks. |
| 12. Stream comet path generation | Skipped | Low. Worker and reviewer found the streamed implementation preserved the same point formula, iteration order, cumulative thresholds, visible path selection, symmetric expansion, and RNG consumption. Existing Python-reference comet path fixtures still passed exactly. | Default short benchmark regressed 90,744 -> 88,039 mean steps/sec over two 3,000-step repeats. Low-action comet-spawn workload initially looked close, 222,469 -> 223,710 mean steps/sec, but longer paired low-action testing regressed 221,564 -> 219,254 mean steps/sec over three 10,000-step repeats. | Worker ran `cargo test rules_engine::generation --lib`, `cargo fmt --check`, and `just rs-prepare`; reviewer found no blocking parity issue. Local verification before exclusion: focused generation tests, `cargo test --lib`, `just build-release`, default benchmark, and low-action paired benchmarks. | Code change excluded because performance was worse despite exact behavior parity. |
| 13. Return `[Planet; 4]` from `symmetric_planets` | Accepted | Very low. The array-return implementation uses the same `fourfold_symmetric_points` order, assigns IDs by the same index order, and call sites still extend `Vec<Planet>` in group order. | Low-action reset/spawn-heavy benchmark improved 219,599 -> 222,458 mean steps/sec over two 3,000-step repeats with 30-second cooldown and `--launch-prob 0.001`. | Worker ran `cargo fmt --check`, `cargo test rules_engine::generation --lib`, and `just rs-prepare`; reviewer found no blocking behavior issues. Local verification: `cargo test rules_engine::generation --lib`, `cargo fmt --check`, `just build-release`, and cooled repeated benchmark. Added an order-specific symmetric group test. | Kept. |
| 14. Cache generation/orbit metadata | Skipped | Low for the attempted generation-only `valid_group` metadata cache: it stayed private to generated planets, kept metadata aligned by appending only on accepted groups, and reused the same `distance` and `is_orbiting` formulas. The broader per-step orbit metadata cache was skipped because it would touch public `State` alignment and risks changing manual invalid states with missing or duplicate `initial_planets` IDs. | Short low-action reset/spawn-heavy run was flat: 222,458 -> 222,394 mean steps/sec over two 3,000-step repeats with 30-second cooldown and `--launch-prob 0.001`. Longer low-action paired run was only mildly positive: 222,749 -> 223,342 mean steps/sec over three 10,000-step repeats with 45-second cooldown. Default random-policy run regressed 91,179 -> 90,409 mean steps/sec over two 3,000-step repeats with 30-second cooldown. | Worker ran focused generation tests and `just rs-prepare`; reviewer found no blocking behavior issues for the generation-only subset and confirmed fixture parity/RNG order. Local verification before exclusion: `cargo test rules_engine::generation --lib`, `cargo fmt --check`, `just rs-prepare`, `just build-release`, and paired cooled benchmarks. | Code change excluded because performance was not reliably improved. |
| 15. Math cleanup and broad-phase collision culling | Accepted | Low for valid finite states with nonnegative radii. The broad-phase AABB check is only a conservative prefilter expanded by `radius + 1e-12`; final collision behavior still uses the exact strict `point_to_segment_distance(...) < radius` predicate, preserving exact-boundary non-collision semantics. Manual invalid NaN/negative-radius states should not create newly accepted collisions because the final exact predicate remains decisive. `powi(2)` was replaced with explicit multiplication in geometry helpers; any discrepancy would be limited to tiny floating-point lowering differences, with existing boundary tests and replay tests still passing. | Default random-policy benchmark improved 90,680 -> 103,393 mean steps/sec over two 3,000-step repeats with 30-second cooldown. High-launch fleet-pressure benchmark improved 53,306 -> 63,858 mean steps/sec over two 3,000-step repeats with 30-second cooldown, `--launch-prob 1.0 --max-entities 512`; warning I/O still occurred in both runs, so this is a noisy stress workload. | Worker ran focused utils/env tests and `just rs-prepare`; reviewer found no blocking boundary, collision-order, invalid-value, or duplicate-ID issues. Local verification: `cargo test rules_engine::utils --lib`, `cargo test rules_engine::env --lib`, `cargo fmt --check`, `just rs-prepare`, `just build-release`, and cooled default/fleet-pressure benchmarks. Added AABB tests for exact-boundary, endpoint-collision, and far-point cases. | Kept. |
