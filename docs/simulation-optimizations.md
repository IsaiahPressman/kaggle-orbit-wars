# Simulation Optimizations

This is a standing log of simulator and vector-environment optimization work.
It records what was tried, what was kept, what was reverted or skipped, and why.
The main goal is to help future contributors continue optimizing without
repeating noisy measurements or reopening behavior-sensitive changes without
context.

## Ground Rules

Behavior preservation is the primary constraint. Any change with a meaningful
risk of changing generated-game behavior should be reviewed carefully before it
is kept. If a change intentionally narrows behavior for manual or invalid
states, document that discrepancy here.

Tiny floating-point differences from algebraically equivalent rewrites can be
acceptable when existing boundary and replay tests still pass. Changes that
alter the measurement of collision thresholds, RNG streams, action decoding,
reward timing, or observation layout need stronger proof that the end behavior
still matches the original rules.

## Measurement Protocol

- Build and benchmark release mode only.
- Use `just build-release` before timing Rust/PyO3 changes.
- After release builds or long benchmarks, wait for CPU temperature to settle
 before the next timed run.
- Default benchmark:
  `uv run python scripts/benchmark_envs.py --n-envs 256 --steps 1000 --target rust --no-progress`
- If results are close, run paired before/after measurements with longer timing
  and cooldown before deciding.
- Use `--repeats` to report mean, standard deviation, min, and max throughput
  when run-to-run variance may be material.
- The accepted-change checkpoint should be approximately monotonic over time:
  a post-change result should be close to the next change's pre-change result
  when measured with the same command and on the same hardware.
- Run focused tests for the touched behavior, then `just rs-prepare` after Rust
  edits and `just py-prepare` after Python edits.

## Benchmark Coverage Notes

The default benchmark uses a random valid-launch policy. It is useful for broad
RL vector-env throughput, but it can under- or over-represent some workloads:

- Fleet-heavy games: trained policies may create fewer large fleets or many
  coordinated small fleets. Add an always-launch/high-launch-probability run and
  a handcrafted fleet-heavy state benchmark for collision, combat, and fleet
  observation changes.
- Reset-heavy and spawn-heavy paths: random rollouts include resets and comet
  spawns, but do not isolate them. Add reset-only and comet-spawn-step
  benchmarks before judging generation/comet optimizations.
- Low-action policies: no-op or conservative policies stress production,
  planet movement, observation writing, and action masks without much fleet
  churn. Add a no-launch benchmark for changes outside combat and fleet motion.
- Player-count mix: the default run is 4-player. Also run `--players 2` for
  changes involving player results, player mapping, action masks, or outer
  player slots.
- Observation capacity pressure: benchmark states with fleets below, near, and
  above `max_fleets`, because overflow logging and fleet sorting/truncation can
  dominate differently from ordinary random rollouts.
- Overflow-warning volume: the random-policy benchmark can emit many
  `max_entities exceeded` warnings during timed runs. Compare default-capacity
  and high-`--max-entities` runs before attributing regressions to simulator
  logic instead of observation-capacity pressure and warning I/O.

## Baselines

| Label | Command | Result | Notes |
| --- | --- | --- | --- |
| Initial branch baseline | `uv run python scripts/benchmark_envs.py --n-envs 256 --steps 1000 --target rust --no-progress` | 84,951 steps/sec; 3.014 seconds for 256,000 env steps; 9.418 launches/step | Captured before the first optimization, using an installed release build. |

## Optimization Log

| Work tried | Outcome | Behavior notes | Benchmark impact | Verification and follow-up |
| --- | --- | --- | --- | --- |
| Borrow planets during fleet movement instead of cloning the planet vector | Applied later as cleanup; original standalone timing was not convincing | Very low risk when field borrows preserve planet order, collision order, fleet math, combat contents, and removal behavior. A later cleanup also changed `initial_by_id` to store `&Planet` rather than cloned `Planet` values. | Original short run improved 84,951 -> 85,468 steps/sec, but a longer rerun did not confirm the win: 81,259 -> 80,839 steps/sec. The later cleanup was not independently benchmarked in this log. | Focused env tests and full Rust tests passed in the original attempt. Later cleanup is present in `014a9b8`. |
| Replace per-combat owner `HashMap + sort` with a fixed owner scan | Reverted | Low for generated states, but the optimized path assumed fleet owners are in `0..=3`. Manual invalid states with out-of-range fleet owners would panic instead of being accepted as arbitrary `HashMap` keys. | Early short run looked positive, 82,403 -> 86,941 steps/sec, but cooled per-commit testing did not confirm it: main measured 80,617 steps/sec and the fixed-owner scan measured 80,574 steps/sec. | Focused env tests and `just rs-prepare` passed. Four-player top-tie and tied-second regression tests were kept. Code was reverted because performance was flat and the invalid-state discrepancy was avoidable. |
| Track swept fleet IDs once per step instead of rebuilding removal sets inside every sweep | Kept | Low for generated states. The optimized path assumes live fleet IDs are unique; manual states with duplicate live fleet IDs can now skip the second same-id fleet within a single sweep. This follows the `next_fleet_id` uniqueness invariant and was approved before keeping. | The cooled audit attributed the first confirmed gain to this area: main measured 80,617 steps/sec while the state including tracked swept fleets measured 85,226 steps/sec, and the fixed-owner scan alone was flat. | Focused env tests and `just rs-prepare` passed. Added sweep precedence tests for first sweep target and planet-before-comet ordering. |
| Replace cloned fleet combat lists with compact combat accumulators | Skipped | Low for generated states under the unique-fleet-ID invariant. The attempted implementation also changed invalid overflow timing by summing ships at queue time, even for combats that might later target a missing planet. | Default release benchmark regressed 95,670 -> 90,984 steps/sec. | Focused env tests and `just rs-prepare` passed, but the code was excluded because it was slower. Likely cause: removing fleet clones did not offset repeated linear lookup over compact planet accumulators. |
| Remove small per-step maps/sets for comet and initial-planet lookups | Reverted or skipped by subpart | Direct comet lookup was skipped because duplicate planet IDs would change from `HashMap` last-wins to linear-search first-wins. Ordered `initial_planets` zip was skipped to preserve manual states where current and initial planet order differ. The accepted subset was later reverted because the benchmark did not hold. | Early short run looked positive, 93,540 -> 96,680 steps/sec, but cooled testing showed a regression: 85,226 -> 82,703 steps/sec. | Focused rules/action/obs tests and `just rs-prepare` passed. Code was reverted after cooled audit. |
| Rewrite player-result computation with fixed arrays | Kept | Low for generated states. Uses `MAX_PLAYERS = 4` fixed buffers and intentionally panics for invalid 5+ player states. Scores are still computed only after terminal status is known, preserving old nonterminal overflow timing. | Cooled interaction testing was roughly flat to mildly positive within observed variance: the current branch measured 85,181 steps/sec on a 10,000-step run and 84,882 mean steps/sec with 3,532 std over three 5,000-step repeats. | Focused env/RL tests, `just rs-prepare`, `just build-release`, and cooled benchmarks passed. Added a regression test proving nonterminal result computation does not sum scores. |
| Avoid RNG creation on non-comet steps | Skipped | Low for generated states. Spawn-step random streams were preserved by still constructing the real RNG on comet-spawn steps. Manual invalid `state.step == u32::MAX` would overflow slightly earlier because the public `step` checked `state.step + 1` before later validation. | Default short run regressed 92,646 -> 88,106 steps/sec. Longer 10,000-step A/B with cooldown was effectively flat: 85,363 without the change vs 85,844 with it. | Focused env tests and `just rs-prepare` passed. Code was excluded because performance was not reliably improved. |
| Cache action entity slots per vector environment | Kept | Low for valid vector-env states. Action slots are refreshed while writing observations and consumed by the next `step` decode, preserving the intended prior-observation mapping. Manual duplicate planet IDs remain ambiguous because actions ultimately carry only `from_planet_id`; a stale/reordered cached slot could validate one duplicate before the rules engine resolves the first matching ID. | Short cooled run improved 89,831 -> 90,401 mean steps/sec over two 3,000-step repeats. Longer paired worktree run improved 86,678 -> 88,372 mean steps/sec over three 5,000-step repeats with 30-second cooldown. | Focused RL action/vec-env/obs tests, `cargo test --lib`, `just rs-prepare`, `just build-release`, and paired cooled benchmarks passed. Added cached-slot and stale-slot action decode tests. |
| Reuse decoded action buffers | Skipped | Low. The attempted implementation preserved the two-phase decode-before-mutation contract, but failed decodes could leave partial actions in internal scratch buffers until the next decode cleared them. Review found no external behavior leak. | Short cooled run regressed 91,312 -> 90,673 mean steps/sec. Longer paired run was flat: 89,052 -> 89,135 mean steps/sec over three 5,000-step repeats. | Focused RL tests, `cargo test --lib`, and `just rs-prepare` passed. Code was excluded because performance was not reliably improved and the extra per-env scratch state did not justify itself. |
| Optimize fleet observation sorting and scratch reuse | Skipped | Low for the attempted partial-sort subset. The implementation used a total comparator `(ships desc, id asc, original index)` so `select_nth_unstable_by` plus prefix sort preserved the previous full stable sort order, including manual duplicate `(ships, id)` fleets. Reusable scratch was not pursued because this exact subset was already slower. | Default paired benchmark regressed 88,893 -> 87,737 mean steps/sec over three 5,000-step repeats. High-launch fleet-pressure run also regressed 53,745 -> 52,268 mean steps/sec, with warning I/O dominating both runs. | Focused obs tests, `cargo test --lib`, and `just rs-prepare` passed. Code was excluded because performance was worse in both workloads. |
| Stream comet path generation instead of allocating dense temporary points | Skipped | Low. The streamed implementation preserved the same point formula, iteration order, cumulative thresholds, visible path selection, symmetric expansion, and RNG consumption. Python-reference comet path fixtures still passed exactly. | Default short benchmark regressed 90,744 -> 88,039 mean steps/sec. Low-action comet-spawn workload initially looked close, 222,469 -> 223,710 mean steps/sec, but longer paired low-action testing regressed 221,564 -> 219,254 mean steps/sec. | Focused generation tests, `cargo test --lib`, and `just rs-prepare` passed. Code was excluded because performance was worse despite exact behavior parity. |
| Return fixed-size symmetric planet groups instead of allocating a vector | Kept | Very low. The implementation uses the same `fourfold_symmetric_points` order, assigns IDs by the same index order, and call sites still extend `Vec<Planet>` in group order. | Low-action reset/spawn-heavy benchmark improved 219,599 -> 222,458 mean steps/sec over two 3,000-step repeats with 30-second cooldown and `--launch-prob 0.001`. | Focused generation tests, `just rs-prepare`, `just build-release`, and cooled repeated benchmark passed. Added an order-specific symmetric group test. |
| Cache generated-planet or orbit metadata | Skipped | Low for the attempted generation-only `valid_group` metadata cache: it stayed private to generated planets, kept metadata aligned by appending only on accepted groups, and reused the same `distance` and `is_orbiting` formulas. Broader per-step orbit metadata was skipped because it would touch public `State` alignment and risks changing manual states with missing or duplicate `initial_planets` IDs. | Short low-action run was flat: 222,458 -> 222,394 mean steps/sec. Longer low-action paired run was only mildly positive: 222,749 -> 223,342 mean steps/sec. Default random-policy run regressed 91,179 -> 90,409 mean steps/sec. | Focused generation tests, `just rs-prepare`, `just build-release`, and paired cooled benchmarks passed. Code was excluded because performance was not reliably improved. |
| Add conservative collision broad-phase and clean up hot geometry squaring | Kept | Low for finite generated states with nonnegative radii. The broad-phase AABB check is only a conservative prefilter expanded by `radius + 1e-12`; final collision behavior still uses strict `point_to_segment_distance(...) < radius`, preserving exact-boundary non-collision semantics. Manual NaN/negative-radius states should not create newly accepted collisions because the final exact predicate remains decisive. Explicit multiplication replaced `powi(2)` in geometry helpers; any discrepancy should be limited to tiny floating-point lowering differences. | Default random-policy benchmark improved 90,680 -> 103,393 mean steps/sec. High-launch fleet-pressure benchmark improved 53,306 -> 63,858 mean steps/sec, with warning I/O still present. | Focused utils/env tests, replay tests, `just rs-prepare`, `just build-release`, and cooled benchmarks passed. Added AABB tests for exact-boundary, endpoint-collision, and far-point cases. |
| Use direct-indexed combat lists instead of `HashMap<u32, Vec<Fleet>>` | Kept | Low for generated/RL states, with approved manual-state discrepancies. Combat buckets are `Vec<Option<Vec<Fleet>>>` indexed by planet ID, with a `max_planet_id < 100` guard that panics for huge manual IDs. Holes remain `None`; queueing combat into a missing bucket now panics instead of creating a new sweep bucket. Duplicate planet IDs still share one bucket and resolve against the first matching planet. Queued combat for planets removed later in the same step is preserved so collided fleets are still removed before combat resolution skips the missing planet. Combat resolution iteration changes from unspecified `HashMap` order to numeric ID order, but resolution has no cross-planet side effects. | Short default benchmark improved 103,393 -> 104,545 mean steps/sec. Short high-launch fleet-pressure benchmark improved 63,858 -> 65,260 mean steps/sec, but std was high and warning I/O still occurred. Longer paired default benchmark confirmed the win: 100,525 -> 102,516 mean steps/sec over three 5,000-step repeats with 45-second cooldown. | Focused env tests, `just rs-prepare`, `just build-release`, short cooled default/fleet-pressure benchmarks, and longer paired default benchmark passed. Added tests for ID-limit panic, missing buckets, removed planets with queued combat, and duplicate IDs. |
| Sort fleet observations only when truncating, and increase default max entities | Applied as a later follow-up | When all fleets fit in the observation capacity, skipping the sort changes the order of fleet rows from sorted order to state iteration order. That is a possible observation-layout behavior change for consumers that depend on sorted fleet rows even when no truncation occurs. Increasing default `max_entities` from 512 to 1024 reduces overflow pressure and warning I/O in default vector-env use. | No paired benchmark is recorded in this log. | Present in `9c2911e`. Update this row with verification and throughput numbers if this change is benchmarked later. |
| Replace small fleet-ID hash sets with vectors and sweep flags | Kept | Low for generated/RL states under the unique live-fleet-ID invariant. `move_fleets` now uses a `Vec<u32>` for removed IDs, preserving the old ID-membership removal behavior. Moving-planet and comet sweep dedup now uses a `Vec<bool>` indexed by the post-`move_fleets` live fleet slot; `state.fleets` is not inserted, removed, or reordered while those flags are live. Manual duplicate fleet IDs can now both be queued by the same sweep because dedup is per live slot instead of per ID, but final removal remains ID-based. | HashSet baseline at `f90b01f` measured 104,790 mean steps/sec over two 5,000-step repeats. After the vector/sweep-flag change, the default benchmark measured 109,853 mean steps/sec over two 5,000-step repeats. Fleet-pressure benchmark with `--launch-prob 1.0 --max-entities 2048` improved 71,529 -> 83,203 mean steps/sec over two 3,000-step repeats. | Focused env tests passed before the final prep run. Reviewer found no behavior-preservation issue for generated states and recommended documenting duplicate-ID sweep behavior, so a regression test was added. |
| Rebuild live fleets in one pass instead of retaining removed IDs | Skipped | The attempted implementation preserved manual duplicate-ID removal semantics by pruning earlier duplicate survivors when a later same-ID fleet was removed. Generated states were behavior-equivalent, but the duplicate-preserving bookkeeping added repeated `retain` and `contains` scans inside the hot path. | Against the noisy `Vec` removal-list baseline, the usable pre samples were about 103,111 mean steps/sec and the attempted survivor-rebuild measured 103,180 mean steps/sec over three 5,000-step repeats. The result was effectively flat. | Focused env tests passed and reviewer found no semantic bug, but flagged the duplicate-preservation scans as a likely performance trap. The code was reverted and excluded. |
