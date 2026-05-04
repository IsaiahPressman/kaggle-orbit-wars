# AGENTS.md

## Repository Map

- Start with `README.md` for setup, fixture regeneration, and the current
  reference episode IDs.
- Use `docs/rules-engine.md` for rules-engine architecture, current status,
  and known rule risks before changing `src/rules_engine/`.
- Use `docs/rules-parity-coverage.md` as the parity coverage source of truth.
  Update it whenever rules behavior, fixtures, or coverage changes.
- Reference `docs/rl-api-specs.md` before changing `python/owl/rl.py`, `src/rl/`, or
  public RL tensor shapes.
- Reference `docs/model-architecture.md` before changing `python/owl/model/` or model
  config, tensor ordering, actor, critic, or initialization behavior.
- Reference `README.md` and the training config tests before changing
  `python/owl/train/`, `scripts/run_ppo.py`, or `configs/train/`.
- Reference `docs/pr-checklist.md` before creating, recommending, or merging a PR.

## Development Workflow

- Run `just py-prepare` / `just rs-prepare` after any `python` / `rust` code edits, respectively. This handles formatting, linting, static type-checking, and tests.
- Run `just prepare` before creating or recommending a commit or PR.
- Add dependencies with `uv add` / `cargo add`; don't edit `.toml` or `.lock` files directly when adding dependencies.
- Keep `Cargo.lock` and `uv.lock` tracked. Update lockfiles with package-manager
  commands, not manual edits.
- Avoid jumping through hoops for backwards compatibility - don't be afraid of
  refactoring and breaking old APIs in order to improve them.
- Before creating or recommending a PR, complete `docs/pr-checklist.md` and summarize any residual risks.
- Merge PRs with a regular merge commit by default. Do not squash-merge unless
  the user explicitly requests it.
- For Orbit Wars rules changes, keep `docs/rules-engine.md` and
  `docs/rules-parity-coverage.md` current with implementation state, test
  surface, and known gaps.
- `just docs-fresh` requires mapped code changes to update their mapped docs.
  If docs are already current for a small change, rerun the check with
  `DOCS_CURRENT=1` to indicate that the mapped docs were reviewed and are
  still current.

## Error Handling

- Fail fast with explicit, informative errors instead of silent fallbacks.
- When user input is invalid, raise clear exceptions.
- For persisted data schemas owned by this repo, prefer strict key access and explicit validation over backward-compatibility fallbacks.
