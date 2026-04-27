# AGENTS.md

## Repository Map

- Start with `README.md` for setup, fixture regeneration, and the current
  reference episode IDs.
- Use `docs/rules-engine-plan.md` for rules-engine architecture, current status,
  and known rule risks before changing `src/rules_engine/`.
- Use `docs/rules-parity-coverage.md` as the parity coverage source of truth.
  Update it whenever rules behavior, fixtures, or coverage changes.
- Use `docs/rl-api-specs.md` before changing `python/owl/rl.py`, `src/rl/`, or
  public RL tensor shapes.
- Use `docs/pr-checklist.md` before creating, recommending, or merging a PR.

## Development Workflow

- Run `just py-prepare` / `just rs-prepare` after any `python` / `rust` code edits, respectively. This handles formatting, linting, static type-checking, and tests.
- Add dependencies with `uv add` / `cargo add`; don't edit `.toml` files directly when adding dependencies.
- Keep `Cargo.lock` and `uv.lock` tracked. Update lockfiles with package-manager
  commands, not manual edits.
- Avoid jumping through hoops for backwards compatibility - don't be afraid of
  refactoring and breaking old APIs in order to improve them.
- Before creating or recommending a PR, complete `docs/pr-checklist.md` and summarize any residual risks.
- For Orbit Wars rules changes, keep `docs/rules-engine-plan.md` and
  `docs/rules-parity-coverage.md` current with implementation state, test
  surface, and known gaps.

## Error Handling

- Fail fast with explicit, informative errors instead of silent fallbacks.
- When user input is invalid, raise clear exceptions.
- For persisted data schemas owned by this repo, prefer strict key access and explicit validation over backward-compatibility fallbacks.
