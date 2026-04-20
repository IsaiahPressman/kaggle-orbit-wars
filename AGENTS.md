# AGENTS.md

## Development Workflow

- Run `just py-prepare` / `just rs-prepare` after any `python` / `rust` code edits, respectively. This handles formatting, linting, static type-checking, and tests.
- Add dependencies with `uv add` / `cargo add`; don't edit `.toml` files directly when adding dependencies.
- Avoid jumping through hoops for backwards compatibility - don't be afraid of refactoring and breaking old APIs in order to improve them.

## Error Handling

- Fail fast with explicit, informative errors instead of silent fallbacks.
- When user input is invalid, raise clear exceptions.
- For persisted data schemas owned by this repo, prefer strict key access and explicit validation over backward-compatibility fallbacks.
