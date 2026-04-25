# OWL: Orbit Wars (reinforcement) Learning

## Setup instructions

1. Install prerequisites:
    * [Rust](https://www.rust-lang.org/tools/install)
    * [uv](https://docs.astral.sh/uv/getting-started/installation/)
    * [just](https://github.com/casey/just)
2. Add nightly toolchain for `rustfmt`:

```sh
rustup update nightly
rustup component add rustfmt --toolchain nightly
```

3. `just prepare` should run without errors

## Orbit Wars replays

Replay parity tests use Kaggle episode downloads as external fixtures. The raw
`replay-*.json` files are intentionally ignored by Git because they can be large.

The current reference episodes are:

- `75373897`
- `75377525`

If these files are missing, download them into the repo root:

```sh
uv run python scripts/download_replays.py 75373897 75377525 --save-dir .
```

The downloader requires Kaggle API credentials configured for the local user.
