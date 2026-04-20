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
