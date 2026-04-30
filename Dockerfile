# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG TARGETPLATFORM
ARG SKIP_FLASH_ATTN=0
ARG JUST_VERSION=1.50.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PYO3_PYTHON=/opt/venv/bin/python \
    CARGO_HOME=/opt/cargo \
    REQUIRE_PARITY_FIXTURES=0 \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/venv/bin:/opt/cargo/bin:$PATH

RUN test "${TARGETPLATFORM}" = "linux/amd64" || \
    (echo "This image targets linux/amd64 CUDA Slurm nodes. Build with: docker build --platform linux/amd64 -t orbit-wars:dev ." >&2 && exit 1)

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/orbit-wars

COPY rust-toolchain.toml ./

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain none && \
    RUST_TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)" && \
    test -n "${RUST_TOOLCHAIN}" && \
    rustup toolchain install "${RUST_TOOLCHAIN}" --profile minimal \
        --component clippy \
        --component rustfmt && \
    rustup default "${RUST_TOOLCHAIN}" && \
    cargo install just --version "${JUST_VERSION}" --locked

COPY pyproject.toml uv.lock Cargo.toml Cargo.lock rustfmt.toml ./
COPY .cargo/ .cargo/
COPY src/lib.rs src/lib.rs

RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/opt/cargo/git \
    if [ "${SKIP_FLASH_ATTN}" = "1" ]; then \
        uv sync --no-cache --frozen --no-install-project --group dev; \
    else \
        uv sync --no-cache --frozen --no-install-project --group dev --extra flash-attn; \
    fi && \
    cargo fetch --locked

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/opt/cargo/git \
    --mount=type=cache,target=/workspace/orbit-wars/target \
    just prepare-container

CMD ["bash"]
