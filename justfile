docs := "docs/"
py_src := "python/"
py_scripts := "scripts/"
py_tests := "tests/"
all_py_code := f"{{py_src}} {{py_scripts}} {{py_tests}}"
all_rs_code := "src/"

[group: 'docs']
docs-lint:
	uvx pymarkdownlnt scan *.md
	uvx pymarkdownlnt scan --recurse {{all_py_code}} {{all_rs_code}} {{docs}}
[group: 'docs']
docs-fresh:
	uv run python scripts/check_doc_freshness.py

[group: 'python']
py-format:
    uvx ruff check {{all_py_code}} --select I --fix
    uvx ruff format {{all_py_code}}
[group: 'python']
py-lint:
    uv run python scripts/check_python_311_syntax.py
    uvx ruff check {{all_py_code}}
[group: 'python']
py-static:
    uv run mypy {{py_src}} {{py_scripts}}
[group: 'python']
py-test:
    uv run pytest {{py_tests}} -m "not slow"
[group: 'python']
py-test-full:
    uv run pytest {{py_tests}}
[group: 'python']
py-prepare: py-format py-lint py-static py-test docs-fresh

[group: 'rust']
rs-format:
    cargo fmt
[group: 'rust']
rs-lint:
    cargo clippy --all-targets -- -D warnings
[group: 'rust']
rs-test:
	cargo test
[group: 'rust']
rs-prepare: rs-format rs-lint rs-test docs-fresh

[group: 'build']
build:
	uv run maturin develop
[group: 'build']
build-release:
	RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release
[group: 'build']
kaggle-image: prepare
	docker buildx build \
	  --platform linux/amd64 \
	  -f Dockerfile.kaggle \
	  --output type=docker,name=orbit-wars:kaggle,compression=zstd,compression-level=1 \
	  .
[group: 'build']
kaggle-submission model submission="submission": prepare kaggle-image
	#!/usr/bin/env bash
	set -euo pipefail
	submission="{{submission}}"
	if [[ -z "$submission" || "$submission" == *"/"* || "$submission" == "." || "$submission" == ".." ]]; then
	  echo "Submission name must be a non-empty file name, not a path: $submission" >&2
	  exit 2
	fi
	model_abs="$(cd "$(dirname "{{model}}")" && pwd)/$(basename "{{model}}")"
	if [[ "$submission" == *.tar.gz ]]; then
	  output="artifacts/${submission}"
	else
	  output="artifacts/${submission}.tar.gz"
	fi
	output_abs="$(mkdir -p "$(dirname "$output")" && cd "$(dirname "$output")" && pwd)/$(basename "$output")"
	docker run --rm \
	  -v "$(dirname "$model_abs"):/model:ro" \
	  -v "$(dirname "$output_abs"):/artifacts" \
	  orbit-wars:kaggle "/model/$(basename "$model_abs")" "/artifacts/$(basename "$output_abs")"

_prepare_base: build rs-format py-format rs-lint py-lint docs-lint py-static rs-test py-test-full
[group: 'ci']
prepare: _prepare_base docs-fresh
[group: 'ci']
prepare-rl: prepare build-release
[group: 'ci']
prepare-container: _prepare_base build-release

[group: 'misc']
clean:
    cargo clean
    rm -rf .mypy_cache .pytest_cache .ruff_cache .venv/
    rm tests/fixtures/**/*.{json,jsonl}
