docs := "docs/"
py_src := "python/"
py_scripts := "scripts/"
py_tests := "tests/"
all_py_code := f"{{py_src}} {{py_scripts}} {{py_tests}}"
all_rs_code := "src/"

[group: 'python']
py-format:
    uvx ruff check {{all_py_code}} --select I --fix
    uvx ruff format {{all_py_code}}
[group: 'python']
py-lint:
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
py-prepare: py-format py-lint py-static py-test

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
rs-prepare: rs-format rs-lint rs-test

[group: 'build']
build:
	uv run maturin develop
[group: 'build']
build-release:
	RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release

[group: 'misc']
docs-lint:
	uvx pymarkdownlnt scan *.md
	uvx pymarkdownlnt scan --recurse {{all_py_code}} {{all_rs_code}} {{docs}}

[group: 'build']
prepare: build rs-format py-format rs-lint py-lint docs-lint py-static rs-test py-test-full
[group: 'build']
prepare-rl: prepare build-release

[group: 'misc']
clean:
    cargo clean
    rm -rf .mypy_cache .pytest_cache .ruff_cache .venv/
    rm tests/fixtures/**/*.{json,jsonl}
