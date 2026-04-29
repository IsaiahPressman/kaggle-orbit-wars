from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_CURRENT_MARKER = ".updated-docs"

DOC_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "rules-engine",
        ("src/rules_engine/",),
        ("docs/rules-engine.md", "docs/rules-parity-coverage.md"),
    ),
    (
        "rl-api",
        ("python/owl/rl.py", "src/rl/"),
        ("docs/rl-api-specs.md",),
    ),
    (
        "model-architecture",
        ("python/owl/model/",),
        ("docs/model-architecture.md",),
    ),
    (
        "training",
        ("python/owl/train/", "scripts/run_ppo.py", "configs/train/"),
        ("README.md",),
    ),
)


def main() -> int:
    changed_paths = _changed_paths()
    if not changed_paths:
        return 0

    failures: list[str] = []
    changed_existing_set = {
        path for path in changed_paths if (REPO_ROOT / path).exists()
    }
    docs_current_marker_staged = DOCS_CURRENT_MARKER in _staged_paths()
    for rule_name, code_prefixes, required_docs in DOC_RULES:
        touched_code = [
            path for path in changed_paths if _matches_any(path, code_prefixes)
        ]
        if not touched_code:
            continue

        touched_docs = [doc for doc in required_docs if doc in changed_existing_set]
        if touched_docs:
            continue
        if docs_current_marker_staged:
            continue

        code_list = ", ".join(touched_code)
        docs_list = ", ".join(required_docs)
        failures.append(
            f"{rule_name}: changed {code_list}; update {docs_list}, "
            f"or update and stage {DOCS_CURRENT_MARKER} as a docs-current marker"
        )

    if failures:
        print("Doc freshness check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


def _changed_paths() -> list[str]:
    tracked = _git_lines("diff", "--name-only", "HEAD")
    untracked = _git_lines("ls-files", "--others", "--exclude-standard")
    return sorted({*tracked, *untracked})


def _staged_paths() -> list[str]:
    return _git_lines("diff", "--name-only", "--cached", "HEAD")


def _git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _matches_any(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in prefixes)


if __name__ == "__main__":
    raise SystemExit(main())
