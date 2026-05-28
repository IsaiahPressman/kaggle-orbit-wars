from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    roots = (repo_root / "python", repo_root / "scripts")
    failures: list[str] = []

    python_files = sorted(path for root in roots for path in root.rglob("*.py"))
    if not python_files:
        print(f"No python files found in {', '.join(str(root) for root in roots)}")
        return 1

    for path in python_files:
        relative_path = path.relative_to(repo_root)
        try:
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(relative_path),
                feature_version=(3, 11),
            )
        except SyntaxError as exc:
            failures.append(f"{relative_path}:{exc.lineno}:{exc.offset}: {exc.msg}")

    if failures:
        print("Python 3.11-incompatible syntax found:", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
