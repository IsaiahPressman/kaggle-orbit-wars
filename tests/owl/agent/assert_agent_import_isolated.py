import sys

# This script is executed in a subprocess so the test can inspect a fresh
# interpreter's import graph. In-process checks would already have test imports
# in sys.modules and could hide agent-only import regressions.
import owl.agent  # noqa: F401

loaded_train_modules = sorted(
    name for name in sys.modules if name == "owl.train" or name.startswith("owl.train.")
)

if loaded_train_modules:
    print("\n".join(loaded_train_modules))
    raise SystemExit(1)
