from os import environ
from pathlib import Path

OWL_ROOT = Path(__file__).resolve().parent

_native_module_dir = environ.get("OWL_NATIVE_MODULE_DIR")
if _native_module_dir:
    # Keep the image-built Rust extension importable when python/ is mounted.
    __path__.insert(0, _native_module_dir)

__all__ = [
    "OWL_ROOT",
]
