from pathlib import Path


def find_checkpoint_path(root: Path) -> Path:
    checkpoint_paths = sorted(root.glob("*.pt"))
    if len(checkpoint_paths) != 1:
        raise ValueError(
            f"expected exactly one .pt checkpoint adjacent to main.py, "
            f"found {len(checkpoint_paths)} in {root}"
        )
    return checkpoint_paths[0]
