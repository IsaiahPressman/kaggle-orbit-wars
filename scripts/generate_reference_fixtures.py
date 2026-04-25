#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

DEFAULT_OUTFILE = Path("tests/fixtures/generation/reference_generation.json")


class RecordingRandom:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.calls: list[dict[str, int | float | str]] = []

    def randint(self, low: int, high: int) -> int:
        value = self._rng.randint(low, high)
        self.calls.append({"kind": "randint", "low": low, "high": high, "value": value})
        return value

    def uniform(self, low: float, high: float) -> float:
        value = self._rng.uniform(low, high)
        self.calls.append({"kind": "uniform", "low": low, "high": high, "value": value})
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Python-reference fixtures for Rust generation parity."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("orbit_wars.py"),
        help="Path to the reference orbit_wars.py implementation.",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=DEFAULT_OUTFILE,
        help="Fixture JSON path to write.",
    )
    parser.add_argument("--planet-seed", type=int, default=20260425)
    parser.add_argument("--comet-seed", type=int, default=20260426)
    return parser.parse_args()


def load_reference(reference_path: Path) -> dict[str, Any]:
    source = reference_path.read_text(encoding="utf-8")
    cutoff = source.index("\ndef interpreter(")
    namespace: dict[str, Any] = {
        "__file__": str(reference_path),
        "__name__": "orbit_wars_reference_fixture",
    }
    exec(source[:cutoff], namespace)
    return namespace


def run_with_recording_random(
    namespace: dict[str, Any],
    seed: int,
    function_name: str,
    *args: Any,
) -> tuple[Any, list[dict[str, int | float | str]]]:
    recorder = RecordingRandom(seed)
    original_random = namespace["random"]
    namespace["random"] = recorder
    try:
        return namespace[function_name](*args), recorder.calls
    finally:
        namespace["random"] = original_random


def main() -> None:
    args = parse_args()
    namespace = load_reference(args.reference)

    planets, planet_calls = run_with_recording_random(
        namespace,
        args.planet_seed,
        "generate_planets",
    )
    comet_paths, comet_calls = run_with_recording_random(
        namespace,
        args.comet_seed,
        "generate_comet_paths",
        planets,
        0.04,
        50,
        [],
        4.0,
    )
    if comet_paths is None:
        raise RuntimeError("chosen comet seed did not produce a valid comet path")

    fixture = {
        "reference": str(args.reference),
        "planet_generation": {
            "seed": args.planet_seed,
            "random_calls": planet_calls,
            "planets": planets,
        },
        "comet_path_generation": {
            "seed": args.comet_seed,
            "inputs": {
                "angular_velocity": 0.04,
                "spawn_step": 50,
                "comet_planet_ids": [],
                "comet_speed": 4.0,
            },
            "initial_planets": planets,
            "random_calls": comet_calls,
            "paths": comet_paths,
        },
    }

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Reference generation fixture written to: {args.outfile}")


if __name__ == "__main__":
    main()
