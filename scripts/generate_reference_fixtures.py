#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import random
from pathlib import Path
from typing import Any, Protocol, cast

DEFAULT_OUTFILE = Path("tests/fixtures/generation/reference_generation.json")
ORBIT_WARS_MODULE = "kaggle_environments.envs.orbit_wars.orbit_wars"


class OrbitWarsModule(Protocol):
    __file__: str
    random: Any

    def generate_planets(self) -> list[list[int | float]]: ...

    def generate_comet_paths(
        self,
        initial_planets: list[list[int | float]],
        angular_velocity: float,
        spawn_step: int,
        comet_planet_ids: list[int],
        comet_speed: float,
    ) -> list[list[list[float]]] | None: ...


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
        "--outfile",
        type=Path,
        default=DEFAULT_OUTFILE,
        help="Fixture JSON path to write.",
    )
    parser.add_argument("--planet-seed", type=int, default=42)
    parser.add_argument("--comet-seed", type=int, default=43)
    return parser.parse_args()


def load_reference() -> OrbitWarsModule:
    return cast(OrbitWarsModule, importlib.import_module(ORBIT_WARS_MODULE))


def run_with_recording_random(
    module: OrbitWarsModule,
    seed: int,
    function_name: str,
    *args: Any,
) -> tuple[Any, list[dict[str, int | float | str]]]:
    recorder = RecordingRandom(seed)
    original_random = module.random
    module.random = recorder
    try:
        return getattr(module, function_name)(*args), recorder.calls
    finally:
        module.random = original_random


def main() -> None:
    args = parse_args()
    module = load_reference()

    planets, planet_calls = run_with_recording_random(
        module,
        args.planet_seed,
        "generate_planets",
    )
    comet_paths, comet_calls = run_with_recording_random(
        module,
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
        "reference": ORBIT_WARS_MODULE,
        "reference_file": module.__file__,
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
