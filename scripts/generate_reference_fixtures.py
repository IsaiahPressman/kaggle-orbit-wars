#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

DEFAULT_OUTFILE = Path("tests/fixtures/generation/reference_generation.json")
ORBIT_WARS_MODULE = "kaggle_environments.envs.orbit_wars.orbit_wars"
COMET_CASES = [
    {"name": "spawn_50", "seed": 93, "spawn_step": 50, "comet_planet_ids": []},
    {"name": "spawn_150", "seed": 193, "spawn_step": 150, "comet_planet_ids": []},
    {"name": "spawn_250", "seed": 293, "spawn_step": 250, "comet_planet_ids": []},
    {"name": "spawn_350", "seed": 393, "spawn_step": 350, "comet_planet_ids": []},
    {"name": "spawn_450", "seed": 493, "spawn_step": 450, "comet_planet_ids": []},
    {
        "name": "existing_comet_ids",
        "seed": 80,
        "spawn_step": 150,
        "comet_planet_ids": [0, 1, 2, 3],
    },
    {
        "name": "failed_attempts_before_success",
        "seed": 40,
        "spawn_step": 50,
        "comet_planet_ids": [],
    },
]


class OrbitWarsModule(Protocol):
    __file__: str
    random: Any

    def generate_planets(self) -> list[list[int | float]]: ...

    def generate_comet_paths(
        self,
        initial_planets: list[list[int | float]],
        angular_velocity: float,
        _spawn_step: int,
        comet_planet_ids: list[int],
        _comet_speed: float,
    ) -> list[list[list[float]]] | None: ...

    def interpreter(self, state: list[Any], env: Any) -> list[Any]: ...


class RecordingRandom:
    def __init__(self, seed: int | str) -> None:
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


class RecordingRandomFactory:
    def __init__(self) -> None:
        self.recorders: list[RecordingRandom] = []

    def Random(self, seed: int | str) -> RecordingRandom:
        recorder = RecordingRandom(seed)
        self.recorders.append(recorder)
        return recorder


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
    function: Callable[..., Any],
    *args: Any,
) -> tuple[Any, list[dict[str, int | float | str]]]:
    recorder = RecordingRandom(seed)
    original_random = module.random
    module.random = recorder
    try:
        return function(*args), recorder.calls
    finally:
        module.random = original_random


def reset_case(module: OrbitWarsModule, seed: int, players: int) -> dict[str, Any]:
    recorder_factory = RecordingRandomFactory()
    original_random = module.random
    module.random = recorder_factory
    try:
        state = [
            SimpleNamespace(
                observation=SimpleNamespace(),
                action=[],
                status="ACTIVE",
                reward=0,
            )
            for _ in range(players)
        ]
        env = SimpleNamespace(
            configuration=SimpleNamespace(
                episodeSteps=500,
                shipSpeed=6.0,
                cometSpeed=4.0,
                seed=seed,
            ),
            done=False,
        )
        module.interpreter(state, env)
        obs = state[0].observation
    finally:
        module.random = original_random

    return {
        "seed": seed,
        "players": players,
        "random_calls": recorder_factory.recorders[0].calls,
        "state": {
            "angular_velocity": obs.angular_velocity,
            "planets": obs.planets,
            "initial_planets": obs.initial_planets,
            "fleets": obs.fleets,
            "next_fleet_id": obs.next_fleet_id,
            "comets": obs.comets,
            "comet_planet_ids": obs.comet_planet_ids,
            "step": 1,
        },
    }


def comet_path_case(
    module: OrbitWarsModule,
    planets: list[list[int | float]],
    case: dict[str, Any],
) -> dict[str, Any]:
    comet_paths, comet_calls = run_with_recording_random(
        module,
        case["seed"],
        module.generate_comet_paths,
        planets,
        0.04,
        case["spawn_step"],
        case["comet_planet_ids"],
        4.0,
    )
    if comet_paths is None:
        raise RuntimeError(f"comet case {case['name']} did not produce a valid path")

    return {
        "name": case["name"],
        "seed": case["seed"],
        "inputs": {
            "angular_velocity": 0.04,
            "spawn_step": case["spawn_step"],
            "comet_planet_ids": case["comet_planet_ids"],
            "comet_speed": 4.0,
        },
        "initial_planets": planets,
        "random_calls": comet_calls,
        "paths": comet_paths,
    }


def comet_ship_case(seed: int) -> dict[str, Any]:
    recorder = RecordingRandom(seed)
    values = [recorder.randint(1, 99) for _ in range(4)]
    return {
        "seed": seed,
        "random_calls": recorder.calls,
        "ships": min(values),
    }


def no_op_terminal_case(module: OrbitWarsModule) -> dict[str, Any]:
    planets = [
        [0, 0, 20.0, 20.0, 2.0, 10, 0],
        [1, 1, 80.0, 80.0, 2.0, 10, 0],
    ]
    observation = SimpleNamespace(
        angular_velocity=0.0,
        comet_planet_ids=[],
        comets=[],
        fleets=[],
        initial_planets=[planet.copy() for planet in planets],
        next_fleet_id=0,
        planets=[planet.copy() for planet in planets],
        step=2,
    )
    state = [
        SimpleNamespace(
            observation=observation if player == 0 else SimpleNamespace(),
            action=[],
            status="ACTIVE",
            reward=0,
        )
        for player in range(2)
    ]
    env = SimpleNamespace(
        configuration=SimpleNamespace(
            episodeSteps=4,
            shipSpeed=6.0,
            cometSpeed=4.0,
        ),
        done=False,
    )

    module.interpreter(state, env)

    return {
        "players": 2,
        "configuration": {
            "episodeSteps": 4,
            "shipSpeed": 6.0,
            "cometSpeed": 4.0,
        },
        "before": {
            "angular_velocity": 0.0,
            "planets": planets,
            "initial_planets": [planet.copy() for planet in planets],
            "fleets": [],
            "next_fleet_id": 0,
            "comets": [],
            "comet_planet_ids": [],
            "step": 2,
        },
        "rewards": [agent.reward for agent in state],
        "statuses": [agent.status for agent in state],
    }


def main() -> None:
    args = parse_args()
    module = load_reference()

    planets, planet_calls = run_with_recording_random(
        module,
        args.planet_seed,
        module.generate_planets,
    )
    comet_paths, comet_calls = run_with_recording_random(
        module,
        args.comet_seed,
        module.generate_comet_paths,
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
        "reset_cases": [
            reset_case(module, seed=44, players=2),
            reset_case(module, seed=45, players=4),
        ],
        "comet_path_cases": [
            comet_path_case(module, planets, case) for case in COMET_CASES
        ],
        "comet_ship_cases": [
            comet_ship_case(seed=46),
        ],
        "terminal_cases": {
            "no_op_tie": no_op_terminal_case(module),
        },
    }

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile.write_text(
        json.dumps(fixture, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Reference generation fixture written to: {args.outfile}")


if __name__ == "__main__":
    main()
