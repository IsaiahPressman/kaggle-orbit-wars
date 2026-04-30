#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    ActionPureConfig,
    ObsV1Config,
    VectorizedEnv,
)
from owl.rs import assert_release_build
from tqdm import trange

Target = Literal["both", "rust", "kaggle"]
KAGGLE_N_ENVS = 8


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    n_envs: int
    env_steps: int
    elapsed_seconds: float
    launches: int

    @property
    def steps_per_second(self) -> float:
        return self.env_steps / self.elapsed_seconds

    @property
    def launches_per_env_step(self) -> float:
        return self.launches / self.env_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the Rust vectorized Orbit Wars env against the Kaggle "
            "Python environment with random valid launches."
        )
    )
    parser.add_argument(
        "--target",
        choices=["both", "rust", "kaggle"],
        default="both",
        help="Which implementation to benchmark.",
    )
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=128,
        help="Number of parallel Rust sub-envs to run. Kaggle always uses 8.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=200,
        help="Timed environment ticks per sub-env.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Untimed environment ticks per sub-env before measuring.",
    )
    parser.add_argument(
        "--players",
        type=int,
        choices=[2, 4],
        default=4,
        help="Orbit Wars player count for both implementations.",
    )
    parser.add_argument(
        "--launch-prob",
        type=float,
        default=0.5,
        help="Probability of launching from each currently valid source.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for benchmark action sampling and Kaggle env construction.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="Rust observation max_entities. Defaults to ObsV1Config().max_entities.",
    )
    parser.add_argument(
        "--max-per-planet-launches",
        type=int,
        default=1,
        help="Rust action slots per source. The benchmark only fills the first slot.",
    )
    parser.add_argument(
        "--verbose-kaggle-import",
        action="store_true",
        help="Do not suppress Kaggle registry import noise.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of independent timed benchmark repeats to run.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait before each repeat, useful on passively cooled machines.",
    )
    return parser.parse_args()


def benchmark_rust(args: argparse.Namespace) -> BenchmarkResult:
    assert_release_build()
    obs_spec = ObsV1Config()
    if args.max_entities is not None:
        obs_spec = ObsV1Config(max_entities=args.max_entities)

    env = VectorizedEnv(
        n_envs=args.n_envs,
        obs_spec=obs_spec,
        action_spec=ActionPureConfig(
            max_per_planet_launches=args.max_per_planet_launches
        ),
        two_player_weight=1.0 if args.players == 2 else 0.0,
        pin_memory=False,
    )
    env.reset()

    rng = np.random.default_rng(args.seed)
    action_shape = (
        args.n_envs,
        4,
        ACTION_ENTITY_SLOTS,
        args.max_per_planet_launches,
    )
    launch = np.zeros(action_shape, dtype=np.bool_)
    angle = np.zeros(action_shape, dtype=np.float32)
    ships = np.zeros(action_shape, dtype=np.int64)

    launches = 0
    for _ in trange(
        args.warmup_steps,
        desc="rust warmup",
        disable=args.no_progress or args.warmup_steps == 0,
        leave=False,
        unit="tick",
    ):
        launches += sample_rust_actions(
            env, rng, args.launch_prob, launch, angle, ships
        )
        env.step(launch, angle, ships)

    timed_launches = 0
    started_at = time.perf_counter()
    for _ in trange(
        args.steps,
        desc="rust timed",
        disable=args.no_progress,
        unit="tick",
    ):
        timed_launches += sample_rust_actions(
            env, rng, args.launch_prob, launch, angle, ships
        )
        env.step(launch, angle, ships)
    elapsed = time.perf_counter() - started_at

    if launches + timed_launches == 0:
        raise RuntimeError("Rust benchmark sampled only no-op actions")

    return BenchmarkResult(
        name="rust-vectorized",
        n_envs=args.n_envs,
        env_steps=args.n_envs * args.steps,
        elapsed_seconds=elapsed,
        launches=timed_launches,
    )


def sample_rust_actions(
    env: VectorizedEnv,
    rng: np.random.Generator,
    launch_prob: float,
    launch: np.ndarray,
    angle: np.ndarray,
    ships: np.ndarray,
) -> int:
    can_act = env.observations.can_act.numpy()
    max_launch = env.observations.max_launch.numpy()
    selected = can_act & (rng.random(can_act.shape) < launch_prob)

    launch.fill(False)
    launch[..., 0] = selected
    angle[..., 0] = rng.uniform(0.0, math.tau, size=can_act.shape).astype(np.float32)

    high = np.maximum(max_launch + 1, 2)
    sampled_ships = rng.integers(1, high, size=can_act.shape, dtype=np.int64)
    sampled_ships[~selected] = 0
    ships.fill(0)
    ships[..., 0] = sampled_ships

    return int(selected.sum())


def benchmark_kaggle(args: argparse.Namespace) -> BenchmarkResult:
    make = load_kaggle_make(verbose=args.verbose_kaggle_import)
    rng = np.random.default_rng(args.seed)
    envs = [
        make_kaggle_env(make, args.players, args.seed + env_index)
        for env_index in trange(
            KAGGLE_N_ENVS,
            desc="kaggle init",
            disable=args.no_progress,
            leave=False,
            unit="env",
        )
    ]

    launches = 0
    for _ in trange(
        args.warmup_steps,
        desc="kaggle warmup",
        disable=args.no_progress or args.warmup_steps == 0,
        leave=False,
        unit="tick",
    ):
        launches += step_kaggle_envs(envs, rng, args.players, args.launch_prob)

    timed_launches = 0
    started_at = time.perf_counter()
    for _ in trange(
        args.steps,
        desc="kaggle timed",
        disable=args.no_progress,
        unit="tick",
    ):
        timed_launches += step_kaggle_envs(envs, rng, args.players, args.launch_prob)
    elapsed = time.perf_counter() - started_at

    if launches + timed_launches == 0:
        raise RuntimeError("Kaggle benchmark sampled only no-op actions")

    return BenchmarkResult(
        name="kaggle-python",
        n_envs=KAGGLE_N_ENVS,
        env_steps=KAGGLE_N_ENVS * args.steps,
        elapsed_seconds=elapsed,
        launches=timed_launches,
    )


def load_kaggle_make(*, verbose: bool) -> Callable[..., Any]:
    if verbose:
        module = importlib.import_module("kaggle_environments")
    else:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            module = importlib.import_module("kaggle_environments")
    return cast(Callable[..., Any], module.make)


def make_kaggle_env(make: Callable[..., Any], players: int, seed: int) -> Any:
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(players)
    return env


def step_kaggle_envs(
    envs: list[Any],
    rng: np.random.Generator,
    players: int,
    launch_prob: float,
) -> int:
    launches = 0
    for env in envs:
        if env.done:
            env.reset(players)
        actions, action_count = sample_kaggle_actions(env, rng, launch_prob)
        launches += action_count
        env.step(actions)
    return launches


def sample_kaggle_actions(
    env: Any,
    rng: np.random.Generator,
    launch_prob: float,
) -> tuple[list[list[list[int | float]]], int]:
    actions: list[list[list[int | float]]] = []
    launches = 0

    for agent_state in env.state:
        if agent_state.status != "ACTIVE":
            actions.append([])
            continue

        obs = agent_state.observation
        player = int(obs.player)
        moves: list[list[int | float]] = []
        for planet in obs.planets:
            if int(planet[1]) != player or int(planet[5]) <= 0:
                continue
            if rng.random() >= launch_prob:
                continue

            ships = rng.integers(1, int(planet[5]) + 1).item()
            moves.append([int(planet[0]), float(rng.uniform(0.0, math.tau)), ships])

        launches += len(moves)
        actions.append(moves)

    return actions, launches


def validate_args(args: argparse.Namespace) -> None:
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.cooldown_seconds < 0.0:
        raise ValueError("--cooldown-seconds must be non-negative")
    if not 0.0 < args.launch_prob <= 1.0:
        raise ValueError("--launch-prob must be in (0, 1]")
    if args.max_per_planet_launches <= 0:
        raise ValueError("--max-per-planet-launches must be positive")


def print_results(results: list[BenchmarkResult]) -> None:
    print(
        f"{'implementation':<18} {'n_envs':>8} {'env_steps':>10} "
        f"{'seconds':>10} {'steps/sec':>12} {'launches/step':>14}"
    )
    for result in results:
        print(
            f"{result.name:<18} {result.n_envs:>8} {result.env_steps:>10} "
            f"{result.elapsed_seconds:>10.3f} {result.steps_per_second:>12.0f} "
            f"{result.launches_per_env_step:>14.3f}"
        )

    if len(results) == 2:
        rust = next(
            (result for result in results if result.name == "rust-vectorized"), None
        )
        kaggle = next(
            (result for result in results if result.name == "kaggle-python"), None
        )
        if rust is not None and kaggle is not None:
            speedup = rust.steps_per_second / kaggle.steps_per_second
            print(f"\nrust/kaggle env-step throughput speedup: {speedup:.2f}x")

    repeated_names = sorted({result.name for result in results if len(results) > 1})
    summaries = []
    for name in repeated_names:
        name_results = [result for result in results if result.name == name]
        if len(name_results) <= 1:
            continue
        steps_per_second = [result.steps_per_second for result in name_results]
        summaries.append((name, steps_per_second))

    if summaries:
        print(
            f"\n{'implementation':<18} {'runs':>6} {'mean sps':>12} "
            f"{'std sps':>12} {'min sps':>12} {'max sps':>12}"
        )
        for name, steps_per_second in summaries:
            std = statistics.stdev(steps_per_second)
            print(
                f"{name:<18} {len(steps_per_second):>6} "
                f"{statistics.mean(steps_per_second):>12.0f} "
                f"{std:>12.0f} {min(steps_per_second):>12.0f} "
                f"{max(steps_per_second):>12.0f}"
            )


def benchmark_target(target: Target, args: argparse.Namespace) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    if target in ("both", "rust"):
        results.append(benchmark_rust(args))
    if target in ("both", "kaggle"):
        results.append(benchmark_kaggle(args))
    return results


def main() -> None:
    args = parse_args()
    validate_args(args)
    target = cast(Target, args.target)

    results: list[BenchmarkResult] = []
    for repeat_index in range(args.repeats):
        if args.cooldown_seconds > 0.0:
            print(
                f"cooling down for {args.cooldown_seconds:g}s before repeat "
                f"{repeat_index + 1}/{args.repeats}",
                file=sys.stderr,
            )
            time.sleep(args.cooldown_seconds)

        repeat_args = argparse.Namespace(**vars(args))
        repeat_args.seed = args.seed + repeat_index
        results.extend(benchmark_target(target, repeat_args))

    print_results(results)


if __name__ == "__main__":
    main()
