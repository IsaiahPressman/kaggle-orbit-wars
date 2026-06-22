#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from owl.checkpoint_quantization import dequantize_model_state_dict
from owl.int8_emulation import apply_int8_emulation
from owl.model import (
    BaseModelAPI,
    ModelHiddenState,
    ModelOutput,
    apply_lora_to_stateless_transformer,
    create_model,
    fold_lora_adapters,
    load_model_state_dict_allowing_lora,
    lora_config_for_model,
)
from owl.replay import ReplayRecorder
from owl.rl import (
    ActionBundle,
    ActionMask,
    DecodedLaunchActions,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    DiscreteTargetBinActionMask,
    DiscreteTargetBinActions,
    ObsBatch,
    ObsConfig,
    PureActionMask,
    PureActions,
    VectorizedEnv,
)
from owl.rs import assert_release_build
from owl.train import FullConfig, configure_torch
from owl.train.utils import autocast_context
from tqdm import tqdm

MODEL_A = 0
MODEL_B = 1
PLAYER_COUNTS = (2, 4)


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    config: FullConfig
    model: BaseModelAPI
    env_steps: int | None
    int8_emulation: bool = False


@dataclass(frozen=True)
class CheckpointDeterminism:
    a: bool
    b: bool


@dataclass(frozen=True)
class CheckpointInt8Emulation:
    a: bool
    b: bool


@dataclass
class MatchupStats:
    model_games: list[int]
    wins: list[float]

    @classmethod
    def empty(cls) -> MatchupStats:
        return cls(model_games=[0, 0], wins=[0.0, 0.0])

    def add_game_result(self, winner: int | None) -> None:
        for model_index in range(2):
            self.model_games[model_index] += 1
        if winner is None:
            for model_index in range(2):
                self.wins[model_index] += 0.5
        else:
            self.wins[winner] += 1.0

    def merge(self, other: MatchupStats) -> None:
        for model_index in range(2):
            self.model_games[model_index] += other.model_games[model_index]
            self.wins[model_index] += other.wins[model_index]


@dataclass(frozen=True)
class BenchmarkResult:
    player_count: int
    games: int
    steps: int
    elapsed_seconds: float
    stats: MatchupStats

    @property
    def steps_per_second(self) -> float:
        if self.elapsed_seconds == 0.0:
            return 0.0
        return self.steps / self.elapsed_seconds


def main() -> None:
    args = _parse_args()
    determinism = _deterministic_flags(args.deterministic)
    int8_emulation = _int8_emulation_flags(args.int8_emulation)

    assert_release_build()
    configure_torch()
    device = torch.device(args.device)
    checkpoint_a = _load_checkpoint(
        args.checkpoint_a,
        device=device,
        int8_emulation=int8_emulation.a,
    )
    checkpoint_b = _load_checkpoint(
        args.checkpoint_b,
        device=device,
        int8_emulation=int8_emulation.b,
    )

    game_counts = _player_count_counts(args.n_games, args.two_player_weight)
    replay_counts = _player_count_counts(args.save_replay_games, args.two_player_weight)
    results = [
        run_benchmark(
            checkpoint_a=checkpoint_a,
            checkpoint_b=checkpoint_b,
            player_count=player_count,
            n_games=game_counts[player_count],
            n_envs=args.n_envs,
            device=device,
            determinism=determinism,
            no_progress=args.no_progress,
            replay_games=replay_counts[player_count],
            replay_output_path=_benchmark_replay_path(args, player_count),
            replay_rng=random.Random(),
        )
        for player_count in PLAYER_COUNTS
    ]

    _print_results(checkpoint_a, checkpoint_b, results)


def run_benchmark(
    *,
    checkpoint_a: LoadedCheckpoint,
    checkpoint_b: LoadedCheckpoint,
    player_count: int,
    n_games: int,
    n_envs: int,
    device: torch.device,
    determinism: CheckpointDeterminism,
    no_progress: bool,
    replay_games: int,
    replay_output_path: Path | None,
    replay_rng: random.Random,
) -> BenchmarkResult:
    if n_games == 0:
        return BenchmarkResult(
            player_count=player_count,
            games=0,
            steps=0,
            elapsed_seconds=0.0,
            stats=MatchupStats.empty(),
        )

    cfg = checkpoint_a.config
    n_envs = _benchmark_n_envs(n_envs, n_games)
    env = VectorizedEnv(
        n_envs=n_envs,
        obs_spec=_benchmark_env_obs_spec(checkpoint_a, checkpoint_b),
        action_spec=cfg.env.action_spec,
        two_player_weight=1.0 if player_count == 2 else 0.0,
        pin_memory=device.type == "cuda",
    )
    obs = env.reset()
    assignments = torch.full((n_envs, 4), -1, dtype=torch.int64)
    start_masks = obs.still_playing.clone()
    returns = torch.zeros((n_envs, 4), dtype=torch.float32)
    recorder = (
        ReplayRecorder(
            output_path=replay_output_path,
            source="benchmark_checkpoints",
            player_count=player_count,
            total_games=n_games,
            sample_games=replay_games,
            metadata={
                "checkpoint_a": str(checkpoint_a.path),
                "checkpoint_b": str(checkpoint_b.path),
                "checkpoint_a_env_steps": checkpoint_a.env_steps,
                "checkpoint_b_env_steps": checkpoint_b.env_steps,
                "deterministic_a": determinism.a,
                "deterministic_b": determinism.b,
                "int8_emulation_a": checkpoint_a.int8_emulation,
                "int8_emulation_b": checkpoint_b.int8_emulation,
            },
            rng=replay_rng,
            split_files=True,
        )
        if replay_output_path is not None and replay_games > 0
        else None
    )
    current_game_ordinals: list[int | None] = [None] * n_envs
    started_games = 0
    for env_index in range(n_envs):
        _assign_episode_models(
            assignments,
            env_index,
            active_slots=start_masks[env_index],
            player_count=player_count,
        )
        if started_games < n_games:
            current_game_ordinals[env_index] = started_games
            if recorder is not None:
                recorder.start_episode(
                    env,
                    env_index,
                    game_ordinal=started_games,
                    assignments=assignments[env_index],
                    start_mask=start_masks[env_index],
                )
            started_games += 1

    games = 0
    steps = 0
    stats = MatchupStats.empty()
    hidden_a = checkpoint_a.model.initial_hidden_state(n_envs, device=device)
    hidden_b = checkpoint_b.model.initial_hidden_state(n_envs, device=device)
    progress = tqdm(
        total=None,
        desc=f"{player_count}p steps",
        unit="step",
        disable=no_progress,
    )
    started_at = time.perf_counter()
    try:
        while games < n_games:
            actions, hidden_a, hidden_b = _actions_for_checkpoints(
                env,
                assignments,
                checkpoint_a=checkpoint_a,
                checkpoint_b=checkpoint_b,
                hidden_a=hidden_a,
                hidden_b=hidden_b,
                device=device,
                determinism=determinism,
            )
            obs, rewards, dones, _episode_metrics = env.step_decoded_actions(actions)
            hidden_a = checkpoint_a.model.reset_hidden_state(hidden_a, dones)
            hidden_b = checkpoint_b.model.reset_hidden_state(hidden_b, dones)
            steps += n_envs
            progress.update(n_envs)
            terminal_envs = torch.nonzero(dones.all(dim=1), as_tuple=False).flatten()
            terminal_env_set = {int(env_index.item()) for env_index in terminal_envs}
            if recorder is not None:
                recorder.record_step(
                    env,
                    terminal_envs=terminal_env_set,
                    rewards=rewards,
                    dones=dones,
                )
            returns += rewards
            for env_index_tensor in terminal_envs:
                if games >= n_games:
                    break
                env_index = int(env_index_tensor.item())
                if current_game_ordinals[env_index] is not None:
                    _record_terminal_result(
                        stats,
                        assignments[env_index],
                        start_masks[env_index],
                        returns[env_index],
                    )
                    games += 1

                returns[env_index].zero_()
                start_masks[env_index].copy_(obs.still_playing[env_index])
                _assign_episode_models(
                    assignments,
                    env_index,
                    active_slots=start_masks[env_index],
                    player_count=player_count,
                )
                if started_games < n_games:
                    current_game_ordinals[env_index] = started_games
                    if recorder is not None:
                        recorder.start_episode(
                            env,
                            env_index,
                            game_ordinal=started_games,
                            assignments=assignments[env_index],
                            start_mask=start_masks[env_index],
                        )
                    started_games += 1
                else:
                    current_game_ordinals[env_index] = None
    finally:
        progress.close()

    return BenchmarkResult(
        player_count=player_count,
        games=games,
        steps=steps,
        elapsed_seconds=time.perf_counter() - started_at,
        stats=stats,
    )


def _benchmark_n_envs(n_envs: int, n_games: int) -> int:
    return min(n_envs, n_games)


def _benchmark_env_obs_spec(
    checkpoint_a: LoadedCheckpoint,
    checkpoint_b: LoadedCheckpoint,
) -> ObsConfig:
    obs_spec_a = checkpoint_a.config.env.obs_spec
    obs_spec_b = checkpoint_b.config.env.obs_spec
    max_entities = max(obs_spec_a.max_entities, obs_spec_b.max_entities)
    if obs_spec_a.max_entities == max_entities:
        return obs_spec_a

    return obs_spec_a.model_copy(update={"max_entities": max_entities})


def _benchmark_replay_path(args: argparse.Namespace, player_count: int) -> Path | None:
    if args.save_replay_games == 0:
        return None
    checkpoint_a = args.checkpoint_a.stem
    checkpoint_b = args.checkpoint_b.stem
    return args.replay_dir / f"{checkpoint_a}_vs_{checkpoint_b}_{player_count}p.jsonl"


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    int8_emulation: bool = False,
) -> LoadedCheckpoint:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} must contain a checkpoint mapping")

    config = FullConfig.from_file(_checkpoint_config_path(path))
    model = create_model(
        config.model,
        obs_spec=config.env.obs_spec,
        action_spec=config.env.action_spec,
    ).to(device)
    lora_config = lora_config_for_model(config.model)
    if lora_config is not None:
        apply_lora_to_stateless_transformer(model, lora_config)
    model_state = dequantize_model_state_dict(checkpoint["model"])
    load_model_state_dict_allowing_lora(model, model_state)
    if lora_config is not None:
        fold_lora_adapters(model)
    if int8_emulation:
        apply_int8_emulation(model)
    model.eval()
    env_steps = _checkpoint_env_steps(checkpoint.get("env_steps"), path=path)
    return LoadedCheckpoint(
        path=path,
        config=config,
        model=model,
        env_steps=env_steps,
        int8_emulation=int8_emulation,
    )


def _checkpoint_config_path(checkpoint_path: Path) -> Path:
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"expected checkpoint config at {config_path}")
    return config_path


@torch.inference_mode()
def _actions_for_checkpoints(
    env: VectorizedEnv,
    assignments: torch.Tensor,
    *,
    checkpoint_a: LoadedCheckpoint,
    checkpoint_b: LoadedCheckpoint,
    hidden_a: ModelHiddenState | None,
    hidden_b: ModelHiddenState | None,
    device: torch.device,
    determinism: CheckpointDeterminism,
) -> tuple[DecodedLaunchActions, ModelHiddenState | None, ModelHiddenState | None]:
    obs_spec_a = checkpoint_a.config.env.obs_spec
    obs_spec_b = checkpoint_b.config.env.obs_spec
    action_spec_a = checkpoint_a.config.env.action_spec
    action_spec_b = checkpoint_b.config.env.action_spec
    obs_a = env.observation_for_spec(obs_spec_a, action_spec_a)
    obs_b = env.observation_for_spec(obs_spec_b, action_spec_b)
    output_a = _checkpoint_output(
        checkpoint_a,
        obs_a,
        device=device,
        deterministic=determinism.a,
        hidden_state=hidden_a,
    )
    output_b = _checkpoint_output(
        checkpoint_b,
        obs_b,
        device=device,
        deterministic=determinism.b,
        hidden_state=hidden_b,
    )
    decoded_a = env.decode_actions(
        _model_actions_to_cpu(output_a.actions),
        action_spec=action_spec_a,
    )
    decoded_b = env.decode_actions(
        _model_actions_to_cpu(output_b.actions),
        action_spec=action_spec_b,
    )
    return (
        _select_decoded_actions(decoded_a, decoded_b, assignments.eq(MODEL_A)),
        output_a.next_hidden_state,
        output_b.next_hidden_state,
    )


def _checkpoint_output(
    checkpoint: LoadedCheckpoint,
    obs: ObsBatch,
    *,
    device: torch.device,
    deterministic: bool,
    hidden_state: ModelHiddenState | None,
) -> ModelOutput:
    device_obs = _obs_to_device(obs, device)
    with autocast_context(checkpoint.config.rl, device):
        if hidden_state is None:
            return checkpoint.model(device_obs, deterministic=deterministic)
        return checkpoint.model(
            device_obs,
            deterministic=deterministic,
            hidden_state=hidden_state,
        )


def _obs_to_device(obs: ObsBatch, device: torch.device) -> ObsBatch:
    return ObsBatch(
        **{
            field: getattr(obs, field).to(
                device=device,
                non_blocking=device.type == "cuda",
            )
            for field in ObsBatch.model_fields
            if field
            not in {
                "action_mask",
                "player_features",
                "fleet_target",
                "target_incoming_features",
            }
        },
        action_mask=_action_mask_to_device(obs, device),
        player_features=(
            None
            if obs.player_features is None
            else obs.player_features.to(
                device=device,
                non_blocking=device.type == "cuda",
            )
        ),
        fleet_target=(
            None
            if obs.fleet_target is None
            else obs.fleet_target.to(
                device=device,
                non_blocking=device.type == "cuda",
            )
        ),
        target_incoming_features=(
            None
            if obs.target_incoming_features is None
            else obs.target_incoming_features.to(
                device=device,
                non_blocking=device.type == "cuda",
            )
        ),
    )


def _action_mask_to_device(obs: ObsBatch, device: torch.device) -> ActionMask:
    if isinstance(obs.action_mask, PureActionMask):
        return PureActionMask(
            can_act=obs.action_mask.can_act.to(
                device=device,
                non_blocking=device.type == "cuda",
            ),
            max_launch=obs.action_mask.max_launch.to(
                device=device,
                non_blocking=device.type == "cuda",
            ),
        )
    if isinstance(obs.action_mask, DiscreteTargetActionMask):
        return DiscreteTargetActionMask(
            can_act=obs.action_mask.can_act.to(
                device=device,
                non_blocking=device.type == "cuda",
            ),
            max_launch=obs.action_mask.max_launch.to(
                device=device,
                non_blocking=device.type == "cuda",
            ),
        )
    return DiscreteTargetBinActionMask(
        can_act=obs.action_mask.can_act.to(
            device=device,
            non_blocking=device.type == "cuda",
        )
    )


def _model_actions_to_cpu(actions: ActionBundle) -> ActionBundle:
    if isinstance(actions, PureActions):
        return PureActions(
            launch=actions.launch.cpu(),
            angle=actions.angle.cpu(),
            ships=actions.ships.cpu(),
        )
    if isinstance(actions, DiscreteTargetActions):
        return DiscreteTargetActions(
            launch=actions.launch.cpu(),
            target=actions.target.cpu(),
            ships=actions.ships.cpu(),
        )
    return DiscreteTargetBinActions(
        target=actions.target.cpu(),
        fleet_bin=actions.fleet_bin.cpu(),
    )


def _select_decoded_actions(
    actions_a: DecodedLaunchActions,
    actions_b: DecodedLaunchActions,
    use_a: torch.Tensor,
) -> DecodedLaunchActions:
    max_actions = max(actions_a.valid.shape[2], actions_b.valid.shape[2])
    padded_a = _pad_decoded_actions(actions_a, max_actions)
    padded_b = _pad_decoded_actions(actions_b, max_actions)
    action_mask = use_a[:, :, None]
    return DecodedLaunchActions(
        valid=_select_action(padded_a.valid, padded_b.valid, action_mask),
        from_planet_id=_select_action(
            padded_a.from_planet_id,
            padded_b.from_planet_id,
            action_mask,
        ),
        angle=_select_action(padded_a.angle, padded_b.angle, action_mask),
        ships=_select_action(padded_a.ships, padded_b.ships, action_mask),
    )


def _pad_decoded_actions(
    actions: DecodedLaunchActions,
    max_actions: int,
) -> DecodedLaunchActions:
    return DecodedLaunchActions(
        valid=_pad_decoded_tensor(actions.valid, max_actions),
        from_planet_id=_pad_decoded_tensor(actions.from_planet_id, max_actions),
        angle=_pad_decoded_tensor(actions.angle, max_actions),
        ships=_pad_decoded_tensor(actions.ships, max_actions),
    )


def _pad_decoded_tensor(tensor: torch.Tensor, max_actions: int) -> torch.Tensor:
    if tensor.shape[2] == max_actions:
        return tensor
    padded = torch.zeros(
        (*tensor.shape[:2], max_actions),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, :, : tensor.shape[2]] = tensor
    return padded


def _select_action(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    while mask.ndim < tensor_a.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, tensor_a, tensor_b).cpu()


def _assign_episode_models(
    assignments: torch.Tensor,
    env_index: int,
    *,
    active_slots: torch.Tensor,
    player_count: int,
) -> None:
    active = torch.nonzero(active_slots, as_tuple=False).flatten().tolist()
    if len(active) != player_count:
        raise ValueError(
            f"expected {player_count} active players, got {len(active)} "
            f"for env {env_index}"
        )

    pattern = _assignment_pattern(player_count)
    assignments[env_index].fill_(-1)
    for slot, model_index in zip(active, pattern, strict=True):
        assignments[env_index, slot] = model_index


def _assignment_pattern(player_count: int) -> tuple[int, ...]:
    if player_count == 2:
        return (MODEL_A, MODEL_B)
    if player_count == 4:
        return (MODEL_A, MODEL_B, MODEL_B, MODEL_A)
    raise ValueError(f"player_count must be 2 or 4, got {player_count}")


def _record_terminal_result(
    stats: MatchupStats,
    assignment: torch.Tensor,
    start_mask: torch.Tensor,
    returns: torch.Tensor,
) -> None:
    active_returns = returns[start_mask]
    if active_returns.numel() == 0:
        raise ValueError("cannot record a terminal result without starting players")
    model_returns = [0.0, 0.0]
    for player in torch.nonzero(start_mask, as_tuple=False).flatten().tolist():
        model_index = int(assignment[player].item())
        if model_index not in (MODEL_A, MODEL_B):
            raise ValueError(f"missing model assignment for player slot {player}")
        model_returns[model_index] += float(returns[player].item())
    if model_returns[MODEL_A] > model_returns[MODEL_B]:
        stats.add_game_result(MODEL_A)
    elif model_returns[MODEL_B] > model_returns[MODEL_A]:
        stats.add_game_result(MODEL_B)
    else:
        stats.add_game_result(None)


def _print_results(
    checkpoint_a: LoadedCheckpoint,
    checkpoint_b: LoadedCheckpoint,
    results: list[BenchmarkResult],
) -> None:
    total = MatchupStats.empty()
    print()
    print("Checkpoints")
    print(f"  A: {checkpoint_a.path} ({_env_steps_label(checkpoint_a)})")
    print(f"  B: {checkpoint_b.path} ({_env_steps_label(checkpoint_b)})")
    print()
    for result in results:
        total.merge(result.stats)
        print(
            f"{result.player_count}p: {result.games} games, "
            f"{result.steps:,} env steps, {result.steps_per_second:,.0f} steps/s"
        )
        _print_matrix(result.stats)
        print()

    print("Overall")
    _print_matrix(total)


def _env_steps_label(checkpoint: LoadedCheckpoint) -> str:
    if checkpoint.env_steps is None:
        return "Unknown env steps"

    return f"{checkpoint.env_steps:,} env steps"


def _checkpoint_env_steps(value: Any, *, path: Path) -> int | None:
    if value is None:
        return None

    if not isinstance(value, int):
        raise ValueError(f"checkpoint env_steps must be an integer: {path}")

    return value


def _print_matrix(stats: MatchupStats) -> None:
    print(f"A winrate: {_matrix_cell(stats, MODEL_A)}")
    print(f"B winrate: {_matrix_cell(stats, MODEL_B)}")


def _matrix_cell(stats: MatchupStats, model_index: int) -> str:
    games = stats.model_games[model_index]
    wins = stats.wins[model_index]
    losses = games - wins
    winrate = wins / games if games else 0.0
    return f"{wins:g}-{losses:g} ({winrate:.1%})"


def _player_count_counts(total: int, two_player_weight: float) -> dict[int, int]:
    two_player_games = int(total * two_player_weight + 0.5)
    return {
        2: two_player_games,
        4: total - two_player_games,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark two Orbit Wars model checkpoints against each other in "
            "2-player and 4-player games using the Rust vectorized env."
        )
    )
    parser.add_argument("checkpoint_a", type=Path, help="First checkpoint .pt file")
    parser.add_argument("checkpoint_b", type=Path, help="Second checkpoint .pt file")
    parser.add_argument(
        "--n-games",
        type=int,
        default=512,
        help=(
            "Total completed games to count, split between 2p and 4p according "
            "to --two-player-weight."
        ),
    )
    parser.add_argument(
        "-pw",
        "--two-player-weight",
        type=float,
        default=0.5,
        help="Fraction of games to run as 2-player games.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=256,
        help="Number of parallel Rust sub-envs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for model inference.",
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        nargs="?",
        choices=("both", "a", "b"),
        const="both",
        default="none",
        help=(
            "Use deterministic policy actions instead of sampling. Pass without "
            "a value for both checkpoints, or pass 'a'/'b' for one checkpoint."
        ),
    )
    parser.add_argument(
        "--int8-emulation",
        nargs="?",
        choices=("none", "both", "a", "b"),
        const="both",
        default="none",
        help=(
            "Emulate x86 int8 inference numerics for checkpoint Linear layers "
            "while staying on --device. Pass without a value for both checkpoints, "
            "or pass 'a'/'b' for one checkpoint, or 'none' to disable explicitly. "
            "Final actor/critic output heads stay unquantized."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--save-replay-games",
        type=int,
        default=0,
        help=(
            "Random completed games to save as replay JSONL, split across 2p "
            "and 4p according to --two-player-weight."
        ),
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("replays/benchmark_checkpoints"),
        help="Directory for benchmark replay JSONL files.",
    )
    args = parser.parse_args(_normalize_optional_target_args(sys.argv[1:]))
    _validate_args(args)
    return args


def _normalize_deterministic_args(argv: list[str]) -> list[str]:
    return _normalize_optional_target_args(argv)


def _normalize_optional_target_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    target_modes = {"none", "a", "b", "both"}
    optional_target_flags = {
        "-d": "--deterministic",
        "--deterministic": "--deterministic",
        "--int8-emulation": "--int8-emulation",
    }
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in optional_target_flags and (
            index + 1 == len(argv) or argv[index + 1] not in target_modes
        ):
            normalized.append(f"{optional_target_flags[arg]}=both")
        else:
            normalized.append(arg)
        index += 1
    return normalized


def _deterministic_flags(mode: str) -> CheckpointDeterminism:
    if mode == "none":
        return CheckpointDeterminism(a=False, b=False)
    if mode == "both":
        return CheckpointDeterminism(a=True, b=True)
    if mode == "a":
        return CheckpointDeterminism(a=True, b=False)
    if mode == "b":
        return CheckpointDeterminism(a=False, b=True)
    raise ValueError(f"unknown deterministic mode: {mode}")


def _int8_emulation_flags(mode: str) -> CheckpointInt8Emulation:
    if mode == "none":
        return CheckpointInt8Emulation(a=False, b=False)
    if mode == "both":
        return CheckpointInt8Emulation(a=True, b=True)
    if mode == "a":
        return CheckpointInt8Emulation(a=True, b=False)
    if mode == "b":
        return CheckpointInt8Emulation(a=False, b=True)
    raise ValueError(f"unknown int8 emulation mode: {mode}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_games <= 0:
        raise ValueError("--n-games must be positive")
    if not 0.0 <= args.two_player_weight <= 1.0:
        raise ValueError("--two-player-weight must be in [0, 1]")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if args.save_replay_games < 0:
        raise ValueError("--save-replay-games must be non-negative")
    if args.save_replay_games > args.n_games:
        raise ValueError("--save-replay-games must be <= --n-games")
    game_counts = _player_count_counts(args.n_games, args.two_player_weight)
    replay_counts = _player_count_counts(
        args.save_replay_games,
        args.two_player_weight,
    )
    if any(
        replay_counts[player_count] > game_counts[player_count]
        for player_count in PLAYER_COUNTS
    ):
        raise ValueError(
            "--save-replay-games requests more games than one player count runs"
        )


if __name__ == "__main__":
    main()
