#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from owl.model import StatelessTransformerV1
from owl.replay import ReplayRecorder
from owl.rl import (
    ActionBundle,
    ActionConfig,
    ActionMask,
    DecodedLaunchActions,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    DiscreteTargetBinActionMask,
    DiscreteTargetBinActions,
    ObsBatch,
    PureActionMask,
    PureActions,
    VectorizedEnv,
)
from owl.rs import assert_release_build
from owl.train import FullConfig, configure_torch
from owl.train.utils import DTypeConfig, autocast_context
from tqdm import tqdm

MODEL_A = 0
MODEL_B = 1
PLAYER_COUNTS = (2, 4)


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    config: FullConfig
    model: StatelessTransformerV1
    env_steps: int | None


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
        return self.steps / self.elapsed_seconds


def main() -> None:
    args = _parse_args()

    assert_release_build()
    configure_torch()
    device = torch.device(args.device)
    checkpoint_a = _load_checkpoint(args.checkpoint_a, device=device)
    checkpoint_b = _load_checkpoint(args.checkpoint_b, device=device)
    _validate_compatible_checkpoints(checkpoint_a, checkpoint_b)

    per_player_count_games = args.n_games // len(PLAYER_COUNTS)
    results = [
        run_benchmark(
            checkpoint_a=checkpoint_a,
            checkpoint_b=checkpoint_b,
            player_count=player_count,
            n_games=per_player_count_games,
            n_envs=args.n_envs,
            device=device,
            deterministic=args.deterministic,
            no_progress=args.no_progress,
            replay_games=args.save_replay_games // len(PLAYER_COUNTS),
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
    deterministic: bool,
    no_progress: bool,
    replay_games: int,
    replay_output_path: Path | None,
    replay_rng: random.Random,
) -> BenchmarkResult:
    cfg = checkpoint_a.config
    env = VectorizedEnv(
        n_envs=n_envs,
        obs_spec=cfg.env.obs_spec,
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
                "deterministic": deterministic,
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
    progress = tqdm(
        total=n_games,
        desc=f"{player_count}p games",
        unit="game",
        disable=no_progress,
    )
    started_at = time.perf_counter()
    try:
        while games < n_games:
            actions = _actions_for_assignments(
                env,
                assignments,
                model_a=checkpoint_a.model,
                model_b=checkpoint_b.model,
                action_spec_a=checkpoint_a.config.env.action_spec,
                action_spec_b=checkpoint_b.config.env.action_spec,
                config_a=checkpoint_a.config.rl,
                config_b=checkpoint_b.config.rl,
                device=device,
                deterministic=deterministic,
            )
            obs, rewards, dones, _episode_metrics = env.step_decoded_actions(actions)
            steps += n_envs
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
                    progress.update(1)

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


def _benchmark_replay_path(args: argparse.Namespace, player_count: int) -> Path | None:
    if args.save_replay_games == 0:
        return None
    checkpoint_a = args.checkpoint_a.stem
    checkpoint_b = args.checkpoint_b.stem
    return args.replay_dir / f"{checkpoint_a}_vs_{checkpoint_b}_{player_count}p.jsonl"


def _load_checkpoint(path: Path, *, device: torch.device) -> LoadedCheckpoint:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} must contain a checkpoint mapping")

    config = FullConfig.from_file(_checkpoint_config_path(path))
    model = StatelessTransformerV1(
        config.model,
        obs_spec=config.env.obs_spec,
        action_spec=config.env.action_spec,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    env_steps = checkpoint.get("env_steps")
    if env_steps is not None:
        env_steps = int(env_steps)
    return LoadedCheckpoint(path=path, config=config, model=model, env_steps=env_steps)


def _checkpoint_config_path(checkpoint_path: Path) -> Path:
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"expected checkpoint config at {config_path}")
    return config_path


def _validate_compatible_checkpoints(
    checkpoint_a: LoadedCheckpoint,
    checkpoint_b: LoadedCheckpoint,
) -> None:
    env_a = checkpoint_a.config.env
    env_b = checkpoint_b.config.env
    if env_a.obs_spec != env_b.obs_spec:
        raise ValueError("checkpoint observation specs must match")


@torch.inference_mode()
def _actions_for_assignments(
    env: VectorizedEnv,
    assignments: torch.Tensor,
    *,
    model_a: StatelessTransformerV1,
    model_b: StatelessTransformerV1,
    action_spec_a: ActionConfig,
    action_spec_b: ActionConfig,
    config_a: DTypeConfig,
    config_b: DTypeConfig,
    device: torch.device,
    deterministic: bool,
) -> DecodedLaunchActions:
    obs_a = env.observation_for_action_spec(action_spec_a)
    obs_b = env.observation_for_action_spec(action_spec_b)
    device_obs_a = _obs_to_device(obs_a, device)
    device_obs_b = _obs_to_device(obs_b, device)
    with autocast_context(config_a, device):
        output_a = model_a(device_obs_a, deterministic=deterministic)
    with autocast_context(config_b, device):
        output_b = model_b(device_obs_b, deterministic=deterministic)
    decoded_a = env.decode_actions(
        _model_actions_to_cpu(output_a.actions),
        action_spec=action_spec_a,
    )
    decoded_b = env.decode_actions(
        _model_actions_to_cpu(output_b.actions),
        action_spec=action_spec_b,
    )
    return _select_decoded_actions(decoded_a, decoded_b, assignments.eq(MODEL_A))


def _obs_to_device(obs: ObsBatch, device: torch.device) -> ObsBatch:
    return ObsBatch(
        **{
            field: getattr(obs, field).to(
                device=device,
                non_blocking=device.type == "cuda",
            )
            for field in ObsBatch.model_fields
            if field != "action_mask"
        },
        action_mask=_action_mask_to_device(obs, device),
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


def _select_actions(
    actions_a: ActionBundle,
    actions_b: ActionBundle,
    use_a: torch.Tensor,
) -> ActionBundle:
    if type(actions_a) is not type(actions_b):
        raise ValueError("checkpoint action bundles must have matching kinds")
    if isinstance(actions_a, PureActions) and isinstance(actions_b, PureActions):
        action_mask = use_a[:, :, None, None]
        return PureActions(
            launch=_select_action(actions_a.launch, actions_b.launch, action_mask),
            angle=_select_action(actions_a.angle, actions_b.angle, action_mask),
            ships=_select_action(actions_a.ships, actions_b.ships, action_mask),
        )
    if isinstance(actions_a, DiscreteTargetActions) and isinstance(
        actions_b,
        DiscreteTargetActions,
    ):
        action_mask = use_a[:, :, None, None]
        return DiscreteTargetActions(
            launch=_select_action(actions_a.launch, actions_b.launch, action_mask),
            target=_select_action(actions_a.target, actions_b.target, action_mask),
            ships=_select_action(actions_a.ships, actions_b.ships, action_mask),
        )
    if isinstance(actions_a, DiscreteTargetBinActions) and isinstance(
        actions_b,
        DiscreteTargetBinActions,
    ):
        action_mask = use_a[:, :, None]
        return DiscreteTargetBinActions(
            target=_select_action(actions_a.target, actions_b.target, action_mask),
            fleet_bin=_select_action(
                actions_a.fleet_bin,
                actions_b.fleet_bin,
                action_mask,
            ),
        )
    raise ValueError("unsupported checkpoint action bundle kind")


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
        return "env_steps unknown"
    return f"{checkpoint.env_steps:,} env steps"


def _print_matrix(stats: MatchupStats) -> None:
    print(f"A winrate: {_matrix_cell(stats, MODEL_A)}")
    print(f"B winrate: {_matrix_cell(stats, MODEL_B)}")


def _matrix_cell(stats: MatchupStats, model_index: int) -> str:
    games = stats.model_games[model_index]
    wins = stats.wins[model_index]
    losses = games - wins
    winrate = wins / games if games else 0.0
    return f"{wins:g}-{losses:g} ({winrate:.1%})"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark two Orbit Wars model checkpoints against each other in "
            "balanced 2-player and 4-player games using the Rust vectorized env."
        )
    )
    parser.add_argument("checkpoint_a", type=Path, help="First checkpoint .pt file")
    parser.add_argument("checkpoint_b", type=Path, help="Second checkpoint .pt file")
    parser.add_argument(
        "--n-games",
        type=int,
        default=512,
        help="Total completed games to count, split evenly between 2p and 4p.",
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
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions instead of sampling.",
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
            "Random completed games to save as replay JSONL, split evenly "
            "across 2p and 4p."
        ),
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("replays/benchmark_checkpoints"),
        help="Directory for benchmark replay JSONL files.",
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_games <= 0:
        raise ValueError("--n-games must be positive")
    if args.n_games % len(PLAYER_COUNTS) != 0:
        raise ValueError("--n-games must be even so 2p and 4p games are balanced")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if args.save_replay_games < 0:
        raise ValueError("--save-replay-games must be non-negative")
    if args.save_replay_games % len(PLAYER_COUNTS) != 0:
        raise ValueError(
            "--save-replay-games must be even so 2p and 4p replays are balanced"
        )
    if args.save_replay_games > args.n_games:
        raise ValueError("--save-replay-games must be <= --n-games")
    per_player_count_games = args.n_games // len(PLAYER_COUNTS)
    if args.save_replay_games // len(PLAYER_COUNTS) > per_player_count_games:
        raise ValueError(
            "--save-replay-games requests more games than one player count runs"
        )
    if per_player_count_games % args.n_envs != 0:
        raise ValueError("(--n-games / 2) must be divisible by --n-envs")


if __name__ == "__main__":
    main()
