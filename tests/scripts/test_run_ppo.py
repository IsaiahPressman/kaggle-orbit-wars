from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import torch
from owl.train import FullConfig

_RUN_PPO_PATH = Path(__file__).parents[2] / "scripts" / "run_ppo.py"
_RUN_PPO_SPEC = importlib.util.spec_from_file_location("run_ppo", _RUN_PPO_PATH)
assert _RUN_PPO_SPEC is not None
assert _RUN_PPO_SPEC.loader is not None
run_ppo = importlib.util.module_from_spec(_RUN_PPO_SPEC)
_RUN_PPO_SPEC.loader.exec_module(run_ppo)


def _full_config() -> FullConfig:
    return FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
            },
            "optimizer": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "lr_schedule": {
                    "warmup_steps": 1,
                    "decay_steps": 4,
                    "lr_min_ratio": 0.1,
                },
            },
            "rl": {
                "horizon": 4,
                "n_envs": 2,
            },
        }
    )


def test_next_periodic_checkpoint_step_handles_crossed_cadence() -> None:
    assert run_ppo._next_periodic_checkpoint_step(checkpoint_every_env_steps=0) is None
    assert run_ppo._next_periodic_checkpoint_step(checkpoint_every_env_steps=100) == 100
    assert (
        run_ppo._next_periodic_checkpoint_step(
            checkpoint_every_env_steps=100,
            env_steps=256,
        )
        == 300
    )


def test_should_stop_training_checks_step_and_runtime_limits() -> None:
    assert run_ppo._should_stop_training(
        env_steps=128,
        started_at=time.monotonic(),
        max_env_steps=128,
        max_runtime_seconds=None,
    )
    assert run_ppo._should_stop_training(
        env_steps=1,
        started_at=time.monotonic() - 2.0,
        max_env_steps=None,
        max_runtime_seconds=1.0,
    )
    assert not run_ppo._should_stop_training(
        env_steps=1,
        started_at=time.monotonic(),
        max_env_steps=10,
        max_runtime_seconds=10.0,
    )


def test_write_checkpoint_includes_training_state(tmp_path: Path) -> None:
    cfg = _full_config()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    path = tmp_path / "checkpoint.pt"

    run_ppo._write_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        cfg=cfg,
        config_path=Path("config.yaml"),
        env_steps=512,
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["env_steps"] == 512
    assert checkpoint["config"]["env"]["n_envs"] == 2
    assert checkpoint["config_path"] == "config.yaml"
    assert checkpoint["model"].keys() == model.state_dict().keys()
    assert "state" in checkpoint["optimizer"]
    assert checkpoint["lr_scheduler"] == scheduler.state_dict()
    assert set(checkpoint["rng_state"]) == {
        "python",
        "numpy",
        "torch",
        "torch_cuda",
    }
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()
