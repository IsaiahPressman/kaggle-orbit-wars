# ruff: noqa: E402
import os
from pathlib import Path
from typing import Any

THREADS = 1
os.environ.setdefault("OMP_NUM_THREADS", f"{THREADS}")
os.environ.setdefault("MKL_NUM_THREADS", f"{THREADS}")

import torch

torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)

from owl import OWL_ROOT
from owl.agent import Agent
from owl.rs import assert_release_build

assert_release_build()
ROOT = OWL_ROOT.parent
PRIMARY_MODEL_ROOT = ROOT / "models" / "primary"
FALLBACK_MODEL_ROOT = ROOT / "models" / "fallback"
fallback_checkpoint_config_path: Path | None
fallback_checkpoint_path: Path | None
if FALLBACK_MODEL_ROOT.is_dir():
    fallback_checkpoint_config_path = FALLBACK_MODEL_ROOT / "config.yaml"
    fallback_checkpoint_path = FALLBACK_MODEL_ROOT / "checkpoint.pt"
else:
    fallback_checkpoint_config_path = None
    fallback_checkpoint_path = None

AGENT = Agent(
    checkpoint_config_path=PRIMARY_MODEL_ROOT / "config.yaml",
    checkpoint_path=PRIMARY_MODEL_ROOT / "checkpoint.pt",
    fallback_checkpoint_config_path=fallback_checkpoint_config_path,
    fallback_checkpoint_path=fallback_checkpoint_path,
)


def agent_fn(observation: Any) -> list[list[float]]:
    return AGENT.act(observation)
