# ruff: noqa: E402
import os
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from owl import OWL_ROOT
from owl.agent import Agent, KaggleObservation, find_checkpoint_path
from owl.rs import assert_release_build

assert_release_build()
ROOT = OWL_ROOT.parent
CONFIG_PATH = ROOT / "config.yaml"
AGENT = Agent(
    checkpoint_config_path=CONFIG_PATH,
    checkpoint_path=find_checkpoint_path(ROOT),
)


def agent_fn(observation: Any) -> list[list[float]]:
    return AGENT.act(KaggleObservation.model_validate(observation))
