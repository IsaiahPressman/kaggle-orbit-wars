from pathlib import Path
from typing import Any

from owl.agent import Agent, KaggleObservation

AGENT: Agent | None = None
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def _checkpoint_path() -> Path:
    checkpoint_paths = sorted(ROOT.glob("*.pt"))
    if len(checkpoint_paths) != 1:
        raise ValueError(
            f"expected exactly one .pt checkpoint adjacent to main.py, "
            f"found {len(checkpoint_paths)} in {ROOT}"
        )
    return checkpoint_paths[0]


def agent_fn(observation: Any) -> list[list[float]]:
    global AGENT
    if AGENT is None:
        AGENT = Agent(config_path=CONFIG_PATH, checkpoint_path=_checkpoint_path())

    return AGENT.act(KaggleObservation.model_validate(observation))
