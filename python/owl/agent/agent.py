from pathlib import Path

import torch

from owl.config import BaseConfig
from owl.model import StatelessTransformerV1
from owl.rl import (
    ActionDiscreteTargetsConfig,
    ObsBatch,
    actions_to_kaggle,
    encode_python_observation,
)
from owl.rs import assert_release_build
from owl.train.config import FullConfig

from .kaggle_observation import KaggleObservation


class AgentConfig(BaseConfig):
    deterministic: bool


class Agent:
    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
    ) -> None:
        assert_release_build()
        self.config_path = config_path
        if not self.config_path.is_file():
            raise ValueError(f"expected Kaggle config at {self.config_path}")

        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path.is_file():
            raise ValueError(f"expected Kaggle checkpoint at {self.checkpoint_path}")

        self.agent_config_path = Path(__file__).with_name("agent_config.yaml")
        if not self.agent_config_path.is_file():
            raise ValueError(f"expected agent config at {self.agent_config_path}")

        self.agent_config = AgentConfig.from_file(self.agent_config_path)

        self.config = FullConfig.from_file(self.config_path)
        if (
            isinstance(self.config.env.action_spec, ActionDiscreteTargetsConfig)
            and self.config.env.action_spec.max_per_planet_launches != 1
        ):
            raise ValueError(
                "Kaggle discrete_targets checkpoints must use max_per_planet_launches=1"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StatelessTransformerV1(
            self.config.model,
            obs_spec=self.config.env.obs_spec,
            action_spec=self.config.env.action_spec,
        ).to(self.device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint must be a dictionary: {self.checkpoint_path}")
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @torch.inference_mode()
    def act(self, observation: KaggleObservation) -> list[list[float]]:
        obs_dict = observation.to_rl_observation()
        obs = encode_python_observation(
            obs_dict,
            obs_spec=self.config.env.obs_spec,
            action_spec=self.config.env.action_spec,
        )
        output = self.model(
            self._obs_to_device(obs),
            deterministic=self.agent_config.deterministic,
        )
        actions = output.actions
        return actions_to_kaggle(
            obs_dict,
            observation.player,
            actions.launch.cpu(),
            actions.action_value().cpu(),
            actions.ships.cpu(),
            action_spec=self.config.env.action_spec,
        )

    def _obs_to_device(self, obs: ObsBatch) -> ObsBatch:
        return ObsBatch(
            **{
                field: getattr(obs, field).to(device=self.device)
                for field in ObsBatch.model_fields
            }
        )
