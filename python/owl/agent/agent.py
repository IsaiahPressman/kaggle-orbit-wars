from pathlib import Path

import torch

from owl.model import StatelessTransformerV1
from owl.rl import (
    ActionDiscreteTargetsConfig,
    ObsBatch,
    actions_to_kaggle,
    encode_python_observation,
)
from owl.rs import assert_release_build

from .config import AgentConfig
from .kaggle_observation import KaggleObservation


class Agent:
    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        agent_config_path: Path | None = None,
    ) -> None:
        assert_release_build()
        self.config_path = config_path
        if not self.config_path.is_file():
            raise ValueError(f"expected Kaggle config at {self.config_path}")
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path.is_file():
            raise ValueError(f"expected Kaggle checkpoint at {self.checkpoint_path}")
        self.agent_config_path = agent_config_path or Path(__file__).with_name(
            "agent_config.yaml"
        )
        if not self.agent_config_path.is_file():
            raise ValueError(f"expected agent config at {self.agent_config_path}")
        self.agent_config = AgentConfig.from_file(self.agent_config_path)
        from owl.train.config import FullConfig

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
        obs = self._obs_for_player(obs, player=observation.player)
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

    def _obs_for_player(self, obs: ObsBatch, *, player: int) -> ObsBatch:
        player_mask = torch.zeros((1, 4), dtype=torch.bool)
        player_mask[0, player] = True
        can_act = obs.can_act & player_mask[(...,) + (None,) * (obs.can_act.ndim - 2)]
        max_launch = torch.where(
            player_mask[:, :, None],
            obs.max_launch,
            torch.zeros_like(obs.max_launch),
        )
        return obs.model_copy(
            update={
                "still_playing": player_mask,
                "can_act": can_act,
                "max_launch": max_launch,
            }
        )

    def _obs_to_device(self, obs: ObsBatch) -> ObsBatch:
        return ObsBatch(
            **{
                field: getattr(obs, field).to(device=self.device)
                for field in ObsBatch.model_fields
            }
        )
