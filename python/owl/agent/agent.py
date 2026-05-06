from pathlib import Path
from time import perf_counter

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
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint must be a dictionary: {self.checkpoint_path}")
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @torch.inference_mode()
    def act(self, observation: KaggleObservation) -> list[list[float]]:
        total_start = perf_counter()

        encode_start = perf_counter()
        obs_dict = observation.to_rl_observation()
        obs = encode_python_observation(
            obs_dict,
            obs_spec=self.config.env.obs_spec,
            action_spec=self.config.env.action_spec,
        )
        device_obs = self._obs_to_device(obs)
        self._synchronize_device()
        encode_ms = _elapsed_ms(encode_start)

        inference_start = perf_counter()
        output = self.model(
            device_obs,
            deterministic=self.agent_config.deterministic,
        )
        self._synchronize_device()
        values = output.values.detach().cpu()[0]
        inference_ms = _elapsed_ms(inference_start)

        conversion_start = perf_counter()
        actions = actions_to_kaggle(
            obs_dict,
            observation.player,
            output.actions.launch.detach().cpu(),
            output.actions.action_value().detach().cpu(),
            output.actions.ships.detach().cpu(),
            action_spec=self.config.env.action_spec,
        )
        conversion_ms = _elapsed_ms(conversion_start)
        total_ms = _elapsed_ms(total_start)

        self.log(
            total_ms=total_ms,
            encode_ms=encode_ms,
            inference_ms=inference_ms,
            conversion_ms=conversion_ms,
            self_value=float(values[observation.player].item()),
            player_values=[float(value) for value in values.tolist()],
            entity_count=int(obs.entity_mask.sum().item()),
            remaining_overage_time=observation.remaining_overage_time,
        )
        return actions

    def _obs_to_device(self, obs: ObsBatch) -> ObsBatch:
        return ObsBatch(
            **{
                field: getattr(obs, field).to(device=self.device)
                for field in ObsBatch.model_fields
            }
        )

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def log(
        self,
        *,
        total_ms: int,
        encode_ms: int,
        inference_ms: int,
        conversion_ms: int,
        self_value: float,
        player_values: list[float],
        entity_count: int,
        remaining_overage_time: float,
    ) -> None:
        values = ",".join(f"{value:.3f}" for value in player_values)
        print(
            "agent "
            f"total_ms={total_ms} "
            f"encode_ms={encode_ms} "
            f"inference_ms={inference_ms} "
            f"conversion_ms={conversion_ms} "
            f"value_self={self_value:.3f} "
            f"values=[{values}] "
            f"entities={entity_count} "
            f"remaining_overage_s={remaining_overage_time:.3f}"
        )


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)
