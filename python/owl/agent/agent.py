from pathlib import Path
from typing import Any

import torch

from owl.model import StatelessTransformerV1
from owl.rl import (
    ActionDiscreteTargetsConfig,
    ObsBatch,
    actions_to_kaggle,
    encode_python_observation,
)
from owl.rs import assert_release_build
from owl.train import FullConfig

from .kaggle_observation import KaggleObservation


class Agent:
    def __init__(self) -> None:
        assert_release_build()
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = self.root / "config.yaml"
        if not self.config_path.is_file():
            raise ValueError(f"expected Kaggle config at {self.config_path}")
        checkpoint_paths = sorted(self.root.glob("*.pt"))
        if len(checkpoint_paths) != 1:
            raise ValueError(
                f"expected exactly one .pt checkpoint adjacent to main.py, "
                f"found {len(checkpoint_paths)} in {self.root}"
            )
        self.checkpoint_path = checkpoint_paths[0]
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

    def act(self, observation: KaggleObservation) -> list[list[float]]:
        obs_dict = observation.to_rl_observation()
        encoded = encode_python_observation(
            obs_dict,
            obs_spec=self.config.env.obs_spec,
            action_spec=self.config.env.action_spec,
        )
        obs = self._obs_batch(encoded, player=observation.player)
        with torch.inference_mode():
            output = self.model(self._obs_to_device(obs), deterministic=True)
        actions = output.actions
        return actions_to_kaggle(
            obs_dict,
            observation.player,
            actions.launch.cpu(),
            actions.action_value().cpu(),
            actions.ships.cpu(),
            action_spec=self.config.env.action_spec,
        )

    def _obs_batch(
        self,
        encoded: tuple[
            Any,
            Any,
            Any,
            Any,
            Any,
            Any,
            Any,
            Any,
        ],
        *,
        player: int,
    ) -> ObsBatch:
        (
            planets,
            orbiting_planets,
            fleets,
            comets,
            entity_mask,
            global_features,
            can_act,
            max_launch,
        ) = encoded
        can_act_tensor = torch.as_tensor(can_act, dtype=torch.bool).unsqueeze(0)
        max_launch_tensor = torch.as_tensor(max_launch, dtype=torch.int64).unsqueeze(0)
        player_mask = torch.zeros((1, 4), dtype=torch.bool)
        player_mask[0, player] = True
        can_act_tensor &= player_mask[(...,) + (None,) * (can_act_tensor.ndim - 2)]
        max_launch_tensor = torch.where(
            player_mask[:, :, None],
            max_launch_tensor,
            torch.zeros_like(max_launch_tensor),
        )
        still_playing = torch.zeros((1, 4), dtype=torch.bool)
        still_playing[0, player] = True
        return ObsBatch(
            planets=torch.as_tensor(planets, dtype=torch.float32).unsqueeze(0),
            orbiting_planets=torch.as_tensor(
                orbiting_planets, dtype=torch.bool
            ).unsqueeze(0),
            fleets=torch.as_tensor(fleets, dtype=torch.float32).unsqueeze(0),
            comets=torch.as_tensor(comets, dtype=torch.float32).unsqueeze(0),
            entity_mask=torch.as_tensor(entity_mask, dtype=torch.bool).unsqueeze(0),
            still_playing=still_playing,
            global_features=torch.as_tensor(
                global_features, dtype=torch.float32
            ).unsqueeze(0),
            can_act=can_act_tensor,
            max_launch=max_launch_tensor,
        )

    def _obs_to_device(self, obs: ObsBatch) -> ObsBatch:
        return ObsBatch(
            **{
                field: getattr(obs, field).to(device=self.device)
                for field in ObsBatch.model_fields
            }
        )
