from pathlib import Path
from time import perf_counter

import torch
from pydantic import ConfigDict, Field

from owl.config import BaseConfig
from owl.model import ModelConfig, ModelHiddenState, create_model
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    ActionBundle,
    ActionMask,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    DiscreteTargetBinActionMask,
    DiscreteTargetBinActions,
    EntityBasedBaseConfig,
    EnvConfig,
    ObsBatch,
    PureActionMask,
    PureActions,
    TargetingMode,
    actions_to_kaggle,
    encode_python_observation,
)

from .checkpoint_quantization import dequantize_model_state_dict
from .kaggle_observation import KaggleObservation

AGENT_CONFIG_PATH = Path(__file__).with_name("agent_config.yaml")


class AgentConfig(BaseConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    deterministic: bool
    max_entities_override: int | None = None
    targeting_mode_override: TargetingMode | None = None
    min_overage_time: float = Field(default=0.0, ge=0.0, le=60.0)


class AgentCheckpointConfig(BaseConfig):
    model_config = ConfigDict(extra="ignore", frozen=True)

    env: EnvConfig
    model: ModelConfig


class Agent:
    def __init__(
        self,
        *,
        checkpoint_config_path: Path,
        checkpoint_path: Path,
    ) -> None:
        init_start = perf_counter()
        if not checkpoint_config_path.is_file():
            raise ValueError(f"expected Kaggle config at {checkpoint_config_path}")

        if not checkpoint_path.is_file():
            raise ValueError(f"expected Kaggle checkpoint at {checkpoint_path}")

        self.config = AgentConfig.from_file(AGENT_CONFIG_PATH)
        self.checkpoint_config = AgentCheckpointConfig.from_file(checkpoint_config_path)
        self.checkpoint_config = apply_max_entities_override(
            self.checkpoint_config,
            self.config.max_entities_override,
        )
        self.checkpoint_config = apply_targeting_mode_override(
            self.checkpoint_config,
            self.config.targeting_mode_override,
        )
        self._last_turn_value = float("nan")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(
            self.checkpoint_config.model,
            obs_spec=self.checkpoint_config.env.obs_spec,
            action_spec=self.checkpoint_config.env.action_spec,
        ).to(self.device)
        self.hidden_state: ModelHiddenState | None = None
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(
                f"checkpoint must be a dictionary with key 'model': {checkpoint_path}"
            )

        self.model.load_state_dict(dequantize_model_state_dict(checkpoint["model"]))
        self.model.eval()
        print(f"init_s={perf_counter() - init_start:.2f} - ", end="", flush=True)

    @torch.inference_mode()
    def act(self, observation: KaggleObservation) -> list[list[float]]:
        if observation.remaining_overage_time < self.config.min_overage_time:
            return []

        total_start = perf_counter()

        encode_start = perf_counter()
        obs_dict = observation.to_rl_observation()
        obs = encode_python_observation(
            obs_dict,
            obs_spec=self.checkpoint_config.env.obs_spec,
            action_spec=self.checkpoint_config.env.action_spec,
        )
        obs = compact_entities(obs)
        device_obs = self._obs_to_device(obs)
        self._synchronize_device()
        encode_ms = _elapsed_ms(encode_start)

        inference_start = perf_counter()
        hidden_state = getattr(self, "hidden_state", None)
        initial_hidden_state = getattr(self.model, "initial_hidden_state", None)
        if callable(initial_hidden_state):
            if observation.step == 0 or hidden_state is None:
                hidden_state = initial_hidden_state(1, device=self.device)
            output = (
                self.model(device_obs, deterministic=self.config.deterministic)
                if hidden_state is None
                else self.model(
                    device_obs,
                    deterministic=self.config.deterministic,
                    hidden_state=hidden_state,
                )
            )
            self.hidden_state = getattr(output, "next_hidden_state", None)
        else:
            output = self.model(device_obs, deterministic=self.config.deterministic)
            self.hidden_state = None
        self._synchronize_device()
        values = output.values.detach().cpu()[0]
        self_value = float(values[observation.player].item())
        if observation.step == 0:
            n_players = _observation_player_count(observation)
            self._last_turn_value = (2.0 - n_players) / n_players

        advantage = self_value - self._last_turn_value
        inference_ms = _elapsed_ms(inference_start)

        conversion_start = perf_counter()
        actions = actions_to_kaggle(
            obs_dict,
            observation.player,
            _model_actions_to_cpu(output.actions),
            action_spec=self.checkpoint_config.env.action_spec,
        )
        conversion_ms = _elapsed_ms(conversion_start)
        total_ms = _elapsed_ms(total_start)

        self.log(
            step=observation.step,
            total_ms=total_ms,
            encode_ms=encode_ms,
            inference_ms=inference_ms,
            conversion_ms=conversion_ms,
            self_value=self_value,
            advantage=advantage,
            player_values=[float(value) for value in values.tolist()],
            entity_count=obs.entity_mask.shape[1],
            remaining_overage_time=observation.remaining_overage_time,
        )
        self._last_turn_value = self_value
        return actions

    def _obs_to_device(self, obs: ObsBatch) -> ObsBatch:
        return ObsBatch(
            **{
                field: getattr(obs, field).to(device=self.device)
                for field in ObsBatch.model_fields
                if field != "action_mask"
            },
            action_mask=_action_mask_to_device(obs.action_mask, self.device),
        )

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def log(
        self,
        *,
        step: int,
        total_ms: int,
        encode_ms: int,
        inference_ms: int,
        conversion_ms: int,
        self_value: float,
        advantage: float,
        player_values: list[float],
        entity_count: int,
        remaining_overage_time: float,
    ) -> None:
        values = ",".join(f"{value:.3f}" for value in player_values)
        print(
            f"step={step} - "
            f"total_ms={total_ms} - "
            f"encode_ms={encode_ms} - "
            f"inference_ms={inference_ms} - "
            f"conversion_ms={conversion_ms} - "
            f"value_self={self_value:.3f} - "
            f"advantage={advantage:.3f} - "
            f"values=[{values}] - "
            f"entities={entity_count} - "
            f"remaining_overage_s={remaining_overage_time:.1f}",
            flush=True,
        )


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def _observation_player_count(observation: KaggleObservation) -> int:
    player_indexes = [observation.player]
    player_indexes.extend(
        owner
        for _, owner, *_ in [
            *observation.initial_planets,
            *observation.planets,
            *observation.fleets,
        ]
        if owner >= 0
    )
    return 4 if max(player_indexes) >= 2 else 2


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


def _action_mask_to_device(action_mask: ActionMask, device: torch.device) -> ActionMask:
    if isinstance(action_mask, PureActionMask):
        return PureActionMask(
            can_act=action_mask.can_act.to(device=device),
            max_launch=action_mask.max_launch.to(device=device),
        )
    if isinstance(action_mask, DiscreteTargetActionMask):
        return DiscreteTargetActionMask(
            can_act=action_mask.can_act.to(device=device),
            max_launch=action_mask.max_launch.to(device=device),
        )
    return DiscreteTargetBinActionMask(can_act=action_mask.can_act.to(device=device))


def compact_entities(obs: ObsBatch) -> ObsBatch:
    """Drop inactive fleet rows from a single-row observation batch."""
    batch_size = obs.entity_mask.shape[0]
    if batch_size != 1:
        raise ValueError(
            f"runtime entity compaction requires batch size 1, got {batch_size}"
        )

    fleet_mask = obs.entity_mask[0, ACTION_ENTITY_SLOTS:]
    active_fleet_indexes = torch.nonzero(fleet_mask, as_tuple=True)[0]
    if active_fleet_indexes.numel() == obs.fleets.shape[1]:
        return obs

    return ObsBatch(
        planets=obs.planets,
        orbiting_planets=obs.orbiting_planets,
        fleets=obs.fleets[:, active_fleet_indexes, :],
        comets=obs.comets,
        entity_mask=torch.cat(
            (
                obs.entity_mask[:, :ACTION_ENTITY_SLOTS],
                obs.entity_mask[:, ACTION_ENTITY_SLOTS:][:, active_fleet_indexes],
            ),
            dim=1,
        ),
        still_playing=obs.still_playing,
        global_features=obs.global_features,
        action_mask=obs.action_mask,
    )


def apply_max_entities_override(
    config: AgentCheckpointConfig,
    max_entities_override: int | None,
) -> AgentCheckpointConfig:
    if max_entities_override is None:
        return config

    obs_spec = config.env.obs_spec
    if not isinstance(obs_spec, EntityBasedBaseConfig):
        raise TypeError(
            "max_entities_override requires entity-based obs_spec, "
            f"got {type(obs_spec).__name__}"
        )

    override_obs_spec = type(obs_spec).model_validate(
        {**obs_spec.model_dump(mode="python"), "max_entities": max_entities_override}
    )
    env = config.env.model_copy(
        update={
            "obs_spec": override_obs_spec,
        }
    )
    return config.model_copy(update={"env": env})


def apply_targeting_mode_override(
    config: AgentCheckpointConfig,
    targeting_mode_override: TargetingMode | None,
) -> AgentCheckpointConfig:
    if targeting_mode_override is None:
        return config

    action_spec = config.env.action_spec
    if action_spec.action_spec == "pure":
        print(
            "warning: targeting_mode_override is ignored for pure action_spec",
            flush=True,
        )
        return config

    override_action_spec = type(action_spec).model_validate(
        {
            **action_spec.model_dump(mode="python"),
            "targeting_mode": targeting_mode_override,
        }
    )
    env = config.env.model_copy(
        update={
            "action_spec": override_action_spec,
        }
    )
    return config.model_copy(update={"env": env})
