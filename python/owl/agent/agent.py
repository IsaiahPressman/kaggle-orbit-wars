from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal

import torch
from pydantic import ConfigDict, Field

from owl.config import BaseConfig
from owl.model import (
    BaseModelAPI,
    ModelConfig,
    ModelHiddenState,
    RecurrentTransformerV1Config,
    create_model,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    MAX_PLANETS,
    ActionBundle,
    ActionConfig,
    ActionDiscreteTargetBinsConfig,
    ActionDiscreteTargetsConfig,
    ActionMask,
    ActionPureConfig,
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
    encode_python_observation_with_metrics,
)

from .checkpoint_quantization import dequantize_model_state_dict
from .kaggle_observation import KaggleObservation

AGENT_CONFIG_PATH = Path(__file__).with_name("agent_config.yaml")
_OBS_REQUIRED_TENSOR_FIELDS = tuple(
    field
    for field in ObsBatch.model_fields
    if field
    not in {
        "action_mask",
        "player_features",
        "fleet_target",
        "target_incoming_features",
    }
)


@dataclass(frozen=True)
class CompactedObservation:
    obs: ObsBatch
    action_entity_indices: torch.Tensor


class AgentConfig(BaseConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    deterministic: bool
    max_entities_override: int | None = None
    targeting_mode_override: TargetingMode | None = None
    min_fleet_size: Literal["match"] | Annotated[int, Field(ge=1)] = "match"
    min_overage_time: float = Field(default=0.0, ge=0.0, le=60.0)
    fallback_min_overage_time: float | None = Field(default=None, ge=0.0, le=60.0)


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
        fallback_checkpoint_config_path: Path | None = None,
        fallback_checkpoint_path: Path | None = None,
    ) -> None:
        init_start = perf_counter()
        if (fallback_checkpoint_config_path is None) != (
            fallback_checkpoint_path is None
        ):
            raise ValueError(
                "fallback checkpoint config and checkpoint path must be provided "
                "together"
            )

        self.config = AgentConfig.from_file(AGENT_CONFIG_PATH)
        self._last_turn_value = float("nan")
        self._peak_total_ms = 0
        self._peak_entities = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_config, self.model = self._load_model(
            checkpoint_config_path=checkpoint_config_path,
            checkpoint_path=checkpoint_path,
        )
        self.hidden_state: ModelHiddenState | None = None

        self.fallback_checkpoint_config: AgentCheckpointConfig | None = None
        self.fallback_model: BaseModelAPI | None = None
        if fallback_checkpoint_config_path is not None:
            assert fallback_checkpoint_path is not None
            self.fallback_checkpoint_config, self.fallback_model = self._load_model(
                checkpoint_config_path=fallback_checkpoint_config_path,
                checkpoint_path=fallback_checkpoint_path,
                allow_recurrent=False,
            )
            if self.config.fallback_min_overage_time is None:
                print(
                    "warning: fallback model is packaged but "
                    "fallback_min_overage_time is null",
                    flush=True,
                )
        print(f"init_s={perf_counter() - init_start:.2f} - ", end="", flush=True)

    def _load_model(
        self,
        *,
        checkpoint_config_path: Path,
        checkpoint_path: Path,
        allow_recurrent: bool = True,
    ) -> tuple[AgentCheckpointConfig, BaseModelAPI]:
        if not checkpoint_config_path.is_file():
            raise ValueError(f"expected Kaggle config at {checkpoint_config_path}")

        if not checkpoint_path.is_file():
            raise ValueError(f"expected Kaggle checkpoint at {checkpoint_path}")

        checkpoint_config = AgentCheckpointConfig.from_file(checkpoint_config_path)
        checkpoint_config = apply_max_entities_override(
            checkpoint_config,
            self.config.max_entities_override,
        )
        checkpoint_config = apply_targeting_mode_override(
            checkpoint_config,
            self.config.targeting_mode_override,
        )
        if not allow_recurrent and isinstance(
            checkpoint_config.model, RecurrentTransformerV1Config
        ):
            raise ValueError("fallback model cannot be recurrent")

        model = create_model(
            checkpoint_config.model,
            obs_spec=checkpoint_config.env.obs_spec,
            action_spec=checkpoint_config.env.action_spec,
        ).to(self.device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(
                f"checkpoint must be a dictionary with key 'model': {checkpoint_path}"
            )

        model.load_state_dict(dequantize_model_state_dict(checkpoint["model"]))
        model.eval()
        return checkpoint_config, model

    @torch.inference_mode()
    def act(self, observation: Any) -> list[list[float]]:
        total_start = perf_counter()
        encode_start = total_start
        observation = KaggleObservation.model_validate(observation)
        if observation.remaining_overage_time < self.config.min_overage_time:
            return []

        model = self.model
        checkpoint_config = self.checkpoint_config
        hidden_state = self.hidden_state
        use_fallback = self._should_use_fallback(observation.remaining_overage_time)
        if use_fallback:
            assert self.fallback_model is not None
            assert self.fallback_checkpoint_config is not None
            model = self.fallback_model
            checkpoint_config = self.fallback_checkpoint_config
            hidden_state = None

        min_fleet_size = _resolve_min_fleet_size(
            self.config,
            checkpoint_config.env.action_spec,
        )
        obs_dict = observation.to_rl_observation()
        encoded = encode_python_observation_with_metrics(
            obs_dict,
            obs_spec=checkpoint_config.env.obs_spec,
            action_spec=checkpoint_config.env.action_spec,
            fleet_filter_min_size=min_fleet_size,
        )
        obs = encoded.obs
        compacted = compact_entities(
            obs,
            compact_planets=_should_compact_planets(checkpoint_config.model),
        )
        obs = compacted.obs
        device_obs = self._obs_to_device(obs)
        self._synchronize_device()
        encode_ms = _elapsed_ms(encode_start)

        inference_start = perf_counter()
        if not use_fallback and (observation.step == 0 or hidden_state is None):
            hidden_state = model.initial_hidden_state(1, device=self.device)
        output = model.serve(
            device_obs,
            deterministic=self.config.deterministic,
            hidden_state=hidden_state,
        )
        if not use_fallback:
            self.hidden_state = output.next_hidden_state
        self._synchronize_device()
        values = output.values.detach().cpu()[0]
        self_value = float(values[observation.player].item())
        if observation.step == 0:
            n_players = _observation_player_count(observation)
            self._last_turn_value = (2.0 - n_players) / n_players

        advantage = self_value - self._last_turn_value
        inference_ms = _elapsed_ms(inference_start)

        conversion_start = perf_counter()
        actions_cpu = expand_actions_to_full_action_slots(
            _model_actions_to_cpu(output.actions),
            compacted.action_entity_indices,
            action_spec=checkpoint_config.env.action_spec,
        )
        actions = actions_to_kaggle(
            obs_dict,
            observation.player,
            actions_cpu,
            action_spec=checkpoint_config.env.action_spec,
        )
        conversion_ms = _elapsed_ms(conversion_start)
        total_ms = _elapsed_ms(total_start)
        entity_count = obs.entity_mask.shape[1]
        peak_total_ms, peak_entities = self._update_peak_metrics(
            step=observation.step,
            total_ms=total_ms,
            entity_count=entity_count,
        )

        self.log(
            step=observation.step,
            total_ms=total_ms,
            peak_total_ms=peak_total_ms,
            encode_ms=encode_ms,
            inference_ms=inference_ms,
            conversion_ms=conversion_ms,
            self_value=self_value,
            advantage=advantage,
            player_values=[float(value) for value in values.tolist()],
            entity_count=entity_count,
            peak_entities=peak_entities,
            filtered_fleets=encoded.filtered_fleets,
            remaining_overage_time=observation.remaining_overage_time,
            fallback_triggered=use_fallback,
        )
        self._last_turn_value = self_value
        return actions

    def _should_use_fallback(self, remaining_overage_time: float) -> bool:
        return (
            self.fallback_model is not None
            and self.config.fallback_min_overage_time is not None
            and remaining_overage_time < self.config.fallback_min_overage_time
        )

    def _obs_to_device(self, obs: ObsBatch) -> ObsBatch:
        return ObsBatch(
            **{
                field: getattr(obs, field).to(device=self.device)
                for field in _OBS_REQUIRED_TENSOR_FIELDS
            },
            action_mask=_action_mask_to_device(obs.action_mask, self.device),
            player_features=(
                None
                if obs.player_features is None
                else obs.player_features.to(device=self.device)
            ),
            fleet_target=(
                None
                if obs.fleet_target is None
                else obs.fleet_target.to(device=self.device)
            ),
            target_incoming_features=(
                None
                if obs.target_incoming_features is None
                else obs.target_incoming_features.to(device=self.device)
            ),
        )

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def log(
        self,
        *,
        step: int,
        total_ms: int,
        peak_total_ms: int,
        encode_ms: int,
        inference_ms: int,
        conversion_ms: int,
        self_value: float,
        advantage: float,
        player_values: list[float],
        entity_count: int,
        peak_entities: int,
        filtered_fleets: int,
        remaining_overage_time: float,
        fallback_triggered: bool,
    ) -> None:
        values = ",".join(f"{value:.3f}" for value in player_values)
        prefix = "fallback triggered - " if fallback_triggered else ""
        print(
            f"{prefix}"
            f"step={step} - "
            f"total_ms={total_ms} - "
            f"peak_total_ms={peak_total_ms} - "
            f"encode_ms={encode_ms} - "
            f"inference_ms={inference_ms} - "
            f"conversion_ms={conversion_ms} - "
            f"value_self={self_value:.3f} - "
            f"advantage={advantage:.3f} - "
            f"values=[{values}] - "
            f"entities={entity_count} - "
            f"peak_entities={peak_entities} - "
            f"filtered_fleets={filtered_fleets} - "
            f"remaining_overage_s={remaining_overage_time:.1f}",
            flush=True,
        )

    def _update_peak_metrics(
        self,
        *,
        step: int,
        total_ms: int,
        entity_count: int,
    ) -> tuple[int, int]:
        peak_total_ms = self._peak_total_ms
        if step != 0:
            peak_total_ms = max(peak_total_ms, total_ms)
        self._peak_total_ms = peak_total_ms

        peak_entities = max(self._peak_entities, entity_count)
        self._peak_entities = peak_entities
        return peak_total_ms, peak_entities


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def _resolve_min_fleet_size(config: AgentConfig, action_spec: ActionConfig) -> int:
    if config.min_fleet_size == "match":
        return action_spec.min_fleet_size
    return config.min_fleet_size


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


def compact_entities(
    obs: ObsBatch, *, compact_planets: bool = True
) -> CompactedObservation:
    """Drop inactive entity rows from a single-row observation batch."""
    batch_size = obs.entity_mask.shape[0]
    if batch_size != 1:
        raise ValueError(
            f"runtime entity compaction requires batch size 1, got {batch_size}"
        )

    planet_mask = obs.entity_mask[0, :MAX_PLANETS]
    comet_mask = obs.entity_mask[0, MAX_PLANETS:ACTION_ENTITY_SLOTS]
    fleet_mask = obs.entity_mask[0, ACTION_ENTITY_SLOTS:]
    active_planet_indexes = (
        torch.nonzero(planet_mask, as_tuple=True)[0]
        if compact_planets
        else torch.arange(MAX_PLANETS, device=obs.entity_mask.device)
    )
    active_comet_indexes = torch.nonzero(comet_mask, as_tuple=True)[0]
    active_fleet_indexes = torch.nonzero(fleet_mask, as_tuple=True)[0]
    action_entity_indices = torch.cat(
        (
            active_planet_indexes,
            active_comet_indexes + MAX_PLANETS,
        )
    )
    if action_entity_indices.numel() == 0:
        raise ValueError(
            "runtime entity compaction requires at least one action entity"
        )
    compact_fleet_target = None
    if obs.fleet_target is not None:
        selected_targets = obs.fleet_target[:, active_fleet_indexes]
        target_remap = torch.full(
            (ACTION_ENTITY_SLOTS,),
            -1,
            dtype=obs.fleet_target.dtype,
            device=obs.fleet_target.device,
        )
        target_remap[action_entity_indices] = torch.arange(
            action_entity_indices.numel(),
            dtype=obs.fleet_target.dtype,
            device=obs.fleet_target.device,
        )
        compact_fleet_target = torch.full_like(selected_targets, -1)
        valid_targets = selected_targets >= 0
        compact_fleet_target[valid_targets] = target_remap[
            selected_targets[valid_targets]
        ]

    compacted = ObsBatch(
        planets=obs.planets[:, active_planet_indexes, :],
        orbiting_planets=obs.orbiting_planets[:, active_planet_indexes],
        fleets=obs.fleets[:, active_fleet_indexes, :],
        fleet_target=compact_fleet_target,
        target_incoming_features=(
            None
            if obs.target_incoming_features is None
            else obs.target_incoming_features[:, action_entity_indices, :]
        ),
        comets=obs.comets[:, active_comet_indexes, :],
        entity_mask=torch.cat(
            (
                obs.entity_mask[:, :MAX_PLANETS][:, active_planet_indexes],
                obs.entity_mask[:, MAX_PLANETS:ACTION_ENTITY_SLOTS][
                    :, active_comet_indexes
                ],
                obs.entity_mask[:, ACTION_ENTITY_SLOTS:][:, active_fleet_indexes],
            ),
            dim=1,
        ),
        still_playing=obs.still_playing,
        global_features=obs.global_features,
        action_mask=_compact_action_mask(obs.action_mask, action_entity_indices),
        player_features=obs.player_features,
    )
    return CompactedObservation(
        obs=compacted,
        action_entity_indices=action_entity_indices,
    )


def _should_compact_planets(model_config: ModelConfig) -> bool:
    # Recurrent include-planets checkpoints index the fixed planet token prefix
    # directly, so compacting inactive planet rows changes the recurrent layout.
    return not (
        isinstance(model_config, RecurrentTransformerV1Config)
        and model_config.recurrence_mode == "include_planets"
    )


def _compact_action_mask(
    action_mask: ActionMask,
    action_entity_indices: torch.Tensor,
) -> ActionMask:
    if isinstance(action_mask, PureActionMask):
        return PureActionMask(
            can_act=action_mask.can_act.index_select(2, action_entity_indices),
            max_launch=action_mask.max_launch.index_select(2, action_entity_indices),
        )
    if isinstance(action_mask, DiscreteTargetActionMask):
        can_act = action_mask.can_act.index_select(2, action_entity_indices)
        can_act = can_act.index_select(3, action_entity_indices)
        return DiscreteTargetActionMask(
            can_act=can_act,
            max_launch=action_mask.max_launch.index_select(2, action_entity_indices),
        )
    can_act = action_mask.can_act.index_select(2, action_entity_indices)
    can_act = can_act.index_select(3, action_entity_indices)
    return DiscreteTargetBinActionMask(can_act=can_act)


def expand_actions_to_full_action_slots(
    actions: ActionBundle,
    action_entity_indices: torch.Tensor,
    *,
    action_spec: ActionConfig,
) -> ActionBundle:
    if action_entity_indices.numel() == ACTION_ENTITY_SLOTS and torch.equal(
        action_entity_indices,
        torch.arange(ACTION_ENTITY_SLOTS, device=action_entity_indices.device),
    ):
        return actions

    if isinstance(actions, PureActions):
        if not isinstance(action_spec, ActionPureConfig):
            raise TypeError("pure actions require pure action_spec")
        return PureActions(
            launch=_expand_action_tensor(actions.launch, action_entity_indices),
            angle=_expand_action_tensor(actions.angle, action_entity_indices),
            ships=_expand_action_tensor(actions.ships, action_entity_indices),
        )
    if isinstance(actions, DiscreteTargetActions):
        if not isinstance(action_spec, ActionDiscreteTargetsConfig):
            raise TypeError(
                "discrete-target actions require discrete-target action_spec"
            )
        return DiscreteTargetActions(
            launch=_expand_action_tensor(actions.launch, action_entity_indices),
            target=_expand_action_tensor(
                _remap_compact_targets(actions.target, action_entity_indices),
                action_entity_indices,
            ),
            ships=_expand_action_tensor(actions.ships, action_entity_indices),
        )
    if not isinstance(action_spec, ActionDiscreteTargetBinsConfig):
        raise TypeError("target-bin actions require target-bin action_spec")
    return DiscreteTargetBinActions(
        target=_expand_action_tensor(
            _remap_compact_targets(actions.target, action_entity_indices),
            action_entity_indices,
        ),
        fleet_bin=_expand_action_tensor(actions.fleet_bin, action_entity_indices),
    )


def _expand_action_tensor(
    tensor: torch.Tensor,
    action_entity_indices: torch.Tensor,
) -> torch.Tensor:
    full_shape = (*tensor.shape[:2], ACTION_ENTITY_SLOTS, *tensor.shape[3:])
    expanded = tensor.new_zeros(full_shape)
    return expanded.index_copy(
        2,
        action_entity_indices.to(device=tensor.device),
        tensor,
    )


def _remap_compact_targets(
    target: torch.Tensor,
    action_entity_indices: torch.Tensor,
) -> torch.Tensor:
    target_in_range = target.ge(0) & target.lt(action_entity_indices.numel())
    if not target_in_range.all().item():
        raise ValueError("compact target indices must reference compact action slots")
    return action_entity_indices.to(device=target.device)[target]


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
