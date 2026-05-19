from __future__ import annotations

from typing import assert_never

from owl.model.base import BaseModelAPI
from owl.model.config import ModelConfig
from owl.model.recurrent_transformer_v1 import RecurrentTransformerV1
from owl.model.stateless_transformer_v1 import StatelessTransformerV1
from owl.rl import ActionConfig, ObsConfig


def create_model(
    config: ModelConfig,
    *,
    obs_spec: ObsConfig,
    action_spec: ActionConfig,
) -> BaseModelAPI:
    match config.model_arch:
        case "stateless_transformer_v1":
            return StatelessTransformerV1(
                config,
                obs_spec=obs_spec,
                action_spec=action_spec,
            )
        case "recurrent_transformer_v1":
            return RecurrentTransformerV1(
                config,
                obs_spec=obs_spec,
                action_spec=action_spec,
            )
        case _:
            assert_never(config)
