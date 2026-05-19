from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field

from owl.model.recurrent_transformer_v1 import RecurrentTransformerV1Config
from owl.model.stateless_transformer_v1 import StatelessTransformerV1Config

ModelConfig: TypeAlias = Annotated[
    StatelessTransformerV1Config | RecurrentTransformerV1Config,
    Field(discriminator="model_arch"),
]
