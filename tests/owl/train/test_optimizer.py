import pytest
import torch
from owl.model import BaseModelAPI, ModelActions, ModelEvaluation, ModelOutput
from owl.rl import ObsBatch
from owl.train import CompositeOptimizer, PPOConfig, ppo
from torch import nn


class OptimizerTestModel(BaseModelAPI):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(3, 4)
        self.hidden = nn.Linear(4, 4)
        self.output = nn.Linear(4, 2)
        self.norm = nn.LayerNorm(4)

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
    ) -> ModelOutput:
        raise NotImplementedError

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ModelActions,
    ) -> ModelEvaluation:
        raise NotImplementedError

    def get_input_layers(self) -> tuple[nn.Module, ...]:
        return (self.input,)

    def get_output_layers(self) -> tuple[nn.Module, ...]:
        return (self.output,)


def test_create_optimizer_supports_adamw_and_muon() -> None:
    adamw_model = OptimizerTestModel()
    muon_model = OptimizerTestModel()

    adamw = ppo.create_optimizer(
        adamw_model,
        PPOConfig(optimizer="adamw", learning_rate=3e-4, weight_decay=0.01),
    )
    muon = ppo.create_optimizer(
        muon_model,
        PPOConfig(optimizer="muon", learning_rate=3e-4, muon_lr=0.02),
    )

    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.defaults["lr"] == pytest.approx(3e-4)
    assert adamw.defaults["weight_decay"] == pytest.approx(0.01)
    assert isinstance(muon, CompositeOptimizer)
    assert isinstance(muon, torch.optim.Optimizer)
    assert any(isinstance(inner, torch.optim.Muon) for inner in muon.optimizers)
    assert any(isinstance(inner, torch.optim.AdamW) for inner in muon.optimizers)
    muon_param_ids = {
        id(param)
        for inner in muon.optimizers
        if isinstance(inner, torch.optim.Muon)
        for group in inner.param_groups
        for param in group["params"]
    }
    assert id(muon_model.hidden.weight) in muon_param_ids
    assert id(muon_model.input.weight) not in muon_param_ids
    assert id(muon_model.output.weight) not in muon_param_ids


def test_composite_optimizer_round_trips_nested_state_dict() -> None:
    model = OptimizerTestModel()
    optimizer = ppo.create_optimizer(
        model,
        PPOConfig(optimizer="muon", learning_rate=3e-4, muon_lr=0.02),
    )
    assert isinstance(optimizer, CompositeOptimizer)

    loss = sum(param.square().sum() for param in model.parameters())
    loss.backward()
    optimizer.step()
    state = optimizer.state_dict()

    replacement = ppo.create_optimizer(
        model,
        PPOConfig(optimizer="muon", learning_rate=3e-4, muon_lr=0.02),
    )
    assert isinstance(replacement, CompositeOptimizer)
    replacement.load_state_dict(state)

    assert replacement.state_dict()["optimizers"]


def test_create_optimizer_rejects_unknown_optimizer_contract() -> None:
    model = OptimizerTestModel()
    config = PPOConfig.model_construct(optimizer="sgd")

    with pytest.raises(AssertionError, match="Expected code to be unreachable"):
        ppo.create_optimizer(model, config)
