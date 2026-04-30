import pytest
import torch
from owl.model import BaseModelAPI, ModelActions, ModelEvaluation, ModelOutput
from owl.rl import ObsBatch
from owl.train import (
    AdamWConfig,
    CompositeOptimizer,
    LRScheduleConfig,
    MuonConfig,
    create_lr_scheduler,
    create_optimizer,
    lr_multiplier,
)
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

    adamw = create_optimizer(
        adamw_model,
        AdamWConfig(learning_rate=3e-4, weight_decay=0.01),
    )
    muon = create_optimizer(
        muon_model,
        MuonConfig(adamw_lr=3e-4, muon_lr=0.02),
    )

    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.defaults["lr"] == pytest.approx(3e-4)
    assert adamw.defaults["weight_decay"] == pytest.approx(0.01)
    assert isinstance(muon, CompositeOptimizer)
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
    optimizer = create_optimizer(
        model,
        MuonConfig(adamw_lr=3e-4, muon_lr=0.02),
    )
    assert isinstance(optimizer, CompositeOptimizer)

    loss = sum(param.square().sum() for param in model.parameters())
    loss.backward()
    optimizer.step()
    state = optimizer.state_dict()

    replacement = create_optimizer(
        model,
        MuonConfig(adamw_lr=3e-4, muon_lr=0.02),
    )
    assert isinstance(replacement, CompositeOptimizer)
    replacement.load_state_dict(state)

    assert replacement.state_dict()["optimizers"]


def test_create_optimizer_rejects_unknown_optimizer_contract() -> None:
    model = OptimizerTestModel()
    config = AdamWConfig.model_construct(optimizer="sgd")

    with pytest.raises(AssertionError, match="Expected code to be unreachable"):
        create_optimizer(model, config)


def test_lr_multiplier_linear_warmup_then_cosine_decay() -> None:
    config = LRScheduleConfig(warmup_steps=2, decay_steps=4, lr_min_ratio=0.1)

    assert lr_multiplier(config, 0) == pytest.approx(0.0)
    assert lr_multiplier(config, 1) == pytest.approx(0.5)
    assert lr_multiplier(config, 2) == pytest.approx(1.0)
    assert lr_multiplier(config, 6) == pytest.approx(0.1)
    assert lr_multiplier(config, 20) == pytest.approx(0.1)


def test_lr_scheduler_scales_optimizer_param_groups() -> None:
    model = OptimizerTestModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = create_lr_scheduler(
        optimizer,
        LRScheduleConfig(warmup_steps=2, decay_steps=2, lr_min_ratio=0.25),
    )
    assert scheduler is not None
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert scheduler.get_last_lr() == pytest.approx([0.0])

    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.005])
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.01])
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.00625])


def test_composite_lr_scheduler_round_trips_nested_state_dict() -> None:
    model = OptimizerTestModel()
    optimizer = create_optimizer(
        model,
        MuonConfig(
            adamw_lr=0.01,
            muon_lr=0.02,
            lr_schedule=LRScheduleConfig(warmup_steps=1, decay_steps=2),
        ),
    )
    scheduler = create_lr_scheduler(
        optimizer,
        LRScheduleConfig(warmup_steps=1, decay_steps=2),
    )
    assert scheduler is not None

    optimizer.step()
    scheduler.step()
    state = scheduler.state_dict()
    replacement = create_lr_scheduler(
        optimizer,
        LRScheduleConfig(warmup_steps=1, decay_steps=2),
    )
    assert replacement is not None
    replacement.load_state_dict(state)

    assert replacement.state_dict()["schedulers"]
