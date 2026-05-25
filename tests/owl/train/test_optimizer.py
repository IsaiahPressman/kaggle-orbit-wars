import pytest
import torch
from owl.model import (
    BaseModelAPI,
    InputLayer,
    ModelActions,
    ModelEvaluation,
    ModelOutput,
)
from owl.rl import ObsBatch
from owl.train import (
    AdamConfig,
    AdamWConfig,
    CosineLRScheduleConfig,
    LinearWarmupCosineDecayLRScheduleConfig,
    LRScheduleConfig,
    MuonConfig,
    create_lr_scheduler,
    create_optimizer,
)
from owl.train.optimizer import CompositeOptimizer, lr_multiplier
from pydantic import TypeAdapter
from torch import nn


class OptimizerTestModel(BaseModelAPI):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(3, 4)
        self.hidden = nn.Linear(4, 4)
        self.output = nn.Linear(4, 2)
        self.norm = nn.LayerNorm(4)
        self.token_state = nn.Parameter(torch.zeros(4, 4))

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

    def compute_value(self, obs: ObsBatch) -> torch.Tensor:
        raise NotImplementedError

    def reset_parameters(self) -> None:
        self.input.reset_parameters()
        self.hidden.reset_parameters()
        self.output.reset_parameters()
        self.norm.reset_parameters()
        nn.init.zeros_(self.token_state)

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return (self.input, self.token_state)

    def get_output_layers(self) -> tuple[nn.Module, ...]:
        return (self.output,)


def test_create_optimizer_supports_adam_adamw_and_muon() -> None:
    adam_model = OptimizerTestModel()
    adamw_model = OptimizerTestModel()
    muon_model = OptimizerTestModel()

    adam = create_optimizer(
        adam_model,
        AdamConfig(
            learning_rate=3e-4,
            betas=(0.85, 0.95),
            eps=1e-6,
            weight_decay=0.01,
        ),
    )
    adamw = create_optimizer(
        adamw_model,
        AdamWConfig(learning_rate=3e-4, betas=(0.8, 0.9), weight_decay=0.01),
    )
    muon = create_optimizer(
        muon_model,
        MuonConfig(adamw_lr=3e-4, muon_lr=0.02),
    )

    assert isinstance(adam, torch.optim.Adam)
    assert adam.defaults["lr"] == pytest.approx(3e-4)
    assert adam.defaults["betas"] == pytest.approx((0.85, 0.95))
    assert adam.defaults["eps"] == pytest.approx(1e-6)
    assert adam.defaults["weight_decay"] == pytest.approx(0.01)
    assert adam.defaults["fused"] is True
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.defaults["lr"] == pytest.approx(3e-4)
    assert adamw.defaults["betas"] == pytest.approx((0.8, 0.9))
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
    assert id(muon_model.token_state) not in muon_param_ids


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
    config = LinearWarmupCosineDecayLRScheduleConfig(
        warmup_steps=2,
        decay_steps=4,
        lr_min_ratio=0.1,
    )

    assert lr_multiplier(config, 0) == pytest.approx(0.0)
    assert lr_multiplier(config, 1) == pytest.approx(0.5)
    assert lr_multiplier(config, 2) == pytest.approx(1.0)
    assert lr_multiplier(config, 6) == pytest.approx(0.1)
    assert lr_multiplier(config, 20) == pytest.approx(0.1)


def test_lr_multiplier_cosine_oscillates_between_bounds() -> None:
    config = CosineLRScheduleConfig(full_cycle_steps=4, lr_min_ratio=0.2)

    assert lr_multiplier(config, 0) == pytest.approx(1.0)
    assert lr_multiplier(config, 1) == pytest.approx(0.6)
    assert lr_multiplier(config, 2) == pytest.approx(0.2)
    assert lr_multiplier(config, 3) == pytest.approx(0.6)
    assert lr_multiplier(config, 4) == pytest.approx(1.0)
    assert lr_multiplier(config, 6) == pytest.approx(0.2)


def test_lr_multiplier_cosine_config_validation() -> None:
    config = CosineLRScheduleConfig(full_cycle_steps=2, lr_min_ratio=0.0)
    assert lr_multiplier(config, 1) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="less than 1"):
        CosineLRScheduleConfig(lr_min_ratio=1.0)
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        CosineLRScheduleConfig(full_cycle_steps=1)


def test_lr_schedule_config_discriminates_allowed_fields() -> None:
    adapter = TypeAdapter(LRScheduleConfig)

    cosine_config = adapter.validate_python(
        {"schedule": "cosine", "full_cycle_steps": 4, "lr_min_ratio": 0.0}
    )
    assert isinstance(cosine_config, CosineLRScheduleConfig)

    linear_config = adapter.validate_python(
        {
            "schedule": "linear_warmup_cosine_decay",
            "warmup_steps": 2,
            "decay_steps": 4,
            "lr_min_ratio": 0.1,
        }
    )
    assert isinstance(linear_config, LinearWarmupCosineDecayLRScheduleConfig)

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        adapter.validate_python({"schedule": "cosine", "warmup_steps": 2})
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        adapter.validate_python(
            {"schedule": "linear_warmup_cosine_decay", "full_cycle_steps": 4}
        )


def test_lr_scheduler_scales_optimizer_param_groups() -> None:
    model = OptimizerTestModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = create_lr_scheduler(
        optimizer,
        LinearWarmupCosineDecayLRScheduleConfig(
            warmup_steps=2,
            decay_steps=2,
            lr_min_ratio=0.025,
        ),
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
    assert scheduler.get_last_lr() == pytest.approx([0.005125])
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.00025])
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.00025])


def test_composite_lr_scheduler_round_trips_nested_state_dict() -> None:
    model = OptimizerTestModel()
    optimizer = create_optimizer(
        model,
        MuonConfig(
            adamw_lr=0.01,
            muon_lr=0.02,
            lr_schedule=LinearWarmupCosineDecayLRScheduleConfig(
                warmup_steps=1,
                decay_steps=2,
            ),
        ),
    )
    scheduler = create_lr_scheduler(
        optimizer,
        LinearWarmupCosineDecayLRScheduleConfig(warmup_steps=1, decay_steps=2),
    )
    assert scheduler is not None

    optimizer.step()
    scheduler.step()
    state = scheduler.state_dict()
    replacement = create_lr_scheduler(
        optimizer,
        LinearWarmupCosineDecayLRScheduleConfig(warmup_steps=1, decay_steps=2),
    )
    assert replacement is not None
    replacement.load_state_dict(state)

    assert replacement.state_dict()["schedulers"]
