"""Run the preregistered G14 tied-versus-decoupled gate-law screen."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

Arm: TypeAlias = Literal["gated_delta_v1", "gated_delta_v2"]
SEEDS = (14_001, 14_002, 14_003)
VALUE_DIM = 8


class TiedGateController(nn.Module):
    """GDN-v1 scalar beta used for both erasure and value injection."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(VALUE_DIM + 1, 1)

    def forward(
        self, controls: torch.Tensor, values: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta = torch.sigmoid(self.gate(controls))
        return (1.0 - beta) * state + beta * values, beta, beta


class DecoupledGateController(nn.Module):
    """GDN2 scalar address erasure and channel-wise value writing."""

    def __init__(self) -> None:
        super().__init__()
        self.erase = nn.Linear(VALUE_DIM + 1, 1)
        self.write = nn.Linear(VALUE_DIM + 1, VALUE_DIM)

    def forward(
        self, controls: torch.Tensor, values: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        erase = torch.sigmoid(self.erase(controls))
        write = torch.sigmoid(self.write(controls))
        return (1.0 - erase) * state + write * values, erase, write


@dataclass(frozen=True)
class ScreenConfig:
    updates: int = 600
    batch_size: int = 128
    learning_rate: float = 3e-2
    evaluation_batches: int = 32


def _batch(
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.rand(batch_size, VALUE_DIM, generator=generator).argsort(dim=-1)
    features = order[:, :length].to(device)
    values = F.one_hot(features, num_classes=VALUE_DIM).to(torch.float32)
    target = values.sum(dim=1)
    return values, target


def _rollout(
    model: nn.Module, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = values.new_zeros(values.shape[0], VALUE_DIM)
    erases = []
    writes = []
    denominator = max(values.shape[1] - 1, 1)
    for position in range(values.shape[1]):
        position_feature = values.new_full((values.shape[0], 1), position / denominator)
        controls = torch.cat((values[:, position], position_feature), dim=-1)
        state, erase, write = model(controls, values[:, position], state)
        erases.append(erase)
        writes.append(write)
    return state, torch.stack(erases, dim=1), torch.stack(writes, dim=1)


def run_arm(
    arm: Arm,
    seed: int,
    config: ScreenConfig,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model: nn.Module
    if arm == "gated_delta_v1":
        model = TiedGateController()
    elif arm == "gated_delta_v2":
        model = DecoupledGateController()
    else:
        raise ValueError(f"unknown arm: {arm}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    losses = []
    finite_gradients = True
    for update in range(config.updates):
        length = 2 + update % 6
        values, target = _batch(
            config.batch_size,
            length,
            generator=generator,
            device=device,
        )
        state, _, _ = _rollout(model, values)
        loss = F.binary_cross_entropy(state.clamp(1e-6, 1.0 - 1e-6), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        finite_gradients = finite_gradients and bool(
            gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        )
        optimizer.step()
        losses.append(float(loss.detach()))

    squared_error = 0.0
    correct = 0
    bits = 0
    erase_sum = 0.0
    erase_count = 0
    write_sum = 0.0
    write_count = 0
    with torch.no_grad():
        for _ in range(config.evaluation_batches):
            values, target = _batch(
                config.batch_size,
                VALUE_DIM,
                generator=generator,
                device=device,
            )
            state, erase, write = _rollout(model, values)
            squared_error += float((state - target).square().sum())
            correct += int(((state >= 0.5) == target.bool()).sum())
            bits += target.numel()
            erase_sum += float(erase.sum())
            erase_count += erase.numel()
            active_write = (write * values).sum(dim=-1)
            write_sum += float(active_write.sum())
            write_count += active_write.numel()
    mse = squared_error / bits
    return {
        "arm": arm,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_training_loss": losses[0],
        "final_training_loss": losses[-1],
        "finite_losses": all(torch.isfinite(torch.tensor(losses))),
        "finite_gradients": finite_gradients,
        "held_out_length": VALUE_DIM,
        "held_out_state_mse": mse,
        "held_out_bit_accuracy": correct / bits,
        "mean_erase_gate": erase_sum / erase_count,
        "mean_active_write_gate": write_sum / write_count,
    }


def run_screen(config: ScreenConfig, device: torch.device) -> dict[str, object]:
    rows = [
        run_arm(arm, seed, config, device)
        for seed in SEEDS
        for arm in ("gated_delta_v1", "gated_delta_v2")
    ]
    by_seed = {
        seed: {str(row["arm"]): row for row in rows if int(row["seed"]) == seed}
        for seed in SEEDS
    }
    finite = all(
        bool(row["finite_losses"]) and bool(row["finite_gradients"]) for row in rows
    )
    v2_rows = [row for row in rows if row["arm"] == "gated_delta_v2"]
    gates = {
        "finite_all_runs": finite,
        "v2_mse_at_most_0_01_all_seeds": all(
            float(row["held_out_state_mse"]) <= 0.01 for row in v2_rows
        ),
        "v2_bit_accuracy_at_least_0_99_all_seeds": all(
            float(row["held_out_bit_accuracy"]) >= 0.99 for row in v2_rows
        ),
        "v2_beats_v1_mse_all_paired_seeds": all(
            float(by_seed[seed]["gated_delta_v2"]["held_out_state_mse"])
            < float(by_seed[seed]["gated_delta_v1"]["held_out_state_mse"])
            for seed in SEEDS
        ),
        "semantic_contract_tests_required": True,
    }
    return {
        "schema_version": 1,
        "screen": "g14_decoupled_erase_write_gate_law",
        "claim_boundary": (
            "controlled representational and optimization screen only; no "
            "natural-text, long-context, throughput, or model-quality claim"
        ),
        "runtime": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
        "config": {
            "seeds": list(SEEDS),
            "value_dim": VALUE_DIM,
            **config.__dict__,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
        },
        "rows": rows,
        "gates": gates,
        "mechanism_gate_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    report = run_screen(
        ScreenConfig(
            updates=arguments.updates,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            evaluation_batches=arguments.evaluation_batches,
        ),
        device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
