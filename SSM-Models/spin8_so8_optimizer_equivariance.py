"""Causal audit of optimizer equivariance across equivalent SO(8) charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from compare_recurrences import GROUPS, make_group_batches
from mechanistic_group_actions import PureGroupActionModel
from q8_spinor_center_experiment import INPUT_ELEMENTS
from spin8_triality import so8_chart_equivalence_diagnostics


def _paired_models(seed: int, *, dtype: torch.dtype) -> tuple[
    PureGroupActionModel, PureGroupActionModel, torch.Tensor
]:
    report = so8_chart_equivalence_diagnostics(seed)
    mapping = torch.tensor(report["coefficient_map"], dtype=dtype)
    torch.manual_seed(seed)
    positive = PureGroupActionModel(
        len(INPUT_ELEMENTS), 8, family="pure_spin8_positive", channels=2
    ).to(dtype=dtype)
    torch.manual_seed(seed)
    generic = PureGroupActionModel(
        len(INPUT_ELEMENTS), 8, family="pure_so8_exponential", channels=2
    ).to(dtype=dtype)
    with torch.no_grad():
        generic.action_parameters.copy_(positive.action_parameters @ mapping)
    return positive, generic, mapping


def _nonaction_parameter_error(
    positive: PureGroupActionModel, generic: PureGroupActionModel
) -> float:
    first = dict(positive.named_parameters())
    second = dict(generic.named_parameters())
    return max(
        float((first[name] - second[name]).detach().abs().max())
        for name in first
        if name != "action_parameters"
    )


def run_optimizer_pair(
    optimizer_name: str,
    *,
    seed: int = 20260803,
    steps: int = 12,
    batch_size: int = 64,
) -> dict[str, object]:
    """Train chart-related models on identical batches and measure divergence."""

    if optimizer_name not in ("sgd", "adamw"):
        raise ValueError("optimizer_name must be 'sgd' or 'adamw'")
    positive, generic, mapping = _paired_models(seed, dtype=torch.float64)
    optimizer_type = torch.optim.SGD if optimizer_name == "sgd" else torch.optim.AdamW
    kwargs = {"lr": 3e-3, "weight_decay": 0.0}
    positive_optimizer = optimizer_type(positive.parameters(), **kwargs)
    generic_optimizer = optimizer_type(generic.parameters(), **kwargs)
    batches = make_group_batches(
        GROUPS["q8"],
        steps,
        batch_size,
        16,
        seed + 10_000,
        input_elements=INPUT_ELEMENTS,
    )
    trajectory = []
    for step, (tokens, targets) in enumerate(batches, start=1):
        endpoint = targets[:, -1]
        positive_optimizer.zero_grad(set_to_none=True)
        generic_optimizer.zero_grad(set_to_none=True)
        positive_logits = positive(tokens)[:, -1]
        generic_logits = generic(tokens)[:, -1]
        positive_loss = nn.functional.cross_entropy(positive_logits, endpoint)
        generic_loss = nn.functional.cross_entropy(generic_logits, endpoint)
        positive_loss.backward()
        generic_loss.backward()
        mapped_gradient = positive.action_parameters.grad @ mapping
        gradient_covariance_error = float(
            (generic.action_parameters.grad - mapped_gradient).abs().max()
        )
        positive_optimizer.step()
        generic_optimizer.step()
        mapped_parameters = positive.action_parameters @ mapping
        coefficient_error = float(
            (generic.action_parameters - mapped_parameters).detach().abs().max()
        )
        action_error = float(
            (positive.action_matrices() - generic.action_matrices())
            .detach()
            .abs()
            .max()
        )
        with torch.no_grad():
            post_positive = positive(tokens)[:, -1]
            post_generic = generic(tokens)[:, -1]
        trajectory.append(
            {
                "step": step,
                "preupdate_logit_max_abs_error": float(
                    (positive_logits - generic_logits).detach().abs().max()
                ),
                "gradient_covariance_max_abs_error": gradient_covariance_error,
                "coefficient_map_max_abs_error": coefficient_error,
                "action_max_abs_error": action_error,
                "postupdate_logit_max_abs_error": float(
                    (post_positive - post_generic).abs().max()
                ),
                "nonaction_parameter_max_abs_error": _nonaction_parameter_error(
                    positive, generic
                ),
            }
        )
    keys = (
        "preupdate_logit_max_abs_error",
        "gradient_covariance_max_abs_error",
        "coefficient_map_max_abs_error",
        "action_max_abs_error",
        "postupdate_logit_max_abs_error",
        "nonaction_parameter_max_abs_error",
    )
    return {
        "optimizer": optimizer_name,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "maxima": {
            key: max(float(row[key]) for row in trajectory) for key in keys
        },
        "trajectory": trajectory,
    }


def optimizer_equivariance_audit(seed: int = 20260803) -> dict[str, object]:
    sgd = run_optimizer_pair("sgd", seed=seed)
    adamw = run_optimizer_pair("adamw", seed=seed)
    checks = {
        "sgd_preserves_coefficient_map": (
            sgd["maxima"]["coefficient_map_max_abs_error"] <= 1e-10
        ),
        "sgd_preserves_actions": sgd["maxima"]["action_max_abs_error"] <= 1e-10,
        "sgd_preserves_logits": (
            sgd["maxima"]["postupdate_logit_max_abs_error"] <= 1e-10
        ),
        "adamw_breaks_coefficient_map": (
            adamw["maxima"]["coefficient_map_max_abs_error"] >= 1e-4
        ),
        "adamw_breaks_functional_equivalence": (
            adamw["maxima"]["postupdate_logit_max_abs_error"] >= 1e-4
        ),
        "initial_adamw_gradient_is_covariant": (
            adamw["trajectory"][0]["gradient_covariance_max_abs_error"] <= 1e-10
        ),
    }
    return {
        "experiment": "optimizer equivariance across exact Spin8/SO8 charts",
        "seed": seed,
        "sgd": sgd,
        "adamw": adamw,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    report = optimizer_equivariance_audit(args.seed)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
