"""Controlled equivariant-identification benchmark for Intertwiner SchurScan.

The task isolates the bilinear drive in a triangular recurrence.  A structured
row learns the scalar multiplying a known one-dimensional intertwiner family;
unrestricted and group-augmented bilinear fits provide deterministic least-
squares controls.  Spin(8) triality and the SO(3) cross product are evaluated
under the same protocol.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from intertwiner_schurscan import (
    recurrent_intertwiner_scan,
    so3_cross_product_tensor,
    staged_intertwiner_scan,
)
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    spin8_actions,
    torch_triality_generators,
)
from spin8_triality_lift import triality_tensor

DTYPE = torch.float64
DECAY = 0.995


@dataclass(frozen=True)
class Instance:
    name: str
    beta: torch.Tensor
    support_dimension: int

    @property
    def output_dimension(self) -> int:
        return int(self.beta.shape[0])

    @property
    def first_dimension(self) -> int:
        return int(self.beta.shape[1])

    @property
    def second_dimension(self) -> int:
        return int(self.beta.shape[2])


@dataclass(frozen=True)
class EndpointDataset:
    first: torch.Tensor
    second: torch.Tensor
    bilinear_features: torch.Tensor
    additive_features: torch.Tensor
    target: torch.Tensor


def make_instance(name: str) -> Instance:
    if name == "spin8_triality":
        # Maintained rho convention: vector x negative x positive.  Hence the
        # generic beta convention (output, first, second) is V x S- x S+.
        return Instance(name, triality_tensor(dtype=DTYPE), 4)
    if name == "so3_cross_product":
        return Instance(name, so3_cross_product_tensor(dtype=DTYPE), 2)
    raise ValueError(f"unknown instance: {name}")


def endpoint_features(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    decay: float = DECAY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted tensor-product and additive endpoint features."""

    if first.ndim != 3 or second.ndim != 3 or first.shape[:2] != second.shape[:2]:
        raise ValueError("source sequences must have shapes (batch, length, dimension)")
    length = first.shape[1]
    weights = first.new_tensor(decay).pow(
        torch.arange(length - 1, -1, -1, device=first.device)
    )
    bilinear = torch.einsum("t,nti,ntj->nij", weights, first, second).flatten(1)
    additive = torch.cat(
        (
            torch.einsum("t,nti->ni", weights, first),
            torch.einsum("t,ntj->nj", weights, second),
        ),
        dim=-1,
    )
    return bilinear, additive


def dataset_from_sequences(
    first: torch.Tensor,
    second: torch.Tensor,
    beta: torch.Tensor,
    *,
    target: torch.Tensor | None = None,
    decay: float = DECAY,
) -> EndpointDataset:
    bilinear, additive = endpoint_features(first, second, decay=decay)
    if target is None:
        target = bilinear @ beta.flatten(1).T
    return EndpointDataset(first, second, bilinear, additive, target)


def random_subspace_dataset(
    instance: Instance,
    *,
    count: int,
    length: int,
    generator: torch.Generator,
) -> EndpointDataset:
    first = torch.zeros(count, length, instance.first_dimension, dtype=DTYPE)
    second = torch.zeros(count, length, instance.second_dimension, dtype=DTYPE)
    first[..., : instance.support_dimension] = torch.randn(
        count,
        length,
        instance.support_dimension,
        dtype=DTYPE,
        generator=generator,
    )
    second[..., : instance.support_dimension] = torch.randn(
        count,
        length,
        instance.support_dimension,
        dtype=DTYPE,
        generator=generator,
    )
    return dataset_from_sequences(first, second, instance.beta)


def sample_actions(
    instance: Instance,
    count: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output, first-source, and second-source group actions."""

    if instance.name == "spin8_triality":
        coefficients = 0.45 * torch.randn(
            count,
            SPIN8_BIVECTOR_DIM,
            dtype=DTYPE,
            generator=generator,
        )
        actions = spin8_actions(coefficients, torch_triality_generators(dtype=DTYPE))
        output_action = actions[:, 0]
        second_action = actions[:, 1]
        first_action = actions[:, 2]
        return output_action, first_action, second_action
    if instance.name == "so3_cross_product":
        raw = 0.45 * torch.randn(count, 3, 3, dtype=DTYPE, generator=generator)
        action = torch.matrix_exp(raw - raw.transpose(-1, -2))
        return action, action, action
    raise ValueError(f"unknown instance: {instance.name}")


def transform_dataset(
    dataset: EndpointDataset,
    instance: Instance,
    *,
    generator: torch.Generator,
) -> tuple[EndpointDataset, float]:
    count = dataset.first.shape[0]
    output_action, first_action, second_action = sample_actions(
        instance, count, generator=generator
    )
    first = torch.einsum("nij,ntj->nti", first_action, dataset.first)
    second = torch.einsum("nij,ntj->nti", second_action, dataset.second)
    expected = torch.einsum("nij,nj->ni", output_action, dataset.target)
    transformed = dataset_from_sequences(first, second, instance.beta, target=expected)
    recomputed = transformed.bilinear_features @ instance.beta.flatten(1).T
    return transformed, float((recomputed - expected).abs().max())


def augment_dataset(
    dataset: EndpointDataset,
    instance: Instance,
    *,
    augmentations: int,
    generator: torch.Generator,
) -> tuple[EndpointDataset, float]:
    if augmentations < 1:
        raise ValueError("augmentations must be positive")
    repeated = EndpointDataset(
        first=dataset.first.repeat_interleave(augmentations, dim=0),
        second=dataset.second.repeat_interleave(augmentations, dim=0),
        bilinear_features=dataset.bilinear_features.repeat_interleave(
            augmentations, dim=0
        ),
        additive_features=dataset.additive_features.repeat_interleave(
            augmentations, dim=0
        ),
        target=dataset.target.repeat_interleave(augmentations, dim=0),
    )
    return transform_dataset(repeated, instance, generator=generator)


def fit_generic_bilinear(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    output_dimension: int,
    first_dimension: int,
    second_dimension: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    singular_values = torch.linalg.svdvals(features)
    solution = torch.linalg.lstsq(features, target, driver="gelsd").solution
    beta = solution.T.reshape(output_dimension, first_dimension, second_dimension)
    rank = int(torch.linalg.matrix_rank(features))
    numerical_tolerance = (
        max(features.shape) * torch.finfo(features.dtype).eps * singular_values.max()
    )
    nonzero = singular_values[singular_values > numerical_tolerance]
    return beta, {
        "rank": rank,
        "feature_dimension": int(features.shape[1]),
        "examples": int(features.shape[0]),
        "largest_singular_value": float(singular_values.max()),
        "smallest_singular_value": float(singular_values.min()),
        "numerical_rank_tolerance": float(numerical_tolerance),
        "smallest_numerically_nonzero_singular_value": float(nonzero.min()),
    }


def fit_additive(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(features, target, driver="gelsd").solution


def fit_structured_scalar(
    features: torch.Tensor, target: torch.Tensor, beta: torch.Tensor
) -> float:
    oracle = features @ beta.flatten(1).T
    denominator = oracle.square().sum()
    if float(denominator) == 0.0:
        raise ValueError("structured design has zero energy")
    return float((oracle * target).sum() / denominator)


def prediction_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    squared_error = (prediction - target).square().sum(dim=-1)
    squared_norm = target.square().sum(dim=-1).clamp_min(torch.finfo(target.dtype).tiny)
    relative = squared_error / squared_norm
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1)
    return {
        "mean_relative_squared_error": float(relative.mean()),
        "maximum_relative_squared_error": float(relative.max()),
        "mean_cosine": float(cosine.mean()),
        "minimum_cosine": float(cosine.min()),
    }


def predict_bilinear(dataset: EndpointDataset, beta: torch.Tensor) -> torch.Tensor:
    return dataset.bilinear_features @ beta.flatten(1).T


def predict_additive(
    dataset: EndpointDataset, coefficients: torch.Tensor
) -> torch.Tensor:
    return dataset.additive_features @ coefficients


def scan_parity(
    instance: Instance,
    *,
    generator: torch.Generator,
) -> dict[str, float | int]:
    dataset = random_subspace_dataset(instance, count=2, length=31, generator=generator)
    dataset, _ = transform_dataset(dataset, instance, generator=generator)
    batch, length, _ = dataset.first.shape
    first_action = torch.zeros(
        batch,
        length,
        instance.first_dimension,
        instance.first_dimension,
        dtype=DTYPE,
    )
    second_action = torch.zeros(
        batch,
        length,
        instance.second_dimension,
        instance.second_dimension,
        dtype=DTYPE,
    )
    output_action = DECAY * torch.eye(instance.output_dimension, dtype=DTYPE).expand(
        batch, length, -1, -1
    )
    output_drive = torch.zeros(batch, length, instance.output_dimension, dtype=DTYPE)
    initial_first = torch.zeros(batch, instance.first_dimension, dtype=DTYPE)
    initial_second = torch.zeros(batch, instance.second_dimension, dtype=DTYPE)
    initial_output = torch.zeros(batch, instance.output_dimension, dtype=DTYPE)
    arguments = (
        first_action,
        dataset.first,
        second_action,
        dataset.second,
        output_action,
        output_drive,
        initial_first,
        initial_second,
        initial_output,
        instance.beta,
    )
    staged = staged_intertwiner_scan(*arguments)
    recurrent = recurrent_intertwiner_scan(*arguments)
    staged_recurrent = max(
        float((left - right).abs().max()) for left, right in zip(staged, recurrent)
    )
    closed_form = dataset.target
    return {
        "length": length,
        "staged_recurrent_max_abs_error": staged_recurrent,
        "staged_closed_form_endpoint_max_abs_error": float(
            (staged[-1][:, -1] - closed_form).abs().max()
        ),
        "recurrent_closed_form_endpoint_max_abs_error": float(
            (recurrent[-1][:, -1] - closed_form).abs().max()
        ),
    }


def run_seed(
    instance: Instance,
    *,
    seed: int,
    train_count: int,
    train_length: int,
    augmentations: int,
    test_count: int,
    test_lengths: Iterable[int],
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    training = random_subspace_dataset(
        instance,
        count=train_count,
        length=train_length,
        generator=generator,
    )
    augmented, augmentation_equivariance = augment_dataset(
        training,
        instance,
        augmentations=augmentations,
        generator=generator,
    )

    structured_scalar = fit_structured_scalar(
        training.bilinear_features, training.target, instance.beta
    )
    structured_beta = structured_scalar * instance.beta
    generic_restricted, restricted_design = fit_generic_bilinear(
        training.bilinear_features,
        training.target,
        output_dimension=instance.output_dimension,
        first_dimension=instance.first_dimension,
        second_dimension=instance.second_dimension,
    )
    generic_augmented, augmented_design = fit_generic_bilinear(
        augmented.bilinear_features,
        augmented.target,
        output_dimension=instance.output_dimension,
        first_dimension=instance.first_dimension,
        second_dimension=instance.second_dimension,
    )
    additive = fit_additive(training.additive_features, training.target)

    training_metrics = {
        "oracle_intertwiner": prediction_metrics(
            predict_bilinear(training, instance.beta), training.target
        ),
        "structured_intertwiner": prediction_metrics(
            predict_bilinear(training, structured_beta), training.target
        ),
        "generic_bilinear_restricted": prediction_metrics(
            predict_bilinear(training, generic_restricted), training.target
        ),
        "generic_bilinear_augmented_on_augmented": prediction_metrics(
            predict_bilinear(augmented, generic_augmented), augmented.target
        ),
        "additive_linear": prediction_metrics(
            predict_additive(training, additive), training.target
        ),
    }

    evaluations: dict[str, object] = {}
    maximum_equivariance_error = augmentation_equivariance
    for length in test_lengths:
        test = random_subspace_dataset(
            instance, count=test_count, length=length, generator=generator
        )
        orbit, equivariance_error = transform_dataset(
            test, instance, generator=generator
        )
        maximum_equivariance_error = max(maximum_equivariance_error, equivariance_error)
        evaluations[str(length)] = {
            "oracle_intertwiner": prediction_metrics(
                predict_bilinear(orbit, instance.beta), orbit.target
            ),
            "structured_intertwiner": prediction_metrics(
                predict_bilinear(orbit, structured_beta), orbit.target
            ),
            "generic_bilinear_restricted": prediction_metrics(
                predict_bilinear(orbit, generic_restricted), orbit.target
            ),
            "generic_bilinear_augmented": prediction_metrics(
                predict_bilinear(orbit, generic_augmented), orbit.target
            ),
            "additive_linear": prediction_metrics(
                predict_additive(orbit, additive), orbit.target
            ),
        }

    parity = scan_parity(instance, generator=generator)
    structured_worst = max(
        float(row["structured_intertwiner"]["mean_relative_squared_error"])
        for row in evaluations.values()
    )
    augmented_worst = max(
        float(row["generic_bilinear_augmented"]["mean_relative_squared_error"])
        for row in evaluations.values()
    )
    restricted_length_8 = float(
        evaluations["8"]["generic_bilinear_restricted"]["mean_relative_squared_error"]
    )
    gates = {
        "tensor_equivariance": maximum_equivariance_error < 1e-10,
        "structured_training": training_metrics["structured_intertwiner"][
            "mean_relative_squared_error"
        ]
        < 1e-16,
        "structured_orbit": structured_worst < 1e-16,
        "restricted_generic_interpolates": training_metrics[
            "generic_bilinear_restricted"
        ]["mean_relative_squared_error"]
        < 1e-16,
        "restricted_generic_fails_orbit": restricted_length_8 > 0.10,
        "augmented_generic_full_rank": augmented_design["rank"]
        == augmented_design["feature_dimension"],
        "augmented_generic_orbit": augmented_worst < 1e-14,
        "additive_is_insufficient": training_metrics["additive_linear"][
            "mean_relative_squared_error"
        ]
        > 0.25,
        "scan_parity": max(
            float(value) for key, value in parity.items() if key.endswith("error")
        )
        < 1e-10,
    }
    return {
        "seed": seed,
        "structured_scalar": structured_scalar,
        "maximum_tensor_equivariance_abs_error": maximum_equivariance_error,
        "designs": {
            "restricted": restricted_design,
            "augmented": augmented_design,
        },
        "training": training_metrics,
        "orbit_evaluation": evaluations,
        "scan_parity": parity,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run_experiment(
    *,
    seeds: Iterable[int] = range(10),
    train_count: int = 64,
    train_length: int = 8,
    augmentations: int = 4,
    test_count: int = 128,
    test_lengths: Iterable[int] = (8, 32, 128, 512),
) -> dict[str, object]:
    seeds = tuple(seeds)
    test_lengths = tuple(test_lengths)
    if 8 not in test_lengths:
        raise ValueError("the frozen gate requires evaluation length 8")
    families: dict[str, object] = {}
    for name in ("spin8_triality", "so3_cross_product"):
        instance = make_instance(name)
        rows = [
            run_seed(
                instance,
                seed=seed,
                train_count=train_count,
                train_length=train_length,
                augmentations=augmentations,
                test_count=test_count,
                test_lengths=test_lengths,
            )
            for seed in seeds
        ]
        structured_errors = [
            float(
                row["orbit_evaluation"][str(length)]["structured_intertwiner"][
                    "mean_relative_squared_error"
                ]
            )
            for row in rows
            for length in test_lengths
        ]
        augmented_errors = [
            float(
                row["orbit_evaluation"][str(length)]["generic_bilinear_augmented"][
                    "mean_relative_squared_error"
                ]
            )
            for row in rows
            for length in test_lengths
        ]
        restricted_length_8 = [
            float(
                row["orbit_evaluation"]["8"]["generic_bilinear_restricted"][
                    "mean_relative_squared_error"
                ]
            )
            for row in rows
        ]
        scan_errors = [
            float(value)
            for row in rows
            for key, value in row["scan_parity"].items()
            if key.endswith("error")
        ]
        families[name] = {
            "dimensions": {
                "first": instance.first_dimension,
                "second": instance.second_dimension,
                "output": instance.output_dimension,
                "training_support": instance.support_dimension,
            },
            "streaming_state_scalars": (
                instance.first_dimension
                + instance.second_dimension
                + instance.output_dimension
            ),
            "fitted_parameter_counts": {
                "structured_intertwiner": 1,
                "generic_bilinear": int(instance.beta.numel()),
                "additive_linear": instance.output_dimension
                * (instance.first_dimension + instance.second_dimension),
            },
            "summary": {
                "maximum_tensor_equivariance_abs_error": max(
                    float(row["maximum_tensor_equivariance_abs_error"]) for row in rows
                ),
                "maximum_structured_orbit_mean_relative_squared_error": max(
                    structured_errors
                ),
                "restricted_generic_length_8_mean_relative_squared_error_range": [
                    min(restricted_length_8),
                    max(restricted_length_8),
                ],
                "maximum_augmented_generic_orbit_mean_relative_squared_error": max(
                    augmented_errors
                ),
                "minimum_additive_training_mean_relative_squared_error": min(
                    float(
                        row["training"]["additive_linear"][
                            "mean_relative_squared_error"
                        ]
                    )
                    for row in rows
                ),
                "maximum_scan_recurrent_closed_form_abs_error": max(scan_errors),
                "restricted_design_ranks": sorted(
                    {int(row["designs"]["restricted"]["rank"]) for row in rows}
                ),
                "augmented_design_ranks": sorted(
                    {int(row["designs"]["augmented"]["rank"]) for row in rows}
                ),
            },
            "seeds": rows,
            "passes": sum(bool(row["passed"]) for row in rows),
            "passed": all(bool(row["passed"]) for row in rows),
        }

    passed = all(bool(family["passed"]) for family in families.values())
    return {
        "experiment": "Intertwiner SchurScan equivariant identification",
        "protocol": {
            "dtype": "float64",
            "decay": DECAY,
            "seeds": list(seeds),
            "train_count": train_count,
            "train_length": train_length,
            "augmentations_per_training_endpoint": augmentations,
            "test_count_per_length": test_count,
            "test_lengths": list(test_lengths),
            "fit": "deterministic minimum-norm least squares",
        },
        "families": families,
        "claim_boundary": {
            "known_equivariant_hypothesis_class": True,
            "triality_specific_advantage_established": False,
            "retrieval_or_language_model_advantage_established": False,
            "parameter_counts_matched": False,
            "group_augmented_labels_matched": False,
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/intertwiner_schurscan_equivariant_identification_20260810.json"
        ),
    )
    args = parser.parse_args()
    report = run_experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
