"""Test whether viable low-cardinality state actions quotient the k=8 action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from mechanistic_group_actions import PureGroupActionModel
from spin8_state_cardinality_audit import candidate_action
from spin8_state_only_compiler import collect_state_paths, squared_distances


def quotient_map(
    points: np.ndarray,
    coarse_centers: np.ndarray,
    fine_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse = squared_distances(points, coarse_centers).argmin(axis=1)
    fine = squared_distances(points, fine_centers).argmin(axis=1)
    contingency = np.zeros((len(fine_centers), len(coarse_centers)), dtype=np.int64)
    np.add.at(contingency, (fine, coarse), 1)
    mapping = contingency.argmax(axis=1)
    purity = contingency.max(axis=1) / contingency.sum(axis=1)
    return mapping, purity, contingency


def intertwining_fraction(
    mapping: np.ndarray, coarse_next: np.ndarray, fine_next: np.ndarray
) -> float:
    expected = coarse_next[mapping[:, None], np.arange(coarse_next.shape[1])]
    actual = mapping[fine_next]
    return float(np.mean(expected == actual))


def align_map(
    primary_coarse: np.ndarray,
    primary_fine: np.ndarray,
    audit_coarse: np.ndarray,
    audit_fine: np.ndarray,
    audit_mapping: np.ndarray,
) -> np.ndarray:
    def audit_to_primary(audit: np.ndarray, primary: np.ndarray) -> np.ndarray:
        rows, columns = linear_sum_assignment(
            np.sqrt(squared_distances(audit, primary))
        )
        result = np.empty(len(audit), dtype=np.int64)
        result[rows] = columns
        return result

    coarse_alignment = audit_to_primary(audit_coarse, primary_coarse)
    fine_alignment = audit_to_primary(audit_fine, primary_fine)
    aligned = np.empty_like(audit_mapping)
    aligned[fine_alignment] = coarse_alignment[audit_mapping]
    return aligned


def audit_checkpoint(path: Path, *, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    seed = int(config["seed"])
    model = PureGroupActionModel(
        4, 8, family=checkpoint["family"], channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    datasets = [
        collect_state_paths(
            model, seed_base=base + 20_000 * seed,
            token_count=4, device=device,
        )
        for base in (13_500_000, 13_510_000)
    ]
    sections = []
    for corpus, base in zip(datasets, (13_500_000, 13_510_000)):
        actions = {}
        for k in (2, 8):
            actions[k] = candidate_action(
                *corpus, clusters=k,
                seed=base + 20_000 * seed + 8_000_003,
            )
        coarse, coarse_centers, coarse_next, _ = actions[2]
        fine, fine_centers, fine_next, _ = actions[8]
        if coarse_centers is None or fine_centers is None:
            sections.append({"available": False, "coarse": coarse, "fine": fine})
            continue
        mapping, purity, contingency = quotient_map(
            corpus[0], coarse_centers, fine_centers
        )
        fibres = np.bincount(mapping, minlength=2)
        token_nonidentity = [
            not np.array_equal(coarse_next[:, token], np.arange(2))
            for token in range(coarse_next.shape[1])
        ]
        sections.append({
            "available": True,
            "coarse": coarse,
            "fine": fine,
            "mapping": mapping.tolist(),
            "minimum_fine_cluster_purity": float(purity.min()),
            "contingency": contingency.tolist(),
            "quotient_fibre_sizes": fibres.tolist(),
            "intertwining_fraction": intertwining_fraction(
                mapping, coarse_next, fine_next
            ),
            "all_tokens_nonidentity_on_quotient": all(token_nonidentity),
            "coarse_centers": coarse_centers,
            "fine_centers": fine_centers,
        })
    primary, audit = sections
    aligned_agreement = False
    if primary["available"] and audit["available"]:
        aligned = align_map(
            primary["coarse_centers"], primary["fine_centers"],
            audit["coarse_centers"], audit["fine_centers"],
            np.asarray(audit["mapping"]),
        )
        aligned_agreement = bool(np.array_equal(aligned, primary["mapping"]))
    for section in sections:
        section.pop("coarse_centers", None)
        section.pop("fine_centers", None)
    pass_seed = bool(
        primary["available"] and audit["available"]
        and primary["coarse"]["local_viability"]
        and audit["coarse"]["local_viability"]
        and primary["minimum_fine_cluster_purity"] >= 0.99
        and audit["minimum_fine_cluster_purity"] >= 0.99
        and min(primary["quotient_fibre_sizes"]) > 0
        and min(audit["quotient_fibre_sizes"]) > 0
        and primary["intertwining_fraction"] == 1.0
        and audit["intertwining_fraction"] == 1.0
        and aligned_agreement
        and primary["all_tokens_nonidentity_on_quotient"]
        and audit["all_tokens_nonidentity_on_quotient"]
    )
    return {
        "seed": seed, "source": str(path), "primary": primary,
        "audit": audit, "aligned_quotient_maps_agree": aligned_agreement,
        "quotient_hypothesis_passed": pass_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.use_deterministic_algorithms(True)
    results = [audit_checkpoint(path, device=device) for path in args.sources]
    report = {
        "experiment": "Spin8 state quotient-lattice audit",
        "exploratory_not_gate_repair": True,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
