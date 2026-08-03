"""Select the largest replicated metric action, then compile it into Spin(8).

The historical filename is retained for artifact compatibility. This scan fits
one Euclidean K-means candidate per cardinality and certifies quotient relations
only among those discovered candidates; it does not enumerate the complete
congruence lattice. See ``spin8_exact_congruence_lattice_audit.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mechanistic_group_actions import PureGroupActionModel
from spin8_state_cardinality_audit import candidate_action, replicated_candidate
from spin8_state_only_compiler import (
    collect_state_paths,
    compile_state_only_checkpoint,
    posthoc_state_only_score,
    squared_distances,
)


SEARCH_CARDINALITIES = tuple(range(2, 13))


def quotient_certificate(
    points: np.ndarray,
    coarse: tuple[dict[str, object], np.ndarray | None, np.ndarray | None, int | None],
    fine: tuple[dict[str, object], np.ndarray | None, np.ndarray | None, int | None],
) -> dict[str, object]:
    _, coarse_centers, coarse_next, _ = coarse
    _, fine_centers, fine_next, _ = fine
    if (
        coarse_centers is None or coarse_next is None
        or fine_centers is None or fine_next is None
    ):
        return {"is_quotient": False, "reason": "candidate unavailable"}
    coarse_labels = squared_distances(points, coarse_centers).argmin(axis=1)
    fine_labels = squared_distances(points, fine_centers).argmin(axis=1)
    contingency = np.zeros(
        (len(fine_centers), len(coarse_centers)), dtype=np.int64
    )
    np.add.at(contingency, (fine_labels, coarse_labels), 1)
    mapping = contingency.argmax(axis=1)
    purity = contingency.max(axis=1) / contingency.sum(axis=1)
    fibre_sizes = np.bincount(mapping, minlength=len(coarse_centers))
    expected = coarse_next[mapping[:, None], np.arange(coarse_next.shape[1])]
    actual = mapping[fine_next]
    intertwining = float(np.mean(expected == actual))
    is_quotient = bool(
        float(purity.min()) >= 0.99
        and int(fibre_sizes.min()) > 0
        and intertwining == 1.0
    )
    return {
        "fine_to_coarse": mapping.tolist(),
        "minimum_mapping_purity": float(purity.min()),
        "fibre_sizes": fibre_sizes.tolist(),
        "intertwining_fraction": intertwining,
        "is_quotient": is_quotient,
    }


def discover_finest_congruence(
    model: PureGroupActionModel,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    primary_seed = 13_500_000 + 20_000 * seed
    audit_seed = 13_510_000 + 20_000 * seed
    primary_data = collect_state_paths(
        model, seed_base=primary_seed, token_count=4, device=device
    )
    audit_data = collect_state_paths(
        model, seed_base=audit_seed, token_count=4, device=device
    )
    raw: dict[int, dict[str, object]] = {}
    summaries = []
    for k in SEARCH_CARDINALITIES:
        primary = candidate_action(
            *primary_data, clusters=k, seed=primary_seed + 8_000_003
        )
        audit = candidate_action(
            *audit_data, clusters=k, seed=audit_seed + 8_000_003
        )
        replicated = replicated_candidate(primary, audit, clusters=k)
        raw[k] = {"primary": primary, "audit": audit, "replicated": replicated}
        summaries.append(replicated)
        print(
            f"seed={seed} k={k} viable={replicated['viable']}", flush=True
        )
    viable = [k for k in SEARCH_CARDINALITIES if raw[k]["replicated"]["viable"]]
    if not viable:
        raise ValueError("no independently replicated regular state congruence")
    selected = max(viable)
    certificates = {}
    for coarse_k in viable:
        if coarse_k == selected:
            continue
        certificates[str(coarse_k)] = {
            "primary": quotient_certificate(
                primary_data[0], raw[coarse_k]["primary"], raw[selected]["primary"]
            ),
            "audit": quotient_certificate(
                audit_data[0], raw[coarse_k]["audit"], raw[selected]["audit"]
            ),
        }
    all_quotients = all(
        certificate[side]["is_quotient"]
        for certificate in certificates.values()
        for side in ("primary", "audit")
    )
    if not all_quotients:
        raise ValueError(
            "discovered viable metric actions are incomparable"
        )
    return {
        "search_cardinalities": list(SEARCH_CARDINALITIES),
        "viable_cardinalities": viable,
        "selected_cardinality": selected,
        "selection_rule": (
            "largest replicated regular K-means candidate whose other "
            "discovered viable candidates are exact quotients"
        ),
        "quotient_certificates": certificates,
        "all_other_viable_actions_are_quotients": all_quotients,
        "candidate_summaries": summaries,
    }


def compile_finest_congruence_checkpoint(
    source: Path, destination: Path, *, device: torch.device
) -> tuple[dict[str, object], object]:
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint["family"] != "pure_spin8_positive":
        raise ValueError("finest-congruence compiler requires positive Spin(8)")
    config = checkpoint["config"]
    model = PureGroupActionModel(
        4, 8, family=checkpoint["family"], channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    discovery = discover_finest_congruence(
        model, seed=int(config["seed"]), device=device
    )
    selected = int(discovery["selected_cardinality"])
    result, recovered = compile_state_only_checkpoint(
        source,
        destination,
        device=device,
        state_count=selected,
        minimum_separation_ratio=None,
        method=(
            "state-only finest-congruence discovery plus shared regular "
            "Spin8 retraction"
        ),
        state_cardinality_supplied=False,
        compiler_config_key="spin8_finest_congruence_compiler",
        extra_result={"finest_congruence_discovery": discovery},
        extra_gates={
            "maximal_discovered_metric_candidate_identified": True,
            "all_other_discovered_viable_actions_are_quotients": bool(
                discovery["all_other_viable_actions_are_quotients"]
            ),
        },
    )
    return result, recovered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.use_deterministic_algorithms(True)
    result, recovered = compile_finest_congruence_checkpoint(
        args.source, args.destination, device=device
    )
    result = posthoc_state_only_score(
        result, recovered, args.destination, device=device
    )
    report = {
        "experiment": "Spin8 state-only finest-congruence compiler",
        "result": result,
        "passed": result["passed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
