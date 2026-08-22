"""Apply frozen decisions to the Spin-Delta query-gradient audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from spin_delta_query_gradient_topology import MODES, PROTOCOL, SEEDS


def summarize(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload["stage"] != "spin_delta_query_gradient_topology"
        or payload["protocol"] != PROTOCOL
        or payload["seeds"] != list(SEEDS)
        or payload["modes"] != list(MODES)
    ):
        raise ValueError("audit does not match the frozen protocol")
    local = {row["mode"]: row for row in payload["local_float64"]}
    full = {
        (row["seed"], path_row["mode"]): path_row
        for row in payload["full_model"]
        for path_row in row["paths"]
    }
    finite = all(
        path_row["finite_logits"]
        and path_row["finite_router_gradients"]
        and all(
            math.isfinite(path_row[name])
            for name in (
                "loss",
                "query_event_gradient_norm",
                "query_slot_gradient_norm",
                "maximum_absolute_logit_change_from_hard",
            )
        )
        for path_row in full.values()
    )
    dead_path = (
        local["hard_fallback"]["slot_gradient_norm"] == 0.0
        and abs(local["hard_fallback"]["event_logit_gradient"]) > 1.0e-8
        and all(
            full[(seed, "hard_fallback")]["final_query_event_mean"] == 0.0
            and full[(seed, "hard_fallback")]["query_slot_gradient_norm"] <= 1.0e-12
            and full[(seed, "hard_fallback")]["query_event_gradient_norm"] > 1.0e-8
            for seed in SEEDS
        )
    )
    soft_restoration = (
        local["soft_query_event"]["slot_gradient_norm"] > 1.0e-8
        and all(
            full[(seed, "soft_query_event")]["query_slot_gradient_norm"] > 1.0e-8
            for seed in SEEDS
        )
        and finite
    )
    authoritative_restoration = (
        local["authoritative_query"]["slot_gradient_norm"] > 1.0e-8
        and all(
            full[(seed, "authoritative_query")]["query_slot_gradient_norm"] > 1.0e-8
            for seed in SEEDS
        )
        and finite
    )
    soft_smaller = all(
        full[(seed, "soft_query_event")]["maximum_absolute_logit_change_from_hard"]
        < full[(seed, "authoritative_query")]["maximum_absolute_logit_change_from_hard"]
        for seed in SEEDS
    )
    authoritative_smaller = all(
        full[(seed, "authoritative_query")]["maximum_absolute_logit_change_from_hard"]
        < full[(seed, "soft_query_event")]["maximum_absolute_logit_change_from_hard"]
        for seed in SEEDS
    )
    selected = None
    if dead_path and soft_restoration and authoritative_restoration and soft_smaller:
        selected = "soft_query_event"
    elif (
        dead_path
        and soft_restoration
        and authoritative_restoration
        and authoritative_smaller
    ):
        selected = "authoritative_query"
    return {
        "schema_version": 1,
        "dead_path_certificate_pass": dead_path,
        "soft_restoration_pass": soft_restoration,
        "authoritative_restoration_pass": authoritative_restoration,
        "finite_audit_pass": finite,
        "soft_perturbation_strictly_smaller_all_seeds": soft_smaller,
        "authoritative_perturbation_strictly_smaller_all_seeds": authoritative_smaller,
        "selected_repair": selected,
        "rows": payload["full_model"],
        "local_float64": payload["local_float64"],
        "source": path.as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "implementation_sha256": payload["implementation_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
