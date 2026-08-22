"""Run all data-order descendants of one shared learned-router checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from benchmark import file_sha256, package_version, parameter_count
from spin_delta_router_curriculum_transfer import (
    ARMS,
    ROOT,
    RouterTransferConfig,
    evaluation_rows,
    train_core_arm,
    train_router,
)

INIT_SEEDS = (617, 619, 631)
DATA_SEEDS = (641, 643, 647)
PROTOCOL = "SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_V2_PREREGISTRATION.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-seed", type=int, choices=INIT_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    base_config = RouterTransferConfig(args.init_seed, DATA_SEEDS[0])
    rows = evaluation_rows(base_config, device)
    shared, router_phase = train_router(base_config, rows, device)
    execution_id = (
        f"init-{args.init_seed}-router-{router_phase['post_router_router_sha256']}"
    )
    paths = (
        Path(__file__),
        ROOT / "spin_delta_router_curriculum_transfer.py",
        ROOT / "spin_delta_causal_router_gate.py",
        ROOT / "spin_delta_router.py",
        ROOT / "spin_delta_write_curriculum_gate.py",
        ROOT / "model.py",
        ROOT / "spin_delta_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    implementation = {
        path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths
    }
    environment = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "triton": package_version("triton", "triton-windows"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for data_seed in DATA_SEEDS:
        config = RouterTransferConfig(args.init_seed, data_seed)
        arms = [train_core_arm(arm, shared, config, rows, device) for arm in ARMS]
        if arms[0]["initial_state_sha256"] != arms[1]["initial_state_sha256"]:
            raise RuntimeError("core arms did not begin from the same cloned state")
        report = {
            "schema_version": 2,
            "stage": "spin_delta_router_curriculum_transfer",
            "protocol": PROTOCOL,
            "config": config.__dict__,
            "claim_scope": "autonomous learned-router synthetic transfer v2",
            "parameters": parameter_count(shared),
            "router_phase": router_phase,
            "arms": arms,
            "autonomous_evaluation": True,
            "oracle_controls_supplied_to_model": False,
            "cohort_execution": {
                "shared_router_single_execution": True,
                "execution_id": execution_id,
                "init_seed": args.init_seed,
                "data_seeds": list(DATA_SEEDS),
            },
            "environment": environment,
            "implementation_sha256": implementation,
        }
        output = args.output_dir / f"i{args.init_seed}_d{data_seed}.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
