"""Audit source pins and local feasibility without downloading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from huggingface_hub import HfApi

try:
    from .manifest import BASELINES, GITHUB_SOURCES, SCHEMA_VERSION
except ImportError:  # Direct script execution from this directory.
    from manifest import BASELINES, GITHUB_SOURCES, SCHEMA_VERSION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weight_bytes(info: object) -> int:
    suffixes = (".bin", ".gguf", ".pt", ".safetensors")
    return sum(
        sibling.size or 0
        for sibling in info.siblings
        if sibling.rfilename.endswith(suffixes)
    )


def _github_head(url: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", url, "refs/heads/main"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split()[0]


def audit(*, live: bool, metadata_root: Path) -> dict[str, object]:
    large_root = Path(os.environ.get("PURE_SPIN_V12_LARGE_ROOT", ""))
    if not str(large_root).startswith("/mnt/e/"):
        raise RuntimeError("source wsl_env.sh so heavyweight paths resolve to E:")
    if not str(metadata_root).startswith(f"{large_root}/"):
        raise RuntimeError("external metadata must live below PURE_SPIN_V12_LARGE_ROOT")

    api = HfApi() if live else None
    baselines = []
    for expected in BASELINES:
        local_dir = metadata_root / expected["hf_repo"]
        config = local_dir / "config.json"
        if not config.is_file():
            raise FileNotFoundError(
                f"missing pinned config for {expected['id']}: {config}"
            )
        observed = {
            "config_path": str(config),
            "config_sha256": _sha256(config),
            "metadata_bytes": sum(
                path.stat().st_size for path in local_dir.rglob("*") if path.is_file()
            ),
        }
        if live:
            info = api.model_info(
                expected["hf_repo"],
                revision=expected["revision"],
                files_metadata=True,
            )
            observed.update(
                {
                    "hub_revision": info.sha,
                    "hub_weight_bytes": _weight_bytes(info),
                }
            )
            if info.sha != expected["revision"]:
                raise RuntimeError(f"revision mismatch for {expected['id']}")
            if observed["hub_weight_bytes"] != expected["weight_bytes"]:
                raise RuntimeError(f"weight-byte mismatch for {expected['id']}")
        baselines.append({**expected, "observed": observed})

    github = {
        name: {
            **source,
            **({"current_main": _github_head(source["url"])} if live else {}),
        }
        for name, source in GITHUB_SOURCES.items()
    }
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "live_source_verification": live,
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "vram_bytes": vram_bytes,
            "bfloat16_emulation_or_native": torch.cuda.is_bf16_supported(),
            "native_bfloat16": torch.cuda.is_bf16_supported(
                including_emulation=False
            ),
        },
        "comparison_contract": {
            "matched_shakespeare_training": (
                "same byte stream, batches, optimizer budget, parameter tolerance, "
                "precision, and measured local environment"
            ),
            "native_checkpoint_inference": (
                "each official tokenizer and checkpoint; throughput and memory only, "
                "with quality reported on a shared downstream evaluation"
            ),
            "prohibited": (
                "placing pretrained checkpoint loss beside the 300-step byte-LM loss"
            ),
        },
        "github_sources": github,
        "baselines": baselines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    large_root = Path(os.environ.get("PURE_SPIN_V12_LARGE_ROOT", ""))
    metadata_root = args.metadata_root or large_root / "external_metadata"
    report = audit(live=args.live, metadata_root=metadata_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")


if __name__ == "__main__":
    main()
