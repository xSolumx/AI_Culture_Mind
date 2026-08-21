"""Audit recurrent and local-convolution streaming cache sizes."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark import BenchmarkConfig, build_model


def main() -> int:
    config = BenchmarkConfig()
    spin = build_model("pure_spin_v1_2", config)
    mamba = build_model("mamba2_fused", config)
    spin_recurrence = spin.cache_scalars
    spin_conv_fifo = config.layers * config.d_model * (spin.config.d_conv - 1)
    spin_design_total = spin_recurrence + spin_conv_fifo
    mamba_total = 0
    mamba_shapes = []
    for layer in mamba.layers:
        cache = layer.allocate_inference_cache(batch_size=1, max_seqlen=256)
        mamba_shapes.append([list(tensor.shape) for tensor in cache])
        mamba_total += sum(tensor.numel() for tensor in cache)
    report = {
        "schema_version": 1,
        "claim_scope": "per-sequence streaming state scalar count; no latency claim",
        "pure_spin": {
            "implemented_spin_recurrence_scalars": spin_recurrence,
            "required_conv_fifo_scalars": spin_conv_fifo,
            "complete_streaming_design_scalars": spin_design_total,
            "incremental_wrapper_status": "not yet implemented",
        },
        "official_mamba2": {
            "allocated_cache_scalars": mamba_total,
            "per_layer_tensor_shapes": mamba_shapes,
        },
        "mamba_over_complete_spin_design": mamba_total / spin_design_total,
        "parameter_match": {
            "pure_spin_v1_2": sum(parameter.numel() for parameter in spin.parameters()),
            "mamba2_fused": sum(parameter.numel() for parameter in mamba.parameters()),
        },
    }
    output = Path("artifacts/cache_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
