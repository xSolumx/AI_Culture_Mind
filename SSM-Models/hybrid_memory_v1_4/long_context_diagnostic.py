"""Post-hoc causal and positionwise diagnostic for frozen G13 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

if __package__:
    from .long_context_curriculum import (
        EVAL_MACRO_BATCHES,
        FORWARD_CHUNK_SIZE,
        MODEL_SEEDS,
        _load_bpe,
        _macro_batch,
        _mixer_mode,
        _ordinary_evaluation,
    )
    from .model import HybridMemoryConfig, HybridMemoryLM
    from .natural_text_frontier import _sha256, _snapshot_text
    from .tokenization import ByteLevelBPETokenizer, EncodedText
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.long_context_curriculum import (  # type: ignore[no-redef]
        EVAL_MACRO_BATCHES,
        FORWARD_CHUNK_SIZE,
        MODEL_SEEDS,
        _load_bpe,
        _macro_batch,
        _mixer_mode,
        _ordinary_evaluation,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.natural_text_frontier import (  # type: ignore[no-redef]
        _sha256,
        _snapshot_text,
    )
    from hybrid_memory_v1_4.tokenization import (  # type: ignore[no-redef]
        ByteLevelBPETokenizer,
        EncodedText,
    )


SEQUENCE_LENGTH = 4096
POSITION_BIN_ENDS = (256, 512, 1024, 2048, 4096)
ABLATION_MODES = ("full", "gated_delta_off", "attention_off")


def _load_model(run: dict[str, Any], device: torch.device) -> HybridMemoryLM:
    checkpoint = Path(run["checkpoint"])
    if _sha256(checkpoint) != run["checkpoint_sha256"]:
        raise ValueError("G13 checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("stage") != "G13":
        raise ValueError("diagnostic input is not a G13 checkpoint")
    if payload.get("seed") != run["seed"] or payload.get("arm_name") != run["arm_name"]:
        raise ValueError("G13 checkpoint identity mismatch")
    config = HybridMemoryConfig(**payload["model_config"])
    model = HybridMemoryLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def _positionwise_evaluation(
    model: HybridMemoryLM,
    validation: EncodedText,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    bin_ends: tuple[int, ...] = POSITION_BIN_ENDS,
    macro_batches: int = EVAL_MACRO_BATCHES,
    device: torch.device,
) -> list[dict[str, float | int]]:
    if not bin_ends or bin_ends[-1] != sequence_length:
        raise ValueError("position bins must end at sequence_length")
    if tuple(sorted(set(bin_ends))) != bin_ends:
        raise ValueError("position bins must be strictly increasing")
    nll_sums = [0.0] * len(bin_ends)
    raw_byte_sums = [0] * len(bin_ends)
    token_counts = [0] * len(bin_ends)
    model.eval()
    with torch.inference_mode():
        for macro_index in range(macro_batches):
            inputs, targets, byte_lengths, _ = _macro_batch(
                validation,
                namespace="g13:validation",
                macro_index=macro_index,
                sequence_length=sequence_length,
                device=device,
            )
            states = None
            logits = []
            for start in range(0, sequence_length, FORWARD_CHUNK_SIZE):
                output = model(
                    inputs[:, start : start + FORWARD_CHUNK_SIZE],
                    states,
                    delta_scan_mode="parallel",
                )
                logits.append(output["logits"])
                states = output["states"]
            losses = F.cross_entropy(
                torch.cat(logits, dim=1).flatten(0, 1),
                targets.flatten(),
                reduction="none",
            ).view_as(targets)
            lower = 0
            for index, upper in enumerate(bin_ends):
                selected_loss = losses[:, lower:upper]
                selected_bytes = byte_lengths[:, lower:upper]
                nll_sums[index] += float(selected_loss.sum())
                raw_byte_sums[index] += int(selected_bytes.sum())
                token_counts[index] += selected_loss.numel()
                lower = upper
    rows = []
    lower = 0
    for upper, nll_sum, raw_bytes, tokens in zip(
        bin_ends, nll_sums, raw_byte_sums, token_counts, strict=True
    ):
        rows.append(
            {
                "position_start_inclusive": lower + 1,
                "position_end_inclusive": upper,
                "scored_tokens": tokens,
                "scored_raw_bytes": raw_bytes,
                "mean_nats_per_token": nll_sum / tokens,
                "bits_per_raw_byte": nll_sum / raw_bytes / math.log(2.0),
            }
        )
        lower = upper
    return rows


def _memory_controls(
    model: HybridMemoryLM,
    validation: EncodedText,
    *,
    macro_batches: int = EVAL_MACRO_BATCHES,
    device: torch.device,
) -> dict[str, Any]:
    write_sum = None
    retention_sum = None
    written_direction_factor_sum = None
    state_norm_sum = None
    read_norm_sum = None
    count = 0
    model.eval()
    with torch.inference_mode():
        for macro_index in range(macro_batches):
            inputs, _, _, _ = _macro_batch(
                validation,
                namespace="g13:validation",
                macro_index=macro_index,
                sequence_length=SEQUENCE_LENGTH,
                device=device,
            )
            states = None
            for start in range(0, SEQUENCE_LENGTH, FORWARD_CHUNK_SIZE):
                output = model(
                    inputs[:, start : start + FORWARD_CHUNK_SIZE],
                    states,
                    delta_scan_mode="parallel",
                    return_diagnostics=True,
                )
                states = output["states"]
                diagnostic = output["diagnostics"][0]
                write = diagnostic["write_strength"].float()
                retention = diagnostic["retention"].float()
                state_norm = diagnostic["state_norm"].float()
                read_norm = diagnostic["read"].float().norm(dim=-1)
                per_head_write = write.sum(dim=(0, 1))
                per_head_retention = retention.sum(dim=(0, 1))
                per_head_written_direction_factor = (retention * (1.0 - write)).sum(
                    dim=(0, 1)
                )
                per_head_state_norm = state_norm.sum(dim=(0, 2))
                per_head_read_norm = read_norm.sum(dim=(0, 1))
                write_sum = (
                    per_head_write if write_sum is None else write_sum + per_head_write
                )
                retention_sum = (
                    per_head_retention
                    if retention_sum is None
                    else retention_sum + per_head_retention
                )
                written_direction_factor_sum = (
                    per_head_written_direction_factor
                    if written_direction_factor_sum is None
                    else written_direction_factor_sum
                    + per_head_written_direction_factor
                )
                state_norm_sum = (
                    per_head_state_norm
                    if state_norm_sum is None
                    else state_norm_sum + per_head_state_norm
                )
                read_norm_sum = (
                    per_head_read_norm
                    if read_norm_sum is None
                    else read_norm_sum + per_head_read_norm
                )
                count += write.shape[0] * write.shape[1]
    assert write_sum is not None
    assert retention_sum is not None
    assert written_direction_factor_sum is not None
    assert state_norm_sum is not None
    assert read_norm_sum is not None
    return {
        "scored_tokens": count,
        "mean_write_strength_per_head": (write_sum / count).tolist(),
        "mean_retention_per_head": (retention_sum / count).tolist(),
        "mean_written_direction_transition_factor_per_head": (
            written_direction_factor_sum / count
        ).tolist(),
        "mean_state_norm_per_head": (state_norm_sum / count).tolist(),
        "mean_read_norm_per_head": (read_norm_sum / count).tolist(),
        "residual_scales": [
            {
                "kind": block.kind,
                "raw": float(block.residual_scale.detach()),
                "sigmoid": float(torch.sigmoid(block.residual_scale).detach()),
            }
            for block in model.blocks
        ],
    }


def _mean_vectors(vectors: list[list[float]]) -> list[float]:
    return [statistics.mean(values) for values in zip(*vectors, strict=True)]


def _summaries(runs: list[dict[str, Any]]) -> dict[str, Any]:
    arm_rows = []
    for arm_name in ("fixed_256_control", "long_context_curriculum"):
        selected = [run for run in runs if run["arm_name"] == arm_name]
        ablations = {}
        for mode in ABLATION_MODES:
            values = [run["ordinary_ablation_bprb"][mode] for run in selected]
            ablations[mode] = {
                "mean_bprb": statistics.mean(values),
                "worst_bprb": max(values),
            }
        full = ablations["full"]["mean_bprb"]
        arm_rows.append(
            {
                "arm_name": arm_name,
                "ordinary_4096_ablations": ablations,
                "bprb_increase_without_gated_delta": ablations["gated_delta_off"][
                    "mean_bprb"
                ]
                - full,
                "bprb_increase_without_attention": ablations["attention_off"][
                    "mean_bprb"
                ]
                - full,
                "mean_write_strength_per_head": _mean_vectors(
                    [
                        run["memory_controls"]["mean_write_strength_per_head"]
                        for run in selected
                    ]
                ),
                "mean_retention_per_head": _mean_vectors(
                    [
                        run["memory_controls"]["mean_retention_per_head"]
                        for run in selected
                    ]
                ),
                "mean_written_direction_transition_factor_per_head": _mean_vectors(
                    [
                        run["memory_controls"][
                            "mean_written_direction_transition_factor_per_head"
                        ]
                        for run in selected
                    ]
                ),
                "mean_state_norm_per_head": _mean_vectors(
                    [
                        run["memory_controls"]["mean_state_norm_per_head"]
                        for run in selected
                    ]
                ),
                "mean_read_norm_per_head": _mean_vectors(
                    [
                        run["memory_controls"]["mean_read_norm_per_head"]
                        for run in selected
                    ]
                ),
                "mean_residual_sigmoid_by_kind": {
                    kind: statistics.mean(
                        next(
                            row["sigmoid"]
                            for row in run["memory_controls"]["residual_scales"]
                            if row["kind"] == kind
                        )
                        for run in selected
                    )
                    for kind in ("gated_delta", "attention")
                },
            }
        )

    by_arm_seed = {(run["arm_name"], run["seed"]): run for run in runs}
    positionwise = []
    for bin_index, upper in enumerate(POSITION_BIN_ENDS):
        lower = 1 if bin_index == 0 else POSITION_BIN_ENDS[bin_index - 1] + 1
        control = [
            by_arm_seed[("fixed_256_control", seed)]["positionwise"][bin_index][
                "bits_per_raw_byte"
            ]
            for seed in MODEL_SEEDS
        ]
        curriculum = [
            by_arm_seed[("long_context_curriculum", seed)]["positionwise"][bin_index][
                "bits_per_raw_byte"
            ]
            for seed in MODEL_SEEDS
        ]
        deltas = [
            candidate - baseline
            for candidate, baseline in zip(curriculum, control, strict=True)
        ]
        positionwise.append(
            {
                "position_start_inclusive": lower,
                "position_end_inclusive": upper,
                "fixed_control_mean_bprb": statistics.mean(control),
                "curriculum_mean_bprb": statistics.mean(curriculum),
                "curriculum_minus_control_mean_bprb": statistics.mean(deltas),
                "curriculum_wins": sum(delta < 0.0 for delta in deltas),
            }
        )
    return {"arms": arm_rows, "positionwise_4096": positionwise}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g13-report", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    started = time.perf_counter()
    report = json.loads(args.g13_report.read_text(encoding="utf-8"))
    if report.get("stage") != "G13" or not report["decision"]["passed_integrity_gate"]:
        raise ValueError(
            "diagnostic requires the complete integrity-passing G13 report"
        )
    _, validation_text, snapshot = _snapshot_text(args.snapshot)
    tokenizer: ByteLevelBPETokenizer = _load_bpe(args.tokenizer_audit)
    validation = tokenizer.encode(validation_text)
    runs = []
    for source_run in report["runs"]:
        model = _load_model(source_run, device)
        ordinary_ablation_bprb = {}
        for mode in ABLATION_MODES:
            with _mixer_mode(model, mode):
                evaluation = _ordinary_evaluation(
                    model,
                    validation,
                    sequence_length=SEQUENCE_LENGTH,
                    device=device,
                )
            ordinary_ablation_bprb[mode] = evaluation["bits_per_raw_byte"]
        runs.append(
            {
                "seed": source_run["seed"],
                "arm_name": source_run["arm_name"],
                "checkpoint": source_run["checkpoint"],
                "checkpoint_sha256": source_run["checkpoint_sha256"],
                "ordinary_ablation_bprb": ordinary_ablation_bprb,
                "positionwise": _positionwise_evaluation(
                    model, validation, device=device
                ),
                "memory_controls": _memory_controls(model, validation, device=device),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summaries = _summaries(runs)
    output = {
        "schema_version": 1,
        "stage": "G13D",
        "claim_status": "post-hoc causal and positionwise diagnostic",
        "summaries": summaries,
        "runs": runs,
        "inputs": {
            "g13_report": str(args.g13_report),
            "g13_report_sha256": _sha256(args.g13_report),
            "snapshot": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "tokenizer_audit": str(args.tokenizer_audit),
            "tokenizer_audit_sha256": _sha256(args.tokenizer_audit),
            "hub_sha": snapshot["hub_sha_at_snapshot"],
        },
        "protocol": {
            "sequence_length_tokens": SEQUENCE_LENGTH,
            "position_bin_ends": list(POSITION_BIN_ENDS),
            "macro_batches": EVAL_MACRO_BATCHES,
            "scored_tokens_per_evaluation": EVAL_MACRO_BATCHES * SEQUENCE_LENGTH,
            "ablation_modes": list(ABLATION_MODES),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "post-hoc frozen-checkpoint diagnostic; mixer suppression and position "
            "bins explain G13 but do not alter its preregistered decisions"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
