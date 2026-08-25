"""Evaluate G12C checkpoints beyond training length and on paired recall facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

if __package__:
    from .model import HybridMemoryConfig, HybridMemoryLM
    from .natural_text_frontier import _offset, _sha256, _snapshot_text
    from .tokenization import (
        ByteLevelBPETokenizer,
        EncodedText,
        LosslessTextTokenizer,
        RawByteTokenizer,
        tokenizer_fingerprint,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.natural_text_frontier import (  # type: ignore[no-redef]
        _offset,
        _sha256,
        _snapshot_text,
    )
    from hybrid_memory_v1_4.tokenization import (  # type: ignore[no-redef]
        ByteLevelBPETokenizer,
        EncodedText,
        LosslessTextTokenizer,
        RawByteTokenizer,
        tokenizer_fingerprint,
    )


PREREGISTRATION = Path(__file__).with_name("G12_PREREGISTRATION.md")
CONTEXT_TOKEN_LENGTHS = (256, 512, 1024)
RECALL_RAW_BYTE_DISTANCES = (128, 256, 512, 1024)
RECALL_SEED = 1901
RECALL_EXAMPLES = 12
EVAL_BATCH_SIZE = 4
EVAL_BATCHES = 8
NAMES = ("Lily", "Milo", "Nora", "Toby")
VALUES = ("lion", "bear", "frog", "duck")


def _git() -> tuple[str, list[str]]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, status


def _tokenizer_for_run(
    run: dict[str, Any], bpe: ByteLevelBPETokenizer
) -> LosslessTextTokenizer:
    name = run["tokenizer"]["name"]
    tokenizer: LosslessTextTokenizer
    if name == "raw_utf8_bytes":
        tokenizer = RawByteTokenizer()
    elif name == "bytelevel_bpe":
        tokenizer = bpe
    else:
        raise ValueError(f"unknown checkpoint tokenizer {name!r}")
    if tokenizer_fingerprint(tokenizer) != run["tokenizer"]["sha256"]:
        raise ValueError("checkpoint tokenizer fingerprint mismatch")
    return tokenizer


def _batch_at_length(
    stream: EncodedText,
    *,
    namespace: str,
    batch_index: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = stream.token_count - sequence_length
    tokens = []
    byte_lengths = []
    for row in range(EVAL_BATCH_SIZE):
        index = batch_index * EVAL_BATCH_SIZE + row
        start = _offset(namespace, index, maximum)
        tokens.append(stream.token_ids[start : start + sequence_length + 1])
        byte_lengths.append(
            stream.token_byte_lengths[start : start + sequence_length + 1]
        )
    token_batch = torch.stack(tokens).to(device)
    byte_batch = torch.stack(byte_lengths).to(device)
    return token_batch[:, :-1], token_batch[:, 1:], byte_batch[:, 1:]


def _ordinary_evaluation(
    model: HybridMemoryLM,
    validation: EncodedText,
    *,
    namespace: str,
    sequence_length: int,
    device: torch.device,
) -> dict[str, float | int | bool]:
    nll_sum = 0.0
    raw_bytes = 0
    token_count = 0
    model.eval()
    with torch.inference_mode():
        for batch_index in range(EVAL_BATCHES):
            inputs, targets, byte_lengths = _batch_at_length(
                validation,
                namespace=namespace,
                batch_index=batch_index,
                sequence_length=sequence_length,
                device=device,
            )
            logits = model(inputs, delta_scan_mode="parallel")["logits"]
            nll_sum += float(
                F.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), reduction="sum"
                )
            )
            raw_bytes += int(byte_lengths.sum())
            token_count += targets.numel()
    return {
        "sequence_length_tokens": sequence_length,
        "mean_sequence_span_raw_bytes": raw_bytes / (EVAL_BATCH_SIZE * EVAL_BATCHES),
        "scored_tokens": token_count,
        "scored_raw_bytes": raw_bytes,
        "bits_per_raw_byte": nll_sum / raw_bytes / math.log(2.0),
        "finite": math.isfinite(nll_sum),
    }


def _ascii_filler(validation_text: str) -> str:
    filler = "".join(
        character if character.isascii() else " " for character in validation_text
    )
    if len(filler.encode("utf-8")) != len(filler):
        raise RuntimeError("recall filler must contain only one-byte characters")
    return filler


def _filler_offset(distance: int, example: int, maximum: int) -> int:
    payload = f"g12d:{RECALL_SEED}:{distance}:{example}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % maximum


def _recall_pair(
    filler_source: str,
    *,
    distance: int,
    example: int,
) -> dict[str, str | int]:
    name = NAMES[example % len(NAMES)]
    value_index = (example // len(NAMES)) % len(VALUES)
    value = VALUES[value_index]
    mismatch_value = VALUES[(value_index + 1) % len(VALUES)]
    first_tail = ".\n\n"
    query = f"\n\nAfter the story, {name}'s secret word was"
    fixed_distance = len((first_tail + query).encode("utf-8"))
    filler_length = distance - fixed_distance
    if filler_length < 1:
        raise ValueError("recall distance is too short for the fixed prompt")
    maximum = len(filler_source) - filler_length
    if maximum < 1:
        raise ValueError("held-out filler source is too short")
    offset = _filler_offset(distance, example, maximum)
    filler = filler_source[offset : offset + filler_length]
    matching_prompt = f"{name}'s secret word was {value}{first_tail}{filler}{query}"
    mismatched_prompt = (
        f"{name}'s secret word was {mismatch_value}{first_tail}{filler}{query}"
    )
    continuation = f" {value}"
    measured = len((first_tail + filler + query).encode("utf-8"))
    if measured != distance:
        raise RuntimeError("constructed recall pair has the wrong byte distance")
    if len(matching_prompt.encode("utf-8")) != len(mismatched_prompt.encode("utf-8")):
        raise RuntimeError("counterfactual recall prompts must have equal byte length")
    return {
        "name": name,
        "supported_value": value,
        "counterfactual_value": mismatch_value,
        "distance_raw_bytes": measured,
        "filler_offset": offset,
        "matching_prompt": matching_prompt,
        "mismatched_prompt": mismatched_prompt,
        "continuation": continuation,
    }


def _continuation_log_probability(
    model: HybridMemoryLM,
    tokenizer: LosslessTextTokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> tuple[float, int, int]:
    prompt_ids = tokenizer.encode(prompt).token_ids
    full_ids = tokenizer.encode(prompt + continuation).token_ids
    if full_ids[: prompt_ids.numel()].tolist() != prompt_ids.tolist():
        raise RuntimeError("tokenizer continuation does not preserve the prompt prefix")
    continuation_ids = full_ids[prompt_ids.numel() :]
    if not continuation_ids.numel():
        raise RuntimeError("recall continuation encoded to no tokens")
    inputs = full_ids[:-1].unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(inputs, delta_scan_mode="parallel")["logits"][0]
        start = prompt_ids.numel() - 1
        continuation_logits = logits[start : start + continuation_ids.numel()]
        log_probabilities = F.log_softmax(continuation_logits.float(), dim=-1)
        selected = log_probabilities.gather(
            -1, continuation_ids.to(device).unsqueeze(-1)
        )
    return (
        float(selected.sum()),
        prompt_ids.numel(),
        continuation_ids.numel(),
    )


def _recall_evaluation(
    model: HybridMemoryLM,
    tokenizer: LosslessTextTokenizer,
    filler_source: str,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for distance in RECALL_RAW_BYTE_DISTANCES:
        for example in range(RECALL_EXAMPLES):
            pair = _recall_pair(filler_source, distance=distance, example=example)
            matching, matching_tokens, continuation_tokens = (
                _continuation_log_probability(
                    model,
                    tokenizer,
                    pair["matching_prompt"],  # type: ignore[arg-type]
                    pair["continuation"],  # type: ignore[arg-type]
                    device,
                )
            )
            mismatched, mismatched_tokens, mismatched_continuation_tokens = (
                _continuation_log_probability(
                    model,
                    tokenizer,
                    pair["mismatched_prompt"],  # type: ignore[arg-type]
                    pair["continuation"],  # type: ignore[arg-type]
                    device,
                )
            )
            if continuation_tokens != mismatched_continuation_tokens:
                raise RuntimeError("counterfactual continuation token counts differ")
            rows.append(
                {
                    "distance_raw_bytes": distance,
                    "example": example,
                    "name": pair["name"],
                    "supported_value": pair["supported_value"],
                    "counterfactual_value": pair["counterfactual_value"],
                    "filler_offset": pair["filler_offset"],
                    "matching_prompt_tokens": matching_tokens,
                    "mismatched_prompt_tokens": mismatched_tokens,
                    "continuation_tokens": continuation_tokens,
                    "continuation_raw_bytes": len(
                        pair["continuation"].encode("utf-8")  # type: ignore[union-attr]
                    ),
                    "matching_log_probability_nats": matching,
                    "mismatched_log_probability_nats": mismatched,
                    "matching_minus_mismatched_nats": matching - mismatched,
                }
            )
    return rows


def _summaries(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary: dict[tuple[str, int], list[float]] = defaultdict(list)
    recall_seed_means: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for run in runs:
        arm = run["arm_name"]
        seed = run["seed"]
        for row in run["ordinary_long_context"]:
            ordinary[(arm, row["sequence_length_tokens"])].append(
                row["bits_per_raw_byte"]
            )
        for row in run["recall"]:
            recall_seed_means[(arm, seed, row["distance_raw_bytes"])].append(
                row["matching_minus_mismatched_nats"]
            )
    ordinary_rows = []
    for (arm, length), values in sorted(ordinary.items()):
        ordinary_rows.append(
            {
                "arm_name": arm,
                "sequence_length_tokens": length,
                "mean_bprb_across_seeds": statistics.mean(values),
                "worst_bprb_across_seeds": max(values),
            }
        )
    recall_rows = []
    arm_distance: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (arm, seed, distance), values in recall_seed_means.items():
        arm_distance[(arm, distance)].append(statistics.mean(values))
    for (arm, distance), seed_means in sorted(arm_distance.items()):
        recall_rows.append(
            {
                "arm_name": arm,
                "distance_raw_bytes": distance,
                "mean_gain_nats_across_seed_means": statistics.mean(seed_means),
                "minimum_seed_mean_gain_nats": min(seed_means),
                "positive_seed_means": sum(value > 0.0 for value in seed_means),
                "seeds": len(seed_means),
                "descriptive_support": all(value > 0.0 for value in seed_means),
            }
        )
    return {"ordinary_long_context": ordinary_rows, "recall": recall_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g12c-report", type=Path, required=True)
    parser.add_argument("--tokenizer-audit", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    git_commit, git_status = _git()
    _, validation_text, snapshot = _snapshot_text(args.snapshot)
    validation_ascii = _ascii_filler(validation_text)
    g12c = json.loads(args.g12c_report.read_text(encoding="utf-8"))
    tokenizer_audit = json.loads(args.tokenizer_audit.read_text(encoding="utf-8"))
    tokenizer_path = Path(tokenizer_audit["selection_rule"]["selected_path"])
    bpe = ByteLevelBPETokenizer.from_serialized(
        tokenizer_path.read_text(encoding="utf-8")
    )
    started = time.perf_counter()
    runs = []
    encoded_validation: dict[str, EncodedText] = {}
    for source_run in g12c["runs"]:
        tokenizer = _tokenizer_for_run(source_run, bpe)
        fingerprint = tokenizer_fingerprint(tokenizer)
        if fingerprint not in encoded_validation:
            encoded_validation[fingerprint] = tokenizer.encode(validation_text)
        checkpoint = Path(source_run["checkpoint"])
        if _sha256(checkpoint) != source_run["checkpoint_sha256"]:
            raise ValueError("G12C checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        config = HybridMemoryConfig(**payload["model_config"])
        model = HybridMemoryLM(config).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        ordinary = [
            _ordinary_evaluation(
                model,
                encoded_validation[fingerprint],
                namespace=f"g12d:{source_run['seed']}:{fingerprint}:{length}",
                sequence_length=length,
                device=device,
            )
            for length in CONTEXT_TOKEN_LENGTHS
        ]
        recall = _recall_evaluation(model, tokenizer, validation_ascii, device=device)
        runs.append(
            {
                "seed": source_run["seed"],
                "arm_name": source_run["arm_name"],
                "optimizer_name": source_run["optimizer_name"],
                "parameter_count": source_run["parameter_count"],
                "tokenizer": source_run["tokenizer"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": source_run["checkpoint_sha256"],
                "ordinary_long_context": ordinary,
                "recall": recall,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summaries = _summaries(runs)
    output = {
        "schema_version": 1,
        "stage": "G12D",
        "claim_status": (
            "post-pretraining long-context execution and counterfactual recall probe"
        ),
        "ordinary_context_token_lengths": list(CONTEXT_TOKEN_LENGTHS),
        "recall_raw_byte_distances": list(RECALL_RAW_BYTE_DISTANCES),
        "recall_examples_per_distance": RECALL_EXAMPLES,
        "recall_seed": RECALL_SEED,
        "summaries": summaries,
        "runs": runs,
        "dataset": {
            "snapshot": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "hub_sha": snapshot["hub_sha_at_snapshot"],
            "validation_only": True,
        },
        "inputs": {
            "g12c_report": str(args.g12c_report),
            "g12c_report_sha256": _sha256(args.g12c_report),
            "tokenizer_audit": str(args.tokenizer_audit),
            "tokenizer_audit_sha256": _sha256(args.tokenizer_audit),
        },
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "git_commit_at_start": git_commit,
        "git_status_at_start": git_status,
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
            "ordinary next-token pretrained checkpoints; templated paired recall "
            "probe is not corpus perplexity, instruction following, or a trained "
            "retrieval task"
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
