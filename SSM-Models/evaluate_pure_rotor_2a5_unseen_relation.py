"""Evaluate frozen ``2.A5`` checkpoints on a deterministically unseen relation.

This runner performs no optimization.  It reconstructs the original training
schedules, selects the frozen protocol's shortest absent reduced identity/
center word pair, verifies checkpoint hashes, and evaluates the saved cohort on
new paired contexts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from benchmark_pure_rotor_2a5 import (
    ORACLE_CANDIDATES,
    BinaryA5BenchmarkConfig,
    BinaryA5Task,
    CentralPairBatch,
    _prefix_targets,
    _sample_pair_free_contexts,
    binary_icosahedral_task,
    build_models,
    evaluate_central_pairs,
    make_training_batches,
)
from benchmark_pure_rotor_a5 import batch_schedule_sha256

FORBIDDEN_LOCAL_PAIRS = ((0, 0), (1, 2), (2, 1))
SOURCE_ARTIFACT_SHA256 = (
    "911815d9e104fa08e632161f97f41a966991a9102c70ca65e52a5f07d28d4476"
)


def _config_for_seed(
    source_report: dict[str, object], seed: int
) -> BinaryA5BenchmarkConfig:
    values = dict(source_report["config"])
    values["seed"] = seed
    values["evaluation_lengths"] = tuple(values["evaluation_lengths"])
    return BinaryA5BenchmarkConfig(**values)


def _word_product(task: BinaryA5Task, word: tuple[int, ...]) -> int:
    state = 0
    for token in word:
        state = int(task.group.table[state, task.input_elements[token]])
    return state


def _is_locally_reduced(word: tuple[int, ...]) -> bool:
    return all(
        tuple(word[position : position + 2]) not in FORBIDDEN_LOCAL_PAIRS
        for position in range(len(word) - 1)
    )


def select_unseen_relation_words(
    task: BinaryA5Task,
    source_report: dict[str, object],
) -> dict[str, object]:
    """Apply the predeclared input-only lexicographic selection rule."""

    training_by_seed: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    maximum_length = 0
    for seed in source_report["seeds"]:
        config = _config_for_seed(source_report, int(seed))
        training = make_training_batches(task, config)
        expected_hash = source_report["task"]["split"][str(seed)]["training"][
            "training_schedule_sha256"
        ]
        actual_hash = batch_schedule_sha256(training)
        if actual_hash != expected_hash:
            raise RuntimeError(f"training schedule replay mismatch for seed {seed}")
        training_by_seed[int(seed)] = training
        maximum_length = max(maximum_length, config.training_length)

    occurrence_counts: dict[tuple[int, ...], dict[int, int]] = {}
    selected_length = None
    identity_word = None
    center_word = None
    for length in range(2, maximum_length + 1):
        seen = set()
        for training in training_by_seed.values():
            for inputs, _ in training:
                for row in inputs.tolist():
                    row_tuple = tuple(map(int, row))
                    seen.update(
                        row_tuple[start : start + length]
                        for start in range(len(row_tuple) - length + 1)
                    )
        identity_candidates = []
        center_candidates = []
        for word in itertools.product(range(len(task.input_elements)), repeat=length):
            if word in seen or not _is_locally_reduced(word):
                continue
            product = _word_product(task, word)
            if product == 0:
                identity_candidates.append(word)
            elif product == task.center_index:
                center_candidates.append(word)
        if identity_candidates and center_candidates:
            selected_length = length
            identity_word = identity_candidates[0]
            center_word = center_candidates[0]
            break
    if selected_length is None or identity_word is None or center_word is None:
        raise RuntimeError("no unseen identity/center relation pair was found")

    for word in (identity_word, center_word):
        occurrence_counts[word] = {}
        for seed, training in training_by_seed.items():
            count = 0
            for inputs, _ in training:
                for row in inputs.tolist():
                    row_tuple = tuple(map(int, row))
                    count += sum(
                        row_tuple[start : start + selected_length] == word
                        for start in range(len(row_tuple) - selected_length + 1)
                    )
            occurrence_counts[word][seed] = count

    return {
        "selection_rule": (
            "shortest joint length, then lexicographically first locally reduced "
            "words absent from every realized training input"
        ),
        "forbidden_local_pairs": [list(pair) for pair in FORBIDDEN_LOCAL_PAIRS],
        "length": selected_length,
        "identity_word": list(identity_word),
        "center_word": list(center_word),
        "identity_product": _word_product(task, identity_word),
        "center_product": _word_product(task, center_word),
        "training_occurrences": {
            "identity_word": occurrence_counts[identity_word],
            "center_word": occurrence_counts[center_word],
        },
    }


def make_unseen_relation_batches(
    task: BinaryA5Task,
    config: BinaryA5BenchmarkConfig,
    *,
    length: int,
    relation_position: str,
    identity_word: tuple[int, ...],
    center_word: tuple[int, ...],
) -> list[CentralPairBatch]:
    """Place an equal-length identity/center word in shared random contexts."""

    relation_length = len(identity_word)
    if len(center_word) != relation_length or length < relation_length:
        raise ValueError("relation words must have equal length within the sequence")
    if relation_position not in {"early", "late"}:
        raise ValueError("relation_position must be early or late")
    batches = []
    for batch_index in range(config.validation_batches):
        generator = np.random.default_rng(
            80_000
            + 10_000 * config.seed
            + 1_000 * length
            + 100 * (relation_position == "late")
            + batch_index
        )
        context = _sample_pair_free_contexts(
            config.validation_pairs_per_batch,
            length - relation_length,
            generator,
            forbid_first_a=(relation_position == "early" and center_word[-1] == 0),
            forbid_last_a=(relation_position == "late" and center_word[0] == 0),
        )
        center_block = np.tile(np.asarray(center_word), (len(context), 1))
        identity_block = np.tile(np.asarray(identity_word), (len(context), 1))
        if relation_position == "early":
            center_words = np.concatenate((center_block, context), axis=1)
            identity_words = np.concatenate((identity_block, context), axis=1)
            post_start = relation_length - 1
        else:
            center_words = np.concatenate((context, center_block), axis=1)
            identity_words = np.concatenate((context, identity_block), axis=1)
            post_start = length - 1
        interleaved = np.empty((2 * len(context), length), dtype=np.int64)
        interleaved[0::2] = center_words
        interleaved[1::2] = identity_words
        targets = _prefix_targets(task, interleaved)
        mask = np.zeros_like(interleaved, dtype=bool)
        mask[:, post_start:] = True
        batches.append(
            CentralPairBatch(
                inputs=torch.from_numpy(interleaved),
                targets=torch.from_numpy(targets),
                post_relation_mask=torch.from_numpy(mask),
                relation_position=relation_position,
            )
        )
    return batches


def unseen_relation_audit(
    task: BinaryA5Task,
    batches: list[CentralPairBatch],
    *,
    identity_word: tuple[int, ...],
    center_word: tuple[int, ...],
) -> dict[str, object]:
    """Verify forced blocks and exact post-block central pairing."""

    digest = hashlib.sha256()
    paired_sequences = 0
    post_positions = 0
    central_partner_checks = 0
    projective_checks = 0
    forced_center_checks = 0
    forced_identity_checks = 0
    relation_length = len(identity_word)
    for batch in batches:
        for tensor in (batch.inputs, batch.targets, batch.post_relation_mask):
            values = tensor.detach().cpu().contiguous()
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.numpy().tobytes())
        inputs = batch.inputs.numpy()
        targets = batch.targets.numpy()
        mask = batch.post_relation_mask.numpy()
        pairs = len(inputs) // 2
        paired_sequences += pairs
        relation_start = (
            0
            if batch.relation_position == "early"
            else inputs.shape[1] - relation_length
        )
        forced_center_checks += int(
            np.all(
                inputs[0::2, relation_start : relation_start + relation_length]
                == np.asarray(center_word),
                axis=1,
            ).sum()
        )
        forced_identity_checks += int(
            np.all(
                inputs[1::2, relation_start : relation_start + relation_length]
                == np.asarray(identity_word),
                axis=1,
            ).sum()
        )
        for center_row in range(0, len(inputs), 2):
            identity_row = center_row + 1
            for position in np.flatnonzero(mask[center_row]):
                center_target = int(targets[center_row, position])
                identity_target = int(targets[identity_row, position])
                post_positions += 1
                central_partner_checks += int(
                    task.central_partner[identity_target] == center_target
                )
                projective_checks += int(
                    task.projective_label[identity_target]
                    == task.projective_label[center_target]
                )
    passed = (
        forced_center_checks == paired_sequences
        and forced_identity_checks == paired_sequences
        and central_partner_checks == post_positions
        and projective_checks == post_positions
    )
    return {
        "paired_sequences": paired_sequences,
        "post_relation_pair_positions": post_positions,
        "forced_center_word_checks": forced_center_checks,
        "forced_identity_word_checks": forced_identity_checks,
        "exact_central_partner_checks": central_partner_checks,
        "projective_match_checks": projective_checks,
        "evaluation_schedule_sha256": digest.hexdigest(),
        "passed": passed,
    }


def run_evaluation(
    source_path: Path,
    *,
    device: torch.device,
    evaluation_lengths: tuple[int, ...] = (16, 64, 128),
    microbatch_size: int = 16,
) -> dict[str, object]:
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != SOURCE_ARTIFACT_SHA256:
        raise RuntimeError("source cohort SHA-256 does not match the frozen protocol")
    source = json.loads(source_bytes)
    task = binary_icosahedral_task()
    selection = select_unseen_relation_words(task, source)
    identity_word = tuple(selection["identity_word"])
    center_word = tuple(selection["center_word"])
    if (
        selection["identity_product"] != 0
        or selection["center_product"] != task.center_index
    ):
        raise RuntimeError("selected words do not have the required exact products")
    if any(
        count
        for word_counts in selection["training_occurrences"].values()
        for count in word_counts.values()
    ):
        raise RuntimeError("selected relation word occurred in training")

    rows_by_seed = {
        int(seed): {
            row["name"]: row
            for row in source["results"]
            if int(row["seed"]) == int(seed)
        }
        for seed in source["seeds"]
    }
    all_results = []
    all_oracles: dict[str, object] = {}
    all_audits: dict[str, object] = {}
    for raw_seed in source["seeds"]:
        seed = int(raw_seed)
        config = _config_for_seed(source, seed)
        evaluations = {}
        audits = {}
        for length in evaluation_lengths:
            if length < len(identity_word):
                raise ValueError(
                    "evaluation length is shorter than the selected relation"
                )
            for position in ("early", "late"):
                key = f"{position}_L{length}"
                evaluations[key] = make_unseen_relation_batches(
                    task,
                    config,
                    length=length,
                    relation_position=position,
                    identity_word=identity_word,
                    center_word=center_word,
                )
                audits[key] = unseen_relation_audit(
                    task,
                    evaluations[key],
                    identity_word=identity_word,
                    center_word=center_word,
                )
                if not audits[key]["passed"]:
                    raise RuntimeError(
                        f"unseen relation audit failed for seed {seed} {key}"
                    )
        all_audits[str(seed)] = audits
        all_oracles[str(seed)] = {
            name: {
                key: evaluate_central_pairs(
                    name,
                    None,
                    batches,
                    task,
                    device,
                    "parallel",
                    "parallel",
                    microbatch_size,
                )
                for key, batches in evaluations.items()
            }
            for name in ORACLE_CANDIDATES
        }

        models = build_models(task, config)
        for name, source_row in rows_by_seed[seed].items():
            checkpoint_path = Path(source_row["checkpoint"])
            if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
                checkpoint_path = (
                    Path(__file__).resolve().parent.parent / checkpoint_path
                )
            checkpoint_bytes = checkpoint_path.read_bytes()
            checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
            if checkpoint_sha256 != source_row["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint SHA-256 mismatch: {checkpoint_path}")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if payload["candidate"] != name:
                raise RuntimeError(f"checkpoint candidate mismatch: {checkpoint_path}")
            model = models[name]
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to(device).eval()
            metrics = {
                key: evaluate_central_pairs(
                    name,
                    model,
                    batches,
                    task,
                    device,
                    source["rotor_scan_mode"],
                    source["quaternion_scan_mode"],
                    microbatch_size,
                )
                for key, batches in evaluations.items()
            }
            all_results.append(
                {
                    "name": name,
                    "seed": seed,
                    "parameters": source_row["parameters"],
                    "source_checkpoint": str(checkpoint_path),
                    "source_checkpoint_sha256": checkpoint_sha256,
                    "metrics": metrics,
                }
            )
            model.to("cpu")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return {
        "experiment": "frozen-checkpoint unseen 2.A5 relation generalization",
        "status": "completed post-pilot exploratory evaluation; no retraining",
        "source_artifact": str(source_path),
        "source_artifact_sha256": source_sha256,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "torch_version": torch.__version__,
        "config": {
            "seeds": list(map(int, source["seeds"])),
            "evaluation_lengths": list(evaluation_lengths),
            "validation_batches": source["config"]["validation_batches"],
            "validation_pairs_per_batch": source["config"][
                "validation_pairs_per_batch"
            ],
            "evaluation_microbatch_size": microbatch_size,
            "rotor_scan_mode": source["rotor_scan_mode"],
            "quaternion_scan_mode": source["quaternion_scan_mode"],
        },
        "task": {
            "group": "binary icosahedral group 2.A5",
            "order": task.group.order,
            "center_index": task.center_index,
            "group_table_sha256": task.group_table_sha256,
            "selection": selection,
            "evaluation_audits": all_audits,
        },
        "oracle_results": all_oracles,
        "results": all_results,
        "claim_scope": {
            "empirical": [
                "fixed-checkpoint generalization to one deterministically selected unseen central relation pair"
            ],
            "not_claimed": [
                "a preregistered result",
                "general relation or language-model generalization",
                "a theorem about any trained model family",
            ],
        },
    }


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item.strip())
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("evaluation lengths must be nonempty and distinct")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--evaluation-lengths", default="16,64,128")
    parser.add_argument("--evaluation-microbatch-size", type=int, default=16)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    report = run_evaluation(
        args.source,
        device=torch.device(args.device),
        evaluation_lengths=parse_lengths(args.evaluation_lengths),
        microbatch_size=args.evaluation_microbatch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
