"""Run the frozen G12 Stage-A lossless tokenizer audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .natural_text_data import SEPARATOR
    from .tokenization import (
        ByteLevelBPETokenizer,
        RawByteTokenizer,
        tokenizer_report,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.natural_text_data import SEPARATOR  # type: ignore[no-redef]
    from hybrid_memory_v1_4.tokenization import (  # type: ignore[no-redef]
        ByteLevelBPETokenizer,
        RawByteTokenizer,
        tokenizer_report,
    )


PREREGISTRATION = Path(__file__).with_name("G12_PREREGISTRATION.md")
VOCAB_SIZES = (512, 1024)
MINIMUM_BYTES_PER_TOKEN = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(rows: list[dict[str, Any]]) -> str:
    values = [row.get("text") for row in rows]
    if not values or not all(isinstance(value, str) for value in values):
        raise ValueError("snapshot rows must contain nonempty text strings")
    return SEPARATOR.join(values)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    train_text = _text(snapshot["train"]["rows"])
    validation_text = _text(snapshot["validation"]["rows"])
    args.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    raw = RawByteTokenizer()
    candidates.append(
        tokenizer_report(raw, raw.encode(train_text), raw.encode(validation_text))
    )
    for vocab_size in VOCAB_SIZES:
        tokenizer = ByteLevelBPETokenizer.train(train_text, vocab_size=vocab_size)
        train = tokenizer.encode(train_text)
        validation = tokenizer.encode(validation_text)
        path = args.tokenizer_dir / f"tinystories_train_only_bytelevel_bpe_{vocab_size}.json"
        path.write_text(tokenizer.serialized() + "\n", encoding="utf-8")
        report = tokenizer_report(tokenizer, train, validation)
        report["path"] = str(path)
        report["file_sha256"] = _sha256(path)
        candidates.append(report)

    eligible = [
        candidate
        for candidate in candidates
        if candidate["name"] == "bytelevel_bpe"
        and candidate["train"]["bytes_per_token"] >= MINIMUM_BYTES_PER_TOKEN
        and candidate["train"]["round_trip"]
        and candidate["validation"]["round_trip"]
    ]
    selected = min(eligible, key=lambda candidate: candidate["vocab_size"], default=None)
    root = Path(__file__).resolve().parents[2]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    output = {
        "schema_version": 1,
        "stage": "G12A",
        "passed": selected is not None,
        "claim_status": (
            "selected a lossless training-only compressed tokenizer candidate"
            if selected is not None
            else "rejected fitted tokenizer candidates"
        ),
        "selection_rule": {
            "minimum_training_bytes_per_token": MINIMUM_BYTES_PER_TOKEN,
            "rule": "smallest exact ByteLevel BPE vocabulary meeting threshold",
            "selected_vocab_size": None if selected is None else selected["vocab_size"],
            "selected_sha256": None if selected is None else selected["sha256"],
            "selected_path": None if selected is None else selected["path"],
        },
        "candidates": candidates,
        "dataset": {
            "snapshot": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "hub_sha": snapshot["hub_sha_at_snapshot"],
            "train_fit_only": True,
            "validation_used_to_fit": False,
        },
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "git_commit_at_start": git_commit,
        "git_status_at_start": git_status,
        "claim_boundary": (
            "tokenizer compression and losslessness only; no model-quality result"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(output["selection_rule"], sort_keys=True))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
