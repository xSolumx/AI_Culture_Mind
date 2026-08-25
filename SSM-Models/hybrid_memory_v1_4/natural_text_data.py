"""Snapshot fixed TinyStories rows through the Hugging Face Dataset Viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "roneneldan/TinyStories"
CONFIG = "default"
VIEWER_BASE = "https://datasets-server.huggingface.co"
HUB_API = "https://huggingface.co/api/datasets/roneneldan/TinyStories"
SEPARATOR = "\n\n<|story|>\n\n"


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hybrid-memory-v1.4-research"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _viewer_url(endpoint: str, **parameters: object) -> str:
    return f"{VIEWER_BASE}/{endpoint}?{urllib.parse.urlencode(parameters)}"


def _fetch_rows(split: str, count: int) -> list[dict[str, object]]:
    rows = []
    for offset in range(0, count, 100):
        length = min(100, count - offset)
        payload = _get_json(
            _viewer_url(
                "rows",
                dataset=DATASET,
                config=CONFIG,
                split=split,
                offset=offset,
                length=length,
            )
        )
        page = payload.get("rows")
        if not isinstance(page, list) or len(page) != length:
            raise RuntimeError(f"incomplete Dataset Viewer page at {split}:{offset}")
        for expected_index, item in enumerate(page, start=offset):
            if item.get("row_idx") != expected_index:
                raise RuntimeError("Dataset Viewer returned a noncontiguous row index")
            if item.get("truncated_cells"):
                raise RuntimeError("Dataset Viewer truncated a selected text cell")
            row = item.get("row")
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text:
                raise RuntimeError("selected TinyStories row has no text")
            rows.append({"row_idx": expected_index, "text": text})
    return rows


def rows_to_bytes(rows: list[dict[str, object]]) -> bytes:
    texts = [row["text"] for row in rows]
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("every snapshot row must contain string text")
    return SEPARATOR.join(texts).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=2000)
    parser.add_argument("--validation-rows", type=int, default=256)
    args = parser.parse_args()
    if args.train_rows < 1 or args.validation_rows < 1:
        parser.error("row counts must be positive")
    started = time.perf_counter()
    validity = _get_json(_viewer_url("is-valid", dataset=DATASET))
    if not validity.get("viewer"):
        raise RuntimeError("TinyStories Dataset Viewer is not available")
    splits = _get_json(_viewer_url("splits", dataset=DATASET))
    available = {
        (item.get("config"), item.get("split")) for item in splits.get("splits", [])
    }
    if {(CONFIG, "train"), (CONFIG, "validation")} - available:
        raise RuntimeError("required TinyStories splits are unavailable")
    hub = _get_json(HUB_API)
    train_rows = _fetch_rows("train", args.train_rows)
    validation_rows = _fetch_rows("validation", args.validation_rows)
    train_bytes = rows_to_bytes(train_rows)
    validation_bytes = rows_to_bytes(validation_rows)
    train_texts = {row["text"] for row in train_rows}
    validation_texts = {row["text"] for row in validation_rows}
    report = {
        "schema_version": 1,
        "dataset": DATASET,
        "config": CONFIG,
        "hub_sha_at_snapshot": hub.get("sha"),
        "license": (hub.get("cardData") or {}).get("license"),
        "source": "Hugging Face Dataset Viewer rows API",
        "viewer_base": VIEWER_BASE,
        "row_selection": "zero-based contiguous prefix of each official split",
        "separator": SEPARATOR,
        "train": {
            "split": "train",
            "rows": train_rows,
            "row_count": len(train_rows),
            "byte_count": len(train_bytes),
            "bytes_sha256": hashlib.sha256(train_bytes).hexdigest(),
        },
        "validation": {
            "split": "validation",
            "rows": validation_rows,
            "row_count": len(validation_rows),
            "byte_count": len(validation_bytes),
            "bytes_sha256": hashlib.sha256(validation_bytes).hexdigest(),
        },
        "train_validation_byte_streams_distinct": train_bytes != validation_bytes,
        "exact_story_overlap_count": len(train_texts & validation_texts),
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "hub_sha": report["hub_sha_at_snapshot"],
                "train_bytes": len(train_bytes),
                "validation_bytes": len(validation_bytes),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
