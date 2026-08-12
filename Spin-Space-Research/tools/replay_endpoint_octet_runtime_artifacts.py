"""Regenerate the twenty endpoint-octet outputs found outside the manifest.

Nineteen inputs are 2026-08-10 resource-supervisor records.  Their recorded
commands are replayed into an isolated runtime directory and the generated
mathematical JSON is compared byte-for-byte with its canonical artifact.  The
twentieth input is the completed 32-child ``00001`` atlas, for which the full
command is reconstructed explicitly.

Runtime duration, peak RSS, and return codes are intentionally recorded as new
evidence rather than compared bytewise: those fields are machine-state
dependent. Strict byte equality and mathematical-payload equality are reported
separately. The latter may ignore only the frozen audit-engine provenance keys
listed in ``SCHEMA_ONLY_KEYS``; every mathematical field must still agree.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from spin8_resource_limits import run_bounded

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_GLOB = "spin8_dirac_endpoint_octet_cubic_*runtime_20260810.json"
NESTED_CANONICAL = (
    ROOT
    / "artifacts"
    / "spin8_dirac_endpoint_octet_cubic_atlas_nested_00001_complete_20260810.json"
)
DEFAULT_SOURCE_RUNTIME_DIR = ROOT / "runtime" / "endpoint-octet" / "2026-08-10"
DEFAULT_REPLAY_DIR = ROOT / "runtime" / "endpoint-octet" / "replay-2026-08-11"
SCHEMA_ONLY_KEYS = frozenset({"audit_engine", "batch_entry_limit"})
CONDITIONAL_SCHEMA_KEYS = frozenset(
    {"parent_path", "post_zero_path", "selected_zero_face"}
)
LEGACY_UI_FACE_LABEL = "ui=0 after the tangent radial selector"
CURRENT_UI_FACE_LABEL = "ui"
LEGACY_UI_SCOPE = (
    "This atlas certifies only the selected ui=0 face core. The full chart also "
    "requires the exact selector identity and nonnegative complement recorded by "
    "the parent artifacts."
)
CURRENT_SELECTED_FACE_SCOPE = (
    "This atlas certifies only one selected face core. The full chart also "
    "requires the exact selector identities and nonnegative complements recorded "
    "by the parent artifacts."
)


@dataclass(frozen=True)
class ReplayJob:
    job_id: str
    command: tuple[str, ...]
    canonical_output: Path
    source_runtime: Path | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _strip_schema_metadata(value: object) -> object:
    if isinstance(value, dict):
        normalized = {
            key: _strip_schema_metadata(item)
            for key, item in value.items()
            if key not in SCHEMA_ONLY_KEYS
        }
        if normalized.get("post_zero_path", object()) == []:
            normalized.pop("post_zero_path")
        if normalized.get("parent_path", object()) == []:
            normalized.pop("parent_path")
        if normalized.get("selected_zero_face", object()) is None:
            normalized.pop("selected_zero_face")
        if normalized.get("selected_face") == LEGACY_UI_FACE_LABEL:
            normalized["selected_face"] = CURRENT_UI_FACE_LABEL
        if normalized.get("scope_boundary") == LEGACY_UI_SCOPE:
            normalized["scope_boundary"] = CURRENT_SELECTED_FACE_SCOPE
        return normalized
    if isinstance(value, list):
        return [_strip_schema_metadata(item) for item in value]
    return value


def _difference_paths(left: object, right: object, *, path: str = "$") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_difference_paths(left[key], right[key], path=child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}.length"]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            paths.extend(
                _difference_paths(left_item, right_item, path=f"{path}[{index}]")
            )
        return paths
    return [] if left == right else [path]


def _compare_outputs(canonical: Path, fresh: Path) -> dict[str, object]:
    if not fresh.is_file():
        return {
            "fresh_sha256": None,
            "byte_output_match": False,
            "mathematical_output_match": False,
            "json_difference_paths": ["$ (fresh output missing)"],
            "schema_compatibility_used": False,
        }
    canonical_sha = _sha256(canonical)
    fresh_sha = _sha256(fresh)
    byte_match = fresh_sha == canonical_sha
    canonical_payload = json.loads(canonical.read_text(encoding="utf-8"))
    fresh_payload = json.loads(fresh.read_text(encoding="utf-8"))
    difference_paths = _difference_paths(canonical_payload, fresh_payload)
    mathematical_match = _strip_schema_metadata(
        canonical_payload
    ) == _strip_schema_metadata(fresh_payload)
    return {
        "fresh_sha256": fresh_sha,
        "byte_output_match": byte_match,
        "mathematical_output_match": mathematical_match,
        "json_difference_paths": difference_paths,
        "schema_compatibility_used": bool(mathematical_match and not byte_match),
    }


def _refresh_result_comparison(result: dict[str, object]) -> dict[str, object]:
    canonical = ROOT / str(result["canonical_output"])
    fresh = ROOT / str(result["fresh_output"])
    comparison = _compare_outputs(canonical, fresh)
    result.update(comparison)
    # Retain the original field for readers of the first campaign schema. It
    # means strict byte equality and is never widened by compatibility logic.
    result["exact_output_match"] = comparison["byte_output_match"]
    return result


def _historical_jobs(source_runtime_dir: Path) -> list[ReplayJob]:
    source_files = sorted(source_runtime_dir.glob(HISTORICAL_GLOB))
    if not source_files and source_runtime_dir == DEFAULT_SOURCE_RUNTIME_DIR:
        source_files = sorted((ROOT / "artifacts").glob(HISTORICAL_GLOB))
    if len(source_files) != 19:
        raise RuntimeError(
            f"expected 19 historical runtime records, found {len(source_files)} "
            f"under {source_runtime_dir} (with the artifacts fallback)"
        )

    jobs: list[ReplayJob] = []
    for runtime_path in source_files:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        command = list(payload["command"])
        if command[:1] != ["python"]:
            raise ValueError(f"{runtime_path.name}: frozen command does not use python")
        try:
            output_index = command.index("--output") + 1
        except ValueError as error:
            raise ValueError(f"{runtime_path.name}: command has no --output") from error
        canonical_output = ROOT / command[output_index]
        if canonical_output.parent != ROOT / "artifacts":
            raise ValueError(
                f"{runtime_path.name}: output escapes the canonical artifact directory"
            )
        if not canonical_output.is_file():
            raise FileNotFoundError(canonical_output)
        command[0] = sys.executable
        jobs.append(
            ReplayJob(
                job_id=runtime_path.stem.removesuffix("_runtime_20260810"),
                command=tuple(command),
                canonical_output=canonical_output,
                source_runtime=runtime_path,
            )
        )
    return jobs


def _nested_atlas_job() -> ReplayJob:
    command = [
        sys.executable,
        "src/spin8_dirac_endpoint_octet_cubic_atlas.py",
        "--delegate-corner-path",
        "00001/00001",
        "--flint-threads",
        "6",
    ]
    for bits in itertools.product("01", repeat=5):
        command.extend(("--box", f"00001/{''.join(bits)}"))
    command.extend(("--output", NESTED_CANONICAL.as_posix()))
    return ReplayJob(
        job_id="spin8_dirac_endpoint_octet_cubic_atlas_nested_00001_complete",
        command=tuple(command),
        canonical_output=NESTED_CANONICAL,
        source_runtime=None,
    )


def build_jobs(source_runtime_dir: Path) -> list[ReplayJob]:
    # Start the long independent atlas beside the historical queue so a
    # two-job campaign does not leave it running alone at the end.
    jobs = [_nested_atlas_job(), *_historical_jobs(source_runtime_dir)]
    if len(jobs) != 20 or len({job.job_id for job in jobs}) != 20:
        raise AssertionError("endpoint replay inventory is not exactly twenty jobs")
    return jobs


def _fresh_command(job: ReplayJob, output: Path) -> list[str]:
    command = list(job.command)
    output_index = command.index("--output") + 1
    command[output_index] = output.relative_to(ROOT).as_posix()
    return command


def _run_one(
    job: ReplayJob,
    *,
    replay_dir: Path,
    workers: int,
    memory_gib: float,
) -> dict[str, object]:
    output = replay_dir / "outputs" / job.canonical_output.name
    runtime_report = replay_dir / "reports" / f"{job.job_id}_runtime.json"
    command = _fresh_command(job, output)
    canonical_sha = _sha256(job.canonical_output)

    existing_comparison = _compare_outputs(job.canonical_output, output)
    if existing_comparison["mathematical_output_match"]:
        report: dict[str, object] = {
            "command": command,
            "passed": True,
            "return_code": 0,
            "skipped_exact_existing_output": bool(
                existing_comparison["byte_output_match"]
            ),
            "skipped_mathematically_matching_existing_output": True,
        }
    else:
        report = run_bounded(command, workers=workers, memory_gib=memory_gib)
        report["skipped_exact_existing_output"] = False
        report["skipped_mathematically_matching_existing_output"] = False
    _atomic_json(runtime_report, report)

    comparison = _compare_outputs(job.canonical_output, output)
    source_runtime_sha = _sha256(job.source_runtime) if job.source_runtime else None
    historical_runtime = (
        json.loads(job.source_runtime.read_text(encoding="utf-8"))
        if job.source_runtime is not None
        else None
    )
    return {
        "job_id": job.job_id,
        "source_runtime": (
            job.source_runtime.relative_to(ROOT).as_posix()
            if job.source_runtime is not None
            else None
        ),
        "source_runtime_sha256": source_runtime_sha,
        "historical_runtime": (
            {
                "return_code": historical_runtime["return_code"],
                "passed": historical_runtime["passed"],
                "elapsed_seconds": historical_runtime["elapsed_seconds"],
                "peak_process_tree_rss_gib": historical_runtime[
                    "peak_process_tree_rss_gib"
                ],
                "memory_limit_gib": historical_runtime["memory_limit_gib"],
                "memory_limit_exceeded": historical_runtime["memory_limit_exceeded"],
            }
            if historical_runtime is not None
            else None
        ),
        "canonical_output": job.canonical_output.relative_to(ROOT).as_posix(),
        "canonical_sha256": canonical_sha,
        "fresh_output": output.relative_to(ROOT).as_posix(),
        **comparison,
        "exact_output_match": comparison["byte_output_match"],
        "runtime_report": runtime_report.relative_to(ROOT).as_posix(),
        "runtime_passed": bool(report["passed"]),
        "return_code": report["return_code"],
        "memory_limit_exceeded": report.get("memory_limit_exceeded", False),
        "skipped_exact_existing_output": report["skipped_exact_existing_output"],
        "skipped_mathematically_matching_existing_output": report[
            "skipped_mathematically_matching_existing_output"
        ],
    }


def _campaign_payload(
    *,
    jobs: list[ReplayJob],
    results: dict[str, dict[str, object]],
    workers: int,
    parallel_jobs: int,
    memory_gib: float,
) -> dict[str, object]:
    ordered = [results[job.job_id] for job in jobs if job.job_id in results]
    complete = len(ordered) == len(jobs)
    return {
        "campaign": "Spin(8) endpoint-octet twenty-output exact replay",
        "schema_version": 2,
        "expected_job_count": len(jobs),
        "completed_job_count": len(ordered),
        "complete": complete,
        "exact_outputs_passed": complete
        and all(bool(row["byte_output_match"]) for row in ordered),
        "byte_outputs_passed": complete
        and all(bool(row["byte_output_match"]) for row in ordered),
        "mathematical_outputs_passed": complete
        and all(bool(row["mathematical_output_match"]) for row in ordered),
        "all_runtime_processes_passed": complete
        and all(bool(row["runtime_passed"]) for row in ordered),
        "resource_contract": {
            "parallel_jobs": parallel_jobs,
            "workers_per_job": workers,
            "memory_gib_per_job": memory_gib,
            "aggregate_memory_gib_cap": parallel_jobs * memory_gib,
        },
        "results": ordered,
        "interpretation": (
            "Strict byte equality is reported independently. The mathematical "
            "acceptance gate applies only the separately enumerated provenance-key "
            "removals, empty or null optional defaults, and exact old/current value "
            "aliases before structural equality; every coefficient, degree, sign "
            "count, route, and theorem field must agree. Runtime measurements are "
            "new machine evidence and are not expected to reproduce historical "
            "elapsed time or peak RSS byte-for-byte."
        ),
        "schema_only_keys": sorted(SCHEMA_ONLY_KEYS),
        "conditional_schema_keys": sorted(CONDITIONAL_SCHEMA_KEYS),
        "conditional_schema_rule": (
            "Omit parent_path and post_zero_path only when they are the empty "
            "default [] and selected_zero_face only when it is the null "
            "default. Nonempty paths and substantive face selectors remain."
        ),
        "exact_value_aliases": {
            "selected_face": {LEGACY_UI_FACE_LABEL: CURRENT_UI_FACE_LABEL},
            "scope_boundary": {LEGACY_UI_SCOPE: CURRENT_SELECTED_FACE_SCOPE},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-runtime-dir", type=Path, default=DEFAULT_SOURCE_RUNTIME_DIR
    )
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--memory-gib-per-job", type=float, default=7.4)
    arguments = parser.parse_args()
    if not 1 <= arguments.jobs <= 2:
        raise ValueError("at most two exact jobs may run concurrently on this host")
    if arguments.jobs * arguments.memory_gib_per_job >= 15.0:
        raise ValueError("aggregate replay memory cap must remain below 15 GiB")

    replay_dir = arguments.replay_dir.resolve()
    if ROOT not in replay_dir.parents:
        raise ValueError("replay directory must remain inside the repository")
    jobs = build_jobs(arguments.source_runtime_dir.resolve())
    campaign_path = replay_dir / "campaign.json"
    results: dict[str, dict[str, object]] = {}
    lock = threading.Lock()

    if campaign_path.is_file():
        previous = json.loads(campaign_path.read_text(encoding="utf-8"))
        results = {
            row["job_id"]: _refresh_result_comparison(row)
            for row in previous.get("results", [])
        }

    pending = [
        job
        for job in jobs
        if not results.get(job.job_id, {}).get("mathematical_output_match", False)
    ]
    with ThreadPoolExecutor(max_workers=arguments.jobs) as executor:
        future_jobs = {
            executor.submit(
                _run_one,
                job,
                replay_dir=replay_dir,
                workers=arguments.workers,
                memory_gib=arguments.memory_gib_per_job,
            ): job
            for job in pending
        }
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            result = future.result()
            with lock:
                results[job.job_id] = result
                payload = _campaign_payload(
                    jobs=jobs,
                    results=results,
                    workers=arguments.workers,
                    parallel_jobs=arguments.jobs,
                    memory_gib=arguments.memory_gib_per_job,
                )
                _atomic_json(campaign_path, payload)
            print(
                f"{payload['completed_job_count']:02d}/{len(jobs)} "
                f"{job.job_id}: byte_match={result['byte_output_match']} "
                f"mathematical_match={result['mathematical_output_match']} "
                f"runtime_passed={result['runtime_passed']}",
                flush=True,
            )

    final = _campaign_payload(
        jobs=jobs,
        results=results,
        workers=arguments.workers,
        parallel_jobs=arguments.jobs,
        memory_gib=arguments.memory_gib_per_job,
    )
    _atomic_json(campaign_path, final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final["mathematical_outputs_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
