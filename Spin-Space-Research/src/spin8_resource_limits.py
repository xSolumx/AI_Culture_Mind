"""Resource contracts for the reference Spin(8) workstation.

The exact proof pipeline is intentionally staged.  This module provides a
small subprocess supervisor that limits CPU affinity and thread-pool fanout,
records peak resident memory for the complete process tree, and terminates a
stage before it crosses a conservative RAM threshold.

The watchdog is a safety mechanism, not a proof primitive.  Exact theorem
status continues to come only from the arithmetic certificates themselves.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

DEFAULT_WORKERS = 6
DEFAULT_MEMORY_GIB = 15.0
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def bounded_environment(workers: int = DEFAULT_WORKERS) -> dict[str, str]:
    """Return a child environment with every common native pool capped."""

    if not 1 <= workers <= 7:
        raise ValueError("workers must leave at least one i7-9700K core free")
    environment = os.environ.copy()
    for name in THREAD_ENVIRONMENT:
        environment[name] = str(workers)
    environment["SYMPY_GROUND_TYPES"] = "flint"
    environment.setdefault("PYTHONHASHSEED", "0")
    return environment


def constrain_current_process(workers: int = DEFAULT_WORKERS) -> list[int]:
    """Restrict the current process to the first available logical CPUs."""

    if not 1 <= workers <= 7:
        raise ValueError("workers must leave at least one i7-9700K core free")
    process = psutil.Process()
    available = process.cpu_affinity()
    if workers > len(available):
        raise ValueError(
            f"requested {workers} workers but only {len(available)} CPUs are available"
        )
    selected = available[:workers]
    process.cpu_affinity(selected)
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = str(workers)
    return selected


def _tree(process: psutil.Process) -> list[psutil.Process]:
    try:
        return [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _resident_bytes(process: psutil.Process) -> int:
    total = 0
    for member in _tree(process):
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _apply_affinity(process: psutil.Process, cpus: list[int]) -> None:
    for member in _tree(process):
        try:
            if member.cpu_affinity() != cpus:
                member.cpu_affinity(cpus)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            pass


def _terminate_tree(process: psutil.Process) -> None:
    members = list(reversed(_tree(process)))
    for member in members:
        try:
            member.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(members, timeout=3.0)
    for member in alive:
        try:
            member.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def run_bounded(
    command: list[str],
    *,
    workers: int = DEFAULT_WORKERS,
    memory_gib: float = DEFAULT_MEMORY_GIB,
    poll_seconds: float = 0.1,
) -> dict[str, object]:
    """Run one stage under affinity, thread, and process-tree RSS limits."""

    if not command:
        raise ValueError("a child command is required")
    if not 0 < memory_gib < 16:
        raise ValueError("memory_gib must be positive and strictly below 16")
    current = psutil.Process()
    available_cpus = current.cpu_affinity()
    if workers > len(available_cpus):
        raise ValueError(
            f"requested {workers} workers but only {len(available_cpus)} CPUs are available"
        )
    cpus = available_cpus[:workers]
    limit_bytes = int(memory_gib * 2**30)
    started = time.perf_counter()
    child = subprocess.Popen(command, env=bounded_environment(workers))
    process = psutil.Process(child.pid)
    peak = 0
    exceeded = False
    while child.poll() is None:
        _apply_affinity(process, cpus)
        resident = _resident_bytes(process)
        peak = max(peak, resident)
        if resident >= limit_bytes:
            exceeded = True
            _terminate_tree(process)
            break
        time.sleep(poll_seconds)
    return_code = child.wait()
    elapsed = time.perf_counter() - started
    report = {
        "command": command,
        "workers": workers,
        "cpu_affinity": cpus,
        "memory_limit_gib": memory_gib,
        "peak_process_tree_rss_gib": peak / 2**30,
        "memory_limit_exceeded": exceeded,
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "passed": return_code == 0 and not exceeded,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--memory-gib", type=float, default=DEFAULT_MEMORY_GIB)
    parser.add_argument("--report", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    report = run_bounded(
        command,
        workers=arguments.workers,
        memory_gib=arguments.memory_gib,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(
            137 if report["memory_limit_exceeded"] else report["return_code"]
        )


if __name__ == "__main__":
    main()
