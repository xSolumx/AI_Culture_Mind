"""Generate the publication-facing memory benchmark figure atlas.

The figures are derived from frozen aggregate artifacts.  This module does not
run or retime a benchmark; it renders already-qualified evidence and keeps
quality, systems, and implementation-status comparisons in separate panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPOSITORY_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "figures" / "memory_benchmark_atlas"

SOURCE_ARTIFACTS = {
    "hierarchy": "large_slot_semantic_hierarchy_seeds30_39.json",
    "fused": "fused_gathered_block_memory_cuda_aggregate_20260810.json",
    "matched": "matched_memory_cores_cuda_rtx2070s_frozen_aggregate_20260810.json",
    "fla": "fla_delta_rule_cuda_rtx2070s_frozen_aggregate_20260810.json",
    "comoving": "comoving_fla_frozen_aggregate_20260810.json",
    "task_b_strict": "task_b_delta_action_replay_seeds0_9.json",
    "task_b_paired": "task_b_paired_action_replication_seeds20_29.json",
}

INK = "#16202A"
MUTED = "#5D6975"
GRID = "#D9E0E7"
PAPER = "#FAFBFD"
BLUE = "#246BCE"
TEAL = "#159A9C"
ORANGE = "#E07A2D"
PURPLE = "#7656C9"
RED = "#C94C4C"
GREEN = "#2B8A57"
GOLD = "#B88912"
PALETTE = [BLUE, TEAL, ORANGE, PURPLE, RED, GREEN, GOLD]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing the frozen aggregate JSON artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for rendered figures and the source manifest.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg"),
        default=("png", "svg"),
        help="One or both deterministic output formats.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "svg.hashsalt": "spin8-memory-benchmark-atlas-20260810",
        }
    )


def finish_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.06, y=0.985, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.06, 0.935, subtitle, ha="left", va="top", color=MUTED, fontsize=10.5)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> list[Path]:
    output_paths: list[Path] = []
    for output_format in formats:
        path = output_dir / f"{stem}.{output_format}"
        metadata = {"Creator": "Spin(8) Triality Research memory atlas generator"}
        if output_format == "svg":
            metadata["Date"] = None
        fig.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata)
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def mean_min_max(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty series")
    return float(array.mean()), float(array.min()), float(array.max())


def add_range_series(
    ax: plt.Axes,
    x: Sequence[float],
    seed_values: Sequence[Sequence[float]],
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
    alpha: float = 0.12,
) -> None:
    summaries = [mean_min_max(values) for values in seed_values]
    means = np.asarray([summary[0] for summary in summaries])
    minima = np.asarray([summary[1] for summary in summaries])
    maxima = np.asarray([summary[2] for summary in summaries])
    ax.fill_between(x, minima, maxima, color=color, alpha=alpha, linewidth=0)
    ax.plot(
        x,
        means,
        color=color,
        marker=marker,
        linewidth=2.2,
        markersize=6,
        linestyle=linestyle,
        label=label,
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.4,
    radius: float = 0.02,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def figure_fla_fit_map(
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(14, 8.2))
    add_figure_header(
        fig,
        "Where the local memory work fits FLA hybrid models",
        "FLA schedules mixers across depth; the local programme adds a candidate sparse persistent-memory mixer and an operator-level transport compiler.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.055, 0.84, "FLA model shell", fontsize=14, fontweight="bold")
    layer_specs = [
        ("Layer 0", "Native linear / recurrent mixer", "#EAF2FD", BLUE),
        ("Layer 1", "Local attention", "#FFF2E8", ORANGE),
        ("Layer 2", "Native linear / recurrent mixer", "#EAF2FD", BLUE),
        ("Layer 3", "Selected-block memory (candidate)", "#E9F7F3", TEAL),
        ("Layer 4", "Full or local attention", "#FFF2E8", ORANGE),
    ]
    for index, (layer, label, face, edge) in enumerate(layer_specs):
        y = 0.72 - index * 0.115
        rounded_box(ax, (0.055, y), 0.34, 0.078, facecolor=face, edgecolor=edge)
        ax.text(0.075, y + 0.039, layer, va="center", fontsize=9, color=MUTED)
        ax.text(0.15, y + 0.039, label, va="center", fontsize=10.5, fontweight="bold")

    arrow = FancyArrowPatch(
        (0.405, 0.416),
        (0.492, 0.416),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2,
        color=TEAL,
    )
    ax.add_patch(arrow)

    ax.text(
        0.5, 0.84, "Candidate memory mixer internals", fontsize=14, fontweight="bold"
    )
    stages = [
        ("Hidden state", "token features", "#F1F4F7", MUTED),
        ("Shared canonical router", "coarse block + fine slot", "#F1ECFB", PURPLE),
        ("Actual block gather", "read only selected state", "#E9F7F3", TEAL),
        ("Fused direct / delta", "update + read in one kernel", "#EAF2FD", BLUE),
        ("Residual output", "return to model stream", "#F1F4F7", MUTED),
    ]
    stage_x = [0.5, 0.5, 0.5, 0.5, 0.5]
    stage_y = [0.72, 0.605, 0.49, 0.375, 0.26]
    for index, ((name, detail, face, edge), x, y) in enumerate(
        zip(stages, stage_x, stage_y)
    ):
        rounded_box(ax, (x, y), 0.43, 0.078, facecolor=face, edgecolor=edge)
        ax.text(
            x + 0.02, y + 0.051, name, va="center", fontsize=10.5, fontweight="bold"
        )
        ax.text(x + 0.02, y + 0.024, detail, va="center", fontsize=9, color=MUTED)
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.215, y - 0.003),
                    (x + 0.215, stage_y[index + 1] + 0.083),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.5,
                    color=MUTED,
                )
            )

    status = [
        ("ESTABLISHED", "FLA DeltaRule\noperator integration", GREEN),
        ("ESTABLISHED", "standalone fused\ngathered inference", GREEN),
        ("OPEN", "FLA cache/layer API\nand backward kernel", RED),
        ("OPEN", "trained hybrid-model\nadvantage", RED),
    ]
    x_positions = [0.055, 0.29, 0.525, 0.76]
    for x, (tag, label, color) in zip(x_positions, status):
        rounded_box(ax, (x, 0.075), 0.185, 0.095, facecolor="#FFFFFF", edgecolor=color)
        ax.text(x + 0.012, 0.143, tag, fontsize=8.5, color=color, fontweight="bold")
        ax.text(
            x + 0.012,
            0.108,
            label,
            fontsize=8.5,
            color=INK,
            va="center",
            linespacing=1.25,
        )

    return save_figure(fig, output_dir, "00_fla_hybrid_fit_map", formats)


def figure_campaign_dashboard(
    hierarchy: dict[str, Any],
    fused: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    summary = hierarchy["summary"]
    fused_rows = fused["rows"]
    fused_latencies_us = [
        row["timing"][key]["median_of_process_medians_ms"] * 1000.0
        for row in fused_rows
        for key in ("direct_triton_fused_gathered", "delta_triton_fused_gathered")
    ]
    direct_speedups = [
        row["comparisons"]["direct"]["fused_speedup_over_eager_dense"]
        for row in fused_rows
    ]
    delta_speedups = [
        row["comparisons"]["delta"]["fused_speedup_over_eager_dense"]
        for row in fused_rows
    ]

    fig, ax = plt.subplots(figsize=(14, 7.7))
    add_figure_header(
        fig,
        "Memory campaign: the result in one page",
        "Frozen quality cohorts and a separate three-process RTX 2070 SUPER systems campaign; numbers are not cross-protocol scores.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cards = [
        (
            "SHARED ROUTING",
            f"{summary['shared_heldout_hard_accuracy_mean']:.3f}",
            "held-out cross-view hard-route accuracy\n10 fixed seeds; 64 overlapping slots",
            PURPLE,
        ),
        (
            "DIRECT QUALITY",
            f"+{summary['direct_block_improvement_mean']:.3f}",
            "mean cosine gain: block top-1 vs dense soft\nfrozen semantic hierarchy cohort",
            BLUE,
        ),
        (
            "DELTA QUALITY",
            f"+{summary['delta_block_improvement_mean']:.3f}",
            "mean cosine gain: block top-1 vs dense soft\nfrozen semantic hierarchy cohort",
            TEAL,
        ),
        (
            "FUSED LATENCY",
            f"{min(fused_latencies_us):.1f}–{max(fused_latencies_us):.1f} µs",
            "one-step update + read\nall tested slot/batch cells",
            ORANGE,
        ),
        (
            "DIRECT SPEEDUP",
            f"{min(direct_speedups):.2f}–{max(direct_speedups):.2f}×",
            "fused gather over eager dense direct\nmedian of three process medians",
            BLUE,
        ),
        (
            "DELTA SPEEDUP",
            f"{min(delta_speedups):.2f}–{max(delta_speedups):.2f}×",
            "fused gather over eager dense delta\nmedian of three process medians",
            TEAL,
        ),
    ]
    positions = [
        (0.055, 0.55),
        (0.365, 0.55),
        (0.675, 0.55),
        (0.055, 0.20),
        (0.365, 0.20),
        (0.675, 0.20),
    ]
    for (tag, value, detail, color), (x, y) in zip(cards, positions):
        rounded_box(
            ax, (x, y), 0.27, 0.25, facecolor="#FFFFFF", edgecolor=color, linewidth=1.7
        )
        ax.text(x + 0.02, y + 0.205, tag, color=color, fontsize=9, fontweight="bold")
        ax.text(x + 0.02, y + 0.125, value, color=INK, fontsize=25, fontweight="bold")
        ax.text(
            x + 0.02, y + 0.045, detail, color=MUTED, fontsize=9.2, linespacing=1.35
        )

    ax.text(
        0.055,
        0.105,
        "Boundary: this establishes a routing prior and a standalone inference kernel—not extra storage capacity, a training kernel, or model-level language quality.",
        color=RED,
        fontsize=9.5,
        fontweight="bold",
    )
    return save_figure(fig, output_dir, "01_campaign_dashboard", formats)


def find_stream(run: dict[str, Any], radius: float, cohort: str) -> dict[str, Any]:
    for stream in run["streams"]:
        if (
            abs(float(stream["radius"]) - radius) < 1e-12
            and stream["query_cohort"] == cohort
        ):
            return stream
    raise KeyError(f"No stream for radius={radius} cohort={cohort}")


def figure_router_completion(
    hierarchy: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    runs = hierarchy["results"]
    radii = [float(value) for value in runs[0]["world"]["radii"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), sharey=True)
    add_figure_header(
        fig,
        "Shared action structure completes an unseen view",
        "Mean hard top-1 route accuracy; shaded bands show the full min–max range over 10 fixed seeds.",
    )
    for ax, cohort, title in zip(
        axes, ("observed", "heldout"), ("Observed actions", "Held-out action/view")
    ):
        cases = [
            ("Shared action family", "shared", PURPLE, "o", "-"),
            (
                "Independent fitted actions",
                "independent",
                ORANGE if cohort == "observed" else RED,
                "s",
                "--",
            ),
        ]
        for label, family, color, marker, linestyle in cases:
            by_radius: list[list[float]] = []
            for radius in radii:
                cell_name = f"radius_{radius:.2f}_{cohort}"
                by_radius.append(
                    [
                        float(
                            run["router_diagnostics"]["cells"][cell_name][family][
                                "hard_top1"
                            ]["accuracy"]
                        )
                        for run in runs
                    ]
                )
            add_range_series(
                ax,
                radii,
                by_radius,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
            )
        ax.set_title(title)
        ax.set_xlabel("Semantic overlap radius")
        ax.set_xticks(radii)
        ax.set_ylim(-0.04, 1.04)
        ax.legend(
            loc="lower left" if cohort == "observed" else "center left", fontsize=9
        )
        finish_axes(ax)
    axes[0].set_ylabel("Hard top-1 route accuracy")
    axes[1].text(
        0.98,
        0.24,
        "Independent held-out has no parameters trained\nfor that view: a completion control, not a generic baseline.",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.89), w_pad=2.5)
    return save_figure(fig, output_dir, "02_router_completion", formats)


def figure_retrieval_quality(
    hierarchy: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    runs = hierarchy["results"]
    radii = [float(value) for value in runs[0]["world"]["radii"]]
    methods = [
        ("Direct · dense soft", "direct_dense_soft", BLUE, "o", "--"),
        ("Direct · block top-1", "direct_block_top1", BLUE, "s", "-"),
        ("Delta · dense soft", "delta_dense_soft", TEAL, "o", "--"),
        ("Delta · block top-1", "delta_block_top1", TEAL, "s", "-"),
        ("Hard top-1 · direct = delta", "direct_hard_top1", PURPLE, "D", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), sharey=True)
    add_figure_header(
        fig,
        "Hierarchical selection reduces soft-memory interference",
        "Length 2,048, shared router, 64 slots. Lines are seed means; bands are full seed ranges. Hard direct and delta coincide numerically.",
    )
    for ax, cohort, title in zip(
        axes, ("observed", "heldout"), ("Observed actions", "Held-out action/view")
    ):
        for label, method, color, marker, linestyle in methods:
            by_radius: list[list[float]] = []
            for radius in radii:
                values = []
                for run in runs:
                    stream = find_stream(run, radius, cohort)
                    snapshot = stream["snapshots"]["2048"]["shared"]
                    values.append(float(snapshot[method]["mean_query_cosine"]))
                by_radius.append(values)
            add_range_series(
                ax,
                radii,
                by_radius,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                alpha=0.09,
            )
        ax.set_title(title)
        ax.set_xlabel("Semantic overlap radius")
        ax.set_xticks(radii)
        ax.set_ylim(0.0, 1.03)
        finish_axes(ax)
    axes[0].set_ylabel("Mean query cosine")
    axes[1].legend(loc="lower left", fontsize=8.6)
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.89), w_pad=2.4)
    return save_figure(fig, output_dir, "03_retrieval_quality", formats)


def figure_task_b_action_evidence(
    strict: dict[str, Any],
    paired: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
    add_figure_header(
        fig,
        "Task B separates a representation prior from the memory write law",
        "Length 2,048 mean cosine by seed. In both cohorts direct and delta agree under the same hard route; the gap is action completion.",
    )
    panels = [
        (
            axes[0],
            strict["rows"],
            "Strict replay (seeds 0–9)",
            "shared_direct_length2048_mean_cosine",
            "independent_delta_length2048_mean_cosine",
            "Replay implementation gate: 0/10; prior win: 10/10",
        ),
        (
            axes[1],
            paired["rows"],
            "Prospective paired replication (seeds 20–29)",
            "shared_delta_length2048_mean_cosine",
            "independent_delta_length2048_mean_cosine",
            "Implementation gate: 10/10; prior win: 10/10",
        ),
    ]
    for ax, rows, title, shared_key, independent_key, note in panels:
        seeds = [int(row["seed"]) for row in rows]
        shared_values = [float(row["decision"][shared_key]) for row in rows]
        independent_values = [float(row["decision"][independent_key]) for row in rows]
        ax.plot(
            seeds,
            shared_values,
            color=PURPLE,
            marker="o",
            linewidth=2.2,
            label="Shared Spin(8) action prior",
        )
        ax.plot(
            seeds,
            independent_values,
            color=ORANGE,
            marker="s",
            linewidth=2.2,
            label="Independent fitted actions",
        )
        ax.fill_between(
            seeds, independent_values, shared_values, color=PURPLE, alpha=0.08
        )
        ax.set_title(title)
        ax.set_xlabel("Frozen seed")
        ax.set_xticks(seeds)
        ax.set_ylim(0.38, 1.035)
        finish_axes(ax)
        ax.text(0.02, 0.05, note, transform=ax.transAxes, fontsize=8.8, color=MUTED)
    axes[0].set_ylabel("Mean query cosine")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.10, 0.98, 0.89), w_pad=2.5)
    return save_figure(fig, output_dir, "04_task_b_action_evidence", formats)


def figure_fused_latency(
    fused: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    rows = sorted(
        (row for row in fused["rows"] if int(row["batch"]) == 16),
        key=lambda row: int(row["slots"]),
    )
    slots = [int(row["slots"]) for row in rows]
    panels = [
        (
            "Direct overwrite",
            [
                ("Eager dense", "direct_eager_dense", BLUE, "o"),
                ("Eager gathered", "direct_eager_gathered", ORANGE, "s"),
                ("Triton fused gathered", "direct_triton_fused_gathered", PURPLE, "D"),
            ],
        ),
        (
            "Delta overwrite",
            [
                ("Eager dense", "delta_eager_dense", TEAL, "o"),
                ("Eager gathered", "delta_eager_gathered", ORANGE, "s"),
                ("Triton fused gathered", "delta_triton_fused_gathered", PURPLE, "D"),
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
    add_figure_header(
        fig,
        "Fusing gather + update + read turns selection into a systems win",
        "Batch 16, RTX 2070 SUPER, fp32, median of three independent process medians. One recurrent inference step; logarithmic latency axis.",
    )
    for ax, (title, methods) in zip(axes, panels):
        for label, key, color, marker in methods:
            values = [
                float(row["timing"][key]["median_of_process_medians_ms"])
                for row in rows
            ]
            ax.plot(
                slots,
                values,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.3,
                markersize=6,
            )
        ax.set_title(title)
        ax.set_xlabel("Logical slots")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(slots, [str(value) for value in slots])
        finish_axes(ax, grid_axis="both")
    axes[0].set_ylabel("Median step latency (ms, log scale)")
    axes[1].legend(loc="center right")
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.89), w_pad=2.5)
    return save_figure(fig, output_dir, "05_fused_latency", formats)


def figure_fused_speedup_heatmap(
    fused: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    rows = fused["rows"]
    slots = sorted({int(row["slots"]) for row in rows})
    batches = sorted({int(row["batch"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    add_figure_header(
        fig,
        "Fused gathered speedup is stable across the frozen grid",
        "Speedup over the matching eager dense update/read. Each cell is a median-of-process-medians ratio; same RTX 2070 SUPER protocol.",
    )
    images = []
    for ax, family, title in zip(
        axes, ("direct", "delta"), ("Direct overwrite", "Delta overwrite")
    ):
        matrix = np.empty((len(batches), len(slots)), dtype=float)
        for batch_index, batch in enumerate(batches):
            for slot_index, slot in enumerate(slots):
                row = next(
                    item
                    for item in rows
                    if int(item["batch"]) == batch and int(item["slots"]) == slot
                )
                matrix[batch_index, slot_index] = float(
                    row["comparisons"][family]["fused_speedup_over_eager_dense"]
                )
        image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=7.0, vmax=13.5)
        images.append(image)
        for batch_index in range(len(batches)):
            for slot_index in range(len(slots)):
                value = matrix[batch_index, slot_index]
                text_color = "white" if value < 10.0 else INK
                ax.text(
                    slot_index,
                    batch_index,
                    f"{value:.2f}×",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )
        ax.set_title(title)
        ax.set_xlabel("Logical slots")
        ax.set_ylabel("Batch")
        ax.set_xticks(range(len(slots)), [str(value) for value in slots])
        ax.set_yticks(range(len(batches)), [str(value) for value in batches])
    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.024, pad=0.03)
    colorbar.set_label("Speedup over eager dense")
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.11, top=0.83, wspace=0.26)
    return save_figure(fig, output_dir, "06_fused_speedup_grid", formats)


def figure_fused_incremental_memory(
    fused: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    rows = sorted(
        (row for row in fused["rows"] if int(row["batch"]) == 16),
        key=lambda row: int(row["slots"]),
    )
    slots = [int(row["slots"]) for row in rows]
    panels = [
        (
            "Direct overwrite",
            [
                ("Eager dense", "direct_eager_dense", BLUE, "o"),
                ("Eager gathered", "direct_eager_gathered", ORANGE, "s"),
                ("Triton fused gathered", "direct_triton_fused_gathered", PURPLE, "D"),
            ],
        ),
        (
            "Delta overwrite",
            [
                ("Eager dense", "delta_eager_dense", TEAL, "o"),
                ("Eager gathered", "delta_eager_gathered", ORANGE, "s"),
                ("Triton fused gathered", "delta_triton_fused_gathered", PURPLE, "D"),
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
    add_figure_header(
        fig,
        "The fused path also suppresses incremental allocation",
        "Batch 16; CUDA incremental peak allocated memory for the measured step. Logarithmic KiB axis; this is not total model memory.",
    )
    for ax, (title, methods) in zip(axes, panels):
        for label, key, color, marker in methods:
            values_kib = [
                float(row["timing"][key]["median_incremental_peak_bytes"]) / 1024.0
                for row in rows
            ]
            ax.plot(
                slots,
                values_kib,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.3,
                markersize=6,
            )
        ax.set_title(title)
        ax.set_xlabel("Logical slots")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(slots, [str(value) for value in slots])
        finish_axes(ax, grid_axis="both")
    axes[0].set_ylabel("Incremental peak allocated (KiB, log scale)")
    axes[1].legend(loc="upper left")
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.89), w_pad=2.5)
    return save_figure(fig, output_dir, "07_fused_incremental_memory", formats)


def process_median(row: dict[str, Any], section: str, method: str) -> float:
    return float(row[section][method]["process_median_summary_ms"]["median"])


def figure_matched_core_scaling(
    matched: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    rows = sorted(matched["rows"], key=lambda row: int(row["length"]))
    lengths = [int(row["length"]) for row in rows]
    methods = [
        ("Direct slots", "direct_slot_hybrid", BLUE, "o"),
        ("Triality slots", "triality_slot_hybrid", PURPLE, "s"),
        ("Delta chunkwise", "delta_chunkwise", TEAL, "D"),
        ("Fast-weight chunkwise", "fast_weight_chunkwise", ORANGE, "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
    add_figure_header(
        fig,
        "Matched memory cores: triality is not a free capacity or speed win",
        "Corrected 384-parameter core comparison, RTX 2070 SUPER. Three-process medians; logarithmic axes. This protocol predates the selected-block fused kernel.",
    )
    for ax, section, title in zip(
        axes, ("forward", "forward_backward"), ("Forward", "Forward + backward")
    ):
        for label, key, color, marker in methods:
            values = [process_median(row["core"], section, key) for row in rows]
            ax.plot(
                lengths,
                values,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.1,
                markersize=5.5,
            )
        ax.set_title(title)
        ax.set_xlabel("Sequence length")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(lengths, [str(value) for value in lengths])
        finish_axes(ax, grid_axis="both")
    axes[0].set_ylabel("Median latency (ms, log scale)")
    axes[1].legend(loc="upper left", fontsize=8.8)
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.89), w_pad=2.5)
    return save_figure(fig, output_dir, "08_matched_core_scaling", formats)


def figure_fla_delta_scaling(
    fla: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    rows = sorted(fla["rows"], key=lambda row: int(row["length"]))
    lengths = [int(row["length"]) for row in rows]
    methods = [
        ("Direct slots", "direct_slot", BLUE, "o"),
        ("Triality slots", "triality_slot", PURPLE, "s"),
        ("Local eager delta", "local_eager_delta", RED, "x"),
        ("Official FLA chunk delta", "fla_chunk_delta", TEAL, "D"),
        ("Official FLA recurrent delta", "fla_fused_recurrent_delta", ORANGE, "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
    add_figure_header(
        fig,
        "Official FLA kernels are the production baseline, not an afterthought",
        "Frozen RTX 2070 SUPER fp16 operator comparison. Three-process medians; logarithmic axes. Shape and state budgets are those of the matched FLA campaign.",
    )
    for ax, section, title in zip(
        axes, ("forward", "forward_backward"), ("Forward", "Forward + backward")
    ):
        for label, key, color, marker in methods:
            values = [process_median(row, section, key) for row in rows]
            ax.plot(
                lengths,
                values,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.1,
                markersize=5.5,
            )
        ax.set_title(title)
        ax.set_xlabel("Sequence length")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(lengths, [str(value) for value in lengths])
        finish_axes(ax, grid_axis="both")
    axes[0].set_ylabel("Median latency (ms, log scale)")
    axes[1].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.89), w_pad=2.5)
    return save_figure(fig, output_dir, "09_fla_delta_scaling", formats)


def figure_comoving_fla_scaling(
    comoving: dict[str, Any],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    rows = sorted(comoving["rows"], key=lambda row: int(row["length"]))
    lengths = [int(row["length"]) for row in rows]
    methods = [
        ("Direct slots", "direct_slot", BLUE, "o"),
        ("Native local delta", "native_local_delta", RED, "x"),
        ("Co-moving local delta", "comoving_local_delta", PURPLE, "s"),
        ("Co-moving FLA chunk", "comoving_fla_chunk", TEAL, "D"),
        ("Co-moving FLA recurrent", "comoving_fla_recurrent", ORANGE, "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
    add_figure_header(
        fig,
        "The co-moving compiler carries noncommuting actions into FLA",
        "Frozen full-transport operator campaign, RTX 2070 SUPER fp16. Prefix, inverse-frame solve, FLA kernel, and read frame are included.",
    )
    for ax, section, title in zip(
        axes, ("forward", "forward_backward"), ("Forward", "Forward + backward")
    ):
        for label, key, color, marker in methods:
            values = [process_median(row, section, key) for row in rows]
            ax.plot(
                lengths,
                values,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.1,
                markersize=5.5,
            )
        ax.set_title(title)
        ax.set_xlabel("Sequence length")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(lengths, [str(value) for value in lengths])
        finish_axes(ax, grid_axis="both")
    axes[0].set_ylabel("Median latency (ms, log scale)")
    axes[1].legend(loc="upper left", fontsize=8.4)
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.89), w_pad=2.5)
    return save_figure(fig, output_dir, "10_comoving_fla_scaling", formats)


def validate_sources(payloads: dict[str, dict[str, Any]]) -> None:
    hierarchy_summary = payloads["hierarchy"]["summary"]
    if hierarchy_summary["seeds"] != 10:
        raise ValueError("Expected the frozen 10-seed hierarchy aggregate")
    if not hierarchy_summary["shared_router_completion_supported"]:
        raise ValueError(
            "Hierarchy source does not support the declared shared-router gate"
        )
    if not hierarchy_summary["hierarchical_routing_supported"]:
        raise ValueError(
            "Hierarchy source does not support the declared hierarchy gate"
        )
    if not payloads["fused"]["summary"]["fused_gathered_advantage_supported"]:
        raise ValueError(
            "Fused source does not support its preregistered advantage gate"
        )
    if not payloads["fla"]["passed"]:
        raise ValueError("Official FLA source aggregate failed validation")
    if not payloads["comoving"]["passed"]:
        raise ValueError("Co-moving FLA source aggregate failed validation")
    if payloads["task_b_strict"]["summary"]["implementation_passes"] != 0:
        raise ValueError(
            "Strict Task B replay status changed; review the figure caption"
        )
    if payloads["task_b_paired"]["summary"]["implementation_passes"] != 10:
        raise ValueError(
            "Prospective Task B replication status changed; review the figure caption"
        )


def write_manifest(
    artifact_paths: dict[str, Path],
    figure_paths: Sequence[Path],
    output_dir: Path,
) -> Path:
    manifest_path = output_dir / "figure_manifest.json"
    manifest = {
        "generator": "src/plot_memory_benchmark_atlas.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "status": "derived visualization; benchmark values are unchanged",
        "source_artifacts": {
            key: {"path": f"artifacts/{path.name}", "sha256": sha256(path)}
            for key, path in sorted(artifact_paths.items())
        },
        "figures": [
            {"path": path.name, "sha256": sha256(path)}
            for path in sorted(figure_paths, key=lambda item: item.name)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        key: artifact_dir / filename for key, filename in SOURCE_ARTIFACTS.items()
    }
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required frozen artifacts:\n" + "\n".join(missing)
        )

    payloads = {key: load_json(path) for key, path in artifact_paths.items()}
    validate_sources(payloads)
    configure_style()

    figure_paths: list[Path] = []
    figure_paths += figure_fla_fit_map(output_dir, args.formats)
    figure_paths += figure_campaign_dashboard(
        payloads["hierarchy"], payloads["fused"], output_dir, args.formats
    )
    figure_paths += figure_router_completion(
        payloads["hierarchy"], output_dir, args.formats
    )
    figure_paths += figure_retrieval_quality(
        payloads["hierarchy"], output_dir, args.formats
    )
    figure_paths += figure_task_b_action_evidence(
        payloads["task_b_strict"], payloads["task_b_paired"], output_dir, args.formats
    )
    figure_paths += figure_fused_latency(payloads["fused"], output_dir, args.formats)
    figure_paths += figure_fused_speedup_heatmap(
        payloads["fused"], output_dir, args.formats
    )
    figure_paths += figure_fused_incremental_memory(
        payloads["fused"], output_dir, args.formats
    )
    figure_paths += figure_matched_core_scaling(
        payloads["matched"], output_dir, args.formats
    )
    figure_paths += figure_fla_delta_scaling(payloads["fla"], output_dir, args.formats)
    figure_paths += figure_comoving_fla_scaling(
        payloads["comoving"], output_dir, args.formats
    )
    manifest_path = write_manifest(artifact_paths, figure_paths, output_dir)
    print(f"Rendered {len(figure_paths)} figures to {output_dir}")
    print(f"Wrote provenance manifest to {manifest_path}")


if __name__ == "__main__":
    main()
