from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.combined_validation import (
    COMBINED_SCHEDULE,
    schedule_label_counts,
)


def test_g9_combined_schedule_and_label_budget_are_frozen() -> None:
    assert [phase.updates for phase in COMBINED_SCHEDULE] == [
        1200,
        1200,
        1400,
        1300,
        600,
    ]
    assert [phase.length for phase in COMBINED_SCHEDULE] == [16, 24, 48, 96, 512]
    assert schedule_label_counts() == (1_292_800, 1_408_000)
