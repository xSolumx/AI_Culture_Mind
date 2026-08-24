from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.competence_validation import two_consecutive_mastery


def test_competence_requires_latest_two_consecutive_passing_probes() -> None:
    assert not two_consecutive_mastery([])
    assert not two_consecutive_mastery([0.95])
    assert not two_consecutive_mastery([0.95, 0.89])
    assert not two_consecutive_mastery([0.95, 0.89, 0.96])
    assert two_consecutive_mastery([0.89, 0.96, 0.91])
