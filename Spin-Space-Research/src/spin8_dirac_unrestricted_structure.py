"""Replay the unrestricted Dirac--Gram structural reduction exactly.

This module deliberately stops before positivity.  It certifies the complete
seven-circle sign quotient, the common boundary divisor, and conservative
multidegrees for all sixteen residual polynomial sectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spin8_dirac_final_residual import exact_full_multidegree_certificate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_structure_20260807.json"),
    )
    arguments = parser.parse_args()
    report = exact_full_multidegree_certificate()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("unrestricted structural certificate failed")


if __name__ == "__main__":
    main()
