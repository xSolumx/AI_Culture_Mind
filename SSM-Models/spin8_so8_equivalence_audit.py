"""Write the exact positive-half-spin/generic-SO(8) chart certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spin8_triality import so8_chart_equivalence_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = so8_chart_equivalence_diagnostics()
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
