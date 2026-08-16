"""Tests for the exact central-core diagonal-dominance theorem."""

from __future__ import annotations

import unittest
from fractions import Fraction
from pathlib import Path

from spin8_dirac_endpoint_octet_core_dominance import (
    _ceil_sqrt_fraction,
    _core_box_path,
    verify_report,
)

ROOT = Path(__file__).resolve().parents[1]
COEFFICIENT_DIR = ROOT / "artifacts" / "spin8_dirac_unrestricted_coefficients_20260807"
ARTIFACT = (
    ROOT / "artifacts" / "spin8_dirac_endpoint_octet_core_dominance_20260816.json"
)


class EndpointOctetCoreDominanceTests(unittest.TestCase):
    def test_outward_square_root_is_an_exact_dyadic_ceiling(self) -> None:
        for value in (
            Fraction(0),
            Fraction(1, 3),
            Fraction(2, 7),
            Fraction(9, 16),
            Fraction(81, 5),
        ):
            upper = _ceil_sqrt_fraction(value, bits=40)
            self.assertGreaterEqual(upper**2, value)
            if upper:
                self.assertLess((upper - Fraction(1, 1 << 40)) ** 2, value)

    def test_thirty_two_paths_partition_the_central_core(self) -> None:
        paths = set()
        interval_rows = set()
        for code_value in range(32):
            code = tuple((code_value >> shift) & 1 for shift in range(4, -1, -1))
            path, intervals = _core_box_path(code)
            paths.add(path)
            interval_rows.add(tuple(tuple(interval) for interval in intervals))
        self.assertEqual(len(paths), 32)
        self.assertEqual(len(interval_rows), 32)
        self.assertTrue(
            all(
                interval in (("1/4", "1/2"), ("1/2", "3/4"))
                for row in interval_rows
                for interval in row
            )
        )

    def test_stored_core_certificate_replays_all_exact_bounds(self) -> None:
        verification = verify_report(ARTIFACT, coefficient_dir=COEFFICIENT_DIR)
        self.assertIs(verification["verified"], True)
        self.assertEqual(verification["failures"], [])
        self.assertEqual(verification["replayed_box_count"], 32)
        self.assertIs(verification["strict_diagonal_dominance_proved"], True)
        self.assertGreater(Fraction(verification["minimum_physical_gap_lower"]), 0)


if __name__ == "__main__":
    unittest.main()
