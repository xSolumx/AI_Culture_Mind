"""Exact gates for the 3/5/7 ordered-probe stabilizer tower."""

from __future__ import annotations

from spin8_probe_stabilizer_tower import certificate, ordered_probe_action_matrix


def test_complete_ordered_probe_rank_formula() -> None:
    ranks = [ordered_probe_action_matrix(count).rank() for count in range(9)]
    assert ranks == [0, 7, 13, 18, 22, 25, 27, 28, 28]


def test_three_five_seven_refinement_and_reverse_quotient() -> None:
    report = certificate()
    tower = report["odd_probe_tower"]
    assert report["passed"]
    assert tower["cumulative_ranks"] == [18, 25, 28]
    assert tower["coordinate_split"] == [18, 7, 3]
    assert tower["fiber_dimensions"] == [7, 3]
    assert report["one_three_five_seven_residual_dimensions"] == [21, 10, 3, 0]


def test_su_ladders_remain_representation_specific() -> None:
    atlas = certificate()["cross_representation_atlas"]
    assert "su(3)" in atlas["spin8_mixed_triality"]["binary_rank_3"]
    assert "su(2)" in atlas["spin8_mixed_triality"]["binary_rank_4"]
    assert atlas["spin9_spinor_stabilizer_dimensions"] == {"1": 21, "2": 8, "3": 0}
    assert atlas["spin9_interpretation"] == ["Spin(7)", "SU(3)", "trivial"]
