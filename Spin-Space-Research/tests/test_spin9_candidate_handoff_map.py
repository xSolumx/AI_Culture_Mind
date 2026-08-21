from spin9_candidate_handoff_map import certificate, verify_report


def test_candidate_handoff_map_selects_four_cusp_charts() -> None:
    report = certificate()
    assert verify_report(report)
    assert report["equality_incident_count"] == 16
    assert report["generic_refinement_count"] == 13
    assert report["minimal_shape_independent_cusp_charts"] == 4
