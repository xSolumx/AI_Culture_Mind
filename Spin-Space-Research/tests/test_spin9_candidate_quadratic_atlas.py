from spin9_candidate_quadratic_atlas import certificate, verify_report


def test_exact_quadratic_candidate_atlas_localizes_without_overclaiming() -> None:
    report = certificate(maximum_depth=2)
    assert verify_report(report)
    assert report["unresolved_count"] > 0
    assert report["compact_complement_certified_at_candidate_ratio"] is False
