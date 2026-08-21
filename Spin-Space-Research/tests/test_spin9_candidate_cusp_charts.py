from spin9_candidate_cusp_charts import certificate, verify_report


def test_all_four_candidate_cusp_charts_are_exact() -> None:
    report = certificate()
    assert verify_report(report)
    assert len(report["charts"]) == 4
    assert all(row["z_interval"] == ["0", "1"] for row in report["charts"])
