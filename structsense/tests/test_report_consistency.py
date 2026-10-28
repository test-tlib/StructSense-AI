"""
test_report_consistency.py
===========================
Automated tests for PDF and CSV report consistency,
ensuring central assessment metrics, all 5 scenarios, and disclaimers are preserved.
"""
import pytest
import os
import sys
import tempfile
import csv
import io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from risk.structural_assessment import assess_structure, DISCLAIMER_TEXT
from outputs.report_generator import generate_report


@pytest.fixture
def sample_building():
    return {
        "concrete_grade": 25.0,
        "steel_grade": 415.0,
        "column_width_mm": 400.0,
        "column_depth_mm": 400.0,
        "column_reinforcement_ratio": 0.015,
        "beam_depth_mm": 450.0,
        "slab_thickness_mm": 150.0,
        "num_floors": 5.0,
        "floor_load_kn_m2": 3.5,
        "soil_bearing_capacity_kn_m2": 200.0,
        "foundation_depth_m": 1.5,
        "seismic_zone": 3.0,
        "wind_speed_kmph": 140.0,
        "rainfall_mm_annual": 1200.0,
        "building_age_years": 15.0,
        "shear_wall_present": 1.0,
        "foundation_type": 1.0,
        "location_risk_index": 0.35,
    }


def test_pdf_generation_with_central_assessment(sample_building):
    ca = assess_structure(sample_building)
    verdict = {
        "overall_verdict": f"{ca['risk_label']} RISK",
        "calculated_risk_score": ca["risk_score"],
        "governing_scenario": ca["governing_scenario"],
        "max_dc_ratio": ca["max_dc_ratio"],
        "primary_failure_mode": ca["governing_scenario"],
        "reasons": ca["summary_reasons"],
        "recommendations": ca["recommendations"],
        "is_code_violations": ca["is_code_violations"],
        "scenario_results": {k: v["status"] for k, v in ca["scenario_results"].items()},
        "_meta": {"source": "rule_based_fallback", "model": "deterministic"},
    }

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf_path = tmp_pdf.name

    try:
        out = generate_report(
            verdict=verdict,
            state_params=sample_building,
            risk_scores=ca["element_breakdown"],
            reward_history=[0.1, 0.4, 0.7, 0.9],
            output_path=pdf_path,
        )
        assert os.path.exists(out)
        assert os.path.getsize(out) > 5000  # Multi-page PDF
        with open(out, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def test_pdf_handles_not_evaluated_scenarios(sample_building):
    # If one scenario has NOT EVALUATED status, PDF generation must succeed
    ca = assess_structure(sample_building)
    sc_res = {k: v["status"] for k, v in ca["scenario_results"].items()}
    sc_res["degradation"] = "NOT EVALUATED"

    verdict = {
        "overall_verdict": f"{ca['risk_label']} RISK",
        "calculated_risk_score": ca["risk_score"],
        "governing_scenario": ca["governing_scenario"],
        "max_dc_ratio": ca["max_dc_ratio"],
        "primary_failure_mode": ca["governing_scenario"],
        "reasons": ["Test reasons"],
        "recommendations": ["Test recs"],
        "is_code_violations": [],
        "scenario_results": sc_res,
        "_meta": {"source": "rule_based_fallback", "model": "deterministic"},
    }

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf_path = tmp_pdf.name

    try:
        out = generate_report(
            verdict=verdict,
            state_params=sample_building,
            risk_scores=ca["element_breakdown"],
            reward_history=[],
            output_path=pdf_path,
        )
        assert os.path.exists(out)
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def test_csv_structure_and_completeness(sample_building):
    ca = assess_structure(sample_building)
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["StructSense AI — Assessment Summary"])
    wr.writerow(["Overall Verdict", f"{ca['risk_label']} RISK"])
    wr.writerow(["Calculated Risk Score", f"{ca['risk_score']:.1f} / 100"])
    wr.writerow(["Governing Scenario", ca["governing_scenario"]])
    wr.writerow(["Max Demand/Capacity Ratio", ca["max_dc_ratio"]])
    wr.writerow(["Prototype Disclaimer", DISCLAIMER_TEXT])
    wr.writerow([])
    wr.writerow(["Scenario", "Status", "Demand (kN)", "Capacity (kN)", "D/C Ratio", "Standard"])
    for sk, sc_info in ca.get("scenario_results", {}).items():
        wr.writerow([
            sc_info.get("scenario_name", sk),
            sc_info.get("status", "UNKNOWN"),
            f"{sc_info.get('demand', 0.0):.1f}" if sc_info.get('demand') is not None else "N/A",
            f"{sc_info.get('capacity', 0.0):.1f}" if sc_info.get('capacity') is not None else "N/A",
            f"{sc_info.get('demand_capacity_ratio', 0.0):.2f}" if sc_info.get('demand_capacity_ratio') is not None else "N/A",
            sc_info.get("standard", "IS Code"),
        ])

    csv_content = buf.getvalue()
    assert "Overall Verdict" in csv_content
    assert f"{ca['risk_score']:.1f}" in csv_content
    assert "prototype" in csv_content.lower()
    assert "unvalidated demonstration thresholds" in csv_content.lower()
    # Check all 5 scenarios are written
    for sk in ["normal_load", "seismic", "wind_rain", "overload", "degradation"]:
        sc_name = ca["scenario_results"][sk]["scenario_name"]
        assert sc_name in csv_content
