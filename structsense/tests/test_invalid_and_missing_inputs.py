"""
test_invalid_and_missing_inputs.py
==================================
Automated tests for missing parameters, invalid engineering values,
and validation warnings under IS 456 / IS 1893 / IS 1904.
"""
import pytest
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from risk.structural_assessment import (
    evaluate_normal_load,
    evaluate_seismic_stress,
    evaluate_wind_rain,
    evaluate_overload,
    evaluate_degradation,
    validate_parameters,
    assess_structure,
)


def test_missing_data_returns_not_evaluated():
    # An empty parameter dict must return NOT EVALUATED, never PASS or misleading 0.00
    res_normal = evaluate_normal_load({})
    assert res_normal["status"] == "NOT EVALUATED"
    assert res_normal["passed"] is None
    assert "Missing" in res_normal["notes"] or "Invalid" in res_normal["notes"]

    res_seismic = evaluate_seismic_stress({})
    assert res_seismic["status"] == "NOT EVALUATED"
    assert res_seismic["passed"] is None

    res_wind = evaluate_wind_rain({})
    assert res_wind["status"] == "NOT EVALUATED"

    res_overload = evaluate_overload({})
    assert res_overload["status"] == "NOT EVALUATED"

    res_deg = evaluate_degradation({})
    assert res_deg["status"] == "NOT EVALUATED"


def test_invalid_negative_dimensions():
    invalid_params = {
        "concrete_grade": 25.0,
        "steel_grade": 415.0,
        "column_width_mm": -300.0,  # Invalid negative
        "column_depth_mm": 300.0,
        "column_reinforcement_ratio": 0.015,
        "num_floors": 4.0,
        "floor_load_kn_m2": 3.0,
        "soil_bearing_capacity_kn_m2": 200.0,
    }
    res = evaluate_normal_load(invalid_params)
    assert res["status"] == "NOT EVALUATED"
    assert "positive" in res["notes"].lower() or "validation" in res["notes"].lower()


def test_validation_flags_is_code_minimums():
    # Sub-standard building violating multiple IS clauses
    bad_params = {
        "concrete_grade": 15.0,                 # < M20 min for RC (IS 456 Cl 6.1.2)
        "steel_grade": 250.0,
        "column_width_mm": 180.0,               # < 200 mm min column dimension
        "column_depth_mm": 180.0,
        "column_reinforcement_ratio": 0.005,    # < 0.8% min longitudinal steel (IS 456 Cl 26.5.3.1)
        "num_floors": 8.0,
        "floor_load_kn_m2": 4.0,
        "soil_bearing_capacity_kn_m2": 80.0,    # < 100 kN/m2 low SBC
        "foundation_depth_m": 0.6,              # < 0.9 m min foundation depth (IS 1904)
        "seismic_zone": 5.0,                    # Zone 5 without shear wall
        "shear_wall_present": 0.0,
    }
    is_valid, errors, warnings = validate_parameters(bad_params)
    # Check that critical violations are flagged
    warning_text = " ".join(warnings)
    assert "M15" in warning_text or "concrete" in warning_text.lower()
    assert "reinforcement" in warning_text.lower() or "0.8%" in warning_text
    assert "200 mm" in warning_text or "dimension" in warning_text.lower()
    assert "depth" in warning_text.lower() or "0.9 m" in warning_text
    assert "shear wall" in warning_text.lower() or "Zone 5" in warning_text


def test_assess_structure_handles_partial_inputs():
    partial = {"concrete_grade": 20.0, "num_floors": 3.0}
    assessment = assess_structure(partial)
    assert "risk_score" in assessment
    assert "risk_label" in assessment
    assert assessment["risk_label"] in ("SAFE", "MODERATE", "HIGH")
    assert "scenario_results" in assessment
