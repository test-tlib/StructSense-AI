"""
test_ai_integration_and_fallback.py
===================================
Automated tests for Groq AI integration, offline rule fallback,
API failure handling, and disagreement detection.
"""
import pytest
import os
import sys
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from claude_reasoner.reasoner import (
    GeminiStructuralReasoner,
    rule_based_fallback,
    get_structural_verdict,
)
from risk.structural_assessment import assess_structure


@pytest.fixture
def sample_params():
    return {
        "concrete_grade": 25.0,
        "steel_grade": 415.0,
        "column_width_mm": 400.0,
        "column_depth_mm": 400.0,
        "column_reinforcement_ratio": 0.015,
        "beam_depth_mm": 450.0,
        "slab_thickness_mm": 150.0,
        "num_floors": 4.0,
        "floor_load_kn_m2": 3.0,
        "soil_bearing_capacity_kn_m2": 200.0,
        "foundation_depth_m": 1.5,
        "seismic_zone": 3.0,
        "wind_speed_kmph": 140.0,
        "rainfall_mm_annual": 1200.0,
        "building_age_years": 10.0,
        "shear_wall_present": 1.0,
        "foundation_type": 1.0,
        "location_risk_index": 0.3,
    }


def test_rule_based_fallback_schema(sample_params):
    assessment = assess_structure(sample_params)
    fallback = rule_based_fallback(
        raw_params=sample_params,
        element_risks=assessment["element_breakdown"],
        scenario_results=assessment["scenario_results"],
        overall_risk=assessment["risk_score"],
    )

    # Check schema completeness
    assert "overall_verdict" in fallback
    assert "calculated_risk_score" in fallback
    assert "primary_failure_mode" in fallback
    assert "reasons" in fallback and len(fallback["reasons"]) > 0
    assert "recommendations" in fallback and len(fallback["recommendations"]) > 0
    assert "is_code_violations" in fallback
    assert "scenario_results" in fallback
    assert fallback["_meta"]["source"] in ("rule_based", "rule_based_fallback")

    # Must NOT claim unvalidated probability or confidence
    assert fallback["probability_modeled"] is False
    assert fallback["confidence"] is None
    assert fallback["failure_probability"] is None


def test_groq_api_missing_key_falls_back_gracefully(sample_params):
    assessment = assess_structure(sample_params)
    # Instantiate reasoner with empty API key
    reasoner = GeminiStructuralReasoner(api_key="")
    # analyze() should not crash; must fall back to deterministic engine
    res = reasoner.analyze(
        raw_params=sample_params,
        element_risks=assessment["element_breakdown"],
        scenario_results=assessment["scenario_results"],
        overall_risk=assessment["risk_score"],
        neural_safety_weight=1.0,
    )
    assert res is not None
    assert res["_meta"]["source"] in ("rule_based", "rule_based_fallback")
    assert "calculated_risk_score" in res


def test_groq_api_exception_handled_gracefully(sample_params):
    assessment = assess_structure(sample_params)
    reasoner = GeminiStructuralReasoner(api_key="invalid_dummy_key_12345")

    # Mock groq client call to simulate API network timeout or error
    with patch("claude_reasoner.reasoner.get_structural_verdict", side_effect=Exception("API Connection Refused")):
        res = reasoner.analyze(
            raw_params=sample_params,
            element_risks=assessment["element_breakdown"],
            scenario_results=assessment["scenario_results"],
            overall_risk=assessment["risk_score"],
            neural_safety_weight=1.0,
        )
        # Should gracefully return rule-based fallback without throwing unhandled error
        assert res["_meta"]["source"] in ("rule_based", "rule_based_fallback")
        assert res["calculated_risk_score"] == pytest.approx(assessment["risk_score"], rel=1e-2)


def test_transparent_disagreement_detection(sample_params):
    assessment = assess_structure(sample_params)

    # Mock Groq LLM returning "SAFE" for a "HIGH RISK" building (score 55.8)
    mock_choice = type("Choice", (), {
        "message": type("Message", (), {
            "content": '{"overall_verdict": "SAFE", "primary_failure_mode": "None", "reasons": ["All checks safe"], "recommendations": ["Maintain current state"]}'
        })()
    })()
    mock_completion = type("Completion", (), {"choices": [mock_choice]})()
    mock_client = type("Client", (), {
        "chat": type("Chat", (), {
            "completions": type("Completions", (), {"create": lambda *args, **kwargs: mock_completion})()
        })()
    })()

    with patch("claude_reasoner.reasoner.Groq", return_value=mock_client):
        res = get_structural_verdict("dummy_key", sample_params, assessment["element_breakdown"], assessment["scenario_results"])
        assert res["disagreement_detected"] is True
        assert res["ai_verdict"] == "SAFE"
        assert "HIGH" in res["overall_verdict"]
