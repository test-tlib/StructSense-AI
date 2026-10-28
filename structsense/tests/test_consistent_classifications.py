"""
test_consistent_classifications.py
==================================
Tests to ensure that the risk score, risk classification, and scenario outcomes
are 100% consistent across:
1. Central assessment function (assess_structure)
2. Scorer report (StructuralRiskScorer)
3. AI Reasoner output (GeminiStructuralReasoner)
4. Demonstration threshold boundaries
"""
import pytest
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from risk.structural_assessment import assess_structure, classify_risk_score
from risk.risk_module import StructuralRiskScorer
from claude_reasoner.reasoner import GeminiStructuralReasoner


@pytest.fixture
def moderate_risk_building():
    # Concrete M25, Fe415, 6 floors, zone 3, moderate building age 25y
    return {
        "concrete_grade": 25.0,
        "steel_grade": 415.0,
        "column_width_mm": 350.0,
        "column_depth_mm": 350.0,
        "column_reinforcement_ratio": 0.012,
        "beam_depth_mm": 400.0,
        "slab_thickness_mm": 130.0,
        "num_floors": 6.0,
        "floor_load_kn_m2": 3.5,
        "soil_bearing_capacity_kn_m2": 180.0,
        "foundation_depth_m": 1.5,
        "seismic_zone": 3.0,
        "wind_speed_kmph": 150.0,
        "rainfall_mm_annual": 1400.0,
        "building_age_years": 25.0,
        "shear_wall_present": 0.0,
        "foundation_type": 1.0,
        "location_risk_index": 0.45,
    }


def test_consistent_score_and_label_moderate_building(moderate_risk_building):
    # 1. Central assessment
    ca = assess_structure(moderate_risk_building)
    ca_score = ca["risk_score"]
    ca_label = ca["risk_label"]

    assert ca_label == classify_risk_score(ca_score)

    # 2. Risk Scorer
    scorer = StructuralRiskScorer()
    report = scorer.full_report(raw_params=moderate_risk_building)

    assert pytest.approx(report["risk_score_100"], rel=1e-2) == ca_score
    assert report["risk_label"] == ca_label

    # 3. Reasoner (deterministic fallback mode)
    reasoner = GeminiStructuralReasoner()
    verdict = reasoner.analyze(
        raw_params=moderate_risk_building,
        element_risks=report["element_breakdown"],
        scenario_results=ca["scenario_results"],
        overall_risk=ca_score,
        neural_safety_weight=1.0,
    )

    # The overall verdict string must reflect the central assessment label
    assert ca_label in verdict["overall_verdict"]
    assert pytest.approx(verdict["calculated_risk_score"], rel=1e-2) == ca_score

    # MUST NOT display fabricated probability or confidence percentages
    assert verdict["probability_modeled"] is False
    assert verdict["confidence"] is None
    assert verdict["failure_probability"] is None


def test_consistent_high_risk_building():
    # Severe building with M15 concrete, 15 floors, Zone 5
    high_risk_params = {
        "concrete_grade": 15.0,
        "steel_grade": 250.0,
        "column_width_mm": 210.0,
        "column_depth_mm": 210.0,
        "column_reinforcement_ratio": 0.009,
        "beam_depth_mm": 310.0,
        "slab_thickness_mm": 105.0,
        "num_floors": 15.0,
        "floor_load_kn_m2": 12.0,
        "soil_bearing_capacity_kn_m2": 60.0,
        "foundation_depth_m": 0.5,
        "seismic_zone": 5.0,
        "wind_speed_kmph": 220.0,
        "rainfall_mm_annual": 2800.0,
        "building_age_years": 80.0,
        "shear_wall_present": 0.0,
        "foundation_type": 0.0,
        "location_risk_index": 0.92,
    }
    ca = assess_structure(high_risk_params)
    assert ca["risk_score"] >= 55.0
    assert ca["risk_label"] == "HIGH"

    reasoner = GeminiStructuralReasoner()
    verdict = reasoner.analyze(
        raw_params=high_risk_params,
        element_risks=ca["element_breakdown"],
        scenario_results=ca["scenario_results"],
        overall_risk=ca["risk_score"],
        neural_safety_weight=1.8,
    )
    assert "HIGH" in verdict["overall_verdict"]
    assert verdict["calculated_risk_score"] >= 55.0
