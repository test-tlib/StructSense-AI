"""
test_trajectory.py
==================
Automated tests for sequential stress-test trajectory generation,
central assessment logic consistency, parameter progression, and environment risk alignment.
"""
import pytest
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from risk.structural_assessment import (
    assess_structure,
    classify_risk_score,
    generate_stress_test_trajectory,
    DEFAULT_DEMO_THRESHOLDS,
)
from environment.building_env import BuildingStressEnv


@pytest.fixture
def tnpwd_bridge_params():
    """Reviewed parameters extracted from TNPWD_Structural_Report_NH48_Overbridge.pdf."""
    return {
        "concrete_grade": 30.0,
        "steel_grade": 500.0,
        "column_width_mm": 450.0,
        "column_depth_mm": 600.0,
        "num_floors": 4.0,
        "floor_load_kn_m2": 5.0,
        "soil_bearing_capacity_kn_m2": 280.0,
        "foundation_depth_m": 12.0,
        "seismic_zone": 3.0,
        "wind_speed_kmph": 158.4,
        "rainfall_mm_annual": 1285.0,
        "building_age_years": 0.0,
        "shear_wall_present": 1.0,
        "foundation_type": 2.0,
        "column_reinforcement_ratio": 0.018,
        "beam_depth_mm": 600.0,
        "slab_thickness_mm": 150.0,
        "location_risk_index": 0.35,
    }


def test_trajectory_length_and_structure(tnpwd_bridge_params):
    traj = generate_stress_test_trajectory(tnpwd_bridge_params)
    assert len(traj) == 20, "Trajectory must contain exactly 20 progressive stress steps"

    for idx, step in enumerate(traj):
        assert step["step"] == idx
        assert "name" in step and len(step["name"]) > 0
        assert "category" in step
        assert "description" in step
        assert "param_delta" in step
        assert "risk_score" in step
        assert 0.0 <= step["risk_score"] <= 100.0
        assert step["classification"] in ("SAFE", "MODERATE", "HIGH", "MODERATE RISK", "HIGH RISK")
        assert "governing_scenario" in step
        assert "max_dc_ratio" in step


def test_trajectory_step_zero_matches_base_assessment(tnpwd_bridge_params):
    base_eval = assess_structure(tnpwd_bridge_params)
    traj = generate_stress_test_trajectory(tnpwd_bridge_params)

    step_0 = traj[0]
    assert step_0["name"] == "Baseline As-Designed"
    assert step_0["risk_score"] == pytest.approx(base_eval["risk_score"], rel=1e-2)
    assert step_0["classification"] == base_eval["classification"]
    assert step_0["governing_scenario"] == base_eval["governing_scenario"]


def test_trajectory_overload_progression(tnpwd_bridge_params):
    traj = generate_stress_test_trajectory(tnpwd_bridge_params)

    # Steps 11 to 14 represent progressive gravity overloads (1.25x, 1.50x, 1.75x, 2.00x)
    dc_125 = traj[11]["max_dc_ratio"]
    dc_150 = traj[12]["max_dc_ratio"]
    dc_175 = traj[13]["max_dc_ratio"]
    dc_200 = traj[14]["max_dc_ratio"]

    assert dc_125 is not None and dc_150 is not None and dc_175 is not None and dc_200 is not None
    assert dc_125 < dc_150 < dc_175 < dc_200, "D/C ratio must strictly increase with higher overload factor"


def test_trajectory_aging_progression(tnpwd_bridge_params):
    traj = generate_stress_test_trajectory(tnpwd_bridge_params)

    # Step 0 (0y), Step 15 (+10y), Step 16 (+25y), Step 17 (+50y)
    r_base = traj[0]["risk_score"]
    r_10 = traj[15]["risk_score"]
    r_25 = traj[16]["risk_score"]
    r_50 = traj[17]["risk_score"]

    assert r_base <= r_10 <= r_25 <= r_50, "Risk score must increase or remain constant as structure ages"


def test_trajectory_uses_same_central_logic_as_results_tab(tnpwd_bridge_params):
    traj = generate_stress_test_trajectory(tnpwd_bridge_params)

    for step in traj:
        step_params = dict(tnpwd_bridge_params)
        step_params.update(step["modified_keys"])
        independent_eval = assess_structure(step_params)

        assert step["risk_score"] == pytest.approx(independent_eval["risk_score"], rel=1e-2)
        assert step["classification"] == independent_eval["classification"]
        assert step["governing_scenario"] == independent_eval["governing_scenario"]


def test_building_env_anchored_to_central_assessment(tnpwd_bridge_params):
    env = BuildingStressEnv(tnpwd_bridge_params, max_steps=20)
    _, info = env.reset(seed=42)

    base_eval = assess_structure(tnpwd_bridge_params)
    # Step 0 initial risk score in Gymnasium environment must match central assessment
    assert info["base_risk"] == pytest.approx(base_eval["risk_score"], abs=1.5)


def test_step_18_to_19_compound_transition_and_threshold_crossing(tnpwd_bridge_params):
    traj = generate_stress_test_trajectory(tnpwd_bridge_params)

    step_18 = traj[18]
    step_19 = traj[19]

    # Step 18 is approx 41.6 (MODERATE)
    assert 40.0 <= step_18["risk_score"] <= 43.5
    assert step_18["classification"] == "MODERATE"
    assert step_18["risk_score"] <= DEFAULT_DEMO_THRESHOLDS["moderate_max"]

    # Step 19 is approx 61.0 (HIGH)
    assert 60.0 <= step_19["risk_score"] <= 63.0
    assert step_19["classification"] == "HIGH"
    assert step_19["risk_score"] > DEFAULT_DEMO_THRESHOLDS["moderate_max"]

    # Crossing the threshold of 55 changes classification to HIGH
    assert step_18["risk_score"] <= 55.0 < step_19["risk_score"]
    assert step_18["classification"] != step_19["classification"]
