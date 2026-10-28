"""
test_scenario_calculations.py
==============================
Automated tests for all five structural scenario calculations:
1. Normal Occupancy Load Test (IS 456)
2. Seismic Stress Test (IS 1893)
3. Wind + Rain Combined Test (IS 875)
4. Overload Test 150% (IS 456 / IS 1904)
5. Long-Term Degradation Test
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
    compute_column_axial_capacity_kn,
    compute_applied_gravity_load_kn,
    compute_soil_bearing_capacity,
    assess_structure,
)


@pytest.fixture
def standard_params():
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


def test_column_axial_capacity_standard_formula():
    """Verify IS 456 Cl 39.3: Pu = 0.40 * fck * Ac + 0.67 * fy * Asc"""
    # 400x400 mm column, M25, Fe415, rho = 1.5%
    Ag = 400.0 * 400.0
    Asc = 0.015 * Ag
    Ac = Ag - Asc
    expected_Puz = (0.40 * 25.0 * Ac + 0.67 * 415.0 * Asc) / 1000.0
    calc_Puz = compute_column_axial_capacity_kn(25.0, 415.0, 400.0, 400.0, 0.015)
    assert pytest.approx(expected_Puz, rel=1e-3) == calc_Puz


def test_normal_load_scenario_pass(standard_params):
    res = evaluate_normal_load(standard_params)
    assert res["status"] in ("PASS", "FAIL")
    assert res["demand"] > 0
    assert res["capacity"] > 0
    assert res["demand_capacity_ratio"] > 0
    assert "IS 456" in res["standard"]
    assert len(res["assumptions"]) > 0


def test_seismic_stress_scenario(standard_params):
    res = evaluate_seismic_stress(standard_params)
    assert res["status"] in ("PASS", "FAIL")
    assert res["demand"] > 0
    assert res["capacity"] > 0
    assert "IS 1893" in res["standard"]

    # Shear wall presence should increase lateral capacity and reduce base shear
    params_no_sw = dict(standard_params, shear_wall_present=0.0)
    res_no_sw = evaluate_seismic_stress(params_no_sw)
    assert res_no_sw["demand"] > res["demand"]
    assert res_no_sw["capacity"] < res["capacity"]


def test_wind_rain_scenario(standard_params):
    res = evaluate_wind_rain(standard_params)
    assert res["status"] in ("PASS", "FAIL")
    assert res["demand"] > 0
    assert res["capacity"] > 0
    assert "IS 875" in res["standard"]

    # Extreme wind (250 km/h) and severe rain should increase demand
    params_cyclone = dict(standard_params, wind_speed_kmph=240.0, rainfall_mm_annual=3000.0)
    res_cyclone = evaluate_wind_rain(params_cyclone)
    assert res_cyclone["demand"] > res["demand"]


def test_overload_scenario(standard_params):
    res = evaluate_overload(standard_params)
    assert res["status"] in ("PASS", "FAIL")
    assert res["demand"] > 0
    assert res["capacity"] > 0
    # Overload demand should be exactly 1.5x normal gravity load
    res_normal = evaluate_normal_load(standard_params)
    assert pytest.approx(res["demand"], rel=1e-2) == res_normal["demand"] * 1.5


def test_degradation_scenario(standard_params):
    res_new = evaluate_degradation(dict(standard_params, building_age_years=0.0))
    res_old = evaluate_degradation(dict(standard_params, building_age_years=75.0))
    # Older building must have strictly lower capacity
    assert res_old["capacity"] < res_new["capacity"]
    assert res_old["demand_capacity_ratio"] > res_new["demand_capacity_ratio"]


def test_all_five_scenarios_evaluated_in_central_assessment(standard_params):
    assessment = assess_structure(standard_params)
    scenarios = assessment["scenario_results"]
    expected_keys = {"normal_load", "seismic", "wind_rain", "overload", "degradation"}
    assert set(scenarios.keys()) == expected_keys
    for k in expected_keys:
        sc = scenarios[k]
        assert sc["status"] in ("PASS", "FAIL", "NOT EVALUATED")
        assert sc["demand"] is not None
        assert sc["capacity"] is not None
        assert sc["demand_capacity_ratio"] is not None
        assert sc["standard"] != ""
