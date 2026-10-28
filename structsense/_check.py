"""Quick reasoner check.

1. Offline: forces the rule-based fallback (no network) and validates the schema.
2. Live:    if GROQ_API_KEY is set (env or .env), calls Groq and validates the reply.
"""
import sys
sys.path.insert(0, ".")

import claude_reasoner.reasoner as _mod
from claude_reasoner.reasoner import GeminiStructuralReasoner, ClaudeStructuralReasoner

params = {
    "concrete_grade": 15, "steel_grade": 250,
    "column_width_mm": 210, "column_depth_mm": 210,
    "num_floors": 15, "floor_load_kn_m2": 12,
    "soil_bearing_capacity_kn_m2": 60, "foundation_depth_m": 0.5,
    "seismic_zone": 5, "wind_speed_kmph": 220,
    "rainfall_mm_annual": 2800, "building_age_years": 80,
    "shear_wall_present": 0, "foundation_type": 0,
    "column_reinforcement_ratio": 0.009,
    "beam_depth_mm": 310, "slab_thickness_mm": 105,
    "location_risk_index": 0.92,
}
risks = {"foundation": 0.88, "columns": 0.79, "beams": 0.61, "slab": 0.55}
sim   = {
    "Normal Occupancy Load Test": {"passed": False},
    "Seismic Stress Test":        {"passed": False},
    "Wind + Rain Combined Test":  {"passed": False},
    "Overload Test (150 %)":      {"passed": False},
    "Long-Term Degradation Test": {"passed": False},
}


def check_schema(verdict):
    assert verdict["overall_verdict"] in {"HIGH RISK", "MODERATE RISK", "SAFE"}
    assert set(verdict["scenario_results"]) == {"normal_load", "seismic", "wind_rain", "overload", "degradation"}
    assert set(verdict["risk_zones"])       == {"foundation", "columns", "beams", "slab"}


def show(verdict):
    print(f"verdict          : {verdict['overall_verdict']}")
    print(f"source           : {verdict['_meta']['source']}")
    print(f"failure_prob     : {verdict['failure_probability']}")
    print(f"reasons          : {len(verdict['reasons'])}")
    print(f"IS violations    : {len(verdict['is_code_violations'])}")


# ── 1. Offline fallback ───────────────────────────────────────────────────────
print("── Offline rule-based fallback ──")
_orig = _mod._GROQ_OK
_mod._GROQ_OK = False
verdict = GeminiStructuralReasoner().analyze(raw_params=params, element_risks=risks,
                                             scenario_results=sim, overall_risk=0.88)
_mod._GROQ_OK = _orig
show(verdict)
check_schema(verdict)
assert verdict["overall_verdict"]         == "HIGH RISK"
assert verdict["_meta"]["source"]         == "rule_based"
assert len(verdict["is_code_violations"]) >= 4
assert ClaudeStructuralReasoner is GeminiStructuralReasoner

# ── 2. Live Groq call ─────────────────────────────────────────────────────────
r = GeminiStructuralReasoner()
if r.is_available:
    print("\n── Live Groq call ──")
    verdict = r.analyze(raw_params=params, element_risks=risks,
                        scenario_results=sim, overall_risk=0.88)
    show(verdict)
    assert verdict["_meta"]["source"] == "groq", "Groq call failed and fell back to rules"
    check_schema(verdict)
else:
    print("\n(GROQ_API_KEY not set — skipping live Groq check)")

print("\nALL CHECKS PASSED")
