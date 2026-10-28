"""
claude_reasoner/reasoner.py – StructSense AI · Groq LLM Reasoner
============================================================================
Provides:
  - get_structural_verdict()   : standalone function
  - GeminiStructuralReasoner   : class wrapper keeping the .analyze() interface
    used by app.py, with a deterministic rule-based fallback when no API key is set.

Consistency & Grounding Guarantee:
  - Takes the central backend assessment (assess_structure) as authoritative ground truth.
  - Explains actual calculated engineering numbers (demands, capacities, D/C ratios, IS codes).
  - Does NOT hallucinate confidence percentages or failure probabilities.
  - Transparently indicates AI perspective and any disagreement notes.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

try:
    from groq import Groq
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from risk.structural_assessment import (
    DEFAULT_DEMO_THRESHOLDS,
    DISCLAIMER_TEXT,
    assess_structure,
    classify_risk_score,
)

logger = logging.getLogger(__name__)

# Groq chat model; override with GROQ_MODEL in .env
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule-based fallback (Deterministic ground truth)
# ═══════════════════════════════════════════════════════════════════════════════

def rule_based_fallback(
    state_params: Optional[dict] = None,
    risk_scores: Optional[dict] = None,
    simulation_results: Optional[dict] = None,
    raw_params: Optional[dict] = None,
    element_risks: Optional[dict] = None,
    scenario_results: Optional[dict] = None,
    overall_risk: float = 0.0,
) -> dict:
    """
    Deterministic structural verdict directly from central assessment engine.
    Ensures 100% consistency across Results, AI Verdict, and Report without hallucinated numbers.
    """
    params = raw_params if raw_params is not None else (state_params or {})
    assessment = assess_structure(params)
    classification = assessment["classification"]
    verdict = f"{classification} RISK" if str(classification) in ("HIGH", "MODERATE") else str(classification)
    score = assessment["risk_score"]

    reasons: List[str] = []
    # Ground reasons in actual calculated scenario outcomes and parameters
    scenarios = assessment["scenario_results"]
    for s_key, s_data in scenarios.items():
        if s_data.get("status") == "FAIL":
            reasons.append(f"{s_data['name']} FAILED: {s_data.get('explanation', '')}")
        elif s_data.get("status") == "NOT EVALUATED":
            reasons.append(f"{s_data['name']} NOT EVALUATED: {s_data.get('explanation', '')}")

    # Add material and geometry checks
    conc = float(params.get("concrete_grade", 25))
    rho = float(params.get("column_reinforcement_ratio", 0.02))
    zone = int(round(float(params.get("seismic_zone", 2))))
    sw = float(params.get("shear_wall_present", 1))
    soil = float(params.get("soil_bearing_capacity_kn_m2", 200))
    age = float(params.get("building_age_years", 0))

    if conc < 20:
        reasons.append(f"Concrete grade M{int(conc)} is below IS 456 Cl. 6.1.2 minimum M20 for RCC.")
    if rho < 0.008:
        reasons.append(f"Column longitudinal reinforcement ratio {rho*100:.2f}% is below IS 456 Cl. 26.5.3.1 minimum 0.8%.")
    if sw == 0 and zone >= 3:
        reasons.append(f"No shear walls present in Seismic Zone {zone} — ductile shear system required per IS 1893:2016 Cl. 7.6.")
    if soil < 100:
        reasons.append(f"Soil bearing capacity {soil:.0f} kN/m² is marginal for multi-storey structure without deep foundation.")
    if age > 40:
        reasons.append(f"Building age {age:.0f} years indicates significant cumulative environmental degradation.")

    if not reasons:
        reasons.append(
            f"All structural parameters satisfy basic IS code limits; governing scenario: "
            f"{assessment['governing_scenario']} (max D/C = {assessment.get('max_dc_ratio', 0):.2f})."
        )

    # Scenarios mapping
    scenario_res = {
        k: s.get("status", "NOT EVALUATED")
        for k, s in scenarios.items()
    }

    primary_mode = (
        f"Governed by {assessment['governing_scenario']} (max D/C = {assessment['max_dc_ratio']})"
        if assessment.get("max_dc_ratio") is not None
        else "No critical failure mode identified"
    )

    return {
        "overall_verdict":      verdict,
        "calculated_risk_score": score,
        "confidence":           None,  # Not modeled probabilistically
        "failure_probability":  None,  # Not modeled probabilistically
        "probability_modeled":  False,
        "disagreement_detected": False,
        "ai_verdict":           verdict,
        "ai_perspective":       "AGREES",
        "primary_failure_mode": primary_mode,
        "reasons":              reasons,
        "scenario_results":     scenario_res,
        "risk_zones":           assessment["element_zones"],
        "recommendations":      assessment["recommendations"],
        "is_code_violations":   assessment["is_code_violations"],
        "governing_scenario":   assessment["governing_scenario"],
        "max_dc_ratio":         assessment["max_dc_ratio"],
        "disclaimer":           DISCLAIMER_TEXT,
        "source":               "rule_based",
        "_meta": {
            "source":       "rule_based",
            "model":        "deterministic",
            "raw_response": "",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone function
# ═══════════════════════════════════════════════════════════════════════════════

def get_structural_verdict(
    api_key: str,
    state_params: dict,
    risk_scores: Optional[dict] = None,
    simulation_results: Optional[dict] = None,
) -> dict:
    """
    Groq LLM structural reasoning layer.
    Receives calculated backend metrics, explains engineering results without hallucinating,
    and returns a structured verdict completely aligned with the backend assessment.
    """
    if not api_key or not _GROQ_OK:
        return rule_based_fallback(state_params, risk_scores, simulation_results)

    # Authoritative backend assessment
    assessment = assess_structure(state_params)
    classification = assessment["classification"]
    backend_verdict = f"{classification} RISK" if str(classification) in ("HIGH", "MODERATE") else str(classification)
    backend_score = assessment["risk_score"]

    # Package clean scenario summaries for the prompt
    clean_scenarios = {}
    for sk, sdata in assessment["scenario_results"].items():
        clean_scenarios[sk] = {
            "name": sdata.get("name"),
            "status": sdata.get("status"),
            "demand": f"{sdata.get('demand')} {sdata.get('demand_unit')}" if sdata.get('demand') is not None else "N/A",
            "capacity": f"{sdata.get('capacity')} {sdata.get('capacity_unit')}" if sdata.get('capacity') is not None else "N/A",
            "demand_capacity_ratio": sdata.get("demand_capacity_ratio"),
            "governing_check": sdata.get("governing_check"),
        }

    prompt_context = {
        "building_parameters": state_params,
        "backend_calculated_risk_score": f"{backend_score}/100",
        "backend_demonstration_verdict": backend_verdict,
        "governing_scenario": assessment["governing_scenario"],
        "max_demand_capacity_ratio": assessment["max_dc_ratio"],
        "element_risk_scores": assessment["element_breakdown"],
        "element_risk_zones": assessment["element_zones"],
        "scenario_calculations": clean_scenarios,
        "detected_is_code_violations": assessment["is_code_violations"],
    }

    prompt = f"""You are a certified structural engineering AI reviewer.
Analyze the following CALCULATED structural assessment outputs for a building.
Do NOT invent contradictory values or state unvalidated probability percentages.
Explain the actual engineering numbers, identify governing structural failure mechanisms, and suggest actionable retrofitting/mitigation measures.

ENGINEERING ASSESSMENT DATA:
{json.dumps(prompt_context, indent=2)}

Return a strict JSON object with EXACTLY these keys:
{{
  "ai_perspective": "AGREES or DISAGREES or NUANCED",
  "ai_assessment_summary": "one-paragraph professional technical review of the calculations",
  "primary_failure_mode": "concise description of governing failure mechanism",
  "reasons": [
    "detailed technical explanation 1 based on actual calculated demand vs capacity",
    "detailed technical explanation 2",
    "detailed technical explanation 3"
  ],
  "recommendations": [
    "actionable structural retrofitting / mitigation measure 1",
    "measure 2",
    "measure 3"
  ],
  "disagreement_notes": "None if agreeing, or technical rationale if AI perspective differs from backend demonstration verdict"
}}"""

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional structural engineering AI. Always return valid, well-formed JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=4096,
        )

        raw = response.choices[0].message.content.strip()
        clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean).strip()
        ai_data = json.loads(clean)

        # Merge with central backend assessment to guarantee consistent numbers
        scenario_res = {
            k: s.get("status", "NOT EVALUATED")
            for k, s in assessment["scenario_results"].items()
        }

        reasons = ai_data.get("reasons") or []
        if not isinstance(reasons, list) or not reasons:
            reasons = [
                f"Calculated risk score: {backend_score}/100 ({backend_verdict}).",
                f"Governing scenario: {assessment['governing_scenario']} (max D/C = {assessment.get('max_dc_ratio', 'N/A')}).",
            ]

        recs = ai_data.get("recommendations") or []
        if not isinstance(recs, list) or not recs:
            recs = assessment["recommendations"]

        primary_mode = ai_data.get("primary_failure_mode") or (
            f"Governed by {assessment['governing_scenario']} (D/C = {assessment['max_dc_ratio']})"
            if assessment.get("max_dc_ratio") is not None
            else "No critical failure mode identified"
        )
        ai_verdict = ai_data.get("overall_verdict", "").strip()
        clean_ai = ai_verdict.upper()
        clean_central = str(backend_verdict).upper()
        disagreement_flag = False
        if clean_ai:
            if ("SAFE" in clean_central and "SAFE" not in clean_ai) or \
               ("MODERATE" in clean_central and "MODERATE" not in clean_ai) or \
               ("HIGH" in clean_central and "HIGH" not in clean_ai):
                disagreement_flag = True

        return {
            "overall_verdict":       backend_verdict,
            "calculated_risk_score": backend_score,
            "confidence":            None,  # Do not display unvalidated percentage
            "failure_probability":   None,  # Do not display unvalidated percentage
            "probability_modeled":   False,
            "disagreement_detected": disagreement_flag,
            "ai_verdict":            ai_verdict or backend_verdict,
            "ai_perspective":        ai_data.get("ai_perspective", "AGREES"),
            "ai_assessment_summary": ai_data.get("ai_assessment_summary", ""),
            "primary_failure_mode":  primary_mode,
            "reasons":               reasons,
            "scenario_results":      scenario_res,
            "risk_zones":            assessment["element_zones"],
            "recommendations":       recs,
            "is_code_violations":    assessment["is_code_violations"],
            "governing_scenario":    assessment["governing_scenario"],
            "max_dc_ratio":          assessment["max_dc_ratio"],
            "disagreement_notes":    ai_data.get("disagreement_notes", "None"),
            "disclaimer":            DISCLAIMER_TEXT,
            "source":                "groq",
            "_meta": {
                "source":       "groq",
                "model":        GROQ_MODEL,
                "raw_response": raw,
            },
        }

    except Exception as e:
        logger.warning("Groq call failed, using rule-based fallback: %s", e)
        return rule_based_fallback(state_params, risk_scores, simulation_results)


# ═══════════════════════════════════════════════════════════════════════════════
#  Class wrapper — keeps app.py's .analyze() interface intact
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiStructuralReasoner:
    """
    Drop-in structural reasoning engine backed by Groq LLM with deterministic
    IS-code rule fallback when key is absent or network fails.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            self._api_key = os.environ.get("GROQ_API_KEY", "").strip()

    @property
    def is_available(self) -> bool:
        return bool(_GROQ_OK and self._api_key)

    def analyze(
        self,
        raw_params: Dict[str, Any],
        element_risks: Optional[Dict[str, float]] = None,
        scenario_results: Optional[Dict[str, Any]] = None,
        overall_risk: float = 0.0,
        neural_safety_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform structural safety analysis via Groq or deterministic fallback."""
        if not self._api_key:
            return rule_based_fallback(
                raw_params=raw_params,
                element_risks=element_risks,
                scenario_results=scenario_results,
                overall_risk=overall_risk,
            )
        try:
            return get_structural_verdict(self._api_key, raw_params, element_risks, scenario_results)
        except Exception:
            return rule_based_fallback(
                raw_params=raw_params,
                element_risks=element_risks,
                scenario_results=scenario_results,
                overall_risk=overall_risk,
            )


# Backward-compat alias
ClaudeStructuralReasoner = GeminiStructuralReasoner
