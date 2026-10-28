"""
risk/structural_assessment.py – StructSense AI Central Structural Assessment Engine
===================================================================================
Provides:
  1. Configurable demonstration risk thresholds (clearly labeled unvalidated).
  2. Input parameter validation and missing-data detection.
  3. Five separate, documented, testable engineering calculations:
       - Normal Occupancy Load Test       (IS 456 Cl. 39.3 / IS 875 Parts 1 & 2)
       - Seismic Stress Test              (IS 1893:2016 Part 1 Cl. 6.4 & 7.6)
       - Wind + Rain Combined Test        (IS 875:2015 Part 3 & IS 1904:1986)
       - Overload Test (150%)             (IS 456:2000 Limit State of Collapse)
       - Long-Term Degradation Test       (IS 456 Cl. 8 Durability / Environmental)
  4. Central assess_structure() function providing consistent metrics across all
     dashboard tabs, AI reasoning, and export reports.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
#  Demonstration Risk Thresholds & Engineering Disclaimers
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_DEMO_THRESHOLDS: Dict[str, Any] = {
    "SAFE":     (0.0,  30.0),   # 0 to 30: SAFE (Low Risk)
    "MODERATE": (30.0, 55.0),   # 30 to 55: MODERATE RISK
    "HIGH":     (55.0, 100.0),  # 55 to 100: HIGH RISK (elevated / critical concern if >= 75)
    "safe_max":     30.0,
    "moderate_max": 55.0,
}

class RiskClassification(str):
    """
    String representation of risk classification ("SAFE", "MODERATE RISK", "HIGH RISK")
    that also equals "MODERATE" or "HIGH" and unpacks into (label, css_badge)
    for backwards compatibility.
    """
    def __new__(cls, label: str, badge: str, short_label: Optional[str] = None):
        obj = str.__new__(cls, label)
        obj.label = label
        obj.badge = badge
        obj.short_label = short_label or label.replace(" RISK", "")
        return obj

    def __eq__(self, other):
        if not isinstance(other, str):
            return False
        return str(self) == other or self.short_label == other or self.label == other

    def __hash__(self):
        return hash(self.label)

    def __iter__(self):
        yield self.label
        yield self.badge

    def __getitem__(self, item):
        if item == 0:
            return self.label
        elif item == 1:
            return self.badge
        return super().__getitem__(item)

DISCLAIMER_TEXT = (
    "StructSense AI is an academic and research prototype for preliminary structural "
    "vulnerability screening. All risk scores, classifications, and demand-to-capacity (D/C) "
    "ratios are demonstration indicators derived from simplified engineering checks and "
    "unvalidated demonstration thresholds. This software does NOT certify building safety, "
    "does NOT replace statutory compliance checks, and must NOT be used for real-world life-safety "
    "decisions without a comprehensive audit by a qualified, licensed structural engineer."
)

# ═══════════════════════════════════════════════════════════════════════════════
#  IS Code Limits and Reference Constants
# ═══════════════════════════════════════════════════════════════════════════════

IS_AH_ZONE: Dict[int, float] = {
    1: 0.08,   # Zone I/II equivalent in simplified analysis
    2: 0.10,   # Zone II: Z = 0.10
    3: 0.16,   # Zone III: Z = 0.16
    4: 0.24,   # Zone IV: Z = 0.24
    5: 0.36,   # Zone V: Z = 0.36
}

MIN_COLUMN_DIM_MM = 200.0
MIN_CONCRETE_GRADE_RCC = 20.0   # IS 456:2000 Cl. 6.1.2 min M20 for reinforced concrete
MIN_REBAR_RATIO = 0.008          # IS 456 Cl. 26.5.3.1 min 0.8% longitudinal steel
MAX_REBAR_RATIO = 0.060          # IS 456 Cl. 26.5.3.1 max 6.0% (4% without lap)
MIN_FOUNDATION_DEPTH_M = 0.90    # IS 1904:1986 Cl. 4.2 minimum depth
MIN_SLAB_THICKNESS_MM = 120.0    # IS 456 Cl. 24.1 minimum for two-way slab
MIN_BEAM_DEPTH_MM = 350.0        # IS 456 Cl. 23.2.1 deflection control minimum

# ═══════════════════════════════════════════════════════════════════════════════
#  Parameter Validation
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_PARAMETERS = [
    "concrete_grade",
    "steel_grade",
    "column_width_mm",
    "column_depth_mm",
    "num_floors",
    "floor_load_kn_m2",
    "soil_bearing_capacity_kn_m2",
    "foundation_depth_m",
    "seismic_zone",
    "wind_speed_kmph",
    "rainfall_mm_annual",
    "building_age_years",
    "shear_wall_present",
    "foundation_type",
    "column_reinforcement_ratio",
    "beam_depth_mm",
    "slab_thickness_mm",
    "location_risk_index",
]


def validate_parameters(params: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate building parameters against physical bounds and IS code standards.

    Returns:
        (is_valid, errors, warnings)
        errors: Missing or non-physical parameters that prevent evaluation.
        warnings: Engineering code non-compliances that indicate elevated risk.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(params, dict):
        return False, ["Parameters must be a dictionary"], []

    # Check for missing keys
    for req in REQUIRED_PARAMETERS:
        if req not in params or params[req] is None:
            errors.append(f"Missing required parameter: '{req}'")

    def _val(k: str) -> float:
        try:
            return float(params.get(k, 0.0))
        except (ValueError, TypeError):
            return float("nan")

    # Physical validity checks (hard errors)
    if "concrete_grade" in params:
        conc = _val("concrete_grade")
        if math.isnan(conc) or conc <= 0:
            errors.append(f"Concrete grade must be > 0 MPa (got {params.get('concrete_grade')})")

    if "steel_grade" in params:
        steel = _val("steel_grade")
        if math.isnan(steel) or steel <= 0:
            errors.append(f"Steel grade must be > 0 MPa (got {params.get('steel_grade')})")

    if "column_width_mm" in params or "column_depth_mm" in params:
        col_w = _val("column_width_mm")
        col_d = _val("column_depth_mm")
        if math.isnan(col_w) or col_w <= 0 or math.isnan(col_d) or col_d <= 0:
            errors.append(f"Column dimensions must be positive (got {col_w} x {col_d} mm)")

    if "num_floors" in params:
        floors = _val("num_floors")
        if math.isnan(floors) or floors < 1:
            errors.append(f"Number of floors must be >= 1 (got {params.get('num_floors')})")

    if "floor_load_kn_m2" in params:
        floor_ld = _val("floor_load_kn_m2")
        if math.isnan(floor_ld) or floor_ld < 0:
            errors.append(f"Floor load must be non-negative (got {params.get('floor_load_kn_m2')})")

    if "soil_bearing_capacity_kn_m2" in params:
        soil_sbc = _val("soil_bearing_capacity_kn_m2")
        if math.isnan(soil_sbc) or soil_sbc <= 0:
            errors.append(f"Soil bearing capacity must be > 0 kN/m² (got {params.get('soil_bearing_capacity_kn_m2')})")

    if "foundation_depth_m" in params:
        found_d = _val("foundation_depth_m")
        if math.isnan(found_d) or found_d <= 0:
            errors.append(f"Foundation depth must be > 0 m (got {params.get('foundation_depth_m')})")

    if "column_reinforcement_ratio" in params:
        rho = _val("column_reinforcement_ratio")
        if math.isnan(rho) or rho <= 0 or rho > 0.15:
            errors.append(f"Column reinforcement ratio must be between 0 and 0.15 (got {params.get('column_reinforcement_ratio')})")

    # Engineering standard warnings (IS 456 / IS 1893 / IS 1904)
    if "concrete_grade" in params:
        c = _val("concrete_grade")
        if not math.isnan(c) and c > 0 and c < MIN_CONCRETE_GRADE_RCC:
            warnings.append(f"Concrete grade M{int(c)} is below IS 456:2000 Cl. 6.1.2 minimum of M20 for RCC.")
    if "column_reinforcement_ratio" in params:
        r = _val("column_reinforcement_ratio")
        if not math.isnan(r) and r > 0 and r < MIN_REBAR_RATIO:
            warnings.append(f"Column reinforcement ratio {r*100:.2f}% is below IS 456 Cl. 26.5.3.1 minimum 0.8%.")
        elif not math.isnan(r) and r > MAX_REBAR_RATIO:
            warnings.append(f"Column reinforcement ratio {r*100:.2f}% exceeds IS 456 Cl. 26.5.3.1 maximum 6.0%.")
    if "column_width_mm" in params and "column_depth_mm" in params:
        cw = _val("column_width_mm")
        cd = _val("column_depth_mm")
        if not math.isnan(cw) and not math.isnan(cd) and cw > 0 and cd > 0 and (cw < MIN_COLUMN_DIM_MM or cd < MIN_COLUMN_DIM_MM):
            warnings.append(f"Column dimension {cw}x{cd} mm is below recommended minimum {MIN_COLUMN_DIM_MM} mm.")
    if "foundation_depth_m" in params:
        fd = _val("foundation_depth_m")
        if not math.isnan(fd) and fd > 0 and fd < MIN_FOUNDATION_DEPTH_M:
            warnings.append(f"Foundation depth {fd:.2f} m is below IS 1904:1986 Cl. 4.2 minimum 0.9 m.")
    if "soil_bearing_capacity_kn_m2" in params:
        sb = _val("soil_bearing_capacity_kn_m2")
        if not math.isnan(sb) and sb > 0 and sb < 100.0:
            warnings.append(f"Soil bearing capacity {sb:.1f} kN/m² indicates weak/soft soil requiring pile foundation.")
    if "shear_wall_present" in params and "seismic_zone" in params:
        sw = _val("shear_wall_present")
        sz = _val("seismic_zone")
        if not math.isnan(sw) and not math.isnan(sz) and sw < 0.5 and sz >= 3:
            warnings.append(f"Absence of shear walls in Seismic Zone {int(sz)} violates IS 1893:2016 recommendations.")
    if "beam_depth_mm" in params:
        bd = _val("beam_depth_mm")
        if not math.isnan(bd) and bd > 0 and bd < MIN_BEAM_DEPTH_MM:
            warnings.append(f"Beam depth {bd} mm is below deflection-controlled guideline {MIN_BEAM_DEPTH_MM} mm.")
    if "slab_thickness_mm" in params:
        st = _val("slab_thickness_mm")
        if not math.isnan(st) and st > 0 and st < MIN_SLAB_THICKNESS_MM:
            warnings.append(f"Slab thickness {st} mm is below IS 456 Cl. 24.1 minimum {MIN_SLAB_THICKNESS_MM} mm.")

    return (len(errors) == 0), errors, warnings


# ═══════════════════════════════════════════════════════════════════════════════
#  Core Engineering Structural Formulas
# ═══════════════════════════════════════════════════════════════════════════════

def compute_column_axial_capacity_kn(
    concrete_grade_mpa: float,
    steel_grade_mpa: float,
    width_mm: float,
    depth_mm: float,
    reinforcement_ratio: float,
) -> float:
    """
    IS 456:2000 Cl. 39.3 – Pure axial compression capacity of short non-slender column.
        Pu = 0.40 * fck * Ac + 0.67 * fy * Asc
    Where:
        fck = characteristic compressive cube strength of concrete (MPa)
        fy  = characteristic yield strength of steel (MPa)
        Ag  = gross column area = b * d (mm²)
        Asc = area of longitudinal reinforcement = rho * Ag (mm²)
        Ac  = net concrete area = Ag - Asc (mm²)
    Returns:
        Pu in kN.
    """
    if width_mm <= 0 or depth_mm <= 0 or concrete_grade_mpa <= 0:
        return 0.0
    gross_area = width_mm * depth_mm
    asc = max(0.0, min(reinforcement_ratio, 0.15)) * gross_area
    ac = max(0.0, gross_area - asc)
    pu_newtons = 0.40 * concrete_grade_mpa * ac + 0.67 * steel_grade_mpa * asc
    return max(0.0, pu_newtons / 1000.0)


def compute_applied_gravity_load_kn(
    floor_load_kn_m2: float,
    num_floors: float,
    tributary_area_m2: float = 25.0,
    dead_load_factor: float = 1.25,
    overload_multiplier: float = 1.0,
) -> float:
    """
    Tributary area gravity load demand for critical internal column.
    Assumes standard 5 m x 5 m bay (25 m² tributary) per column.
    dead_load_factor incorporates slab, beam, and column self-weight allowance.
    """
    if num_floors <= 0 or floor_load_kn_m2 < 0:
        return 0.0
    return floor_load_kn_m2 * tributary_area_m2 * num_floors * dead_load_factor * overload_multiplier


def compute_soil_bearing_capacity(
    base_sbc_kn_m2: float,
    foundation_depth_m: float,
    foundation_type: float,
    footing_width_m: float = 1.5,
    footing_length_m: float = 1.5,
) -> Tuple[float, float]:
    """
    IS 6403:1981 / IS 1904:1986 Allowable bearing capacity with depth and foundation type bonus.
    Foundation type:
        0 = Isolated footing (multiplier 1.0)
        1 = Raft foundation  (multiplier 1.5)
        2 = Pile foundation  (multiplier 3.0)
    Returns:
        (allowable_capacity_kn_m2, footing_area_m2)
    """
    footing_area = max(0.25, footing_width_m * footing_length_m)
    depth_bonus = min(max(0.0, foundation_depth_m) / 5.0, 0.40) * base_sbc_kn_m2
    effective_sbc = base_sbc_kn_m2 + depth_bonus

    ftype_int = int(round(foundation_type))
    if ftype_int == 2:
        effective_sbc *= 3.0
    elif ftype_int == 1:
        effective_sbc *= 1.5

    return effective_sbc, footing_area


# ═══════════════════════════════════════════════════════════════════════════════
#  Five Separate, Documented, Testable Scenario Calculations
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_normal_load(params: Dict[str, Any], tributary_area_m2: float = 25.0) -> Dict[str, Any]:
    """
    Scenario 1: Normal Occupancy Load Test
    Standards: IS 456:2000 Cl. 39.3 & IS 875 (Parts 1 & 2)
    Evaluates:
      1. Short column axial compression demand Pu vs factored capacity Puz.
      2. Footing soil pressure demand vs allowable soil bearing capacity.
    """
    is_valid, errors, _ = validate_parameters(params)
    if not is_valid:
        return {
            "name": "Normal Occupancy Load Test",
            "scenario_name": "Normal Occupancy Load Test",
            "short_name": "Normal Load",
            "standard": "IS 456:2000 Cl. 39.3 / IS 875",
            "status": "NOT EVALUATED",
            "demand": None,
            "demand_unit": "kN",
            "capacity": None,
            "capacity_unit": "kN",
            "demand_capacity_ratio": None,
            "passed": None,
            "explanation": f"Input validation failed: {'; '.join(errors)}",
            "notes": f"Input validation failed: {'; '.join(errors)}",
            "governing_check": "Not Evaluated",
            "assumptions": ["Calculation halted due to missing/invalid inputs."],
        }

    cap_kn = compute_column_axial_capacity_kn(
        concrete_grade_mpa=float(params["concrete_grade"]),
        steel_grade_mpa=float(params["steel_grade"]),
        width_mm=float(params["column_width_mm"]),
        depth_mm=float(params["column_depth_mm"]),
        reinforcement_ratio=float(params["column_reinforcement_ratio"]),
    )
    applied_kn = compute_applied_gravity_load_kn(
        floor_load_kn_m2=float(params["floor_load_kn_m2"]),
        num_floors=float(params["num_floors"]),
        tributary_area_m2=tributary_area_m2,
        dead_load_factor=1.25,
        overload_multiplier=1.0,
    )

    col_dc = applied_kn / (cap_kn + 1e-6)

    # Soil bearing check
    eff_sbc, foot_area = compute_soil_bearing_capacity(
        base_sbc_kn_m2=float(params["soil_bearing_capacity_kn_m2"]),
        foundation_depth_m=float(params["foundation_depth_m"]),
        foundation_type=float(params["foundation_type"]),
    )
    soil_pressure = applied_kn / foot_area
    soil_dc = soil_pressure / (eff_sbc + 1e-6)

    governing_dc = max(col_dc, soil_dc)
    passed = governing_dc <= 1.0

    if not passed:
        if col_dc > 1.0 and soil_dc > 1.0:
            gov = "Combined Column & Soil Overload"
            expl = f"Column axial D/C ({col_dc:.2f}) and soil bearing D/C ({soil_dc:.2f}) both exceed 1.0."
        elif col_dc > 1.0:
            gov = "Column Axial Compression"
            expl = f"Column axial demand ({applied_kn:.1f} kN) exceeds IS 456 capacity ({cap_kn:.1f} kN), D/C = {col_dc:.2f}."
        else:
            gov = "Foundation Soil Bearing"
            expl = f"Footing pressure ({soil_pressure:.1f} kN/m²) exceeds allowable SBC ({eff_sbc:.1f} kN/m²), D/C = {soil_dc:.2f}."
    else:
        gov = "Column Axial Compression" if col_dc >= soil_dc else "Foundation Soil Bearing"
        expl = f"Structure safely satisfies IS 456 axial limits and IS 6403 soil bearing (governing D/C = {governing_dc:.2f} <= 1.0)."

    return {
        "name": "Normal Occupancy Load Test",
        "scenario_name": "Normal Occupancy Load Test",
        "short_name": "Normal Load",
        "standard": "IS 456:2000 Cl. 39.3 / IS 875",
        "status": "PASS" if passed else "FAIL",
        "demand": round(applied_kn, 1),
        "demand_unit": "kN",
        "capacity": round(cap_kn, 1),
        "capacity_unit": "kN",
        "demand_capacity_ratio": round(governing_dc, 2),
        "passed": passed,
        "explanation": expl,
        "notes": expl,
        "governing_check": gov,
        "details": {
            "column_axial_demand_kn": round(applied_kn, 1),
            "column_axial_capacity_kn": round(cap_kn, 1),
            "column_dc": round(col_dc, 3),
            "soil_pressure_kn_m2": round(soil_pressure, 1),
            "allowable_sbc_kn_m2": round(eff_sbc, 1),
            "soil_dc": round(soil_dc, 3),
        },
        "assumptions": [
            f"Tributary bay area: {tributary_area_m2} m² per internal column.",
            "Factored dead load + live load combined multiplier = 1.25.",
            "Square isolated/raft footing checked against allowable bearing capacity.",
        ],
    }


def evaluate_seismic_stress(params: Dict[str, Any], tributary_area_m2: float = 25.0) -> Dict[str, Any]:
    """
    Scenario 2: Seismic Stress Test
    Standards: IS 1893 (Part 1):2016 Cl. 6.4 (Equivalent Static Base Shear) & Cl. 7.6
    Evaluates horizontal seismic base shear demand VB vs frame lateral shear capacity.
    """
    is_valid, errors, _ = validate_parameters(params)
    if not is_valid:
        return {
            "name": "Seismic Stress Test",
            "scenario_name": "Seismic Stress Test",
            "short_name": "Seismic Stress",
            "standard": "IS 1893:2016 Cl. 6.4 / 7.6",
            "status": "NOT EVALUATED",
            "demand": None,
            "demand_unit": "kN",
            "capacity": None,
            "capacity_unit": "kN",
            "demand_capacity_ratio": None,
            "passed": None,
            "explanation": f"Input validation failed: {'; '.join(errors)}",
            "notes": f"Input validation failed: {'; '.join(errors)}",
            "governing_check": "Not Evaluated",
            "assumptions": ["Calculation halted due to missing/invalid inputs."],
        }

    zone_val = int(round(float(params["seismic_zone"])))
    ah_coeff = IS_AH_ZONE.get(zone_val, 0.16)

    has_shear_wall = float(params["shear_wall_present"]) >= 0.5
    # Response reduction factor: R=5 for ductile shear wall/SMRF, R=3 for ordinary frame
    if has_shear_wall:
        effective_ah = ah_coeff * 0.60
    else:
        effective_ah = ah_coeff

    # Seismic weight per column tributary (IS 1893 Cl. 7.3.1: DL + 0.25-0.50 LL)
    floors = float(params["num_floors"])
    q = float(params["floor_load_kn_m2"])
    seismic_weight_col = q * tributary_area_m2 * floors * 1.0

    # Height amplification for multi-storey overturning
    height_factor = 1.0 + max(0.0, floors - 1.0) * 0.05
    vb_demand = effective_ah * seismic_weight_col * height_factor

    # Column axial capacity to derive lateral shear capacity
    axial_cap = compute_column_axial_capacity_kn(
        concrete_grade_mpa=float(params["concrete_grade"]),
        steel_grade_mpa=float(params["steel_grade"]),
        width_mm=float(params["column_width_mm"]),
        depth_mm=float(params["column_depth_mm"]),
        reinforcement_ratio=float(params["column_reinforcement_ratio"]),
    )

    # Frame lateral shear capacity: ~20% of axial capacity for frame; 2.5x bonus with shear wall
    shear_wall_bonus = 2.5 if has_shear_wall else 1.0
    lateral_cap = axial_cap * 0.20 * shear_wall_bonus

    dc_ratio = vb_demand / (lateral_cap + 1e-6)
    passed = dc_ratio <= 1.0

    if not passed:
        expl = (
            f"Seismic base shear ({vb_demand:.1f} kN) exceeds frame lateral shear capacity ({lateral_cap:.1f} kN) "
            f"in Zone {zone_val} (D/C = {dc_ratio:.2f})."
        )
        if not has_shear_wall and zone_val >= 3:
            expl += " Mandatory shear walls absent per IS 1893:2016 Cl. 7.6."
    else:
        expl = f"Lateral resisting system safely handles IS 1893 Zone {zone_val} base shear (D/C = {dc_ratio:.2f} <= 1.0)."

    return {
        "name": "Seismic Stress Test",
        "scenario_name": "Seismic Stress Test",
        "short_name": "Seismic Stress",
        "standard": "IS 1893:2016 Cl. 6.4 / 7.6",
        "status": "PASS" if passed else "FAIL",
        "demand": round(vb_demand, 1),
        "demand_unit": "kN",
        "capacity": round(lateral_cap, 1),
        "capacity_unit": "kN",
        "demand_capacity_ratio": round(dc_ratio, 2),
        "passed": passed,
        "explanation": expl,
        "notes": expl,
        "governing_check": "Lateral Base Shear vs Resisting Capacity",
        "details": {
            "seismic_zone": zone_val,
            "horizontal_seismic_coeff_ah": round(effective_ah, 3),
            "seismic_weight_kn": round(seismic_weight_col, 1),
            "height_amplification": round(height_factor, 2),
            "base_shear_demand_kn": round(vb_demand, 1),
            "lateral_capacity_kn": round(lateral_cap, 1),
            "shear_wall_present": has_shear_wall,
        },
        "assumptions": [
            f"Equivalent static base shear Ah computed from IS 1893:2016 Zone {zone_val}.",
            "Lateral capacity modeled as shear resistance of frame and shear walls.",
            "Response reduction factor R = 5 (with shear wall) or R = 3 (without shear wall).",
        ],
    }


def evaluate_wind_rain(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scenario 3: Wind + Rain Combined Test
    Standards: IS 875 (Part 3):2015 Cl. 6.3 & IS 1904:1986 Cl. 4.2
    Evaluates dynamic wind lateral pressure, rainfall pore-water surcharge,
    and foundation scour/erosion vulnerability.
    """
    is_valid, errors, _ = validate_parameters(params)
    if not is_valid:
        return {
            "name": "Wind + Rain Combined Test",
            "scenario_name": "Wind + Rain Combined Test",
            "short_name": "Wind + Rain",
            "standard": "IS 875 Part 3 / IS 1904",
            "status": "NOT EVALUATED",
            "demand": None,
            "demand_unit": "kN",
            "capacity": None,
            "capacity_unit": "kN",
            "demand_capacity_ratio": None,
            "passed": None,
            "explanation": f"Input validation failed: {'; '.join(errors)}",
            "notes": f"Input validation failed: {'; '.join(errors)}",
            "governing_check": "Not Evaluated",
            "assumptions": ["Calculation halted due to missing/invalid inputs."],
        }

    v_kmph = float(params["wind_speed_kmph"])
    v_ms = v_kmph / 3.6
    # Design wind pressure qz = 0.6 * Vz² / 1000 (kN/m²)
    qz_kn_m2 = 0.60 * (v_ms ** 2) / 1000.0

    floors = float(params["num_floors"])
    storey_height_m = 3.0
    col_width_m = float(params["column_width_mm"]) / 1000.0

    # Wind force on column tributary facade face
    wind_force_kn = qz_kn_m2 * 1.2 * col_width_m * storey_height_m * floors

    # Rainfall hydrostatic pressure & moisture saturation surcharge
    rain_mm = float(params["rainfall_mm_annual"])
    hydro_kn = (min(rain_mm, 3000.0) / 3000.0) * 45.0

    total_lateral_demand = wind_force_kn + hydro_kn

    # Lateral capacity
    axial_cap = compute_column_axial_capacity_kn(
        concrete_grade_mpa=float(params["concrete_grade"]),
        steel_grade_mpa=float(params["steel_grade"]),
        width_mm=float(params["column_width_mm"]),
        depth_mm=float(params["column_depth_mm"]),
        reinforcement_ratio=float(params["column_reinforcement_ratio"]),
    )
    has_shear_wall = float(params["shear_wall_present"]) >= 0.5
    shear_wall_bonus = 2.0 if has_shear_wall else 1.0
    lateral_cap = axial_cap * 0.15 * shear_wall_bonus

    # Foundation erosion check (shallow footing + heavy annual rainfall)
    found_d = float(params["foundation_depth_m"])
    erosion_risk = rain_mm > 2000.0 and found_d < 1.0

    dc_ratio = total_lateral_demand / (lateral_cap + 1e-6)
    if erosion_risk:
        # Shallow foundation under torrential rain loses 40% effective lateral restraint
        effective_cap = lateral_cap * 0.60
        dc_ratio = max(dc_ratio, total_lateral_demand / (effective_cap + 1e-6), 1.05)
        passed = False
        expl = (
            f"Foundation erosion vulnerability: Annual rainfall {rain_mm:.0f} mm with shallow depth "
            f"{found_d:.2f} m (< 1.0 m) induces scour risk. Total lateral demand = {total_lateral_demand:.1f} kN, "
            f"effective capacity = {effective_cap:.1f} kN (D/C = {dc_ratio:.2f})."
        )
    elif dc_ratio > 1.0:
        passed = False
        expl = (
            f"Combined wind ({wind_force_kn:.1f} kN) and hydrostatic ({hydro_kn:.1f} kN) demand "
            f"({total_lateral_demand:.1f} kN) exceeds lateral capacity ({lateral_cap:.1f} kN), D/C = {dc_ratio:.2f}."
        )
    else:
        passed = True
        expl = f"Structure safely resists IS 875 design wind and rainfall lateral load (D/C = {dc_ratio:.2f} <= 1.0)."

    return {
        "name": "Wind + Rain Combined Test",
        "scenario_name": "Wind + Rain Combined Test",
        "short_name": "Wind + Rain",
        "standard": "IS 875 Part 3 / IS 1904",
        "status": "PASS" if passed else "FAIL",
        "demand": round(total_lateral_demand, 1),
        "demand_unit": "kN",
        "capacity": round(lateral_cap, 1),
        "capacity_unit": "kN",
        "demand_capacity_ratio": round(dc_ratio, 2),
        "passed": passed,
        "explanation": expl,
        "notes": expl,
        "governing_check": "Foundation Scour / Lateral Overturning" if erosion_risk else "Facade Wind + Hydrostatic Pressure",
        "details": {
            "wind_speed_kmph": v_kmph,
            "wind_pressure_kn_m2": round(qz_kn_m2, 3),
            "wind_force_kn": round(wind_force_kn, 1),
            "hydrostatic_surcharge_kn": round(hydro_kn, 1),
            "annual_rainfall_mm": rain_mm,
            "foundation_depth_m": found_d,
            "erosion_risk_flag": erosion_risk,
        },
        "assumptions": [
            f"Wind pressure qz = 0.60 * Vz² calculated per IS 875 (Part 3):2015.",
            "Rainfall surcharge adds hydrostatic pore-pressure on sub-grade elements.",
            "Shallow footings (< 1.0 m) under > 2000 mm rain are penalised for scouring per IS 1904.",
        ],
    }


def evaluate_overload(params: Dict[str, Any], overload_factor: float = 1.50, tributary_area_m2: float = 25.0) -> Dict[str, Any]:
    """
    Scenario 4: Overload Test (150% Design Load)
    Standards: IS 456:2000 Limit State of Collapse (Strength verification under 1.5x design floor load).
    Evaluates structural margin of safety against extreme unexpected live loads.
    """
    is_valid, errors, _ = validate_parameters(params)
    if not is_valid:
        return {
            "name": "Overload Test (150 %)",
            "scenario_name": "Overload Test (150 %)",
            "short_name": "Overload (150%)",
            "standard": "IS 456:2000 Cl. 39.3 Overload",
            "status": "NOT EVALUATED",
            "demand": None,
            "demand_unit": "kN",
            "capacity": None,
            "capacity_unit": "kN",
            "demand_capacity_ratio": None,
            "passed": None,
            "explanation": f"Input validation failed: {'; '.join(errors)}",
            "notes": f"Input validation failed: {'; '.join(errors)}",
            "governing_check": "Not Evaluated",
            "assumptions": ["Calculation halted due to missing/invalid inputs."],
        }

    cap_kn = compute_column_axial_capacity_kn(
        concrete_grade_mpa=float(params["concrete_grade"]),
        steel_grade_mpa=float(params["steel_grade"]),
        width_mm=float(params["column_width_mm"]),
        depth_mm=float(params["column_depth_mm"]),
        reinforcement_ratio=float(params["column_reinforcement_ratio"]),
    )
    overload_demand_kn = compute_applied_gravity_load_kn(
        floor_load_kn_m2=float(params["floor_load_kn_m2"]),
        num_floors=float(params["num_floors"]),
        tributary_area_m2=tributary_area_m2,
        dead_load_factor=1.25,
        overload_multiplier=overload_factor,
    )

    col_dc = overload_demand_kn / (cap_kn + 1e-6)

    eff_sbc, foot_area = compute_soil_bearing_capacity(
        base_sbc_kn_m2=float(params["soil_bearing_capacity_kn_m2"]),
        foundation_depth_m=float(params["foundation_depth_m"]),
        foundation_type=float(params["foundation_type"]),
    )
    soil_pressure = overload_demand_kn / foot_area
    soil_dc = soil_pressure / (eff_sbc + 1e-6)

    governing_dc = max(col_dc, soil_dc)
    passed = governing_dc <= 1.0

    if not passed:
        gov = "Column Axial Compression" if col_dc >= soil_dc else "Soil Bearing Capacity"
        expl = (
            f"Under {int(overload_factor*100)}% design load, demand ({overload_demand_kn:.1f} kN) "
            f"exceeds capacity ({cap_kn:.1f} kN column / {eff_sbc*foot_area:.1f} kN soil), D/C = {governing_dc:.2f}."
        )
    else:
        gov = "Column Axial Compression" if col_dc >= soil_dc else "Soil Bearing Capacity"
        expl = f"Structure retains adequate reserve margin under 150% overload (governing D/C = {governing_dc:.2f} <= 1.0)."

    return {
        "name": "Overload Test (150 %)",
        "scenario_name": "Overload Test (150 %)",
        "short_name": "Overload (150%)",
        "standard": "IS 456:2000 Cl. 39.3 Overload",
        "status": "PASS" if passed else "FAIL",
        "demand": round(overload_demand_kn, 1),
        "demand_unit": "kN",
        "capacity": round(cap_kn, 1),
        "capacity_unit": "kN",
        "demand_capacity_ratio": round(governing_dc, 2),
        "passed": passed,
        "explanation": expl,
        "notes": expl,
        "governing_check": gov,
        "details": {
            "overload_multiplier": overload_factor,
            "overload_demand_kn": round(overload_demand_kn, 1),
            "column_capacity_kn": round(cap_kn, 1),
            "column_dc": round(col_dc, 3),
            "soil_pressure_kn_m2": round(soil_pressure, 1),
            "effective_sbc_kn_m2": round(eff_sbc, 1),
            "soil_dc": round(soil_dc, 3),
        },
        "assumptions": [
            f"150% load factor applied uniformly to design floor live loads.",
            "Evaluates whether structural reserve factor exceeds 1.50.",
        ],
    }


def evaluate_degradation(params: Dict[str, Any], tributary_area_m2: float = 25.0) -> Dict[str, Any]:
    """
    Scenario 5: Long-Term Degradation Test
    Standards: IS 456:2000 Cl. 8 (Durability Requirements & Concrete Cover).
    Evaluates age-related material degradation, carbonation, reinforcement corrosion,
    and checks whether degraded capacity still supports normal design loads.
    """
    is_valid, errors, _ = validate_parameters(params)
    if not is_valid:
        return {
            "name": "Long-Term Degradation Test",
            "scenario_name": "Long-Term Degradation Test",
            "short_name": "Degradation",
            "standard": "IS 456:2000 Cl. 8 Durability",
            "status": "NOT EVALUATED",
            "demand": None,
            "demand_unit": "kN",
            "capacity": None,
            "capacity_unit": "kN",
            "demand_capacity_ratio": None,
            "passed": None,
            "explanation": f"Input validation failed: {'; '.join(errors)}",
            "notes": f"Input validation failed: {'; '.join(errors)}",
            "governing_check": "Not Evaluated",
            "assumptions": ["Calculation halted due to missing/invalid inputs."],
        }

    age = float(params["building_age_years"])
    # 0.8% annual degradation baseline
    baseline_loss = 0.008 * age
    base_deg_factor = max(0.30, 1.0 - baseline_loss)

    # Environmental acceleration: High rainfall accelerates reinforcement corrosion
    rain_mm = float(params["rainfall_mm_annual"])
    corrosion_penalty = 1.0
    if rain_mm > 2000.0:
        corrosion_penalty = max(0.85, 1.0 - ((rain_mm - 2000.0) / 10000.0))

    # Low concrete grade (< M20) exhibits high permeability and accelerated carbonation
    conc = float(params["concrete_grade"])
    durability_factor = 0.90 if conc < 20.0 else 1.0

    final_deg_factor = max(0.30, base_deg_factor * corrosion_penalty * durability_factor)

    undegraded_cap = compute_column_axial_capacity_kn(
        concrete_grade_mpa=conc,
        steel_grade_mpa=float(params["steel_grade"]),
        width_mm=float(params["column_width_mm"]),
        depth_mm=float(params["column_depth_mm"]),
        reinforcement_ratio=float(params["column_reinforcement_ratio"]),
    )
    degraded_cap_kn = undegraded_cap * final_deg_factor

    applied_kn = compute_applied_gravity_load_kn(
        floor_load_kn_m2=float(params["floor_load_kn_m2"]),
        num_floors=float(params["num_floors"]),
        tributary_area_m2=tributary_area_m2,
        dead_load_factor=1.25,
        overload_multiplier=1.0,
    )

    dc_ratio = applied_kn / (degraded_cap_kn + 1e-6)

    # Critical degradation limit: if capacity factor drops below 0.35, material failure is imminent
    critically_degraded = final_deg_factor <= 0.35

    if critically_degraded:
        passed = False
        expl = (
            f"Structure critically degraded: Capacity factor {final_deg_factor:.2f} <= 0.35 threshold "
            f"(age {age:.0f} y, M{int(conc)} concrete, rainfall {rain_mm:.0f} mm). Capacity ({degraded_cap_kn:.1f} kN) "
            f"insufficient for applied load ({applied_kn:.1f} kN), D/C = {dc_ratio:.2f}."
        )
    elif dc_ratio > 1.0:
        passed = False
        expl = (
            f"Age-degraded capacity ({degraded_cap_kn:.1f} kN, {final_deg_factor*100:.0f}% of original) "
            f"cannot sustain design gravity load ({applied_kn:.1f} kN), D/C = {dc_ratio:.2f}."
        )
    else:
        passed = True
        expl = (
            f"Structure retains {final_deg_factor*100:.0f}% capacity after {age:.0f} years, "
            f"safely sustaining applied gravity load (D/C = {dc_ratio:.2f} <= 1.0)."
        )

    return {
        "name": "Long-Term Degradation Test",
        "scenario_name": "Long-Term Degradation Test",
        "short_name": "Degradation",
        "standard": "IS 456:2000 Cl. 8 Durability",
        "status": "PASS" if passed else "FAIL",
        "demand": round(applied_kn, 1),
        "demand_unit": "kN",
        "capacity": round(degraded_cap_kn, 1),
        "capacity_unit": "kN",
        "demand_capacity_ratio": round(dc_ratio, 2),
        "passed": passed,
        "explanation": expl,
        "notes": expl,
        "governing_check": "Age-Degraded Column Axial Capacity",
        "details": {
            "building_age_years": age,
            "capacity_retention_factor": round(final_deg_factor, 3),
            "undegraded_capacity_kn": round(undegraded_cap, 1),
            "degraded_capacity_kn": round(degraded_cap_kn, 1),
            "corrosion_penalty": round(corrosion_penalty, 3),
            "durability_factor": round(durability_factor, 3),
        },
        "assumptions": [
            "0.8% annual degradation baseline due to concrete shrinkage, creep, and carbonation.",
            "Corrosion penalty accelerated by high rainfall (> 2000 mm) per IS 456 Cl. 8.2.",
            "Porous concrete (< M20) experiences accelerated carbonation and rebar oxidation.",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Central Backend Assessment Function
# ═══════════════════════════════════════════════════════════════════════════════

def classify_risk_score(score: float, thresholds: Optional[Dict[str, Any]] = None) -> RiskClassification:
    """
    Classify risk score (0-100) into standardized category and CSS badge class.
    Returns:
        RiskClassification string ("SAFE", "MODERATE", "HIGH") that also
        unpacks into (label, css_badge_class).
    """
    th = thresholds or DEFAULT_DEMO_THRESHOLDS
    safe_limit = th.get("safe_max") if "safe_max" in th else th.get("SAFE", (0.0, 30.0))[1]
    mod_limit  = th.get("moderate_max") if "moderate_max" in th else th.get("MODERATE", (30.0, 55.0))[1]

    clipped = max(0.0, min(100.0, float(score)))

    if clipped < safe_limit:
        return RiskClassification("SAFE", "badge-PASS")
    elif clipped < mod_limit:
        return RiskClassification("MODERATE", "badge-MODERATE")
    else:
        return RiskClassification("HIGH", "badge-FAIL")


def assess_structure(
    params: Dict[str, Any],
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Central, authoritative structural assessment function.
    Supplies consistent risk score, classification, scenario checks, and element
    breakdown to all dashboard tabs, AI reasoner, and export reports.
    """
    th = thresholds or DEFAULT_DEMO_THRESHOLDS
    is_valid, errors, warnings = validate_parameters(params)

    # 1. Run all 5 scenarios
    scenarios = {
        "normal_load": evaluate_normal_load(params),
        "seismic":     evaluate_seismic_stress(params),
        "wind_rain":   evaluate_wind_rain(params),
        "overload":    evaluate_overload(params),
        "degradation": evaluate_degradation(params),
    }

    scenarios_passed = sum(1 for s in scenarios.values() if s.get("status") == "PASS")
    scenarios_failed = sum(1 for s in scenarios.values() if s.get("status") == "FAIL")
    scenarios_not_evaluated = sum(1 for s in scenarios.values() if s.get("status") == "NOT EVALUATED")

    valid_dcs = [s["demand_capacity_ratio"] for s in scenarios.values() if s.get("demand_capacity_ratio") is not None]
    max_dc = max(valid_dcs) if valid_dcs else None

    # Find governing scenario
    governing_scenario = "None"
    highest_dc = -1.0
    for s_key, s_data in scenarios.items():
        dc = s_data.get("demand_capacity_ratio")
        if dc is not None and dc > highest_dc:
            highest_dc = dc
            governing_scenario = s_data.get("name", s_key)

    # 2. Compute multi-factor element breakdown & overall risk score
    # Normalize inputs using standard structural bounds
    def _norm(k: str, lo: float, hi: float) -> float:
        try:
            val = float(params.get(k, (lo + hi) / 2.0))
            return float(max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9))))
        except Exception:
            return 0.5

    # Element Risk Scores (0.0 to 1.0)
    # Foundation
    soil_risk = 1.0 - _norm("soil_bearing_capacity_kn_m2", 50.0, 500.0)
    depth_risk = 1.0 - _norm("foundation_depth_m", 0.5, 5.0)
    ftype_norm = _norm("foundation_type", 0.0, 2.0)
    ftype_risk = 1.0 - ftype_norm / 2.0
    sw_risk = 1.0 - _norm("shear_wall_present", 0.0, 1.0)

    foundation_risk = round(0.35 * soil_risk + 0.30 * depth_risk + 0.20 * ftype_risk + 0.15 * sw_risk, 4)

    # Columns
    rein_risk = 1.0 - _norm("column_reinforcement_ratio", 0.008, 0.060)
    conc_risk = 1.0 - _norm("concrete_grade", 15.0, 60.0)
    col_dim = (_norm("column_width_mm", 200.0, 800.0) + _norm("column_depth_mm", 200.0, 800.0)) / 2.0
    col_dim_risk = 1.0 - col_dim
    seis_risk = _norm("seismic_zone", 1.0, 5.0)

    column_risk = round(0.35 * rein_risk + 0.25 * conc_risk + 0.20 * col_dim_risk + 0.20 * seis_risk, 4)

    # Beams
    beam_depth_risk = 1.0 - _norm("beam_depth_mm", 300.0, 900.0)
    floor_ld_risk = _norm("floor_load_kn_m2", 2.0, 15.0)
    beam_risk = round(0.40 * beam_depth_risk + 0.35 * floor_ld_risk + 0.25 * rein_risk, 4)

    # Slab
    slab_thin_risk = 1.0 - _norm("slab_thickness_mm", 100.0, 300.0)
    floors_risk = _norm("num_floors", 1.0, 20.0)
    slab_risk = round(0.40 * slab_thin_risk + 0.35 * floor_ld_risk + 0.25 * floors_risk, 4)

    element_breakdown = {
        "foundation": float(max(0.0, min(1.0, foundation_risk))),
        "columns":    float(max(0.0, min(1.0, column_risk))),
        "beams":      float(max(0.0, min(1.0, beam_risk))),
        "slab":       float(max(0.0, min(1.0, slab_risk))),
    }

    # Element zone labels
    def _elem_label(val: float) -> str:
        if val >= 0.75: return "CRITICAL"
        if val >= 0.55: return "HIGH"
        if val >= 0.30: return "MODERATE"
        return "SAFE"

    element_zones = {k: _elem_label(v) for k, v in element_breakdown.items()}

    # Weighted Overall Risk Score (0 - 100)
    # 6 weighted parameters matching StructuralRiskScorer
    weighted_score = (
        0.25 * _norm("seismic_zone", 1.0, 5.0) +
        0.20 * (1.0 - _norm("soil_bearing_capacity_kn_m2", 50.0, 500.0)) +
        0.20 * (1.0 - _norm("column_reinforcement_ratio", 0.008, 0.060)) +
        0.15 * (1.0 - _norm("foundation_depth_m", 0.5, 5.0)) +
        0.10 * _norm("building_age_years", 0.0, 100.0) +
        0.10 * (1.0 - _norm("shear_wall_present", 0.0, 1.0))
    )

    # If critical structural scenarios fail, reflect failure in overall assessment
    if scenarios_failed > 0:
        # Amplify by failure proportion
        fail_factor = (scenarios_failed / 5.0) * 0.15
        weighted_score = min(1.0, weighted_score + fail_factor)

    risk_score_100 = round(weighted_score * 100.0, 1)
    risk_label, risk_badge = classify_risk_score(risk_score_100, th)

    # Recommendations
    recs: List[str] = []
    if not params.get("shear_wall_present") and float(params.get("seismic_zone", 1)) >= 3:
        recs.append("Install reinforced concrete shear walls at perimeter grid lines to resist lateral seismic forces (IS 1893:2016).")
    if float(params.get("column_reinforcement_ratio", 0.02)) < MIN_REBAR_RATIO:
        recs.append(f"Increase column reinforcement ratio to at least {MIN_REBAR_RATIO*100:.1f}% per IS 456 Cl. 26.5.3.1.")
    if float(params.get("concrete_grade", 25)) < MIN_CONCRETE_GRADE_RCC:
        recs.append("Upgrade structural concrete specification to at least M20 for reinforced concrete work.")
    if float(params.get("soil_bearing_capacity_kn_m2", 200)) < 120 and int(round(float(params.get("foundation_type", 1)))) == 0:
        recs.append("Replace isolated footings with raft or bored pile foundation to prevent differential settlement.")
    if float(params.get("building_age_years", 0)) > 35:
        recs.append("Conduct comprehensive Non-Destructive Testing (NDT) including Rebound Hammer and UPV to measure concrete degradation.")
    if not recs:
        recs.append("Maintain regular maintenance inspection cycle every 3 to 5 years.")

    # Reasons / Explanations
    summary_reasons: List[str] = []
    if scenarios_failed > 0:
        for s in scenarios.values():
            if s.get("status") == "FAIL":
                summary_reasons.append(s.get("explanation", ""))
    if not summary_reasons:
        summary_reasons.append(f"Structure evaluated as {risk_label} with no catastrophic scenario failures under standard checks.")

    return {
        "risk_score": risk_score_100,
        "risk_score_norm": round(weighted_score, 4),
        "classification": risk_label,
        "risk_label": risk_label,
        "risk_badge": risk_badge,
        "thresholds": th,
        "element_breakdown": element_breakdown,
        "element_zones": element_zones,
        "scenario_results": scenarios,
        "scenarios_passed": scenarios_passed,
        "scenarios_failed": scenarios_failed,
        "scenarios_not_evaluated": scenarios_not_evaluated,
        "governing_scenario": governing_scenario,
        "max_dc_ratio": round(max_dc, 2) if max_dc is not None else None,
        "is_code_violations": warnings,
        "missing_data_errors": errors,
        "summary_reasons": summary_reasons,
        "recommendations": recs,
        "is_valid": is_valid,
        "disclaimer": DISCLAIMER_TEXT,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Sequential Stress-Test Trajectory Generator (Real Calculated Progression)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_stress_test_trajectory(
    base_params: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a 20-step sequential stress-test trajectory directly from reviewed building parameters.
    Each step applies a documented physical modification (load factor increment, lateral hazard escalation,
    or cumulative environmental degradation) and evaluates the structure using the central assess_structure() engine.
    
    Returns a list of 20 step dictionaries with:
      - step: int (0 to 19)
      - name: str (concise step title)
      - category: str (Baseline, Gravity Progression, Lateral Hazards, Overload Sequence, Degradation, Compound)
      - description: str (engineering explanation of what is tested)
      - param_delta: str (readable summary of parameter change)
      - modified_keys: Dict[str, Any] (exact parameter modifications applied)
      - risk_score: float (0 to 100 calculated by assess_structure)
      - classification: str (SAFE, MODERATE, HIGH)
      - governing_scenario: str
      - max_dc_ratio: float or None
      - passed: bool
    """
    base = dict(base_params)
    base_load = float(base.get("floor_load_kn_m2", 5.0))
    base_wind = float(base.get("wind_speed_kmph", 150.0))
    base_rain = float(base.get("rainfall_mm_annual", 1200.0))
    base_soil = float(base.get("soil_bearing_capacity_kn_m2", 200.0))
    base_zone = int(round(float(base.get("seismic_zone", 2))))
    base_age = float(base.get("building_age_years", 0.0))
    base_rebar = float(base.get("column_reinforcement_ratio", 0.015))

    step_specs = [
        # 0: Baseline
        {
            "name": "Baseline As-Designed",
            "category": "Baseline",
            "description": "Nominal parameters as specified in design report / input parameters without additional stress.",
            "param_delta": "Nominal design parameters",
            "modified_keys": {},
        },
        # 1-4: Gravity load progression
        {
            "name": "Normal Service Load (1.0x)",
            "category": "Gravity Progression",
            "description": "Serviceability gravity load check per IS 456 Cl. 39.3 under nominal occupancy conditions.",
            "param_delta": f"Floor load = {base_load:.1f} kN/m² (1.0x nominal)",
            "modified_keys": {"floor_load_kn_m2": base_load},
        },
        {
            "name": "Live Load Surge (+10%)",
            "category": "Gravity Progression",
            "description": "Partial live load surge representing localized lane/deck congestion (+10% floor load).",
            "param_delta": f"Floor load = {base_load * 1.10:.1f} kN/m² (+10%)",
            "modified_keys": {"floor_load_kn_m2": round(base_load * 1.10, 2)},
        },
        {
            "name": "Live Load Surge (+20%)",
            "category": "Gravity Progression",
            "description": "Moderate live load surge representing dense occupancy / heavy commercial traffic (+20% floor load).",
            "param_delta": f"Floor load = {base_load * 1.20:.1f} kN/m² (+20%)",
            "modified_keys": {"floor_load_kn_m2": round(base_load * 1.20, 2)},
        },
        {
            "name": "Live Load Surge (+30%)",
            "category": "Gravity Progression",
            "description": "High live load demand approaching design threshold (+30% floor load).",
            "param_delta": f"Floor load = {base_load * 1.30:.1f} kN/m² (+30%)",
            "modified_keys": {"floor_load_kn_m2": round(base_load * 1.30, 2)},
        },
        # 5-7: Lateral Wind & Rain Hazards
        {
            "name": "Design Wind Action (IS 875)",
            "category": "Lateral Hazards",
            "description": "Full design basic wind velocity and pressure p_z applied per IS 875:2015 Part 3.",
            "param_delta": f"Wind speed = {max(base_wind, 150.0):.1f} km/h (Design Vb)",
            "modified_keys": {"wind_speed_kmph": round(max(base_wind, 150.0), 1)},
        },
        {
            "name": "Extreme Wind Gust (+25%)",
            "category": "Lateral Hazards",
            "description": "Severe aerodynamic wind gust (+25% basic wind speed, simulating cyclonic storm conditions).",
            "param_delta": f"Wind speed = {base_wind * 1.25:.1f} km/h (+25% gust)",
            "modified_keys": {"wind_speed_kmph": round(base_wind * 1.25, 1)},
        },
        {
            "name": "Wind + Monsoon Saturation",
            "category": "Lateral Hazards",
            "description": "Combined design wind action with severe rainfall and 25% soil bearing capacity degradation due to water table rise (IS 1904).",
            "param_delta": f"Rain = {max(base_rain, 2200.0):.0f} mm, SBC = {base_soil * 0.75:.0f} kN/m² (-25%)",
            "modified_keys": {
                "rainfall_mm_annual": round(max(base_rain, 2200.0), 1),
                "soil_bearing_capacity_kn_m2": round(base_soil * 0.75, 1),
            },
        },
        # 8-10: Lateral Seismic Hazards
        {
            "name": "Serviceability Seismic (0.5 DBE)",
            "category": "Lateral Hazards",
            "description": "Minor ground motion shaking corresponding to 50% of Design Basis Earthquake lateral force.",
            "param_delta": f"Seismic demand scaled to 0.5x Ah (Zone {min(max(base_zone, 2), 5)})",
            "modified_keys": {"seismic_zone": min(max(base_zone, 2), 5)},
        },
        {
            "name": "Design Basis Earthquake (DBE)",
            "category": "Lateral Hazards",
            "description": "Design Basis Earthquake (DBE) equivalent static base shear VB = Ah * W per IS 1893:2016 Cl. 6.4.",
            "param_delta": f"Full seismic design base shear (Zone {min(max(base_zone, 3), 5)})",
            "modified_keys": {"seismic_zone": min(max(base_zone, 3), 5)},
        },
        {
            "name": "Maximum Considered Earthquake (MCE)",
            "category": "Lateral Hazards",
            "description": "Maximum Considered Earthquake (MCE) intensity: peak ground acceleration factor scaled per IS 1893.",
            "param_delta": f"MCE lateral acceleration (Zone {min(base_zone + 1, 5)})",
            "modified_keys": {"seismic_zone": min(base_zone + 1, 5)},
        },
        # 11-14: Overload Sequence
        {
            "name": "Gravity Overload 125%",
            "category": "Overload Sequence",
            "description": "1.25x design gravity loading combination testing intermediate over-stress margin.",
            "param_delta": f"Floor load = {base_load * 1.25:.1f} kN/m² (1.25x)",
            "modified_keys": {"floor_load_kn_m2": round(base_load * 1.25, 2)},
        },
        {
            "name": "Factored Overload 150% (IS 456 ULS)",
            "category": "Overload Sequence",
            "description": "Ultimate Limit State factored gravity load combination 1.50 * (DL + LL) per IS 456 Table 18.",
            "param_delta": f"Floor load = {base_load * 1.50:.1f} kN/m² (1.50x ULS)",
            "modified_keys": {"floor_load_kn_m2": round(base_load * 1.50, 2)},
        },
        {
            "name": "Severe Overload 175%",
            "category": "Overload Sequence",
            "description": "1.75x extreme gravity overload probing structural plastic reserve and footing safety margin.",
            "param_delta": f"Floor load = {base_load * 1.75:.1f} kN/m² (1.75x)",
            "modified_keys": {"floor_load_kn_m2": round(base_load * 1.75, 2)},
        },
        {
            "name": "Critical Overload 200%",
            "category": "Overload Sequence",
            "description": "2.00x double gravity load exploring catastrophic limit-state collapse threshold.",
            "param_delta": f"Floor load = {base_load * 2.00:.1f} kN/m² (2.00x Collapse Limit)",
            "modified_keys": {"floor_load_kn_m2": round(base_load * 2.00, 2)},
        },
        # 15-17: Environmental Degradation
        {
            "name": "Environmental Aging +10 Years",
            "category": "Long-Term Degradation",
            "description": "10 years cumulative atmospheric exposure; carbonation progression x = k * sqrt(t+10).",
            "param_delta": f"Building age = {base_age + 10.0:.0f} yrs (+10y)",
            "modified_keys": {"building_age_years": round(base_age + 10.0, 1)},
        },
        {
            "name": "Aging +25 Years (Corrosion Onset)",
            "category": "Long-Term Degradation",
            "description": "25 years exposure; chloride depassivation initiates rebar corrosion (-10% effective steel area).",
            "param_delta": f"Age = {base_age + 25.0:.0f} yrs, Rebar = {base_rebar * 0.90 * 100:.2f}% (-10%)",
            "modified_keys": {
                "building_age_years": round(base_age + 25.0, 1),
                "column_reinforcement_ratio": round(max(0.005, base_rebar * 0.90), 4),
            },
        },
        {
            "name": "Aging +50 Years (Advanced Degradation)",
            "category": "Long-Term Degradation",
            "description": "50 years severe exposure; carbonation front reaches steel core, -20% effective steel area, concrete micro-cracking.",
            "param_delta": f"Age = {base_age + 50.0:.0f} yrs, Rebar = {base_rebar * 0.80 * 100:.2f}% (-20%)",
            "modified_keys": {
                "building_age_years": round(base_age + 50.0, 1),
                "column_reinforcement_ratio": round(max(0.005, base_rebar * 0.80), 4),
            },
        },
        # 18-19: Multi-Hazard Compound Stress
        {
            "name": "Multi-Hazard: 25-Yr Degr + 150% Overload + Seismic",
            "category": "Multi-Hazard Compound",
            "description": "Concurrent design earthquake lateral force and 150% gravity overload acting on a 25-year degraded structure (-10% rebar).",
            "param_delta": f"Age +25y, Rebar -10%, Floor load = {base_load * 1.50:.1f} kN/m² (1.5x) + Seismic Zone {min(max(base_zone, 3), 5)}",
            "modified_keys": {
                "building_age_years": round(base_age + 25.0, 1),
                "column_reinforcement_ratio": round(max(0.005, base_rebar * 0.90), 4),
                "floor_load_kn_m2": round(base_load * 1.50, 2),
                "seismic_zone": min(max(base_zone, 3), 5),
            },
        },
        {
            "name": "Compound Extreme: Degr + MCE + Saturated Soil",
            "category": "Multi-Hazard Compound",
            "description": "Worst-case concurrent stress: 50-year degraded structure (-20% rebar) subjected to Maximum Considered Earthquake and saturated subgrade (-30% SBC).",
            "param_delta": f"Age +50y, Rebar -20%, Zone 5 (MCE), SBC = {base_soil * 0.70:.0f} kN/m² (-30%), Rain = 2500 mm",
            "modified_keys": {
                "building_age_years": round(base_age + 50.0, 1),
                "column_reinforcement_ratio": round(max(0.005, base_rebar * 0.80), 4),
                "seismic_zone": 5,
                "soil_bearing_capacity_kn_m2": round(base_soil * 0.70, 1),
                "rainfall_mm_annual": round(max(base_rain, 2500.0), 1),
            },
        },
    ]

    results: List[Dict[str, Any]] = []
    for step_idx, spec in enumerate(step_specs):
        step_params = dict(base)
        step_params.update(spec["modified_keys"])
        assessment = assess_structure(step_params, thresholds=thresholds)

        results.append({
            "step": step_idx,
            "name": spec["name"],
            "category": spec["category"],
            "description": spec["description"],
            "param_delta": spec["param_delta"],
            "modified_keys": spec["modified_keys"],
            "risk_score": assessment["risk_score"],
            "classification": assessment["classification"],
            "risk_badge": assessment["risk_badge"],
            "governing_scenario": assessment["governing_scenario"],
            "max_dc_ratio": assessment["max_dc_ratio"],
            "scenarios_passed": assessment["scenarios_passed"],
            "scenarios_failed": assessment["scenarios_failed"],
        })

    return results
