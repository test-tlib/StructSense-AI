"""
test_risk_thresholds.py
=======================
Automated tests for configurable demonstration risk thresholds and classification.
"""
import pytest
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from risk.structural_assessment import (
    DEFAULT_DEMO_THRESHOLDS,
    DISCLAIMER_TEXT,
    classify_risk_score,
)


def test_default_thresholds_values():
    assert DEFAULT_DEMO_THRESHOLDS["safe_max"] == 30.0
    assert DEFAULT_DEMO_THRESHOLDS["moderate_max"] == 55.0


def test_classify_safe_range():
    assert classify_risk_score(0.0) == "SAFE"
    assert classify_risk_score(15.5) == "SAFE"
    assert classify_risk_score(29.99) == "SAFE"


def test_classify_moderate_range():
    # Exactly 30.0 begins MODERATE range
    assert classify_risk_score(30.0) == "MODERATE"
    assert classify_risk_score(38.4) == "MODERATE"
    assert classify_risk_score(54.99) == "MODERATE"


def test_classify_high_range():
    # 55.0 and above is HIGH range
    assert classify_risk_score(55.0) == "HIGH"
    assert classify_risk_score(75.0) == "HIGH"
    assert classify_risk_score(100.0) == "HIGH"


def test_classify_out_of_bounds_clipping():
    # Negative values clip to 0 -> SAFE
    assert classify_risk_score(-10.0) == "SAFE"
    # Over 100 clips to 100 -> HIGH
    assert classify_risk_score(150.0) == "HIGH"


def test_custom_thresholds():
    custom_thresholds = {"safe_max": 20.0, "moderate_max": 70.0}
    assert classify_risk_score(18.0, custom_thresholds) == "SAFE"
    assert classify_risk_score(25.0, custom_thresholds) == "MODERATE"
    assert classify_risk_score(65.0, custom_thresholds) == "MODERATE"
    assert classify_risk_score(72.0, custom_thresholds) == "HIGH"


def test_disclaimer_presence():
    assert "prototype" in DISCLAIMER_TEXT.lower()
    assert "unvalidated demonstration thresholds" in DISCLAIMER_TEXT.lower()
    assert "not certify building safety" in DISCLAIMER_TEXT.lower()
