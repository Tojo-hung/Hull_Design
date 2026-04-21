"""
tests/test_constraints.py — Tests for hull_core.constraints.
==============================================================

Validates the three-tier constraint system:
  - Tier 0: parameter validation (fast, deterministic)
  - Tier 1: PCHIP-based geometric pre-screening
  - Tier 2: placeholder (always passes for now)

No CadQuery dependency.

Run:
    pytest tests/test_constraints.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hull_core.schema import BowSection, HullForm, SectionParams, TransomSection
from hull_core.constraints import ConstraintResult, HullConstraints, Range


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def valid_form() -> HullForm:
    """A known-good hull form based on desktop DEFAULTS."""
    from moth_designer.config import DEFAULTS
    return HullForm.from_flat_dict(
        {k: float(v) for k, v in DEFAULTS.items()}
    )


@pytest.fixture()
def constraints() -> HullConstraints:
    """Default constraint set (matches optimizer bounds)."""
    return HullConstraints()


# ══════════════════════════════════════════════════════════════════════
# Section 1 — Range primitive
# ══════════════════════════════════════════════════════════════════════

class TestRange:

    def test_contains(self):
        r = Range(10, 20)
        assert r.contains(15)
        assert r.contains(10)
        assert r.contains(20)
        assert not r.contains(9.9)
        assert not r.contains(20.1)

    def test_clamp(self):
        r = Range(10, 20)
        assert r.clamp(5) == 10
        assert r.clamp(25) == 20
        assert r.clamp(15) == 15

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="lo.*hi"):
            Range(20, 10)

    def test_repr(self):
        assert repr(Range(1, 2)) == "[1, 2]"


# ══════════════════════════════════════════════════════════════════════
# Section 2 — ConstraintResult
# ══════════════════════════════════════════════════════════════════════

class TestConstraintResult:

    def test_passing_is_truthy(self):
        r = ConstraintResult(passed=True)
        assert r
        assert str(r) == "ok"

    def test_failing_is_falsy(self):
        r = ConstraintResult(passed=False, violations=["bad draft"])
        assert not r
        assert "bad draft" in str(r)

    def test_merge_both_pass(self):
        a = ConstraintResult(passed=True)
        b = ConstraintResult(passed=True)
        merged = a.merge(b)
        assert merged

    def test_merge_one_fails(self):
        a = ConstraintResult(passed=True)
        b = ConstraintResult(passed=False, violations=["x"])
        merged = a.merge(b)
        assert not merged
        assert "x" in merged.violations

    def test_merge_both_fail(self):
        a = ConstraintResult(passed=False, violations=["a"])
        b = ConstraintResult(passed=False, violations=["b"])
        merged = a.merge(b)
        assert not merged
        assert len(merged.violations) == 2


# ══════════════════════════════════════════════════════════════════════
# Section 3 — Tier 0: parameter validation
# ══════════════════════════════════════════════════════════════════════

class TestTier0:
    """Tier 0 checks: pure parameter validation, no interpolation."""

    def test_valid_hull_passes(self, valid_form, constraints):
        result = constraints.check_parameters(valid_form)
        assert result, f"Valid hull failed Tier 0: {result}"

    def test_station_draft_below_floor_fails(self, constraints):
        """If a station draft is below max(bow_draft, transom_draft), fail."""
        form = HullForm(
            bow=BowSection(draft=100, sheer_height=200),
            transom=TransomSection(
                half_beam=195, draft=65, sheer_height=200,
            ),
            stations=[
                SectionParams(x=1000, half_beam=200, draft=150),
                # This station's draft (50) is below the floor (100)
                SectionParams(x=2000, half_beam=200, draft=50),
            ],
        )
        result = constraints.check_parameters(form)
        assert not result
        assert any("draft" in v and "floor" in v for v in result.violations)

    def test_draft_exceeds_class_max_fails(self, constraints):
        form = HullForm(
            bow=BowSection(draft=75, sheer_height=200),
            transom=TransomSection(
                half_beam=195, draft=65, sheer_height=200,
            ),
            stations=[
                SectionParams(x=1000, half_beam=200, draft=400),
            ],
        )
        result = constraints.check_parameters(form)
        assert not result
        assert any("class max" in v for v in result.violations)

    def test_low_freeboard_fails(self, constraints):
        form = HullForm(
            bow=BowSection(draft=75, sheer_height=50),  # too low
            transom=TransomSection(
                half_beam=195, draft=65, sheer_height=200,
            ),
            stations=[
                SectionParams(x=1000, half_beam=200, draft=100),
            ],
        )
        result = constraints.check_parameters(form)
        assert not result
        assert any("min_freeboard" in v for v in result.violations)

    def test_multiple_violations(self, constraints):
        """Multiple issues should all be reported."""
        form = HullForm(
            bow=BowSection(draft=75, sheer_height=50),
            transom=TransomSection(
                half_beam=195, draft=65, sheer_height=200,
            ),
            stations=[
                SectionParams(x=1000, half_beam=200, draft=400),
            ],
        )
        result = constraints.check_parameters(form)
        assert not result
        assert len(result.violations) >= 2  # low freeboard + excess draft


# ══════════════════════════════════════════════════════════════════════
# Section 4 — Tier 1: geometric pre-screening
# ══════════════════════════════════════════════════════════════════════

class TestTier1:
    """Tier 1 checks: PCHIP-based geometric screening."""

    def test_valid_hull_passes(self, valid_form, constraints):
        result = constraints.check_form(valid_form)
        assert result, f"Valid hull failed Tier 1: {result}"

    def test_excessive_rocker_fails(self, constraints):
        """Very deep mid-hull + shallow endpoints = excessive rocker."""
        form = HullForm(
            bow=BowSection(draft=30, sheer_height=200),
            transom=TransomSection(
                half_beam=195, draft=30, sheer_height=200,
            ),
            stations=[
                SectionParams(x=500, half_beam=100, draft=100),
                SectionParams(x=1000, half_beam=200, draft=300),
                SectionParams(x=2000, half_beam=200, draft=300),
                SectionParams(x=3000, half_beam=150, draft=100),
            ],
        )
        result = constraints.check_form(form)
        assert not result
        assert any("rocker" in v for v in result.violations)

    def test_zero_rocker_fails(self, constraints):
        """Perfectly flat keel = rocker ~0, below the 60mm minimum."""
        draft_val = 100.0
        form = HullForm(
            bow=BowSection(draft=draft_val, sheer_height=200),
            transom=TransomSection(
                half_beam=195, draft=draft_val, sheer_height=200,
            ),
            stations=[
                SectionParams(x=500, half_beam=100, draft=draft_val),
                SectionParams(x=1000, half_beam=200, draft=draft_val),
                SectionParams(x=2000, half_beam=200, draft=draft_val),
                SectionParams(x=3000, half_beam=150, draft=draft_val),
            ],
        )
        result = constraints.check_form(form)
        assert not result
        assert any("rocker" in v for v in result.violations)

    def test_beam_reversal_flagged(self, constraints):
        """A local minimum in half_beam between two larger stations."""
        form = HullForm(
            bow=BowSection(draft=75, sheer_height=200),
            transom=TransomSection(
                half_beam=195, draft=65, sheer_height=200,
            ),
            stations=[
                SectionParams(x=500, half_beam=150, draft=100),
                SectionParams(x=1000, half_beam=200, draft=120),
                SectionParams(x=1500, half_beam=80, draft=130),   # reversal
                SectionParams(x=2000, half_beam=200, draft=120),
                SectionParams(x=2800, half_beam=180, draft=100),
            ],
        )
        result = constraints.check_form(form)
        assert not result
        assert any("reversal" in v.lower() for v in result.violations)


# ══════════════════════════════════════════════════════════════════════
# Section 5 — Tier 2: placeholder
# ══════════════════════════════════════════════════════════════════════

class TestTier2:

    def test_placeholder_always_passes(self, valid_form, constraints):
        """Tier 2 is a no-op placeholder — it must always return True."""
        result = constraints.check_solid(None, valid_form)
        assert result


# ══════════════════════════════════════════════════════════════════════
# Section 6 — Full pipeline: Tier 0 + Tier 1
# ══════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Run tiers 0 and 1 together on real parameter sets."""

    def test_desktop_defaults_pass_both_tiers(self, valid_form, constraints):
        r0 = constraints.check_parameters(valid_form)
        assert r0, f"Tier 0 failed: {r0}"
        r1 = constraints.check_form(valid_form)
        assert r1, f"Tier 1 failed: {r1}"

    def test_best_hull_json_passes(self, constraints):
        import json
        path = _REPO_ROOT / "best_hull.json"
        if not path.exists():
            pytest.skip("best_hull.json not found")
        d = json.loads(path.read_text())
        form = HullForm.from_flat_dict(d)
        r0 = constraints.check_parameters(form)
        assert r0, f"Tier 0 failed on best_hull.json: {r0}"
        r1 = constraints.check_form(form)
        assert r1, f"Tier 1 failed on best_hull.json: {r1}"
