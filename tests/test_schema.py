"""
tests/test_schema.py — Tests for hull_core.schema.
====================================================

Validates:
  - Round-trip:  flat dict → HullForm → flat dict (lossless)
  - Invalid inputs raise ValueError
  - Station ordering is enforced
  - Edge cases (keel_width > half_beam, negative draft, etc.)
  - Compatibility with desktop (4-pt) format
  - Real JSON files from the repo parse correctly

Run:
    pytest tests/test_schema.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hull_core.schema import (
    BowSection,
    HullForm,
    SectionParams,
    TransomSection,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def desktop_defaults() -> dict:
    """The DEFAULTS dict from moth_designer/config.py (4-station desktop)."""
    from moth_designer.config import DEFAULTS
    return {k: float(v) for k, v in DEFAULTS.items()}


# ══════════════════════════════════════════════════════════════════════
# Section 1 — Round-trip conversion
# ══════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    """Flat dict → HullForm → flat dict must be lossless."""

    def test_desktop_defaults_round_trip(self, desktop_defaults):
        form = HullForm.from_flat_dict(desktop_defaults)
        out = form.to_flat_dict()

        # Every key from the original must appear with the same value
        for key in desktop_defaults:
            if key in out:
                assert abs(out[key] - desktop_defaults[key]) < 1e-6, (
                    f"Key {key!r}: expected {desktop_defaults[key]}, "
                    f"got {out[key]}"
                )

    def test_desktop_station_count(self, desktop_defaults):
        form = HullForm.from_flat_dict(desktop_defaults)
        assert form.n_stations == 4

    def test_output_contains_all_station_keys(self, desktop_defaults):
        form = HullForm.from_flat_dict(desktop_defaults)
        out = form.to_flat_dict()
        for i in range(1, 5):
            for suffix in ("_x", "_hb", "_d", "_dz", "_kw", "_hz"):
                key = f"p{i}{suffix}"
                assert key in out, f"Missing key {key!r} in output"

    def test_desktop_has_no_dw_keys(self, desktop_defaults):
        """Desktop format does NOT have p{i}_dw keys."""
        form = HullForm.from_flat_dict(desktop_defaults)
        out = form.to_flat_dict()
        for i in range(1, 5):
            assert f"p{i}_dw" not in out


# ══════════════════════════════════════════════════════════════════════
# Section 2 — Real JSON file loading
# ══════════════════════════════════════════════════════════════════════

class TestRealFiles:
    """Parse actual JSON files stored in the repo."""

    def _load_json(self, name: str) -> dict | None:
        path = _REPO_ROOT / name
        if not path.exists():
            pytest.skip(f"{name} not found")
        return json.loads(path.read_text())

    def test_best_hull_json(self):
        d = self._load_json("best_hull.json")
        form = HullForm.from_flat_dict(d)
        assert form.n_stations == 4
        assert form.transom.half_beam > 100

    def test_hull_design_json(self):
        d = self._load_json("hull_design.json")
        form = HullForm.from_flat_dict(d)
        assert form.n_stations == 4

    def test_best_hull_unknown_keys_ignored(self):
        """best_hull.json contains 'ends_draft' which is not a standard key."""
        d = self._load_json("best_hull.json")
        assert "ends_draft" in d  # verify the file has the extra key
        # from_flat_dict must not raise
        form = HullForm.from_flat_dict(d)
        # And it must not appear in the output
        out = form.to_flat_dict()
        assert "ends_draft" not in out


# ══════════════════════════════════════════════════════════════════════
# Section 3 — Station ordering
# ══════════════════════════════════════════════════════════════════════

class TestStationOrdering:

    def test_stations_sorted_by_x(self):
        """Stations provided out of order must be sorted in __post_init__."""
        form = HullForm(stations=[
            SectionParams(x=2000, half_beam=200, draft=100),
            SectionParams(x=500, half_beam=100, draft=80),
            SectionParams(x=1000, half_beam=150, draft=90),
        ])
        xs = [s.x for s in form.stations]
        assert xs == [500.0, 1000.0, 2000.0]

    def test_duplicate_x_raises(self):
        with pytest.raises(ValueError, match="too close"):
            HullForm(stations=[
                SectionParams(x=1000, half_beam=100, draft=80),
                SectionParams(x=1000, half_beam=150, draft=90),
            ])

    def test_station_outside_lwl_raises(self):
        with pytest.raises(ValueError, match="strictly between"):
            HullForm(stations=[
                SectionParams(x=4000, half_beam=100, draft=80),
            ])

    def test_station_at_zero_raises(self):
        with pytest.raises(ValueError, match="strictly between"):
            HullForm(stations=[
                SectionParams(x=0.0, half_beam=100, draft=80),
            ])


# ══════════════════════════════════════════════════════════════════════
# Section 4 — SectionParams validation
# ══════════════════════════════════════════════════════════════════════

class TestSectionValidation:

    def test_keel_width_exceeds_half_beam(self):
        with pytest.raises(ValueError, match="keel_width.*exceeds.*half_beam"):
            SectionParams(x=1000, half_beam=100, draft=80, keel_width=150)

    def test_negative_draft(self):
        with pytest.raises(ValueError, match="draft.*non-negative"):
            SectionParams(x=1000, half_beam=100, draft=-10)

    def test_negative_half_beam(self):
        with pytest.raises(ValueError, match="half_beam.*non-negative"):
            SectionParams(x=1000, half_beam=-5, draft=80)

    def test_sheer_below_beam_height(self):
        with pytest.raises(ValueError, match="sheer_height.*must exceed"):
            SectionParams(
                x=1000, half_beam=100, draft=80,
                sheer_height=50, beam_height=100,
            )

    def test_sheer_equal_beam_height(self):
        with pytest.raises(ValueError, match="sheer_height.*must exceed"):
            SectionParams(
                x=1000, half_beam=100, draft=80,
                sheer_height=100, beam_height=100,
            )

    def test_valid_section(self):
        """A well-formed section should not raise."""
        s = SectionParams(
            x=1000, half_beam=200, draft=150,
            keel_width=50, beam_height=-30, sheer_height=200,
        )
        assert s.effective_deck_half_beam == 200  # defaults to half_beam

    def test_deck_half_beam_explicit(self):
        s = SectionParams(
            x=1000, half_beam=200, draft=150,
            deck_half_beam=220,
        )
        assert s.effective_deck_half_beam == 220

    def test_keel_width_at_boundary(self):
        """keel_width == half_beam is valid (flat-bottomed section)."""
        s = SectionParams(x=1000, half_beam=100, draft=80, keel_width=100)
        assert s.keel_width == 100


# ══════════════════════════════════════════════════════════════════════
# Section 5 — BowSection / TransomSection validation
# ══════════════════════════════════════════════════════════════════════

class TestEndpointValidation:

    def test_bow_negative_draft(self):
        with pytest.raises(ValueError, match="bow draft.*non-negative"):
            BowSection(draft=-10, sheer_height=150)

    def test_bow_zero_sheer(self):
        with pytest.raises(ValueError, match="bow sheer_height.*positive"):
            BowSection(draft=50, sheer_height=0)

    def test_transom_keel_exceeds_beam(self):
        with pytest.raises(ValueError, match="transom keel_width.*exceeds"):
            TransomSection(
                half_beam=100, draft=50, sheer_height=200,
                keel_width=150,
            )

    def test_transom_sheer_below_beam_height(self):
        with pytest.raises(ValueError, match="transom sheer_height.*must exceed"):
            TransomSection(
                half_beam=100, draft=50,
                sheer_height=50, beam_height=100,
            )

    def test_valid_bow(self):
        b = BowSection(draft=75, sheer_height=150)
        assert b.draft == 75

    def test_valid_transom(self):
        t = TransomSection(
            half_beam=195, draft=65, sheer_height=200,
            keel_width=100, beam_height=0,
        )
        assert t.half_beam == 195


# ══════════════════════════════════════════════════════════════════════
# Section 6 — HullForm properties
# ══════════════════════════════════════════════════════════════════════

class TestHullFormProperties:

    def test_n_stations(self, desktop_defaults):
        form = HullForm.from_flat_dict(desktop_defaults)
        assert form.n_stations == 4

    def test_draft_floor(self, desktop_defaults):
        form = HullForm.from_flat_dict(desktop_defaults)
        # draft_floor = max(bow_draft=75, transom_draft=65) = 75
        assert form.draft_floor == 75.0

    def test_repr(self, desktop_defaults):
        form = HullForm.from_flat_dict(desktop_defaults)
        r = repr(form)
        assert "HullForm" in r
        assert "stations=4" in r

    def test_no_stations_raises(self):
        """from_flat_dict with no p{i}_x keys should raise."""
        with pytest.raises(ValueError, match="No station keys"):
            HullForm.from_flat_dict({"bow_draft": 75.0})

    def test_default_hull_form(self):
        """Default HullForm (no args) should be constructible."""
        form = HullForm()
        assert form.n_stations == 0
        assert form.lwl == 3355.0

    def test_custom_lwl(self):
        form = HullForm.from_flat_dict(
            {"p1_x": 500, "p1_hb": 100, "p1_d": 80},
            lwl=4000.0,
        )
        assert form.lwl == 4000.0
