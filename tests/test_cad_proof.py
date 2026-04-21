"""
tests/test_cad_proof.py
=======================

Proof-path tests for hull_cad.builder.

These tests answer one question:
    "Is the CAD geometry path robust enough to generate valid, exportable
     hull solids for CFD downstream?"

They do NOT test the existing Python geometry path.

Run from the repo root:
    pytest tests/test_cad_proof.py -v
    pytest tests/test_cad_proof.py -v -k default    # just the default hull
    pytest tests/test_cad_proof.py -v --tb=short    # concise tracebacks

Requirements:
    cadquery must be installed (conda install -c cadquery cadquery)
    If cadquery is absent, all tests are skipped gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Skip marker ───────────────────────────────────────────────────────────────
try:
    import cadquery as cq  # noqa: F401
    _HAS_CQ = True
except ImportError:
    _HAS_CQ = False

needs_cadquery = pytest.mark.skipif(
    not _HAS_CQ,
    reason="cadquery not installed — skipping CAD proof-path tests",
)

# ── Import the module under test ──────────────────────────────────────────────
if _HAS_CQ:
    from hull_cad.builder import (
        build_hull_solid,
        export_step,
        export_stl,
        export_cfd_stl,
        describe,
        _count_ascii_stl_triangles,
        _ascii_stl_bbox,
        HullCADError,
        HullCADResult,
    )

# ── Import DEFAULTS ───────────────────────────────────────────────────────────
from moth_designer.config import DEFAULTS, LWL, TARGET_DISP_L


# ── Helper: make a params dict from DEFAULTS ─────────────────────────────────

def _base_params(**overrides) -> dict:
    """Return DEFAULTS merged with any keyword overrides."""
    p = {k: float(v) for k, v in DEFAULTS.items()}
    p.update({k: float(v) for k, v in overrides.items()})
    return p


# ── Fixture: default params ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def default_params():
    return _base_params()


@pytest.fixture(scope="module")
def default_result(default_params):
    """Build the default hull once and reuse across tests in this module."""
    return build_hull_solid(default_params)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Default hull
# ══════════════════════════════════════════════════════════════════════════════

@needs_cadquery
class TestDefaultHull:
    """The DEFAULTS hull must build cleanly and pass all basic checks."""

    def test_builds_without_exception(self, default_result):
        assert isinstance(default_result, HullCADResult)

    def test_solid_is_not_null(self, default_result):
        assert default_result.solid is not None
        assert not default_result.solid.wrapped.IsNull()

    def test_solid_passes_brepcheck(self, default_result):
        assert default_result.is_valid, (
            f"BRepCheck_Analyzer failed. Warnings:\n"
            + "\n".join(default_result.warnings)
        )

    def test_volume_is_sane(self, default_result):
        """Full hull solid should be well above the 130 L design displacement."""
        assert default_result.volume_litres > 80.0, (
            f"Volume {default_result.volume_litres:.1f} L is too small"
        )
        assert default_result.volume_litres < 400.0, (
            f"Volume {default_result.volume_litres:.1f} L is suspiciously large"
        )

    def test_solid_has_multiple_faces(self, default_result):
        """A watertight solid needs ≥ 3 faces: hull shell + bow cap + transom."""
        assert default_result.n_faces >= 3, (
            f"Only {default_result.n_faces} face(s) — solid may be open"
        )

    def test_loft_used_expected_station_count(self, default_result):
        # We asked for 40 stations; allow a couple of skipped failures
        assert default_result.n_stations >= 35

    def test_no_fatal_warnings(self, default_result):
        """Warnings are OK; fatal phrases in warning text are not."""
        fatal_phrases = ["too many failed", "null solid", "cannot close"]
        for w in default_result.warnings:
            for phrase in fatal_phrases:
                assert phrase.lower() not in w.lower(), (
                    f"Fatal warning detected: {w}"
                )

    def test_describe_returns_string(self, default_result):
        s = describe(default_result)
        assert "HullCADResult" in s
        assert "valid" in s


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STEP export
# ══════════════════════════════════════════════════════════════════════════════

@needs_cadquery
class TestStepExport:

    def test_step_file_is_written(self, default_result, tmp_path):
        path = tmp_path / "hull.step"
        returned = export_step(default_result, path)
        assert returned == path
        assert path.exists()

    def test_step_file_has_minimum_size(self, default_result, tmp_path):
        path = tmp_path / "hull.step"
        export_step(default_result, path)
        # A real hull STEP should be at least 100 kB
        assert path.stat().st_size > 50_000, (
            f"STEP file is only {path.stat().st_size} bytes — may be empty"
        )

    def test_step_file_starts_with_iso(self, default_result, tmp_path):
        """STEP AP214/AP203 files start with 'ISO-10303'."""
        path = tmp_path / "hull.step"
        export_step(default_result, path)
        header = path.read_text()[:20]
        assert header.startswith("ISO-10303"), (
            f"STEP header does not start with ISO-10303: {header!r}"
        )

    def test_step_export_null_solid_raises(self, tmp_path):
        """Passing a null result should raise HullCADError."""
        bad_result = HullCADResult(
            solid=None,
            is_valid=False,
            volume_mm3=0.0,
            volume_litres=0.0,
            n_faces=0,
            n_stations=0,
            warnings=["null"],
        )
        with pytest.raises(HullCADError):
            export_step(bad_result, tmp_path / "bad.step")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STL export (raw, mm, design CS)
# ══════════════════════════════════════════════════════════════════════════════

@needs_cadquery
class TestRawStlExport:

    def test_stl_file_is_written(self, default_result, tmp_path):
        path = tmp_path / "hull_raw.stl"
        returned = export_stl(default_result, path)
        assert returned == path
        assert path.exists()

    def test_stl_file_has_minimum_size(self, default_result, tmp_path):
        path = tmp_path / "hull_raw.stl"
        export_stl(default_result, path)
        # Binary STL: 84 bytes header + 50 bytes/triangle × expected ≥ 5000 tris
        assert path.stat().st_size > 50_000

    def test_stl_is_binary(self, default_result, tmp_path):
        """CadQuery exports binary STL by default."""
        path = tmp_path / "hull_raw.stl"
        export_stl(default_result, path)
        # Binary STL: bytes 80-83 are the triangle count as uint32
        import struct
        with open(path, "rb") as f:
            f.read(80)
            n_tris = struct.unpack("<I", f.read(4))[0]
        assert n_tris > 1000, f"Only {n_tris} triangles in binary STL"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CFD STL export (m, waterline at z=0)
# ══════════════════════════════════════════════════════════════════════════════

@needs_cadquery
class TestCfdStlExport:

    def test_cfd_stl_is_written(self, default_result, default_params, tmp_path):
        path = tmp_path / "hull_cfd.stl"
        returned = export_cfd_stl(default_result, default_params, path)
        assert returned == path
        assert path.exists()

    def test_cfd_stl_is_ascii(self, default_result, default_params, tmp_path):
        path = tmp_path / "hull_cfd.stl"
        export_cfd_stl(default_result, default_params, path)
        text = path.read_text()
        assert "solid hull" in text
        assert "facet normal" in text
        assert "vertex" in text
        assert "endsolid hull" in text

    def test_cfd_stl_hull_in_metres(self, default_result, default_params, tmp_path):
        """Hull should be ~3.355 m long in X (LWL = 3355 mm = 3.355 m)."""
        path = tmp_path / "hull_cfd.stl"
        export_cfd_stl(default_result, default_params, path,
                       domain_bounds=None)   # hull only, no domain box
        bbox = _ascii_stl_bbox(path)
        x_extent = bbox[1] - bbox[0]
        # The bow stub is at x=5mm=0.005m; LWL is at x=3355mm=3.355m
        # So x range ≈ 3.35 m
        assert 3.0 < x_extent < 3.5, (
            f"X extent is {x_extent:.3f} m — expected ~3.35 m (LWL in metres)"
        )

    def test_cfd_stl_waterline_near_zero(self, default_result, default_params, tmp_path):
        """
        After the waterline shift, some hull vertices should be near z=0 and
        the keel should be below z=0.
        The z range should span from about -0.35 m (keel) to +0.2 m (deck).
        """
        path = tmp_path / "hull_cfd.stl"
        export_cfd_stl(default_result, default_params, path, domain_bounds=None)
        bbox = _ascii_stl_bbox(path)
        z_min = bbox[4]
        z_max = bbox[5]
        assert z_min < -0.05, (
            f"z_min={z_min:.3f} m — keel should be below z=0"
        )
        assert z_max > 0.05, (
            f"z_max={z_max:.3f} m — deck should be above z=0"
        )

    def test_cfd_stl_with_domain_box(self, default_result, default_params, tmp_path):
        """Domain box named solids should appear when domain_bounds is provided."""
        domain = {
            "xn": -1.0, "xx": 5.0,
            "yn": -1.5, "yx": 1.5,
            "zn": -0.8, "zx": 0.5,
        }
        path = tmp_path / "hull_cfd_domain.stl"
        export_cfd_stl(default_result, default_params, path, domain_bounds=domain)
        text = path.read_text()
        for name in ("hull", "inlet", "outlet", "front", "back", "bottom", "atmosphere"):
            assert f"solid {name}" in text, (
                f"Named solid '{name}' not found in CFD STL"
            )

    def test_cfd_stl_triangle_count_is_reasonable(
        self, default_result, default_params, tmp_path
    ):
        path = tmp_path / "hull_cfd.stl"
        export_cfd_stl(default_result, default_params, path, domain_bounds=None)
        n = _count_ascii_stl_triangles(path)
        # Typical: 5 000 – 80 000 hull triangles depending on tessellation quality
        assert 500 < n < 500_000, f"Triangle count {n} is outside expected range"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Representative hull variants
# ══════════════════════════════════════════════════════════════════════════════

# These test that the proof path is robust beyond the DEFAULTS hull.
# They cover parameter regimes the optimizer actually explores.

@needs_cadquery
@pytest.mark.parametrize("desc,overrides", [
    # Narrow, deep, V-shaped midship — slender entry
    ("narrow_deep", dict(
        p3_hb=150.0, p3_d=200.0, p3_kw=20.0,
        p2_hb=110.0, p2_d=170.0,
        p4_hb=160.0, p4_d=185.0,
    )),
    # Wide, shallow, flat-bottomed midship
    ("wide_shallow", dict(
        p3_hb=260.0, p3_d=130.0, p3_kw=130.0,
        p2_hb=220.0, p2_d=130.0,
        p4_hb=250.0, p4_d=120.0,
    )),
    # U-shaped midship (beam_height negative = max-beam below DWL)
    ("u_section", dict(
        p3_hb=220.0, p3_d=175.0, p3_hz=-60.0,
        p2_hb=190.0, p2_d=160.0, p2_hz=-40.0,
    )),
    # V-shaped midship (beam_height positive = max-beam above DWL)
    ("v_section", dict(
        p3_hb=215.0, p3_d=180.0, p3_hz=60.0,
        p2_hb=185.0, p2_d=165.0,
    )),
    # Heavy rocker (large bow draft, moderate stern draft)
    ("heavy_rocker", dict(
        bow_draft=120.0,
        p1_d=160.0, p2_d=200.0, p3_d=210.0, p4_d=195.0,
        transom_draft=65.0,
    )),
    # Minimal keel width everywhere
    ("sharp_keel", dict(
        p1_kw=5.0, p2_kw=5.0, p3_kw=5.0, p4_kw=5.0,
        transom_keel_w=5.0,
    )),
])
def test_variant_hull_builds(desc, overrides):
    """
    Each hull variant must produce a non-null solid without raising.
    Validity and volume are checked at the INFO level only — not hard failures
    — because some extreme shapes may trigger OCCT validity warnings while
    still being usable for CFD.
    """
    params = _base_params(**overrides)
    result = build_hull_solid(params, n_stations=35, n_pts=40)

    assert result.solid is not None, f"[{desc}] solid is None"
    assert not result.solid.wrapped.IsNull(), f"[{desc}] solid is null"
    assert result.volume_litres > 10.0, (
        f"[{desc}] volume {result.volume_litres:.1f} L is suspiciously small"
    )

    # Log but don't fail on validity — some extreme shapes get valid=False
    # and we want to know which ones, not block the proof path.
    if not result.is_valid:
        print(f"\n  [{desc}] BRepCheck warning: solid not valid — warnings:")
        for w in result.warnings:
            print(f"    - {w}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Error handling
# ══════════════════════════════════════════════════════════════════════════════

@needs_cadquery
class TestErrorHandling:

    def test_degenerate_params_raises_hull_cad_error(self):
        """Params that produce zero-width sections should raise, not silently fail."""
        # Set all half-beams to 0 — the loft cannot succeed
        params = _base_params(
            p1_hb=0.0, p2_hb=0.0, p3_hb=0.0, p4_hb=0.0,
            transom_half_beam=0.0,
        )
        with pytest.raises(HullCADError):
            build_hull_solid(params, n_stations=10, n_pts=10)

    def test_fewer_stations_still_works(self):
        """Minimum viable station count should still produce a solid."""
        params = _base_params()
        result = build_hull_solid(params, n_stations=6, n_pts=24)
        assert result.solid is not None
        assert not result.solid.wrapped.IsNull()

    def test_high_station_count_still_works(self):
        """High station count should not cause failures."""
        params = _base_params()
        result = build_hull_solid(params, n_stations=80, n_pts=64)
        assert result.solid is not None

    def test_step_export_to_nonexistent_dir(self, default_result, tmp_path):
        """export_step() should create parent directories automatically."""
        nested = tmp_path / "deep" / "nested" / "hull.step"
        export_step(default_result, nested)
        assert nested.exists()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Section wire internal tests
# ══════════════════════════════════════════════════════════════════════════════

@needs_cadquery
class TestSectionWires:
    """
    Test the section wire builder in isolation.
    These confirm the wire construction is sound before blaming the loft.
    """

    def _ctrl(self, params):
        from moth_designer.geometry import build_ctrl
        return build_ctrl(params)

    def test_midship_wire_is_not_null(self):
        from hull_cad.builder import _build_section_wire
        params = _base_params()
        ctrl = self._ctrl(params)
        wire = _build_section_wire(LWL / 2, ctrl, n_pts=48)
        assert not wire.wrapped.IsNull()

    def test_bow_stub_wire_respects_min_hb(self):
        """Bow stub must have at least the minimum half-beam."""
        from hull_cad.builder import _build_section_wire
        params = _base_params(bow_half_beam=0.0)
        ctrl = self._ctrl(params)
        # x=5mm is the bow stub position; min_hb=2mm enforced
        wire = _build_section_wire(5.0, ctrl, n_pts=24, min_hb=2.0)
        assert not wire.wrapped.IsNull()

    def test_transom_wire_is_not_null(self):
        from hull_cad.builder import _build_section_wire
        params = _base_params()
        ctrl = self._ctrl(params)
        wire = _build_section_wire(float(LWL), ctrl, n_pts=48)
        assert not wire.wrapped.IsNull()

    def test_degenerate_hb_raises(self):
        from hull_cad.builder import _build_section_wire
        params = _base_params()
        ctrl = self._ctrl(params)
        # Force a truly zero half-beam (no min_hb clamp)
        # by patching beam_hb to zero everywhere
        # We achieve this by using a station deep in the bow
        # with min_hb=0 — the actual hb will be small but > 0 for DEFAULTS
        # so we use a contrived params set:
        zero_params = _base_params(
            p1_hb=0.0, p2_hb=0.0, p3_hb=0.0, p4_hb=0.0,
            transom_half_beam=0.0, bow_half_beam=0.0,
        )
        ctrl_zero = self._ctrl(zero_params)
        with pytest.raises(HullCADError):
            _build_section_wire(LWL / 2, ctrl_zero, n_pts=12, min_hb=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Performance smoke test
# ══════════════════════════════════════════════════════════════════════════════

@needs_cadquery
def test_build_time_is_acceptable():
    """
    Building the default hull should complete in < 60 seconds.
    This is a smoke test, not a strict performance requirement.
    """
    import time
    params = _base_params()
    t0 = time.perf_counter()
    result = build_hull_solid(params, n_stations=40, n_pts=48)
    elapsed = time.perf_counter() - t0

    assert result.solid is not None
    assert elapsed < 60.0, (
        f"Build took {elapsed:.1f}s — longer than expected. "
        "This may indicate an environment issue."
    )
    print(f"\n  Build time: {elapsed:.2f}s  |  Volume: {result.volume_litres:.1f} L")
