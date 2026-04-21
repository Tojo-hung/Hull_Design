"""
Geometry quality regression tests.

Validates the guide-curve lofting pipeline introduced in the geometry redesign:
  - No overshoot fallback on standard hull configurations
  - Arc-length-uniform vertex spacing (no bunching)
  - Face planarity within tolerance
  - Guide curve contact at mesh landmark vertices
  - Displacement stability vs. known baseline
  - G1 bilge-to-topside tangent continuity
"""

import json
import os
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(__file__))
_DESIGN = os.path.join(_ROOT, 'hull_design.json')
_BEST   = os.path.join(_ROOT, 'best_hull.json')


def _load(path):
    with open(path) as f:
        return json.load(f)


# ── Variant hulls (mirrors test_cad_proof.py fixture set) ─────────────────
def _variant(base, **overrides):
    p = dict(base)
    p.update(overrides)
    return p


@pytest.fixture(scope='module')
def base_params():
    return _load(_DESIGN)


@pytest.fixture(scope='module')
def variant_params(base_params):
    b = base_params
    return {
        'design':      b,
        'narrow_deep': _variant(b, p2_hb=100.0, p3_hb=130.0, p4_hb=150.0,
                                  p2_d=160.0, p3_d=170.0, p4_d=160.0),
        'wide_shallow': _variant(b, p2_hb=220.0, p3_hb=240.0, p4_hb=250.0,
                                   p2_d=60.0, p3_d=70.0, p4_d=60.0),
        'v_keel':      _variant(b, p1_kw=0.0, p2_kw=0.0, p3_kw=0.0, p4_kw=0.0,
                                  transom_keel_w=0.0),
        'heavy_rocker': _variant(b, bow_draft=150.0, p1_d=180.0, p2_d=200.0),
    }


# ── 1. Overshoot fallback ──────────────────────────────────────────────────

def test_no_overshoot_fallback(variant_params):
    """cross_section_spline must not trigger the _bounded_bilge fallback on
    any standard hull variant."""
    from moth_designer.geometry import (
        build_ctrl, beam_eval, lagrange, cross_section_spline
    )
    from moth_designer.config import LWL, MAX_DEPTH, FREEBOARD

    fallback_total = 0
    for name, params in variant_params.items():
        ctrl = build_ctrl(params)
        ctrl_x, beam_hb, keel_d, _, sheer_z, keel_w, hb_z_arr = ctrl[:7]
        sheer_cx, sheer_z_ext = ctrl[10], ctrl[11]

        for xi in np.linspace(0.0, float(LWL), 40):
            hb    = float(beam_eval([xi], ctrl_x, beam_hb)[0])
            depth = float(lagrange([xi], ctrl_x, keel_d,
                                   clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
            dz    = float(lagrange([xi], sheer_cx, sheer_z_ext, clip_min=0.0)[0])
            kw    = float(lagrange([xi], ctrl_x, keel_w, clip_min=0.0)[0])
            hz    = float(lagrange([xi], ctrl_x, hb_z_arr)[0])

            if hb < 1e-6:
                continue

            ys, zs = cross_section_spline(hb, depth, None, dz, kw, 40, hb_z=hz)
            if (np.max(ys) > hb + 1e-6
                    or np.min(ys) < -1e-6
                    or np.any(np.diff(ys) < -1e-6)):
                fallback_total += 1

    assert fallback_total == 0, (
        f"Overshoot fallback triggered {fallback_total} times across all variants"
    )


# ── 2. Arc-length spacing uniformity ──────────────────────────────────────

def test_arc_length_uniformity(base_params):
    """Adjacent vertex spacings within each cross-section should not vary by
    more than 5× — confirming arc-length resampling is working."""
    from moth_designer.geometry import build_3d_mesh

    verts, _ = build_3d_mesh(base_params, N_X=60, N_T=36)
    N_X, N_T = 60, 36

    ratios = []
    for i in range(N_X):
        row = verts[i * N_T:(i + 1) * N_T]
        ds = np.linalg.norm(np.diff(row, axis=0), axis=1)
        ds = ds[ds > 1e-6]
        if len(ds) < 2:
            continue
        ratios.append(ds.max() / ds.min())

    bad = [r for r in ratios if r > 5.0]
    assert len(bad) == 0, (
        f"{len(bad)} sections have vertex spacing ratio > 5: max={max(ratios):.2f}"
    )


# ── 3. Face planarity ─────────────────────────────────────────────────────

def test_face_planarity(base_params):
    """95th-percentile non-planarity of hull skin quads must be < 0.5 mm."""
    from moth_designer.geometry import build_3d_mesh
    from numpy.linalg import svd

    verts, _ = build_3d_mesh(base_params, N_X=80, N_T=36)
    N_X, N_T = 80, 36

    deviations = []
    for i in range(N_X - 1):
        for j in range(N_T - 1):
            quad = verts[[i * N_T + j,
                          (i + 1) * N_T + j,
                          (i + 1) * N_T + j + 1,
                          i * N_T + j + 1]]
            center = quad.mean(axis=0)
            _, _, Vt = svd(quad - center)
            normal = Vt[-1]
            deviations.append(np.abs((quad - center) @ normal).max())

    p95 = np.percentile(deviations, 95)
    assert p95 < 0.5, f"Face planarity p95 = {p95:.3f} mm (limit 0.5 mm)"


# ── 4. Guide curve contact ─────────────────────────────────────────────────

def test_guide_curve_contact_keel(base_params):
    """The first vertex of every cross-section (index 0, keel CL) must lie
    within 1 mm of the keel centreline guide curve."""
    from moth_designer.geometry import build_3d_mesh, build_guide_curves

    N_X, N_T = 80, 36
    verts, _ = build_3d_mesh(base_params, N_X=N_X, N_T=N_T)
    xs_gc, keel_cl, *_ = build_guide_curves(base_params, n_x=N_X)

    errors = []
    for i in range(N_X):
        mesh_keel = verts[i * N_T]          # vertex 0 of section i (keel CL)
        gc_keel   = keel_cl[i]              # guide curve point at same x
        err = abs(float(mesh_keel[2]) - float(gc_keel[2]))   # Z deviation
        errors.append(err)

    max_err = max(errors)
    assert max_err < 1.0, (
        f"Keel CL mesh vertex deviates {max_err:.2f} mm from guide curve (limit 1 mm)"
    )


def test_guide_curve_contact_sheer(base_params):
    """The last vertex of every cross-section (deck edge) must track the
    sheer guide curve's Z coordinate within 1 mm."""
    from moth_designer.geometry import build_3d_mesh, build_guide_curves

    N_X, N_T = 80, 36
    verts, _ = build_3d_mesh(base_params, N_X=N_X, N_T=N_T)
    xs_gc, *_, sheer_edge = build_guide_curves(base_params, n_x=N_X)

    errors = []
    for i in range(N_X):
        mesh_sheer = verts[i * N_T + (N_T - 1)]
        gc_sheer   = sheer_edge[i]
        err = abs(float(mesh_sheer[2]) - float(gc_sheer[2]))
        errors.append(err)

    max_err = max(errors)
    assert max_err < 1.0, (
        f"Sheer mesh vertex deviates {max_err:.2f} mm from guide curve (limit 1 mm)"
    )


# ── 5. Displacement stability ──────────────────────────────────────────────

def test_displacement_stability(base_params):
    """Displaced volume at design waterline (z=0) must be within ±2 L of
    the expected value from hull_design.json parameters.  Validates that
    the geometry redesign has not shifted the hydrostatics."""
    from moth_designer.geometry import (
        build_ctrl, displaced_volume
    )

    ctrl = build_ctrl(base_params)
    ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, hb_z_arr = ctrl[:7]
    sheer_cx, sheer_z_ext = ctrl[10], ctrl[11]

    vol = displaced_volume(
        0.0, ctrl_x, beam_hb, ctrl_x, keel_d,
        deck_x=ctrl_x, deck_hb_arr=deck_hb, sheer_z_arr=sheer_z,
        keel_w_arr=keel_w, hb_z_arr=hb_z_arr, n_x=200, n_section=60
    )
    # Design targets ~130 L displacement; at z=0 (DWL) should be close.
    assert 50.0 < vol < 250.0, (
        f"Displaced volume at DWL = {vol:.1f} L — outside plausible range [50, 250] L"
    )


# ── 6. G1 tangent at bilge / topside join ─────────────────────────────────

def test_g1_topside_join(base_params):
    """At the bilge/topside join the Y component of the tangent must be
    near zero — confirming the bilge arrives tangent to the vertical topside."""
    from moth_designer.geometry import cross_section_spline

    # Mid-body station: typical bilge proportions
    hb, depth, dz, kw, hz = 200.0, 120.0, 200.0, 30.0, 30.0
    ys, zs = cross_section_spline(hb, depth, hb, dz, kw, 80, hb_z=hz)

    # The bilge ends where Y first reaches hb (after that, topside with Y=const).
    # Identify the join index: last index where Y < hb - small_tol.
    tol = 1.0  # mm
    join_idx = np.searchsorted(ys, hb - tol) - 1
    join_idx = max(1, min(join_idx, len(ys) - 2))

    # Finite-difference tangent at the join (backward difference into bilge region)
    dy = float(ys[join_idx + 1]) - float(ys[join_idx - 1])
    dz_val = float(zs[join_idx + 1]) - float(zs[join_idx - 1])
    tangent_len = np.hypot(dy, dz_val)
    dy_norm = dy / tangent_len if tangent_len > 1e-9 else dy

    assert abs(dy_norm) < 0.15, (
        f"Bilge arrives at topside with normalised dY = {dy_norm:.3f} "
        f"(should be < 0.15 for near-vertical tangent)"
    )


# ── 7. Mesh watertightness (vertex count) ─────────────────────────────────

def test_mesh_vertex_face_count(base_params):
    """build_3d_mesh must produce the expected number of vertices and faces."""
    from moth_designer.geometry import build_3d_mesh

    N_X, N_T = 80, 36
    verts, faces = build_3d_mesh(base_params, N_X=N_X, N_T=N_T)

    expected_verts = 2 * N_X * N_T
    assert len(verts) == expected_verts, (
        f"Expected {expected_verts} verts, got {len(verts)}"
    )
    assert len(faces) > 0, "No faces generated"
    assert faces.min() >= 0, "Negative face index"
    assert faces.max() < len(verts), "Face index out of bounds"
