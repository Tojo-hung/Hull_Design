"""
Generate a corrected hull STEP/STL with:
  1. Flat deck — constant freeboard (200 mm) across all sections
  2. 1 mm bow stub — same 3-edge spline topology as all other sections

Run from repo root:
    python cad_output/generate_flat_deck.py
"""
import os
import sys
from pathlib import Path

_REPO = Path(os.getcwd())
sys.path.insert(0, str(_REPO))

import numpy as np
import cadquery as cq
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE

from moth_designer.config import DEFAULTS, LWL, FREEBOARD
from moth_designer.geometry import (
    build_ctrl, cross_section_spline, beam_eval, lagrange,
)

OUT_DIR = _REPO / "cad_output"
OUT_DIR.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────

def _solid_volume_mm3(solid):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid.wrapped, props)
    return float(props.Mass())

def _count_faces(solid):
    explorer = TopExp_Explorer(solid.wrapped, TopAbs_FACE)
    n = 0
    while explorer.More():
        n += 1
        explorer.Next()
    return n


# ── Parameters: flat deck ────────────────────────────────────────────
DECK_Z = 200.0  # mm — constant freeboard everywhere

params = {k: float(v) for k, v in DEFAULTS.items()}
params["bow_sheer"] = DECK_Z
params["transom_sheer"] = DECK_Z
params["bow_half_beam"] = 1.0   # 1 mm half-beam at bow (must keep)
params["bow_draft"] = 25.0      # raise keel at bow
params["p1_hb"] = 110.0         # widen first station (was 90)
params["p1_hz"] = -40.0         # lower max-beam point below DWL at p1
for i in range(1, 5):
    params[f"p{i}_dz"] = DECK_Z

# ── Station layout ───────────────────────────────────────────────────
BOW_STUB_X = 10.0          # mm from stem
BOW_KW     = 0.5           # mm keel width at bow stub
N_PTS      = 40            # points per half-section

ctrl = build_ctrl(params)

print("Building section wires...", flush=True)


def build_section_wire(xi, ctrl_arrays, n_pts, kw_override=None):
    """Build a closed 3-edge wire for a hull section at x = xi.

    All wires have identical topology: starboard spline + deck line + port spline.
    This is critical for clean OCCT lofting.

    kw_override : if set, forces the keel width to this value (mm).
    """
    (ctrl_x, beam_hb, keel_d, deck_hb,
     sheer_z, keel_w, hb_z_arr, _order,
     _sx, _shb, sheer_ctrl_x, sheer_z_ext) = ctrl_arrays

    hb    = float(beam_eval([xi], ctrl_x, beam_hb)[0])
    depth = float(lagrange([xi], ctrl_x, keel_d,
                           clip_min=0.0, clip_max=350.0)[0])
    dhb   = float(lagrange([xi], ctrl_x, deck_hb, clip_min=0.0)[0])
    dhb   = max(dhb, hb)
    dz    = float(lagrange([xi], sheer_ctrl_x, sheer_z_ext,
                           clip_min=0.0)[0])
    if kw_override is not None:
        kw = kw_override
    else:
        kw = float(lagrange([xi], ctrl_x, keel_w, clip_min=0.0)[0])
    kw    = min(kw, hb)  # keel width cannot exceed half-beam
    hz    = float(lagrange([xi], ctrl_x, hb_z_arr)[0])

    ys_half, zs_half = cross_section_spline(
        hb, depth, dhb, dz, kw, n_pts, hb_z=hz
    )

    stbd_pts = [cq.Vector(float(xi), float(y), float(z))
                for y, z in zip(ys_half, zs_half)]
    port_pts = [cq.Vector(float(xi), -float(y), float(z))
                for y, z in zip(ys_half[::-1], zs_half[::-1])]

    stbd_edge = cq.Edge.makeSpline(stbd_pts)
    deck_edge = cq.Edge.makeLine(stbd_pts[-1], port_pts[0])
    port_edge = cq.Edge.makeSpline(port_pts)

    return cq.Wire.assembleEdges([stbd_edge, deck_edge, port_edge])


# ── Build station x-positions ────────────────────────────────────────
# Cluster stations near the bow for a smooth geometric transition,
# then spread them evenly through the rest of the hull.
bow_cluster = np.array([BOW_STUB_X, 25, 45, 70, 100, 150, 220, 300])
main_body = np.linspace(400, float(LWL), 22)
all_xs = np.concatenate([bow_cluster, main_body])

wires = []
for xi in all_xs:
    # Force keel width at the bow stub station
    kw_ovr = BOW_KW if xi <= BOW_STUB_X + 1.0 else None
    try:
        wire = build_section_wire(xi, ctrl, N_PTS, kw_override=kw_ovr)
        wires.append(wire)
        if xi <= 300:
            hb = float(beam_eval([xi], ctrl[0], ctrl[1])[0])
            print(f"  x={xi:6.0f} mm  hb={hb:6.1f} mm  "
                  f"(width={2*hb:.1f} mm)", flush=True)
    except Exception as e:
        print(f"  SKIP x={xi:.0f}: {e}", flush=True)

print(f"  ... + {len(main_body)} body stations to x={LWL} mm", flush=True)
print(f"  Total wires: {len(wires)}", flush=True)

# ── Loft ────────────────────────────────────────────────────────────
print("Lofting...", flush=True)
solid = cq.Solid.makeLoft(wires, ruled=True)

analyzer = BRepCheck_Analyzer(solid.wrapped)
is_valid = analyzer.IsValid()
vol_mm3 = _solid_volume_mm3(solid)
vol_l = vol_mm3 / 1e6
n_faces = _count_faces(solid)

print(f"  Valid:    {is_valid}", flush=True)
print(f"  Volume:  {vol_l:.1f} L", flush=True)
print(f"  Faces:   {n_faces}", flush=True)

# ── Export STEP ─────────────────────────────────────────────────────
step_path = OUT_DIR / "moth_hull_flat_deck.step"
cq.exporters.export(solid, str(step_path))
print(f"  STEP: {step_path.name}  "
      f"({step_path.stat().st_size // 1024} KB)", flush=True)

# ── Export STL ──────────────────────────────────────────────────────
stl_path = OUT_DIR / "moth_hull_flat_deck.stl"
cq.exporters.export(solid, str(stl_path),
                     tolerance=0.3, angularTolerance=0.2)
print(f"  STL:  {stl_path.name}  "
      f"({stl_path.stat().st_size // 1024} KB)", flush=True)

print("\nDone! Open in Fusion 360 / FreeCAD / any STEP viewer.", flush=True)
