# ─────────────────────────────────────────────────────────────
# exports.py  —  STL, STEP, and XYZ point-table export
# No GUI dependencies.
# ─────────────────────────────────────────────────────────────

import numpy as np
from .config  import LWL, MAX_DEPTH, FREEBOARD
from .geometry import build_ctrl, build_3d_mesh, beam_eval, lagrange, cross_section


def export_txt(filepath, params, n_stations=20, n_t=32):
    """Export hull cross-section points as X Y Z for SolidWorks.

    Origin: centre of transom face at mid-height of transom.
      X  positive forward (bow direction),  0 = transom
      Y  positive starboard,               0 = centreline
      Z  positive up,                      0 = transom mid-height

    Each station is a closed loop of X Y Z points separated by a blank line.
    In SolidWorks: Insert > Curve > Curve Through XYZ Points.
    """
    ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, _, _, _ = build_ctrl(params)
    t_draft = params['transom_draft']

    x_orig = float(LWL)
    z_orig = (float(FREEBOARD) - t_draft) / 2.0

    xs = np.linspace(0.0, float(LWL), n_stations)

    lines = []
    for xi in xs:
        hb    = float(beam_eval([xi], ctrl_x, beam_hb)[0])
        depth = float(lagrange([xi], ctrl_x, keel_d,
                               clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
        dhb   = float(lagrange([xi], ctrl_x, deck_hb, clip_min=0.0)[0])
        dz_   = float(lagrange([xi], ctrl_x, sheer_z, clip_min=0.0)[0])
        kw    = float(lagrange([xi], ctrl_x, keel_w,  clip_min=0.0)[0])
        ys, zs = cross_section(hb, depth, dhb, dz_, kw, n_t)

        x_sw = x_orig - xi
        for y, z in zip(ys, zs):
            lines.append(f'{x_sw:.3f}\t{y:.3f}\t{z - z_orig:.3f}')
        lines.append('')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def export_stl(filepath, params):
    """Write a binary STL file of the hull mesh (units: mm).
    SolidWorks: File > Open, type = STL, set units to mm on import.
    """
    verts, faces = build_3d_mesh(params, N_X=120, N_T=48)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1.0
    normals /= nlen

    n_tri = len(faces)
    with open(filepath, 'wb') as f:
        f.write(b'Moth Hull - exported from moth_designer.py' + b' ' * 38)
        f.write(np.uint32(n_tri).tobytes())
        for i in range(n_tri):
            f.write(normals[i].tobytes())
            f.write(verts[faces[i, 0]].tobytes())
            f.write(verts[faces[i, 1]].tobytes())
            f.write(verts[faces[i, 2]].tobytes())
            f.write(b'\x00\x00')


def export_step(filepath, params, n_stations=40, n_t=24):
    """Loft a solid hull through cross-section wires and write a STEP file.

    Requires cadquery:  conda install -c cadquery cadquery
    """
    import cadquery as cq

    ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, _, _, _ = build_ctrl(params)

    xs = np.linspace(5.0, float(LWL), n_stations)

    wires = []
    for xi in xs:
        hb    = max(float(beam_eval([xi], ctrl_x, beam_hb)[0]), 1.0)
        depth = max(float(lagrange([xi], ctrl_x, keel_d,
                                   clip_min=0.0,
                                   clip_max=float(MAX_DEPTH))[0]), 1.0)
        dhb   = max(float(lagrange([xi], ctrl_x, deck_hb, clip_min=0.0)[0]), 1.0)
        dz_   = max(float(lagrange([xi], ctrl_x, sheer_z, clip_min=0.0)[0]), 1.0)
        kw    = float(lagrange([xi], ctrl_x, keel_w,  clip_min=0.0)[0])

        ys_half, zs_half = cross_section(hb, depth, dhb, dz_, kw, n_t)
        ys_port = -ys_half[::-1]
        zs_port =  zs_half[::-1]
        y_all = np.concatenate([ys_half, ys_port[1:-1]])
        z_all = np.concatenate([zs_half, zs_port[1:-1]])

        pts  = [cq.Vector(float(xi), float(y), float(z))
                for y, z in zip(y_all, z_all)]
        edge = cq.Edge.makeSpline(pts, periodic=True)
        wires.append(cq.Wire.assembleEdges([edge]))

    solid = cq.Solid.makeLoft(wires, ruled=False)
    cq.exporters.export(solid, filepath)


def export_dxf_sections(filepath_or_folder, params, n_pts=60,
                        single_file=False):
    """
    Export cross-section curves as DXF for Onshape sketch import.

    Two modes:
      single_file=False (default)
        Writes one DXF per station into filepath_or_folder/.
        Filename encodes the X offset from the bow so you know which sketch
        plane to offset in Onshape, e.g. station_05_x1677mm.dxf.

      single_file=True
        Writes all sections into one DXF (filepath_or_folder is the file path).
        Each section is drawn at its true X position — useful as a reference
        drawing but not suitable for direct Onshape sketch import.

    DXF coordinate system (matches Onshape sketch plane)
    -----------------------------------------------------
      X (DXF) = Y (hull) = half-breadth, starboard positive  [mm]
      Y (DXF) = Z (hull) = vertical, 0 = design waterline    [mm]

    Each section curve runs port-deck-edge → keel → stbd-deck-edge (open
    polyline) — the shape expected by Onshape's Loft tool.

    Onshape workflow (single_file=False)
    -------------------------------------
      For each DXF file station_NN_xAAAAmm.dxf:
        1. Sketch > offset the Right plane by AAAA mm from the bow face.
        2. Inside the sketch: Insert > DXF/DWG, pick the file.
        3. Confirm the DXF origin aligns with the DWL / centreline corner.
        4. Repeat for all stations, then use Loft across all sketches.
    """
    import ezdxf
    import os

    ctrl_x, beam_hb, keel_d, deck_hb_c, sheer_z_c, keel_w_c, _, _, _ = build_ctrl(params)

    xs = np.array(sorted(set(
        [0.0] + [float(params[f'p{i}_x']) for i in range(1, 6)] + [float(LWL)]
    )))

    def _section_curve(xi):
        """Return full (port→keel→stbd) section as list of (y, z) tuples."""
        hb    = float(beam_eval([xi], ctrl_x, beam_hb)[0])
        depth = float(lagrange([xi], ctrl_x, keel_d,
                               clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
        dhb   = float(lagrange([xi], ctrl_x, deck_hb_c, clip_min=0.0)[0])
        dz    = float(lagrange([xi], ctrl_x, sheer_z_c, clip_min=0.0)[0])
        kw    = float(lagrange([xi], ctrl_x, keel_w_c,  clip_min=0.0)[0])
        ys, zs = cross_section(hb, depth, dhb, dz, kw, n_pts)
        # port side: mirror + reverse
        y_port = (-ys)[::-1];  z_port = zs[::-1]
        y_full = np.concatenate([y_port, ys[1:]])
        z_full = np.concatenate([z_port, zs[1:]])
        return list(zip(y_full.tolist(), z_full.tolist()))

    def _add_section_to_msp(msp, pts, x_offset=0.0):
        shifted = [(x_offset + y, z) for y, z in pts]
        msp.add_lwpolyline(shifted, dxfattribs={'layer': 'SECTION', 'color': 1})

    def _add_dwl_cl(msp, pts, x_offset=0.0):
        max_y = max(abs(p[0]) for p in pts)
        msp.add_line(
            (x_offset - max_y * 1.05, 0.0),
            (x_offset + max_y * 1.05, 0.0),
            dxfattribs={'layer': 'DWL', 'color': 5, 'linetype': 'DASHED'},
        )
        msp.add_line(
            (x_offset, min(p[1] for p in pts) - 5),
            (x_offset, max(p[1] for p in pts) + 5),
            dxfattribs={'layer': 'CL', 'color': 3},
        )

    if single_file:
        doc = ezdxf.new('R2010');  doc.units = 4
        doc.linetypes.new('DASHED', dxfattribs={'description': 'Dashed', 'pattern': [0.6, 0.3, -0.3]})
        msp = doc.modelspace()
        for xi in xs:
            pts = _section_curve(xi)
            _add_section_to_msp(msp, pts, x_offset=xi)
        doc.saveas(filepath_or_folder)
        print(f'Exported all-sections DXF -> {filepath_or_folder}')
    else:
        os.makedirs(filepath_or_folder, exist_ok=True)
        for i, xi in enumerate(xs):
            pts = _section_curve(xi)
            doc = ezdxf.new('R2010');  doc.units = 4
            doc.linetypes.new('DASHED', dxfattribs={'description': 'Dashed', 'pattern': [0.6, 0.3, -0.3]})
            msp = doc.modelspace()
            _add_section_to_msp(msp, pts)
            _add_dwl_cl(msp, pts)
            x_mm   = int(round(xi))
            fname  = os.path.join(filepath_or_folder,
                                  f'station_{i:02d}_x{x_mm:04d}mm.dxf')
            doc.saveas(fname)
        print(f'Exported {len(xs)} section DXFs -> {filepath_or_folder}/')
        print('  Filename X value = offset from bow for Onshape sketch plane.')


def export_dxf_lines_plan(filepath, params, n_wl=6, n_pts=80):
    """
    Export the complete lines plan (body plan + profile + half-breadth) as a
    single reference DXF.  All three views are arranged side-by-side in model
    space with a gap between them.  Units: mm.

    This file is for visual reference; use export_dxf_sections() for the
    per-station files that Onshape can import into individual sketches.

    Layers
    ------
      SECTION    cross-section curves (body plan)
      KEEL       keel / rocker line (profile)
      SHEER      sheer / deck edge (profile)
      WATERLINE  constant-Z waterplane curves (half-breadth)
      DWL        design waterline reference
      CL         centreline / axis reference
      STATION    station tick lines on profile
    """
    import ezdxf

    ctrl_x, beam_hb, keel_d, deck_hb_c, sheer_z_c, keel_w_c, _, _, _ = build_ctrl(params)

    lwl = float(LWL)
    gap = lwl * 0.12

    # X origins for the three panels
    # Body plan centred at 0  (Y = half-breadth, Z = height)
    body_ox = 0.0
    # Profile to the right
    prof_ox = float(np.max(beam_eval(np.linspace(0, lwl, 80), ctrl_x, beam_hb))) + gap
    # Half-breadth to the right of profile
    plan_ox = prof_ox + lwl + gap

    doc = ezdxf.new('R2010');  doc.units = 4
    doc.linetypes.new('DASHED', dxfattribs={'description': 'Dashed', 'pattern': [0.6, 0.3, -0.3]})
    msp = doc.modelspace()

    xs = np.array(sorted(set(
        [0.0] + [float(params[f'p{i}_x']) for i in range(1, 6)] + [lwl]
    )))

    # ── Body plan ──────────────────────────────────────────────
    for xi in xs:
        hb    = float(beam_eval([xi], ctrl_x, beam_hb)[0])
        depth = float(lagrange([xi], ctrl_x, keel_d,
                               clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
        dhb   = float(lagrange([xi], ctrl_x, deck_hb_c, clip_min=0.0)[0])
        dz    = float(lagrange([xi], ctrl_x, sheer_z_c, clip_min=0.0)[0])
        kw    = float(lagrange([xi], ctrl_x, keel_w_c,  clip_min=0.0)[0])
        ys, zs = cross_section(hb, depth, dhb, dz, kw, n_pts)
        y_p = (-ys)[::-1];  z_p = zs[::-1]
        y_f = np.concatenate([y_p, ys[1:]])
        z_f = np.concatenate([z_p, zs[1:]])
        pts = [(body_ox + float(y), float(z)) for y, z in zip(y_f, z_f)]
        msp.add_lwpolyline(pts, dxfattribs={'layer': 'SECTION', 'color': 1})

    max_hb = float(np.max(beam_eval(np.linspace(0, lwl, 80), ctrl_x, beam_hb)))
    msp.add_line((body_ox - max_hb * 1.05, 0), (body_ox + max_hb * 1.05, 0),
                 dxfattribs={'layer': 'DWL', 'color': 5, 'linetype': 'DASHED'})
    msp.add_line((body_ox, -float(MAX_DEPTH) * 1.05), (body_ox, float(FREEBOARD) * 1.05),
                 dxfattribs={'layer': 'CL', 'color': 3})

    # ── Profile ────────────────────────────────────────────────
    x_dense = np.linspace(0.0, lwl, 300)
    keel_z  = -lagrange(x_dense, ctrl_x, keel_d,
                        clip_min=0.0, clip_max=float(MAX_DEPTH))
    sheer_z = lagrange(x_dense, ctrl_x, sheer_z_c, clip_min=0.0)

    msp.add_lwpolyline([(prof_ox + x, z) for x, z in zip(x_dense, keel_z)],
                       dxfattribs={'layer': 'KEEL', 'color': 2})
    msp.add_lwpolyline([(prof_ox + x, z) for x, z in zip(x_dense, sheer_z)],
                       dxfattribs={'layer': 'SHEER', 'color': 6})
    msp.add_line((prof_ox, 0), (prof_ox + lwl, 0),
                 dxfattribs={'layer': 'DWL', 'color': 5, 'linetype': 'DASHED'})

    # Transom line
    t_hb    = params['transom_half_beam']
    t_draft = params['transom_draft']
    t_sheer = params['transom_sheer']
    msp.add_line((prof_ox + lwl, -t_draft), (prof_ox + lwl, t_sheer),
                 dxfattribs={'layer': 'SHEER', 'color': 6})

    # Station ticks on profile
    for xi in xs:
        kz = float(-lagrange([xi], ctrl_x, keel_d,
                             clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
        sz = float(lagrange([xi], ctrl_x, sheer_z_c, clip_min=0.0)[0])
        msp.add_line((prof_ox + xi, kz), (prof_ox + xi, sz),
                     dxfattribs={'layer': 'STATION', 'color': 8})

    # ── Half-breadth plan ──────────────────────────────────────
    z_levels = np.linspace(-float(MAX_DEPTH) * 0.9, -float(MAX_DEPTH) * 0.05, n_wl)
    x_sample = np.linspace(0.0, lwl, 120)

    for z_lev in z_levels:
        stbd = []
        for xi in x_sample:
            hb    = float(beam_eval([xi], ctrl_x, beam_hb)[0])
            depth = float(lagrange([xi], ctrl_x, keel_d,
                                   clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
            dhb   = float(lagrange([xi], ctrl_x, deck_hb_c, clip_min=0.0)[0])
            dz_   = float(lagrange([xi], ctrl_x, sheer_z_c, clip_min=0.0)[0])
            kw    = float(lagrange([xi], ctrl_x, keel_w_c,  clip_min=0.0)[0])
            ys, zs = cross_section(hb, depth, dhb, dz_, kw, 50)
            for k in range(len(zs) - 1):
                if (zs[k] - z_lev) * (zs[k + 1] - z_lev) <= 0:
                    f = (z_lev - zs[k]) / (zs[k + 1] - zs[k] + 1e-12)
                    stbd.append((xi, float(ys[k] + f * (ys[k + 1] - ys[k]))))
                    break
        if not stbd:
            continue
        color = 4 if abs(z_lev) < 1.0 else 1
        msp.add_lwpolyline([(plan_ox + x,  y) for x, y in stbd],
                           dxfattribs={'layer': 'WATERLINE', 'color': color})
        msp.add_lwpolyline([(plan_ox + x, -y) for x, y in reversed(stbd)],
                           dxfattribs={'layer': 'WATERLINE', 'color': color})

    # DWL waterline in half-breadth plan
    msp.add_line((plan_ox, -max_hb * 1.05), (plan_ox, max_hb * 1.05),
                 dxfattribs={'layer': 'CL', 'color': 3})

    doc.saveas(filepath)
    print(f'Exported lines plan DXF -> {filepath}')
    print('  Layers: SECTION  KEEL  SHEER  WATERLINE  DWL  CL  STATION')
