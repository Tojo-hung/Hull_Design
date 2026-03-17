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
