# ─────────────────────────────────────────────────────────────
# geometry.py  —  Pure-numpy hull geometry and hydrostatics
# No GUI dependencies — can be imported or run standalone.
# ─────────────────────────────────────────────────────────────

import numpy as np
from .config import LWL, MAX_DEPTH, FREEBOARD, TARGET_DISP_L


def lagrange(x_eval, ctrl_x, ctrl_y, clip_min=None, clip_max=None):
    """PCHIP interpolation (named lagrange for historic compatibility).
    Monotone-preserving, locally-influenced — no Runge oscillations."""
    from scipy.interpolate import PchipInterpolator
    x_eval = np.asarray(x_eval, dtype=float)
    cx = np.asarray(ctrl_x, dtype=float).copy()
    cy = np.asarray(ctrl_y, dtype=float)
    for i in range(1, len(cx)):
        if cx[i] <= cx[i - 1]:
            cx[i] = cx[i - 1] + 1e-3
    result = PchipInterpolator(cx, cy)(x_eval)
    if clip_min is not None:
        result = np.maximum(result, clip_min)
    if clip_max is not None:
        result = np.minimum(result, clip_max)
    return result


def cross_section(hb, depth, deck_hb=None, deck_z=None, keel_w=0.0, n=40):
    """Quarter-ellipse bilge (with optional flat keel) + straight topsides to deck edge."""
    hb     = max(float(hb),    0.0)
    depth  = max(float(depth), 0.0)
    keel_w = min(max(float(keel_w), 0.0), hb)
    if deck_hb is None: deck_hb = hb
    if deck_z  is None: deck_z  = float(FREEBOARD)
    deck_hb = max(float(deck_hb), 0.0)
    deck_z  = max(float(deck_z),  0.0)

    ell_hb = max(hb - keel_w, 0.001)
    t      = np.linspace(0, np.pi / 2, n)
    y_bilge = keel_w + ell_hb * np.sin(t)
    z_bilge = -depth * np.cos(t)

    if keel_w > 0:
        n_keel = max(n // 8, 3)
        y_sub = np.concatenate([np.linspace(0.0, keel_w, n_keel), y_bilge[1:]])
        z_sub = np.concatenate([np.full(n_keel, -depth),           z_bilge[1:]])
    else:
        y_sub, z_sub = y_bilge, z_bilge

    n_top = max(n // 5, 3)
    y_top = np.linspace(hb,     deck_hb, n_top)
    z_top = np.linspace(0.0,    deck_z,  n_top)
    return (np.concatenate([y_sub, y_top[1:]]),
            np.concatenate([z_sub, z_top[1:]]))


def build_ctrl(p):
    """Build PCHIP control arrays from the parameter dict."""
    t_hb    = p['transom_half_beam']
    t_draft = p['transom_draft']
    b_draft = p['bow_draft']
    raw_x  = np.array([p[f'p{i}_x']  for i in range(1, 6)])
    raw_hb = np.array([p[f'p{i}_hb'] for i in range(1, 6)])
    raw_d  = np.array([p[f'p{i}_d']  for i in range(1, 6)])
    raw_dw = np.array([p[f'p{i}_dw'] for i in range(1, 6)])
    raw_dz = np.array([p[f'p{i}_dz'] for i in range(1, 6)])
    raw_kw = np.array([p[f'p{i}_kw'] for i in range(1, 6)])
    order  = np.argsort(raw_x)
    sx  = raw_x[order];  shb = raw_hb[order];  sd  = raw_d[order]
    sdw = raw_dw[order]; sdz = raw_dz[order];  skw = raw_kw[order]

    ctrl_x  = np.concatenate([[0.0],            sx, [float(LWL)]])
    beam_hb = np.concatenate([[0.0],            shb, [float(t_hb)]])
    keel_d  = np.concatenate([[float(b_draft)], sd,  [float(t_draft)]])
    deck_hb = np.concatenate([[0.0],            sdw, [float(t_hb)]])
    sheer_z = np.concatenate([[p['bow_sheer']], sdz, [p['transom_sheer']]])
    keel_w  = np.concatenate([[0.0],            skw, [float(p['transom_keel_w'])]])

    return ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, order, sx, shb


def beam_eval(x_eval, beam_x, beam_hb):
    x_eval = np.asarray(x_eval, dtype=float)
    return lagrange(x_eval, beam_x, beam_hb, clip_min=0.0)


# ─── Displacement hydrostatics ────────────────────────────────

def _section_area(hb, depth, z_wl):
    """Submerged cross-section area (both sides) below z_wl."""
    if hb <= 0 or depth <= 0:
        return 0.0
    if z_wl <= -depth:
        return 0.0
    if z_wl >= 0:
        return np.pi / 2 * hb * depth
    cos_t0 = np.clip(-z_wl / depth, 0.0, 1.0)
    t0 = np.arccos(cos_t0)
    return 2.0 * hb * depth * (t0 / 2.0 - np.sin(2.0 * t0) / 4.0)


def displaced_volume(z_wl, beam_x, beam_hb, keel_x, keel_d, n_x=200):
    """Volume (litres) displaced below waterline z_wl (z_wl <= 0)."""
    x_arr = np.linspace(0.0, float(LWL), n_x)
    hbs    = beam_eval(x_arr, beam_x, beam_hb)
    depths = lagrange(x_arr, keel_x, keel_d,
                      clip_min=0.0, clip_max=float(MAX_DEPTH))
    areas = np.array([_section_area(hbs[k], depths[k], z_wl)
                      for k in range(n_x)])
    return float(np.trapezoid(areas, x_arr)) / 1e6


def find_disp_waterline(target_l, beam_x, beam_hb, keel_x, keel_d):
    """Bisect to find z_wl (mm, <=0) where displaced_volume == target_l."""
    lo, hi = -float(MAX_DEPTH), 0.0
    vol_hi = displaced_volume(hi, beam_x, beam_hb, keel_x, keel_d)
    if vol_hi <= target_l:
        return hi
    vol_lo = displaced_volume(lo, beam_x, beam_hb, keel_x, keel_d)
    if vol_lo >= target_l:
        return lo
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if displaced_volume(mid, beam_x, beam_hb, keel_x, keel_d) < target_l:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ─── 3-D mesh ─────────────────────────────────────────────────

def build_3d_mesh(params, N_X=80, N_T=36):
    """Return (verts float32 (V,3), faces int32 (F,3)) for GLMeshItem."""
    ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, _, _, _ = build_ctrl(params)

    xs    = np.linspace(0.0, float(LWL), N_X)
    t_new = np.linspace(0.0, 1.0, N_T)

    vs = np.zeros((N_X, N_T, 3), dtype=np.float32)
    vp = np.zeros((N_X, N_T, 3), dtype=np.float32)

    for i, xi in enumerate(xs):
        hb    = float(beam_eval([xi], ctrl_x, beam_hb)[0])
        depth = float(lagrange([xi], ctrl_x, keel_d,
                               clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
        dhb   = float(lagrange([xi], ctrl_x, deck_hb,  clip_min=0.0)[0])
        dz    = float(lagrange([xi], ctrl_x, sheer_z,  clip_min=0.0)[0])
        kw    = float(lagrange([xi], ctrl_x, keel_w,   clip_min=0.0)[0])
        ys, zs = cross_section(hb, depth, dhb, dz, kw, 40)
        t_src  = np.linspace(0.0, 1.0, len(ys))
        yr = np.interp(t_new, t_src, ys).astype(np.float32)
        zr = np.interp(t_new, t_src, zs).astype(np.float32)
        vs[i] = np.column_stack([np.full(N_T, xi, np.float32), yr,  zr])
        vp[i] = np.column_stack([np.full(N_T, xi, np.float32), -yr, zr])

    vs_flat = vs.reshape(-1, 3)
    vp_flat = vp.reshape(-1, 3)
    OFF     = N_X * N_T

    def quad_strip(row_i, row_j, offset_a=0, flip=False):
        tris = []
        for j in range(N_T - 1):
            a = offset_a + row_i * N_T + j
            b = offset_a + row_j * N_T + j
            c = offset_a + row_j * N_T + j + 1
            d = offset_a + row_i * N_T + j + 1
            if not flip:
                tris += [[a, b, c], [a, c, d]]
            else:
                tris += [[a, c, b], [a, d, c]]
        return tris

    fs, fp = [], []
    for i in range(N_X - 1):
        fs += quad_strip(i, i + 1, offset_a=0,   flip=False)
        fp += quad_strip(i, i + 1, offset_a=OFF, flip=True)

    fd = []
    for i in range(N_X - 1):
        a  = i       * N_T + (N_T - 1)
        b  = (i + 1) * N_T + (N_T - 1)
        ap = OFF + i       * N_T + (N_T - 1)
        bp = OFF + (i + 1) * N_T + (N_T - 1)
        fd += [[a, b, bp], [a, bp, ap]]

    sv = vs[-1]
    pv = vp[-1]
    stern_verts = np.vstack([sv, pv]).astype(np.float32)
    S = 2 * OFF
    ft = []
    for j in range(N_T - 1):
        ft += [[S + j, S + j + 1, S + N_T + j + 1],
               [S + j, S + N_T + j + 1, S + N_T + j]]

    verts = np.vstack([vs_flat, vp_flat, stern_verts])
    faces = np.array(fs + fp + fd + ft, dtype=np.int32)
    return verts, faces
