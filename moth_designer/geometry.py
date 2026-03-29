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


def cross_section(hb, depth, deck_hb=None, deck_z=None, keel_w=0.0, n=40, hb_z=0.0):
    """Quarter-ellipse bilge (with optional flat keel) + straight topsides to deck edge.
    hb_z: z-height of the max-beam point (0 = design waterline, negative = submerged)."""
    hb     = max(float(hb),    0.0)
    depth  = max(float(depth), 0.0)
    keel_w = min(max(float(keel_w), 0.0), hb)
    hb_z   = float(hb_z)
    if deck_hb is None: deck_hb = hb
    if deck_z  is None: deck_z  = float(FREEBOARD)
    deck_hb = max(float(deck_hb), 0.0)
    deck_z  = max(float(deck_z),  hb_z)

    ell_hb = max(hb - keel_w, 0.001)
    t      = np.linspace(0, np.pi / 2, n)
    y_bilge = keel_w + ell_hb * np.sin(t)
    # z goes from -depth (keel) to hb_z (max beam point)
    z_bilge = hb_z + (-depth - hb_z) * np.cos(t)

    if keel_w > 0:
        n_keel = max(n // 8, 3)
        y_sub = np.concatenate([np.linspace(0.0, keel_w, n_keel), y_bilge[1:]])
        z_sub = np.concatenate([np.full(n_keel, -depth),           z_bilge[1:]])
    else:
        y_sub, z_sub = y_bilge, z_bilge

    n_top = max(n // 5, 3)
    y_top = np.linspace(hb,  deck_hb, n_top)
    z_top = np.linspace(hb_z, deck_z,  n_top)
    return (np.concatenate([y_sub, y_top[1:]]),
            np.concatenate([z_sub, z_top[1:]]))


def cross_section_spline(hb, depth, deck_hb=None, deck_z=None, keel_w=0.0, n=80, hb_z=0.0):
    """
    Parametric cubic spline cross-section through the three structural points:
      keel (0, -depth)  ->  half-beam (hb, hb_z)  ->  deck edge (deck_hb, deck_z)

    hb_z: z-height of the max-beam point (0 = design waterline, negative = submerged).
    A flat keel of width keel_w is prepended when keel_w > 0.
    Enforces a vertical tangent at the keel centreline (dy/dt = 0 at t=0)
    so the curve meets the symmetry plane smoothly.
    """
    from scipy.interpolate import CubicSpline

    hb     = max(float(hb),    0.0)
    depth  = max(float(depth), 0.0)
    keel_w = min(max(float(keel_w), 0.0), hb)
    hb_z   = float(hb_z)
    if deck_hb is None: deck_hb = hb
    if deck_z  is None: deck_z  = float(FREEBOARD)
    deck_hb = max(float(deck_hb), 0.0)
    deck_z  = max(float(deck_z),  hb_z)

    # Degenerate (bow/zero-beam) — return a vertical centreline stroke
    if hb < 0.5:
        return np.array([0.0, 0.0]), np.array([-depth, deck_z])

    # Structural knot points
    if keel_w > 0:
        key_y = np.array([0.0, keel_w, hb, deck_hb])
        key_z = np.array([-depth, -depth, hb_z, deck_z])
    else:
        key_y = np.array([0.0, hb, deck_hb])
        key_z = np.array([-depth, hb_z, deck_z])

    # Chord-length parameterisation
    pts   = np.column_stack([key_y, key_z])
    chords = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1))
    t      = np.concatenate([[0.0], np.cumsum(chords)])
    t     /= t[-1]

    # Spline for y: enforce dy/dt = 0 at the keel (vertical tangent → bilateral symmetry)
    cs_y = CubicSpline(t, key_y, bc_type=((1, 0.0), 'not-a-knot'))
    cs_z = CubicSpline(t, key_z, bc_type='natural')

    # Dense sampling — give the flat keel its own segment
    if keel_w > 0:
        t_flat = t[1]
        n_flat = max(n // 8, 3)
        t_bilge = np.linspace(t_flat, 1.0, n - n_flat + 1)
        t_dense = np.concatenate([np.linspace(0.0, t_flat, n_flat),
                                  t_bilge[1:]])
    else:
        t_dense = np.linspace(0.0, 1.0, n)

    ys = cs_y(t_dense)
    zs = cs_z(t_dense)

    # Hard-clamp the flat keel so it stays truly flat
    if keel_w > 0:
        flat = t_dense <= t[1] + 1e-9
        ys[flat] = np.linspace(0.0, keel_w, flat.sum())
        zs[flat] = -depth

    return np.clip(ys, 0.0, None), zs


def build_ctrl(p):
    """Build PCHIP control arrays from the parameter dict."""
    t_hb    = p['transom_half_beam']
    t_draft = p['transom_draft']
    b_draft = p['bow_draft']
    raw_x  = np.array([p[f'p{i}_x']  for i in range(1, 5)])
    raw_hb = np.array([p[f'p{i}_hb'] for i in range(1, 5)])
    raw_d  = np.array([p[f'p{i}_d']  for i in range(1, 5)])
    raw_dw = np.array([p[f'p{i}_dw'] for i in range(1, 5)])
    raw_dz = np.array([p[f'p{i}_dz'] for i in range(1, 5)])
    raw_kw = np.array([p[f'p{i}_kw'] for i in range(1, 5)])
    raw_hz = np.array([p.get(f'p{i}_hz', 0.0) for i in range(1, 5)])
    order  = np.argsort(raw_x)
    sx  = raw_x[order];  shb = raw_hb[order];  sd  = raw_d[order]
    sdw = raw_dw[order]; sdz = raw_dz[order];  skw = raw_kw[order]
    shz = raw_hz[order]

    ctrl_x  = np.concatenate([[0.0],            sx, [float(LWL)]])
    beam_hb = np.concatenate([[0.0],            shb, [float(t_hb)]])
    keel_d  = np.concatenate([[float(b_draft)], sd,  [float(t_draft)]])
    deck_hb = np.concatenate([[0.0],            sdw, [float(t_hb)]])
    sheer_z = np.concatenate([[p['bow_sheer']], sdz, [p['transom_sheer']]])
    keel_w  = np.concatenate([[0.0],            skw, [float(p['transom_keel_w'])]])
    hb_z    = np.concatenate([[0.0],            shz, [0.0]])

    return ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, hb_z, order, sx, shb


def beam_eval(x_eval, beam_x, beam_hb):
    x_eval = np.asarray(x_eval, dtype=float)
    return lagrange(x_eval, beam_x, beam_hb, clip_min=0.0)


# ─── Displacement hydrostatics ────────────────────────────────

def _section_area_half_below(ys, zs, z_wl):
    """Shoelace area of the half cross-section polygon below z_wl (mm²).
    Works for any z_wl — below keel, between keel and sheer, or above sheer."""
    z_wl = float(z_wl)
    if z_wl <= float(zs[0]):
        return 0.0

    if z_wl >= float(zs[-1]):
        poly_y = list(ys) + [0.0]
        poly_z = list(zs) + [float(zs[-1])]
    else:
        poly_y, poly_z = [], []
        for i in range(len(zs)):
            if float(zs[i]) <= z_wl:
                poly_y.append(float(ys[i]))
                poly_z.append(float(zs[i]))
            else:
                dz = float(zs[i]) - float(zs[i - 1]) + 1e-12
                f  = (z_wl - float(zs[i - 1])) / dz
                poly_y.append(float(ys[i - 1]) + f * (float(ys[i]) - float(ys[i - 1])))
                poly_z.append(z_wl)
                break
        poly_y.append(0.0)
        poly_z.append(z_wl)

    py = np.array(poly_y + [poly_y[0]])
    pz = np.array(poly_z + [poly_z[0]])
    return 0.5 * abs(float(np.sum(py[:-1] * pz[1:] - py[1:] * pz[:-1])))


def section_area_full(hb, depth, deck_hb, deck_z, keel_w, z_wl, n=60):
    """Full cross-section area (both sides) below z_wl (mm²).
    Includes the topside region above z=0 up to the deck edge."""
    ys, zs = cross_section(hb, depth, deck_hb, deck_z, keel_w, n)
    return 2.0 * _section_area_half_below(ys, zs, z_wl)


def displaced_volume(z_wl, beam_x, beam_hb, keel_x, keel_d,
                     deck_x=None, deck_hb_arr=None, sheer_z_arr=None,
                     keel_w_arr=None, n_x=200):
    """Volume (litres) enclosed by the hull below z_wl.
    Pass deck_x/deck_hb_arr/sheer_z_arr/keel_w_arr to include topsides above z=0."""
    x_arr  = np.linspace(0.0, float(LWL), n_x)
    hbs    = beam_eval(x_arr, beam_x, beam_hb)
    depths = lagrange(x_arr, keel_x, keel_d,
                      clip_min=0.0, clip_max=float(MAX_DEPTH))

    if deck_x is not None:
        dhbs  = lagrange(x_arr, deck_x, deck_hb_arr,  clip_min=0.0)
        sheer = lagrange(x_arr, deck_x, sheer_z_arr,  clip_min=0.0)
        kw    = lagrange(x_arr, deck_x, keel_w_arr,   clip_min=0.0)
        areas = np.array([section_area_full(hbs[k], depths[k],
                                            dhbs[k], sheer[k], kw[k], z_wl)
                          for k in range(n_x)])
    else:
        # Legacy fast path: ellipse-only, below z=0
        from scipy.interpolate import PchipInterpolator  # noqa (already imported via lagrange)
        def _legacy(hb, depth, z):
            if hb <= 0 or depth <= 0 or z <= -depth:
                return 0.0
            if z >= 0:
                return np.pi / 2 * hb * depth
            cos_t0 = np.clip(-z / depth, 0.0, 1.0)
            t0 = np.arccos(cos_t0)
            return 2.0 * hb * depth * (t0 / 2.0 - np.sin(2.0 * t0) / 4.0)
        areas = np.array([_legacy(hbs[k], depths[k], z_wl) for k in range(n_x)])

    return float(np.trapezoid(areas, x_arr)) / 1e6


def find_disp_waterline(target_l, beam_x, beam_hb, keel_x, keel_d,
                        deck_x=None, deck_hb_arr=None, sheer_z_arr=None,
                        keel_w_arr=None):
    """Bisect to find z_wl where displaced_volume == target_l.
    When deck params are supplied the search extends up to the maximum sheer height."""
    kw = dict(deck_x=deck_x, deck_hb_arr=deck_hb_arr,
              sheer_z_arr=sheer_z_arr, keel_w_arr=keel_w_arr)

    lo = -float(MAX_DEPTH)
    hi = float(np.max(sheer_z_arr)) if sheer_z_arr is not None else 0.0

    vol_hi = displaced_volume(hi, beam_x, beam_hb, keel_x, keel_d, **kw)
    if vol_hi <= target_l:
        return hi           # target exceeds total hull volume
    vol_lo = displaced_volume(lo, beam_x, beam_hb, keel_x, keel_d, **kw)
    if vol_lo >= target_l:
        return lo

    for _ in range(42):
        mid = (lo + hi) / 2.0
        if displaced_volume(mid, beam_x, beam_hb, keel_x, keel_d, **kw) < target_l:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ─── Hydrostatic coefficients ─────────────────────────────────

def hydrostatic_coefficients(target_l, beam_x, beam_hb, keel_x, keel_d,
                             deck_x=None, deck_hb_arr=None, sheer_z_arr=None,
                             keel_w_arr=None, hb_z_arr=None, n_x=120):
    """Compute standard hull form coefficients at the waterline for target_l litres.

    Returns a dict:
      Cb   – Block coefficient       = V / (Lwl * Bwl * T)
      Cp   – Prismatic coefficient   = V / (Am * Lwl)
      Cm   – Midship coefficient     = Am / (Bm * Tm)
      Cw   – Waterplane coefficient  = Aw / (Lwl * Bwl)
      z_wl – waterline z (mm)
      T    – max draft to waterline (mm)
      B_wl – max waterline beam (mm)
      A_m  – midship section area (mm²)
      A_wp – waterplane area (mm²)
      L_wl – LWL (mm)
      V    – displacement (litres)
    """
    deck_kw = dict(deck_x=deck_x, deck_hb_arr=deck_hb_arr,
                   sheer_z_arr=sheer_z_arr, keel_w_arr=keel_w_arr)

    z_wl = find_disp_waterline(target_l, beam_x, beam_hb, keel_x, keel_d, **deck_kw)

    x_arr  = np.linspace(0.0, float(LWL), n_x)
    depths = lagrange(x_arr, keel_x, keel_d, clip_min=0.0, clip_max=float(MAX_DEPTH))
    hbs    = beam_eval(x_arr, beam_x, beam_hb)
    dhbs   = lagrange(x_arr, deck_x, deck_hb_arr,  clip_min=0.0) if deck_x  is not None else hbs.copy()
    sheers = lagrange(x_arr, deck_x, sheer_z_arr,  clip_min=0.0) if deck_x  is not None else np.full(n_x, float(FREEBOARD))
    kws    = lagrange(x_arr, deck_x, keel_w_arr,   clip_min=0.0) if deck_x  is not None else np.zeros(n_x)
    hzs    = lagrange(x_arr, deck_x, hb_z_arr)                    if hb_z_arr is not None else np.zeros(n_x)

    # Waterline half-beam and section area at each station
    wl_hb   = np.zeros(n_x)
    sec_area = np.zeros(n_x)
    for k in range(n_x):
        ys, zs = cross_section(hbs[k], depths[k], dhbs[k], sheers[k], kws[k], 60, hb_z=hzs[k])
        sec_area[k] = 2.0 * _section_area_half_below(ys, zs, z_wl)
        for j in range(len(zs) - 1):
            if (zs[j] - z_wl) * (zs[j + 1] - z_wl) <= 0:
                f = (z_wl - zs[j]) / (zs[j + 1] - zs[j] + 1e-12)
                wl_hb[k] = float(ys[j] + f * (ys[j + 1] - ys[j]))
                break

    T    = z_wl + float(np.max(depths))
    B_wl = 2.0 * float(np.max(wl_hb))

    # Waterplane area: integrate waterline half-beam along length
    A_wp = float(np.trapezoid(2.0 * wl_hb, x_arr))

    # Midship section: station closest to LWL/2
    mid_idx = int(n_x // 2)
    A_m  = float(sec_area[mid_idx])
    Bm   = 2.0 * float(wl_hb[mid_idx])
    Tm   = z_wl + float(depths[mid_idx])

    V_mm3 = target_l * 1e6
    Lwl   = float(LWL)

    Cb = V_mm3 / (Lwl * B_wl * T)        if (B_wl > 0 and T > 0)       else float('nan')
    Cp = V_mm3 / (A_m  * Lwl)            if A_m > 0                     else float('nan')
    Cm = A_m   / (Bm   * Tm)             if (Bm > 0 and Tm > 0)         else float('nan')
    Cw = A_wp  / (Lwl  * B_wl)           if (Lwl > 0 and B_wl > 0)      else float('nan')

    return dict(Cb=Cb, Cp=Cp, Cm=Cm, Cw=Cw,
                z_wl=z_wl, T=T, B_wl=B_wl, A_m=A_m, A_wp=A_wp,
                L_wl=Lwl, V=target_l)


# Keep old name as alias for backwards compatibility
def block_coefficient(*args, **kwargs):
    return hydrostatic_coefficients(*args, **kwargs)


# ─── 3-D mesh ─────────────────────────────────────────────────

def build_3d_mesh(params, N_X=80, N_T=36):
    """Return (verts float32 (V,3), faces int32 (F,3)) for GLMeshItem."""
    ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, hb_z_arr, _, _, _ = build_ctrl(params)

    xs    = np.linspace(0.0, float(LWL), N_X)
    t_new = np.linspace(0.0, 1.0, N_T)

    vs = np.zeros((N_X, N_T, 3), dtype=np.float32)
    vp = np.zeros((N_X, N_T, 3), dtype=np.float32)

    for i, xi in enumerate(xs):
        hb    = float(beam_eval([xi], ctrl_x, beam_hb)[0])
        if i == 0:
            hb = max(hb, 0.5)   # 1 mm flat bow face — prevents degenerate tip in STL/CFD mesh
        depth = float(lagrange([xi], ctrl_x, keel_d,
                               clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
        dhb   = float(lagrange([xi], ctrl_x, deck_hb,  clip_min=0.0)[0])
        dz    = float(lagrange([xi], ctrl_x, sheer_z,  clip_min=0.0)[0])
        kw    = float(lagrange([xi], ctrl_x, keel_w,   clip_min=0.0)[0])
        hz    = float(lagrange([xi], ctrl_x, hb_z_arr)[0])
        ys, zs = cross_section(hb, depth, dhb, dz, kw, 40, hb_z=hz)
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

    # Bow cap — 1 mm flat face, normal pointing in -x (forward)
    bv_s = vs[0]
    bv_p = vp[0]
    bow_verts = np.vstack([bv_s, bv_p]).astype(np.float32)
    B = S + 2 * N_T          # index offset: after stern_verts block
    fb = []
    for j in range(N_T - 1):
        fb += [[B + j + 1, B + j,     B + N_T + j],
               [B + j + 1, B + N_T + j, B + N_T + j + 1]]

    verts = np.vstack([vs_flat, vp_flat, stern_verts, bow_verts])
    faces = np.array(fs + fp + fd + ft + fb, dtype=np.int32)
    return verts, faces
