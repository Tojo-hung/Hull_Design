# geometry.py — standalone hull geometry (no PyQt / pyqtgraph)
import numpy as np
from config import LWL, MAX_DEPTH, FREEBOARD, TARGET_DISP_L


def lagrange(x_eval, ctrl_x, ctrl_y, clip_min=None, clip_max=None):
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
    hb     = max(float(hb),    0.0)
    depth  = max(float(depth), 0.0)
    keel_w = min(max(float(keel_w), 0.0), hb)
    hb_z   = float(hb_z)
    if deck_hb is None: deck_hb = hb
    if deck_z  is None: deck_z  = float(FREEBOARD)
    deck_hb = max(float(deck_hb), 0.0)
    deck_z  = max(float(deck_z),  hb_z)

    ell_hb  = max(hb - keel_w, 0.001)
    t       = np.linspace(0, np.pi / 2, n)
    y_bilge = keel_w + ell_hb * np.sin(t)
    z_bilge = hb_z + (-depth - hb_z) * np.cos(t)

    if keel_w > 0:
        n_keel = max(n // 8, 3)
        y_sub  = np.concatenate([np.linspace(0.0, keel_w, n_keel), y_bilge[1:]])
        z_sub  = np.concatenate([np.full(n_keel, -depth),           z_bilge[1:]])
    else:
        y_sub, z_sub = y_bilge, z_bilge

    n_top = max(n // 5, 3)
    y_top = np.linspace(hb,   deck_hb, n_top)
    z_top = np.linspace(hb_z, deck_z,  n_top)
    return (np.concatenate([y_sub, y_top[1:]]),
            np.concatenate([z_sub, z_top[1:]]))


def build_ctrl(p):
    t_hb    = p['transom_half_beam']
    t_draft = p['transom_draft']
    b_draft = p['bow_draft']
    raw_x  = np.array([p[f'p{i}_x']  for i in range(1, 6)])
    raw_hb = np.array([p[f'p{i}_hb'] for i in range(1, 6)])
    raw_d  = np.array([p[f'p{i}_d']  for i in range(1, 6)])
    raw_dw = np.array([p[f'p{i}_dw'] for i in range(1, 6)])
    raw_dz = np.array([p[f'p{i}_dz'] for i in range(1, 6)])
    raw_kw = np.array([p[f'p{i}_kw'] for i in range(1, 6)])
    raw_hz = np.array([p.get(f'p{i}_hz', 0.0) for i in range(1, 6)])
    order  = np.argsort(raw_x)
    sx  = raw_x[order];  shb = raw_hb[order];  sd  = raw_d[order]
    sdw = raw_dw[order]; sdz = raw_dz[order];  skw = raw_kw[order]; shz = raw_hz[order]

    ctrl_x  = np.concatenate([[0.0],            sx, [float(LWL)]])
    beam_hb = np.concatenate([[0.0],            shb, [float(t_hb)]])
    keel_d  = np.concatenate([[float(b_draft)], sd,  [float(t_draft)]])
    deck_hb = np.concatenate([[0.0],            sdw, [float(t_hb)]])
    sheer_z = np.concatenate([[p['bow_sheer']], sdz, [p['transom_sheer']]])
    keel_w  = np.concatenate([[0.0],            skw, [float(p['transom_keel_w'])]])
    hb_z    = np.concatenate([[0.0],            shz, [0.0]])
    return ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, hb_z, order, sx, shb


def beam_eval(x_eval, beam_x, beam_hb):
    return lagrange(np.asarray(x_eval, dtype=float), beam_x, beam_hb, clip_min=0.0)


def _section_area_half_below(ys, zs, z_wl):
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


def section_area_full(hb, depth, deck_hb, deck_z, keel_w, z_wl, hb_z=0.0, n=60):
    ys, zs = cross_section(hb, depth, deck_hb, deck_z, keel_w, n, hb_z=hb_z)
    return 2.0 * _section_area_half_below(ys, zs, z_wl)


def displaced_volume(z_wl, beam_x, beam_hb, keel_x, keel_d,
                     deck_x=None, deck_hb_arr=None, sheer_z_arr=None,
                     keel_w_arr=None, hb_z_arr=None, n_x=200):
    x_arr  = np.linspace(0.0, float(LWL), n_x)
    hbs    = beam_eval(x_arr, beam_x, beam_hb)
    depths = lagrange(x_arr, keel_x, keel_d, clip_min=0.0, clip_max=float(MAX_DEPTH))
    if deck_x is not None:
        dhbs  = lagrange(x_arr, deck_x, deck_hb_arr,  clip_min=0.0)
        sheer = lagrange(x_arr, deck_x, sheer_z_arr,  clip_min=0.0)
        kw    = lagrange(x_arr, deck_x, keel_w_arr,   clip_min=0.0)
        hzs   = lagrange(x_arr, deck_x, hb_z_arr) if hb_z_arr is not None else np.zeros(n_x)
        areas = np.array([section_area_full(hbs[k], depths[k], dhbs[k], sheer[k],
                                            kw[k], z_wl, hb_z=hzs[k])
                          for k in range(n_x)])
    else:
        areas = np.array([section_area_full(hbs[k], depths[k], hbs[k],
                                            float(FREEBOARD), 0.0, z_wl)
                          for k in range(n_x)])
    return float(np.trapezoid(areas, x_arr)) / 1e6


def find_disp_waterline(target_l, beam_x, beam_hb, keel_x, keel_d,
                        deck_x=None, deck_hb_arr=None, sheer_z_arr=None,
                        keel_w_arr=None, hb_z_arr=None):
    kw = dict(deck_x=deck_x, deck_hb_arr=deck_hb_arr,
              sheer_z_arr=sheer_z_arr, keel_w_arr=keel_w_arr, hb_z_arr=hb_z_arr)
    lo = -float(MAX_DEPTH)
    hi = float(np.max(sheer_z_arr)) if sheer_z_arr is not None else 0.0
    if displaced_volume(hi, beam_x, beam_hb, keel_x, keel_d, **kw) <= target_l:
        return hi
    if displaced_volume(lo, beam_x, beam_hb, keel_x, keel_d, **kw) >= target_l:
        return lo
    for _ in range(42):
        mid = (lo + hi) / 2.0
        if displaced_volume(mid, beam_x, beam_hb, keel_x, keel_d, **kw) < target_l:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def hydrostatic_coefficients(target_l, beam_x, beam_hb, keel_x, keel_d,
                             deck_x=None, deck_hb_arr=None, sheer_z_arr=None,
                             keel_w_arr=None, hb_z_arr=None, n_x=120):
    deck_kw = dict(deck_x=deck_x, deck_hb_arr=deck_hb_arr,
                   sheer_z_arr=sheer_z_arr, keel_w_arr=keel_w_arr, hb_z_arr=hb_z_arr)
    z_wl = find_disp_waterline(target_l, beam_x, beam_hb, keel_x, keel_d, **deck_kw)

    x_arr  = np.linspace(0.0, float(LWL), n_x)
    depths = lagrange(x_arr, keel_x, keel_d, clip_min=0.0, clip_max=float(MAX_DEPTH))
    hbs    = beam_eval(x_arr, beam_x, beam_hb)
    dhbs   = lagrange(x_arr, deck_x, deck_hb_arr,  clip_min=0.0) if deck_x  is not None else hbs.copy()
    sheers = lagrange(x_arr, deck_x, sheer_z_arr,  clip_min=0.0) if deck_x  is not None else np.full(n_x, float(FREEBOARD))
    kws    = lagrange(x_arr, deck_x, keel_w_arr,   clip_min=0.0) if deck_x  is not None else np.zeros(n_x)
    hzs    = lagrange(x_arr, deck_x, hb_z_arr)                    if hb_z_arr is not None else np.zeros(n_x)

    wl_hb    = np.zeros(n_x)
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
    A_wp = float(np.trapezoid(2.0 * wl_hb, x_arr))
    mid  = int(n_x // 2)
    A_m  = float(sec_area[mid])
    Bm   = 2.0 * float(wl_hb[mid])
    Tm   = z_wl + float(depths[mid])
    V_mm3 = target_l * 1e6
    Lwl   = float(LWL)

    Cb = V_mm3 / (Lwl * B_wl * T) if (B_wl > 0 and T > 0) else float('nan')
    Cp = V_mm3 / (A_m  * Lwl)     if A_m > 0               else float('nan')
    Cm = A_m   / (Bm   * Tm)      if (Bm > 0 and Tm > 0)   else float('nan')
    Cw = A_wp  / (Lwl  * B_wl)    if (Lwl > 0 and B_wl > 0) else float('nan')

    return dict(Cb=Cb, Cp=Cp, Cm=Cm, Cw=Cw,
                z_wl=z_wl, T=T, B_wl=B_wl, A_m=A_m, A_wp=A_wp,
                L_wl=Lwl, V=target_l)


def export_stl_bytes(params, N_X=120, N_T=48):
    """Build the hull mesh and return it as a binary STL byte string."""
    import struct, io
    verts, faces = build_3d_mesh(params, N_X=N_X, N_T=N_T)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1.0
    normals /= nlen

    buf = io.BytesIO()
    buf.write(b'Moth Hull - moth_designer web app' + b' ' * 47)  # 80-byte header
    buf.write(struct.pack('<I', len(faces)))
    for i in range(len(faces)):
        buf.write(normals[i].tobytes())
        buf.write(verts[faces[i, 0]].tobytes())
        buf.write(verts[faces[i, 1]].tobytes())
        buf.write(verts[faces[i, 2]].tobytes())
        buf.write(b'\x00\x00')
    return buf.getvalue()


def build_3d_mesh(params, N_X=80, N_T=36):
    """Return (verts float32 (V,3), faces int32 (F,3)) for Plotly Mesh3d."""
    ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, hb_z_arr, _, _, _ = build_ctrl(params)

    xs    = np.linspace(0.0, float(LWL), N_X)
    t_new = np.linspace(0.0, 1.0, N_T)

    vs = np.zeros((N_X, N_T, 3), dtype=np.float32)
    vp = np.zeros((N_X, N_T, 3), dtype=np.float32)

    for i, xi in enumerate(xs):
        hb_   = float(beam_eval([xi], ctrl_x, beam_hb)[0])
        depth = float(lagrange([xi], ctrl_x, keel_d, clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
        dhb   = float(lagrange([xi], ctrl_x, deck_hb,     clip_min=0.0)[0])
        dz    = float(lagrange([xi], ctrl_x, sheer_z,     clip_min=0.0)[0])
        kw    = float(lagrange([xi], ctrl_x, keel_w,      clip_min=0.0)[0])
        hz    = float(lagrange([xi], ctrl_x, hb_z_arr)[0])
        ys, zs = cross_section(hb_, depth, dhb, dz, kw, 40, hb_z=hz)
        t_src  = np.linspace(0.0, 1.0, len(ys))
        yr = np.interp(t_new, t_src, ys).astype(np.float32)
        zr = np.interp(t_new, t_src, zs).astype(np.float32)
        vs[i] = np.column_stack([np.full(N_T, xi, np.float32),  yr, zr])
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

    sv = vs[-1]; pv = vp[-1]
    stern_verts = np.vstack([sv, pv]).astype(np.float32)
    S = 2 * OFF
    ft = []
    for j in range(N_T - 1):
        ft += [[S + j, S + j + 1, S + N_T + j + 1],
               [S + j, S + N_T + j + 1, S + N_T + j]]

    verts = np.vstack([vs_flat, vp_flat, stern_verts])
    faces = np.array(fs + fp + fd + ft, dtype=np.int32)
    return verts, faces
