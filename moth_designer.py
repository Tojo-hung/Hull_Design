"""
Moth Hull Designer
==================
Desktop application for interactive Moth hull design.
Built with PyQt5 + pyqtgraph for fast, native rendering.

Type a value into any field and press Enter (or Tab) to update.

Requirements: PyQt5, pyqtgraph, numpy
  pip install PyQt5 pyqtgraph
"""

import sys
import numpy as np

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QLineEdit, QPushButton, QFrame, QSizePolicy,
        QStatusBar, QSplitter,
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPalette, QColor, QFont
    import pyqtgraph as pg
except ImportError:
    print("Missing libraries. Install with:  pip install PyQt5 pyqtgraph")
    sys.exit(1)

# ─── Fixed constants ──────────────────────────────────────────
LWL       = 3355   # mm — Moth class rule
MAX_DEPTH = 250    # mm — canoe body draft
FREEBOARD = 195    # mm — sheer height

# ─── Defaults  (x_mm, half_beam_mm, depth_mm) ─────────────────
_PT = [
    (int(LWL * 0.10),  200,  150),
    (int(LWL * 0.28), 250, 200),
    (int(LWL * 0.50), 300, 220),
    (int(LWL * 0.68), 270, 220),
    (int(LWL * 0.85),  200,  220),
]
DEFAULTS = dict(transom_half_beam=25.0, transom_draft=82.0,
                bow_draft=0.0, bow_radius=50.0)
for _i, (_px, _pb, _pd) in enumerate(_PT, 1):
    DEFAULTS[f'p{_i}_x']  = float(_px)
    DEFAULTS[f'p{_i}_hb'] = float(_pb)
    DEFAULTS[f'p{_i}_d']  = float(_pd)

# ─── Plot colours ─────────────────────────────────────────────
WL_COLORS = ['#00eeff', '#33ccff', '#6699ff', '#9966ff', '#cc33ff', '#ff00ee']
N_WL      = len(WL_COLORS)

SECTION_COLORS = [
    '#00e5ff', '#00ccee', '#00aaff', '#1e90ff', '#4477ff',
    '#6655ff', '#8844ee', '#aa33cc', '#cc2299', '#ee1166',
]
N_SEC = len(SECTION_COLORS)

# ─────────────────────────────────────────────────────────────
# GEOMETRY  (pure numpy — same as before)
# ─────────────────────────────────────────────────────────────

def lagrange(x_eval, ctrl_x, ctrl_y, clip_min=None, clip_max=None):
    x_eval = np.asarray(x_eval, dtype=float)
    result = np.zeros_like(x_eval)
    n = len(ctrl_x)
    for i in range(n):
        basis = np.ones_like(x_eval)
        for j in range(n):
            if j == i:
                continue
            denom = ctrl_x[i] - ctrl_x[j]
            if abs(denom) < 1e-9:
                continue
            basis *= (x_eval - ctrl_x[j]) / denom
        result += ctrl_y[i] * basis
    if clip_min is not None:
        result = np.maximum(result, clip_min)
    if clip_max is not None:
        result = np.minimum(result, clip_max)
    return result


def cross_section(hb, depth, n=40):
    hb    = max(float(hb),    0.0)
    depth = max(float(depth), 0.0)
    t     = np.linspace(0, np.pi / 2, n)
    y_sub = hb * np.sin(t)
    z_sub = -depth * np.cos(t)
    n_top = max(n // 5, 3)
    y_top = np.full(n_top, hb)
    z_top = np.linspace(0.0, FREEBOARD, n_top)
    return (np.concatenate([y_sub, y_top[1:]]),
            np.concatenate([z_sub, z_top[1:]]))


def build_ctrl(p):
    t_hb    = p['transom_half_beam']
    t_draft = p['transom_draft']
    b_draft = p['bow_draft']
    raw_x  = np.array([p[f'p{i}_x']  for i in range(1, 6)])
    raw_hb = np.array([p[f'p{i}_hb'] for i in range(1, 6)])
    raw_d  = np.array([p[f'p{i}_d']  for i in range(1, 6)])
    order  = np.argsort(raw_x)
    sx = raw_x[order];  shb = raw_hb[order];  sd = raw_d[order]
    beam_x  = np.concatenate([[0.0], sx, [float(LWL)]])
    beam_hb = np.concatenate([[0.0], shb, [float(t_hb)]])
    keel_x  = np.concatenate([[0.0], sx, [float(LWL)]])
    keel_d  = np.concatenate([[float(b_draft)], sd, [float(t_draft)]])
    return beam_x, beam_hb, keel_x, keel_d, order, sx, shb


def beam_eval(x_eval, beam_x, beam_hb, bow_radius):
    """Lagrange beam + circular-arc entry clipping for bow radius.
    Near the bow the half-beam is bounded by sqrt(2 * R * x), giving a
    waterplane entry with radius of curvature R (mm) at the stem.
    """
    x_eval = np.asarray(x_eval, dtype=float)
    b = lagrange(x_eval, beam_x, beam_hb, clip_min=0.0)
    if bow_radius > 0:
        arc = np.sqrt(2.0 * bow_radius * np.maximum(x_eval, 0.0))
        b = np.minimum(b, arc)
    return b


# ─── Displacement volume ──────────────────────────────────────
TARGET_DISP_L = 130  # litres

def _section_area(hb, depth, z_wl):
    """Submerged cross-section area (both sides) below z_wl.
    Cross-section is a quarter-ellipse: y=hb*sin(t), z=-depth*cos(t).
    Area = 2 * hb * depth * (t0/2 - sin(2*t0)/4)  where t0=arccos(-z_wl/depth).
    """
    if hb <= 0 or depth <= 0:
        return 0.0
    if z_wl <= -depth:
        return 0.0
    if z_wl >= 0:
        return np.pi / 2 * hb * depth   # full quarter-ellipse × 2 sides
    cos_t0 = np.clip(-z_wl / depth, 0.0, 1.0)
    t0 = np.arccos(cos_t0)
    return 2.0 * hb * depth * (t0 / 2.0 - np.sin(2.0 * t0) / 4.0)


def displaced_volume(z_wl, beam_x, beam_hb, keel_x, keel_d, bow_radius=0.0, n_x=200):
    """Volume (litres) displaced below waterline z_wl (z_wl <= 0)."""
    x_arr = np.linspace(0.0, float(LWL), n_x)
    hbs    = beam_eval(x_arr, beam_x, beam_hb, bow_radius)
    depths = lagrange(x_arr, keel_x, keel_d,
                      clip_min=0.0, clip_max=float(MAX_DEPTH))
    areas = np.array([_section_area(hbs[k], depths[k], z_wl)
                      for k in range(n_x)])
    return float(np.trapezoid(areas, x_arr)) / 1e6   # mm^3 → litres


def find_disp_waterline(target_l, beam_x, beam_hb, keel_x, keel_d, bow_radius=0.0):
    """Bisect to find z_wl (mm, <=0) where displaced_volume == target_l."""
    lo, hi = -float(MAX_DEPTH), 0.0
    vol_hi = displaced_volume(hi, beam_x, beam_hb, keel_x, keel_d, bow_radius)
    if vol_hi <= target_l:
        return hi          # hull too small even at full draft
    vol_lo = displaced_volume(lo, beam_x, beam_hb, keel_x, keel_d, bow_radius)
    if vol_lo >= target_l:
        return lo          # hull exceeds target even at keel
    for _ in range(40):    # 40 iterations → sub-0.001 mm precision
        mid = (lo + hi) / 2.0
        if displaced_volume(mid, beam_x, beam_hb, keel_x, keel_d, bow_radius) < target_l:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ─────────────────────────────────────────────────────────────
# 3-D MESH
# ─────────────────────────────────────────────────────────────

def build_3d_mesh(params, N_X=80, N_T=36):
    """Return (verts float32 (V,3), faces int32 (F,3)) for GLMeshItem.
    Builds a watertight lofted surface: stbd + port panels, deck strip,
    transom cap.  Coordinate system: X=bow→stern, Y=stbd, Z=up.
    """
    beam_x, beam_hb, keel_x, keel_d, _, _, _ = build_ctrl(params)
    bow_r = params['bow_radius']

    xs    = np.linspace(0.0, float(LWL), N_X)
    t_new = np.linspace(0.0, 1.0, N_T)

    vs = np.zeros((N_X, N_T, 3), dtype=np.float32)
    vp = np.zeros((N_X, N_T, 3), dtype=np.float32)

    for i, xi in enumerate(xs):
        hb    = float(beam_eval([xi], beam_x, beam_hb, bow_r)[0])
        depth = float(lagrange([xi], keel_x, keel_d,
                               clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
        ys, zs = cross_section(hb, depth, 40)
        t_src  = np.linspace(0.0, 1.0, len(ys))
        yr = np.interp(t_new, t_src, ys).astype(np.float32)
        zr = np.interp(t_new, t_src, zs).astype(np.float32)
        vs[i] = np.column_stack([np.full(N_T, xi, np.float32), yr,  zr])
        vp[i] = np.column_stack([np.full(N_T, xi, np.float32), -yr, zr])

    vs_flat = vs.reshape(-1, 3)   # (N_X*N_T, 3)
    vp_flat = vp.reshape(-1, 3)
    OFF     = N_X * N_T

    def quad_strip(row_i, row_j, offset_a=0, flip=False):
        """Two triangles for quad (row_i,j)–(row_j,j)–(row_j,j+1)–(row_i,j+1)."""
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

    # Starboard + port panels
    fs, fp = [], []
    for i in range(N_X - 1):
        fs += quad_strip(i, i + 1, offset_a=0,   flip=False)
        fp += quad_strip(i, i + 1, offset_a=OFF, flip=True)

    # Deck strip: connect stbd sheer to port sheer along the hull
    fd = []
    for i in range(N_X - 1):
        a  = i       * N_T + (N_T - 1)
        b  = (i + 1) * N_T + (N_T - 1)
        ap = OFF + i       * N_T + (N_T - 1)
        bp = OFF + (i + 1) * N_T + (N_T - 1)
        fd += [[a, b, bp], [a, bp, ap]]

    # Transom cap: close the stern with a fan from port to stbd sections
    sv = vs[-1]   # stbd stern (N_T, 3)
    pv = vp[-1]   # port stern (N_T, 3)
    stern_verts = np.vstack([sv, pv]).astype(np.float32)  # (2*N_T, 3)
    S = 2 * OFF
    ft = []
    for j in range(N_T - 1):
        # stbd: j,j+1 and mirror port: N_T+j, N_T+j+1
        ft += [[S + j, S + j + 1, S + N_T + j + 1],
               [S + j, S + N_T + j + 1, S + N_T + j]]

    verts = np.vstack([vs_flat, vp_flat, stern_verts])
    faces = np.array(fs + fp + fd + ft, dtype=np.int32)
    return verts, faces


# ─────────────────────────────────────────────────────────────
# DARK THEME
# ─────────────────────────────────────────────────────────────

def apply_dark_theme(app):
    app.setStyle('Fusion')
    p = QPalette()
    bg      = QColor(10,  14,  23)
    panel   = QColor(13,  18,  32)
    text    = QColor(200, 214, 229)
    muted   = QColor(80,  100, 120)
    accent  = QColor(30,  144, 255)
    btn     = QColor(26,  34,  53)
    btn_alt = QColor(36,  50,  80)
    p.setColor(QPalette.Window,          bg)
    p.setColor(QPalette.WindowText,      text)
    p.setColor(QPalette.Base,            panel)
    p.setColor(QPalette.AlternateBase,   btn)
    p.setColor(QPalette.ToolTipBase,     panel)
    p.setColor(QPalette.ToolTipText,     text)
    p.setColor(QPalette.Text,            text)
    p.setColor(QPalette.PlaceholderText, muted)
    p.setColor(QPalette.Button,          btn)
    p.setColor(QPalette.ButtonText,      text)
    p.setColor(QPalette.BrightText,      QColor(255, 100, 100))
    p.setColor(QPalette.Highlight,       accent)
    p.setColor(QPalette.HighlightedText, QColor(0,  0,  0))
    p.setColor(QPalette.Disabled, QPalette.Text,       muted)
    p.setColor(QPalette.Disabled, QPalette.ButtonText, muted)
    app.setPalette(p)


# ─────────────────────────────────────────────────────────────
# INPUT FIELD HELPER
# ─────────────────────────────────────────────────────────────

def make_field(default_val, color, width=80):
    """Styled QLineEdit pre-filled with default_val."""
    edit = QLineEdit(f'{default_val:.0f}')
    edit.setFixedWidth(width)
    edit.setAlignment(Qt.AlignCenter)
    edit.setStyleSheet(f"""
        QLineEdit {{
            color: {color};
            background: #141d30;
            border: 1px solid #1e2d45;
            border-radius: 4px;
            padding: 3px 6px;
            font-size: 11px;
            font-weight: bold;
        }}
        QLineEdit:focus {{
            border-color: {color};
            background: #1a2640;
        }}
    """)
    return edit


# ─────────────────────────────────────────────────────────────
# 3-D VIEW WINDOW
# ─────────────────────────────────────────────────────────────

class Hull3DView(QMainWindow):
    """3-D hull viewer.  Created once; hidden instead of destroyed on close
    so the OpenGL context survives across relaunches."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Moth Hull — 3D View')
        self.resize(1100, 750)
        self._gl       = None   # pyqtgraph.opengl module, set in _init_gl
        self._hull_item = None
        self._wl_item   = None

        try:
            import pyqtgraph.opengl as gl
            self._gl = gl
        except ImportError:
            from PyQt5.QtWidgets import QLabel
            w = QLabel('PyOpenGL not installed.\n\nRun:  pip install PyOpenGL')
            w.setAlignment(Qt.AlignCenter)
            w.setStyleSheet('color:#c8d6e5; font-size:13px;')
            self.setCentralWidget(w)
            return

        self._init_gl()

    def _init_gl(self):
        gl = self._gl
        self.glw = gl.GLViewWidget()
        self.glw.setBackgroundColor('#0a0e17')
        self.setCentralWidget(self.glw)

        self.glw.opts['center'] = pg.Vector(LWL / 2, 0, -MAX_DEPTH / 4)
        self.glw.setCameraPosition(distance=LWL * 1.8, elevation=22, azimuth=-55)

        # Grid and water plane are static — add once
        grid = gl.GLGridItem()
        grid.setSize(LWL * 1.6, LWL * 0.6)
        grid.setSpacing(500, 200)
        grid.setColor((30, 60, 100, 90))
        self.glw.addItem(grid)

        hw = LWL * 0.55
        wv = np.array([[ 0,  -hw, 0], [LWL, -hw, 0],
                       [LWL,  hw, 0], [ 0,   hw, 0]], dtype=np.float32)
        wf = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        wc = np.array([[0.05, 0.25, 0.55, 0.30],
                       [0.05, 0.25, 0.55, 0.30]], dtype=np.float32)
        self.glw.addItem(gl.GLMeshItem(vertexes=wv, faces=wf, faceColors=wc,
                                       smooth=False, glOptions='translucent'))

        ax = gl.GLAxisItem()
        ax.setSize(200, 200, 200)
        self.glw.addItem(ax)

    def refresh(self, params):
        """Rebuild only the hull mesh and displacement waterline."""
        if self._gl is None:
            return
        gl = self._gl

        # Remove previous dynamic items
        if self._hull_item is not None:
            self.glw.removeItem(self._hull_item)
        if self._wl_item is not None:
            self.glw.removeItem(self._wl_item)

        # Hull mesh
        verts, faces = build_3d_mesh(params)
        frac = (verts[faces[:, 0], 0] / LWL).clip(0, 1).astype(np.float32)
        fc = np.zeros((len(faces), 4), dtype=np.float32)
        fc[:, 0] = 0.04 + 0.08 * frac
        fc[:, 1] = 0.28 + 0.22 * frac
        fc[:, 2] = 0.72 - 0.10 * frac
        fc[:, 3] = 1.0
        self._hull_item = gl.GLMeshItem(vertexes=verts, faces=faces,
                                        faceColors=fc, smooth=True,
                                        drawEdges=False, glOptions='opaque')
        self.glw.addItem(self._hull_item)

        # Displacement waterline ring
        beam_x, beam_hb, keel_x, keel_d, _, _, _ = build_ctrl(params)
        bow_r = params['bow_radius']
        dz = find_disp_waterline(TARGET_DISP_L, beam_x, beam_hb,
                                 keel_x, keel_d, bow_r)
        self._wl_item = None
        if dz < 0:
            xs_wl  = np.linspace(0, LWL, 200)
            hbs    = beam_eval(xs_wl, beam_x, beam_hb, bow_r)
            depths = lagrange(xs_wl, keel_x, keel_d,
                              clip_min=0.0, clip_max=float(MAX_DEPTH))
            wl_pts = []
            for xi, hb, depth in zip(xs_wl, hbs, depths):
                if depth <= 0:
                    continue
                cos_t = np.clip(-dz / depth, 0, 1)
                wl_pts.append([xi, hb * np.sqrt(1 - cos_t ** 2), dz])
            if wl_pts:
                stbd = np.array(wl_pts, dtype=np.float32)
                port = stbd.copy(); port[:, 1] *= -1
                pts  = np.vstack([stbd, port[::-1]])
                self._wl_item = gl.GLLinePlotItem(pos=pts, color=(1, 0.84, 0, 1),
                                                  width=2.5, mode='line_strip',
                                                  antialias=True)
                self.glw.addItem(self._wl_item)

    def closeEvent(self, event):
        """Hide instead of destroying so the GL context stays alive."""
        self.hide()
        event.ignore()


# ─────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────

class MothDesigner(QMainWindow):

    def __init__(self):
        super().__init__()
        self.params  = {k: float(v) for k, v in DEFAULTS.items()}
        self.inputs  = {}
        self._3d_win = None
        self._setup_window()
        self._setup_plot_items()
        self._redraw()

    # ── Window / layout ───────────────────────────────────────

    def _setup_window(self):
        self.setWindowTitle('Moth Hull Designer  —  LWL 3355 mm')
        self.resize(1440, 860)

        pg.setConfigOption('background', '#0b1020')
        pg.setConfigOption('foreground', '#7a8fa8')

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 6, 10)
        root_layout.setSpacing(10)

        root_layout.addWidget(self._build_input_panel(), stretch=0)
        root_layout.addWidget(self._build_plot_panel(),  stretch=1)

        self.status = QStatusBar()
        self.status.setStyleSheet('color: #7a8fa8; font-size: 10px;')
        self.setStatusBar(self.status)

    # ── Input panel ───────────────────────────────────────────

    def _build_input_panel(self):
        panel = QWidget()
        panel.setFixedWidth(310)
        outer = QVBoxLayout(panel)
        outer.setSpacing(10)
        outer.setContentsMargins(0, 0, 0, 0)

        # ── Control points ──────────────────────────────────
        cp_box = QFrame()
        cp_box.setStyleSheet("""
            QFrame {
                background: #0d1525;
                border: 1px solid #1e2d45;
                border-radius: 6px;
            }
        """)
        cp_layout = QVBoxLayout(cp_box)
        cp_layout.setContentsMargins(10, 8, 10, 10)
        cp_layout.setSpacing(6)

        sec_label = QLabel('CONTROL POINTS')
        sec_label.setStyleSheet('color: #c8d6e5; font-size: 10px; font-weight: bold;'
                                'border: none; background: transparent;')
        cp_layout.addWidget(sec_label)

        # Header row
        hdr = QWidget()
        hdr.setStyleSheet('background: transparent; border: none;')
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.setSpacing(4)
        hdr_row.addWidget(QLabel(''), stretch=1)  # pt label spacer
        for text, color in [('X from bow', '#ffaa33'),
                             ('Half-beam',  '#4499ff'),
                             ('Depth',      '#ff6b6b')]:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedWidth(80)
            lbl.setStyleSheet(f'color: {color}; font-size: 9px; font-weight: bold;'
                              f'border: none; background: transparent;')
            hdr_row.addWidget(lbl)
        cp_layout.addWidget(hdr)

        # 5 control point rows
        for i in range(1, 6):
            row_w = QWidget()
            row_w.setStyleSheet('background: transparent; border: none;')
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            pt_lbl = QLabel(f'Pt {i}')
            pt_lbl.setStyleSheet('color: #ffaa33; font-size: 10px; font-weight: bold;'
                                 'border: none; background: transparent;')
            pt_lbl.setFixedWidth(32)
            row.addWidget(pt_lbl, stretch=1)

            for suffix, color, vmin, vmax in [
                ('_x',  '#ffaa33', 0,   LWL),
                ('_hb', '#4499ff', 0,   500),
                ('_d',  '#ff6b6b', 0,   MAX_DEPTH),
            ]:
                key  = f'p{i}{suffix}'
                edit = make_field(DEFAULTS[key], color)
                edit.returnPressed.connect(self._make_updater(key, vmin, vmax))
                edit.editingFinished.connect(self._make_updater(key, vmin, vmax))
                self.inputs[key] = edit
                row.addWidget(edit)

            cp_layout.addWidget(row_w)

        outer.addWidget(cp_box)

        # ── Transom ─────────────────────────────────────────
        tr_box = QFrame()
        tr_box.setStyleSheet(cp_box.styleSheet())
        tr_layout = QVBoxLayout(tr_box)
        tr_layout.setContentsMargins(10, 8, 10, 10)
        tr_layout.setSpacing(6)

        tr_title = QLabel('TRANSOM')
        tr_title.setStyleSheet('color: #c8d6e5; font-size: 10px; font-weight: bold;'
                               'border: none; background: transparent;')
        tr_layout.addWidget(tr_title)

        for key, label, vmin, vmax in [
            ('transom_half_beam', 'Half-beam (mm)', 0, 500),
            ('transom_draft',     'Draft (mm)',      0, MAX_DEPTH),
        ]:
            row_w = QWidget()
            row_w.setStyleSheet('background: transparent; border: none;')
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setStyleSheet('color: #c8d6e5; font-size: 10px;'
                              'border: none; background: transparent;')
            edit = make_field(DEFAULTS[key], '#00d4aa', width=90)
            edit.returnPressed.connect(self._make_updater(key, vmin, vmax))
            edit.editingFinished.connect(self._make_updater(key, vmin, vmax))
            self.inputs[key] = edit
            row.addWidget(lbl, stretch=1)
            row.addWidget(edit)
            tr_layout.addWidget(row_w)

        outer.addWidget(tr_box)

        # ── Bow ─────────────────────────────────────────────
        bw_box = QFrame()
        bw_box.setStyleSheet(cp_box.styleSheet())
        bw_layout = QVBoxLayout(bw_box)
        bw_layout.setContentsMargins(10, 8, 10, 10)
        bw_layout.setSpacing(6)

        bw_title = QLabel('BOW')
        bw_title.setStyleSheet('color: #c8d6e5; font-size: 10px; font-weight: bold;'
                               'border: none; background: transparent;')
        bw_layout.addWidget(bw_title)

        for key, label, vmin, vmax in [
            ('bow_draft',  'Draft (mm)',        0, MAX_DEPTH),
            ('bow_radius', 'Entry radius (mm)', 1, 500),
        ]:
            row_w = QWidget()
            row_w.setStyleSheet('background: transparent; border: none;')
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setStyleSheet('color: #c8d6e5; font-size: 10px;'
                              'border: none; background: transparent;')
            edit = make_field(DEFAULTS[key], '#00d4aa', width=90)
            edit.returnPressed.connect(self._make_updater(key, vmin, vmax))
            edit.editingFinished.connect(self._make_updater(key, vmin, vmax))
            self.inputs[key] = edit
            row.addWidget(lbl, stretch=1)
            row.addWidget(edit)
            bw_layout.addWidget(row_w)

        outer.addWidget(bw_box)

        # ── 3D View button ──────────────────────────────────
        view3d_btn = QPushButton('Launch 3D View')
        view3d_btn.setStyleSheet("""
            QPushButton {
                background: #0d2a4a;
                color: #4db8ff;
                border: 1px solid #1e5080;
                border-radius: 5px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover   { background: #133460; border-color: #2e78c0; }
            QPushButton:pressed { background: #1a4070; }
        """)
        view3d_btn.clicked.connect(self._open_3d_view)
        outer.addWidget(view3d_btn)

        # ── Print config button ──────────────────────────────
        cfg_btn = QPushButton('Print Config to Terminal')
        cfg_btn.setStyleSheet("""
            QPushButton {
                background: #141d30;
                color: #c8d6e5;
                border: 1px solid #1e2d45;
                border-radius: 5px;
                padding: 7px;
                font-size: 10px;
            }
            QPushButton:hover   { background: #1e2d45; border-color: #2e4466; }
            QPushButton:pressed { background: #253550; }
        """)
        cfg_btn.clicked.connect(self._print_config)
        outer.addWidget(cfg_btn)

        # ── Reset button ────────────────────────────────────
        reset_btn = QPushButton('Reset to Defaults')
        reset_btn.setStyleSheet("""
            QPushButton {
                background: #141d30;
                color: #c8d6e5;
                border: 1px solid #1e2d45;
                border-radius: 5px;
                padding: 7px;
                font-size: 10px;
            }
            QPushButton:hover { background: #1e2d45; border-color: #2e4466; }
            QPushButton:pressed { background: #253550; }
        """)
        reset_btn.clicked.connect(self._reset)
        outer.addWidget(reset_btn)

        # ── Hint ────────────────────────────────────────────
        hint = QLabel('Press Enter or Tab to update  •  '
                      'Orange = X  •  Blue = beam  •  Red = depth')
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet('color: #3d5068; font-size: 8px; border: none;')
        outer.addWidget(hint)

        outer.addStretch()
        return panel

    # ── Plot panel ────────────────────────────────────────────

    def _build_plot_panel(self):
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        axis_pen = pg.mkPen('#1e2d45')

        def make_plot(title, xlabel, ylabel, aspect_locked=False):
            pw = pg.PlotWidget()
            pw.setTitle(f'<span style="color:#c8d6e5;font-size:11px">{title}</span>')
            pw.setLabel('bottom', xlabel, color='#7a8fa8', size='9pt')
            pw.setLabel('left',   ylabel, color='#7a8fa8', size='9pt')
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.getAxis('bottom').setPen(axis_pen)
            pw.getAxis('left').setPen(axis_pen)
            pw.getAxis('bottom').setTextPen(pg.mkPen('#7a8fa8'))
            pw.getAxis('left').setTextPen(pg.mkPen('#7a8fa8'))
            if aspect_locked:
                pw.setAspectLocked(True)
            return pw

        self.plot_prof = make_plot('Profile', 'X from bow (mm)', 'Z (mm)')
        self.plot_plan = make_plot('Plan view — waterlines', 'X from bow (mm)', 'Y (mm)',
                                   aspect_locked=True)
        self.plot_body = make_plot('Body plan  (fwd right  |  aft left)', 'Y (mm)', 'Z (mm)',
                                   aspect_locked=True)

        self.plot_prof.setXRange(-80, LWL + 80, padding=0)
        self.plot_plan.setXRange(-80, LWL + 80, padding=0)

        # Profile spans full top row; plan + body share bottom row
        grid.addWidget(self.plot_prof, 0, 0, 1, 2)
        grid.addWidget(self.plot_plan, 1, 0)
        grid.addWidget(self.plot_body, 1, 1)
        grid.setRowStretch(0, 5)
        grid.setRowStretch(1, 4)

        return container

    # ── Pre-create plot items ─────────────────────────────────

    def _setup_plot_items(self):
        mk = pg.mkPen
        br = pg.mkBrush

        # ── Profile ─────────────────────────────────────────
        self._p_fill    = pg.PlotCurveItem(pen=None, brush=br(30, 144, 255, 20))
        self._p_keel    = pg.PlotDataItem(pen=mk('#ff6b6b', width=2))
        self._p_sheer   = pg.PlotDataItem(pen=mk('#00d4aa', width=2))
        self._p_wl      = pg.InfiniteLine(pos=0, angle=0,
                                           pen=mk('#4488cc', width=1,
                                                  style=Qt.DashLine))
        self._p_transom = pg.PlotDataItem(pen=mk('#ffaa33', width=2.5))
        self._p_ctrl    = pg.ScatterPlotItem(pen=None, brush=br('#ffaa33'), size=9)
        self._p_disp_wl = pg.InfiniteLine(pos=-50, angle=0,
                                           pen=mk('#ffd700', width=1.5,
                                                  style=Qt.DashLine))
        self._p_disp_lbl = pg.TextItem('', color='#ffd700', anchor=(0, 1))
        self._p_bow_stem  = pg.PlotDataItem(pen=mk('#00d4aa', width=2))
        self._p_bow_keel  = pg.ScatterPlotItem(pen=None, brush=br('#ff6b6b'), size=10)
        for item in [self._p_fill, self._p_keel, self._p_sheer,
                     self._p_wl, self._p_transom, self._p_ctrl,
                     self._p_disp_wl, self._p_disp_lbl,
                     self._p_bow_stem, self._p_bow_keel]:
            self.plot_prof.addItem(item)

        # ── Plan view ───────────────────────────────────────
        self._pl_fill    = pg.PlotCurveItem(pen=None, brush=br(30, 144, 255, 20))
        self._pl_stbd    = pg.PlotDataItem(pen=mk('#00d4aa', width=2))
        self._pl_port    = pg.PlotDataItem(pen=mk('#00d4aa', width=2))
        self._pl_transom = pg.PlotDataItem(pen=mk('#ffaa33', width=2.5))
        self._pl_ctrl_s  = pg.ScatterPlotItem(pen=None, brush=br('#ffaa33'), size=9)
        self._pl_ctrl_p  = pg.ScatterPlotItem(pen=None, brush=br('#ffaa33'), size=9)
        self._pl_labels  = [
            pg.TextItem(f'P{i}', color='#ffaa33', anchor=(0.5, 1.2))
            for i in range(1, 6)
        ]
        self._pl_wl_s = [pg.PlotDataItem(pen=mk(WL_COLORS[j], width=1))
                         for j in range(N_WL)]
        self._pl_wl_p = [pg.PlotDataItem(pen=mk(WL_COLORS[j], width=1))
                         for j in range(N_WL)]
        for item in ([self._pl_fill, self._pl_stbd, self._pl_port,
                      self._pl_transom, self._pl_ctrl_s, self._pl_ctrl_p]
                     + self._pl_labels + self._pl_wl_s + self._pl_wl_p):
            self.plot_plan.addItem(item)

        # ── Body plan ───────────────────────────────────────
        self._b_sections = [pg.PlotDataItem(pen=mk(SECTION_COLORS[i], width=1.6))
                            for i in range(N_SEC)]
        self._b_wl       = pg.InfiniteLine(pos=0, angle=0,
                                            pen=mk('#4488cc', width=1,
                                                   style=Qt.DashLine))
        self._b_center   = pg.InfiniteLine(pos=0, angle=90,
                                            pen=mk('#1e2d45', width=1))
        self._b_transom  = pg.PlotDataItem(pen=mk('#ffaa33', width=2.5,
                                                   style=Qt.DashLine))
        for item in (self._b_sections
                     + [self._b_wl, self._b_center, self._b_transom]):
            self.plot_body.addItem(item)

    # ── Callbacks ─────────────────────────────────────────────

    def _make_updater(self, key, vmin, vmax):
        def update():
            try:
                val = float(self.inputs[key].text())
                val = max(vmin, min(vmax, val))
                self.params[key] = val
                self.inputs[key].setText(f'{val:.0f}')
                self._redraw()
            except ValueError:
                self.inputs[key].setText(f'{self.params[key]:.0f}')
        return update

    def _reset(self):
        for key in self.params:
            self.params[key] = float(DEFAULTS[key])
            self.inputs[key].setText(f'{DEFAULTS[key]:.0f}')
        self._redraw()

    def _open_3d_view(self):
        if self._3d_win is None:
            self._3d_win = Hull3DView(parent=self)
        self._3d_win.refresh(dict(self.params))
        self._3d_win.show()
        self._3d_win.raise_()

    def _print_config(self):
        p = self.params
        lines = [
            '',
            '# --- Paste this into moth_designer.py to update defaults ---',
            '_PT = [',
        ]
        for i in range(1, 6):
            x  = int(p[f'p{i}_x'])
            hb = int(p[f'p{i}_hb'])
            d  = int(p[f'p{i}_d'])
            lines.append(f'    ({x:5d}, {hb:4d}, {d:4d}),   # Pt {i}')
        lines.append(']')
        lines.append(
            f"DEFAULTS = dict(transom_half_beam={p['transom_half_beam']:.0f},"
            f" transom_draft={p['transom_draft']:.0f},"
        )
        lines.append(
            f"                bow_draft={p['bow_draft']:.0f},"
            f" bow_radius={p['bow_radius']:.0f})"
        )
        lines.append('')
        print('\n'.join(lines))

    # ── Redraw (updates data only — no item creation) ─────────

    def _redraw(self):
        p = self.params
        t_hb      = p['transom_half_beam']
        t_draft   = p['transom_draft']
        b_draft   = p['bow_draft']
        bow_r     = p['bow_radius']
        beam_x, beam_hb, keel_x, keel_d, order, sx, shb = build_ctrl(p)

        x = np.linspace(0, LWL, 500)
        beam_arr  = beam_eval(x, beam_x, beam_hb, bow_r)
        depth_arr = lagrange(x, keel_x, keel_d,
                             clip_min=0.0, clip_max=float(MAX_DEPTH))
        kz  = -depth_arr
        shz = np.full(len(x), float(FREEBOARD))

        # ── Profile ─────────────────────────────────────────
        xf = np.concatenate([x, x[::-1]])
        yf = np.concatenate([kz, shz[::-1]])
        self._p_fill.setData(xf, yf)
        self._p_keel.setData(x, kz)
        self._p_sheer.setData(x, shz)
        self._p_transom.setData([LWL, LWL], [-t_draft, FREEBOARD])
        ctrl_d = lagrange(sx, keel_x, keel_d,
                          clip_min=0.0, clip_max=float(MAX_DEPTH))
        self._p_ctrl.setData(x=list(sx), y=list(-ctrl_d))
        # Bow stem line + keel marker
        self._p_bow_stem.setData([0.0, 0.0], [-b_draft, FREEBOARD])
        self._p_bow_keel.setData([0.0], [-b_draft])

        # ── Plan view ───────────────────────────────────────
        xf2 = np.concatenate([x, x[::-1]])
        yf2 = np.concatenate([beam_arr, (-beam_arr)[::-1]])
        self._pl_fill.setData(xf2, yf2)
        self._pl_stbd.setData(x, beam_arr)
        self._pl_port.setData(x, -beam_arr)
        self._pl_transom.setData([LWL, LWL], [-t_hb, t_hb])
        self._pl_ctrl_s.setData(x=list(sx), y=list(shb))
        self._pl_ctrl_p.setData(x=list(sx), y=list(-shb))
        for i, (cx, cy, lbl) in enumerate(zip(sx, shb, self._pl_labels)):
            lbl.setText(f'P{order[i]+1}')
            lbl.setPos(cx, cy + 14)

        # Waterlines
        x_s      = np.linspace(0, LWL, 100)
        z_levels = np.linspace(-MAX_DEPTH * 0.9, -MAX_DEPTH * 0.1, N_WL)
        for ji, zl in enumerate(z_levels):
            px, py = [], []
            for xi in x_s:
                hb_xi    = float(beam_eval([xi], beam_x, beam_hb, bow_r)[0])
                depth_xi = float(lagrange([xi], keel_x, keel_d,
                                          clip_min=0.0,
                                          clip_max=float(MAX_DEPTH))[0])
                ys, zs = cross_section(hb_xi, depth_xi, 50)
                for k in range(len(zs) - 1):
                    if (zs[k] - zl) * (zs[k + 1] - zl) <= 0:
                        f = (zl - zs[k]) / (zs[k + 1] - zs[k] + 1e-12)
                        px.append(xi)
                        py.append(float(ys[k] + f * (ys[k + 1] - ys[k])))
                        break
            if px:
                self._pl_wl_s[ji].setData(px, py)
                self._pl_wl_p[ji].setData(px, [-y for y in py])
            else:
                self._pl_wl_s[ji].setData([], [])
                self._pl_wl_p[ji].setData([], [])

        # ── Body plan ───────────────────────────────────────
        for i in range(N_SEC):
            frac     = (i + 0.5) / N_SEC
            xi       = frac * LWL
            hb_xi    = float(beam_eval([xi], beam_x, beam_hb, bow_r)[0])
            depth_xi = float(lagrange([xi], keel_x, keel_d,
                                      clip_min=0.0,
                                      clip_max=float(MAX_DEPTH))[0])
            ys, zs = cross_section(hb_xi, depth_xi, 40)
            if frac <= 0.5:
                self._b_sections[i].setData(ys, zs)
            else:
                self._b_sections[i].setData(-ys, zs)

        ty = np.array([0.0, t_hb, t_hb])
        tz = np.array([-t_draft, -t_draft, FREEBOARD])
        self._b_transom.setData(-ty, tz)

        # ── Displacement waterline ──────────────────────────
        dz = find_disp_waterline(TARGET_DISP_L, beam_x, beam_hb, keel_x, keel_d, bow_r)
        self._p_disp_wl.setPos(dz)
        max_vol = displaced_volume(0.0, beam_x, beam_hb, keel_x, keel_d, bow_r)
        if max_vol < TARGET_DISP_L:
            lbl_txt = f'{TARGET_DISP_L} L  (hull only {max_vol:.0f} L — too small)'
        else:
            lbl_txt = f'{TARGET_DISP_L} L  (z={dz:.1f} mm)'
        self._p_disp_lbl.setText(lbl_txt)
        self._p_disp_lbl.setPos(LWL * 0.6, dz)

        # ── Status bar ──────────────────────────────────────
        self.status.showMessage(
            f'Max beam: {2 * float(np.max(beam_arr)):.0f} mm   '
            f'Max depth: {float(np.max(depth_arr)):.0f} mm   '
            f'Transom: {2*t_hb:.0f} wide × {t_draft:.0f} deep   '
            f'LWL: {LWL} mm  (fixed)   '
            f'Hull vol: {max_vol:.1f} L   '
            f'{TARGET_DISP_L} L waterline: z={dz:.1f} mm'
        )


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Moth Hull Designer')
    apply_dark_theme(app)
    window = MothDesigner()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
