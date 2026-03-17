"""
Moth Hull Designer
==================
Desktop application for interactive Moth hull design.
Built with PyQt5 + pyqtgraph for fast, native rendering.

Type a value into any field and press Enter (or Tab) to update.

Requirements: PyQt5, pyqtgraph, numpy, scipy
  pip install PyQt5 pyqtgraph scipy
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

from .config import (
    LWL, MAX_DEPTH, TARGET_DISP_L,
    WL_COLORS, N_WL, SECTION_COLORS, N_SEC,
    DEFAULTS,
)
from .geometry import (
    lagrange, cross_section, build_ctrl, beam_eval,
    displaced_volume, find_disp_waterline, build_3d_mesh,
)
from .exports import export_txt, export_stl, export_step


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
        self._gl        = None
        self._hull_item = None
        self._wl_item   = None

        try:
            import pyqtgraph.opengl as gl
            self._gl = gl
        except ImportError:
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
        if self._gl is None:
            return
        gl = self._gl

        if self._hull_item is not None:
            self.glw.removeItem(self._hull_item)
        if self._wl_item is not None:
            self.glw.removeItem(self._wl_item)

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

        ctrl_x, beam_hb, keel_d, _, _, _, _, _, _ = build_ctrl(params)
        dz = find_disp_waterline(TARGET_DISP_L, ctrl_x, beam_hb, ctrl_x, keel_d)
        self._wl_item = None
        if dz < 0:
            xs_wl  = np.linspace(0, LWL, 200)
            hbs    = beam_eval(xs_wl, ctrl_x, beam_hb)
            depths = lagrange(xs_wl, ctrl_x, keel_d,
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

        box_style = """
            QFrame {
                background: #0d1525;
                border: 1px solid #1e2d45;
                border-radius: 6px;
            }
        """

        # ── Control points ──────────────────────────────────
        cp_box = QFrame()
        cp_box.setStyleSheet(box_style)
        cp_layout = QVBoxLayout(cp_box)
        cp_layout.setContentsMargins(10, 8, 10, 10)
        cp_layout.setSpacing(6)

        sec_label = QLabel('CONTROL POINTS')
        sec_label.setStyleSheet('color: #c8d6e5; font-size: 10px; font-weight: bold;'
                                'border: none; background: transparent;')
        cp_layout.addWidget(sec_label)

        hdr = QWidget()
        hdr.setStyleSheet('background: transparent; border: none;')
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.setSpacing(4)
        hdr_row.addWidget(QLabel(''), stretch=1)
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

        # ── Deck & Keel ──────────────────────────────────────
        dk_box = QFrame()
        dk_box.setStyleSheet(box_style)
        dk_layout = QVBoxLayout(dk_box)
        dk_layout.setContentsMargins(10, 8, 10, 10)
        dk_layout.setSpacing(6)

        dk_title = QLabel('DECK & KEEL  (same X positions as above)')
        dk_title.setStyleSheet('color: #c8d6e5; font-size: 10px; font-weight: bold;'
                               'border: none; background: transparent;')
        dk_layout.addWidget(dk_title)

        dk_hdr_w = QWidget(); dk_hdr_w.setStyleSheet('background:transparent;border:none;')
        dk_hdr_r = QHBoxLayout(dk_hdr_w)
        dk_hdr_r.setContentsMargins(0, 0, 0, 0); dk_hdr_r.setSpacing(4)
        dk_hdr_r.addWidget(QLabel(''), stretch=1)
        for text, color in [('Deck width', '#4499ff'),
                             ('Deck height', '#00ffaa'),
                             ('Keel width',  '#ff6b6b')]:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedWidth(80)
            lbl.setStyleSheet(f'color:{color};font-size:9px;font-weight:bold;'
                              f'border:none;background:transparent;')
            dk_hdr_r.addWidget(lbl)
        dk_layout.addWidget(dk_hdr_w)

        for i in range(1, 6):
            row_w = QWidget(); row_w.setStyleSheet('background:transparent;border:none;')
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
            pt_lbl = QLabel(f'Pt {i}')
            pt_lbl.setStyleSheet('color:#ffaa33;font-size:10px;font-weight:bold;'
                                 'border:none;background:transparent;')
            pt_lbl.setFixedWidth(32)
            row.addWidget(pt_lbl, stretch=1)
            for suffix, color, vmin, vmax in [
                ('_dw', '#4499ff', 0,   500),
                ('_dz', '#00ffaa', 0,   500),
                ('_kw', '#ff6b6b', 0,   200),
            ]:
                key  = f'p{i}{suffix}'
                edit = make_field(DEFAULTS[key], color)
                edit.returnPressed.connect(self._make_updater(key, vmin, vmax))
                edit.editingFinished.connect(self._make_updater(key, vmin, vmax))
                self.inputs[key] = edit
                row.addWidget(edit)
            dk_layout.addWidget(row_w)

        outer.addWidget(dk_box)

        # ── Transom ──────────────────────────────────────────
        tr_box = QFrame()
        tr_box.setStyleSheet(box_style)
        tr_layout = QVBoxLayout(tr_box)
        tr_layout.setContentsMargins(10, 8, 10, 10)
        tr_layout.setSpacing(6)

        tr_title = QLabel('TRANSOM')
        tr_title.setStyleSheet('color: #c8d6e5; font-size: 10px; font-weight: bold;'
                               'border: none; background: transparent;')
        tr_layout.addWidget(tr_title)

        for key, label, vmin, vmax in [
            ('transom_half_beam', 'Half-beam (mm)', 0,   500),
            ('transom_draft',     'Draft (mm)',      0,   MAX_DEPTH),
            ('transom_sheer',     'Height (mm)',     0,   500),
            ('transom_keel_w',    'Keel width (mm)', 0,   500),
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

        # ── Bow ──────────────────────────────────────────────
        bw_box = QFrame()
        bw_box.setStyleSheet(box_style)
        bw_layout = QVBoxLayout(bw_box)
        bw_layout.setContentsMargins(10, 8, 10, 10)
        bw_layout.setSpacing(6)

        bw_title = QLabel('BOW')
        bw_title.setStyleSheet('color: #c8d6e5; font-size: 10px; font-weight: bold;'
                               'border: none; background: transparent;')
        bw_layout.addWidget(bw_title)

        for key, label, vmin, vmax in [
            ('bow_draft',  'Draft (mm)',   0,   MAX_DEPTH),
            ('bow_sheer',  'Height (mm)',  0,   500),
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

        # ── Buttons ──────────────────────────────────────────
        def make_btn(label, color, border, handler):
            btn = QPushButton(label)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {color};
                    color: {'#4db8ff' if color == '#0d2a4a' else '#c8d6e5'};
                    border: 1px solid {border};
                    border-radius: 5px;
                    padding: {'8px' if color == '#0d2a4a' else '7px'};
                    font-size: {'11px' if color == '#0d2a4a' else '10px'};
                    {'font-weight: bold;' if color == '#0d2a4a' else ''}
                }}
                QPushButton:hover   {{ background: #1e2d45; border-color: #2e4466; }}
                QPushButton:pressed {{ background: #253550; }}
            """)
            btn.clicked.connect(handler)
            return btn

        outer.addWidget(make_btn('Launch 3D View',                   '#0d2a4a', '#1e5080', self._open_3d_view))
        outer.addWidget(make_btn('Print Config to Terminal',          '#141d30', '#1e2d45', self._print_config))
        outer.addWidget(make_btn('Export XYZ Points  (SolidWorks)',   '#141d30', '#1e2d45', self._export_txt))
        outer.addWidget(make_btn('Export STL  (SolidWorks / CAD)',    '#141d30', '#1e2d45', self._export_stl))
        outer.addWidget(make_btn('Export STEP  (SolidWorks solid)',   '#141d30', '#1e2d45', self._export_step))
        outer.addWidget(make_btn('Reset to Defaults',                 '#141d30', '#1e2d45', self._reset))

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
        self._p_bow_stem = pg.PlotDataItem(pen=mk('#00d4aa', width=2))
        self._p_bow_keel = pg.ScatterPlotItem(pen=None, brush=br('#ff6b6b'), size=10)
        for item in [self._p_fill, self._p_keel, self._p_sheer,
                     self._p_wl, self._p_transom, self._p_ctrl,
                     self._p_disp_wl, self._p_disp_lbl,
                     self._p_bow_stem, self._p_bow_keel]:
            self.plot_prof.addItem(item)

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

    def _export_txt(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export XYZ Points', 'moth_hull_xyz.txt', 'Text files (*.txt)')
        if not path:
            return
        export_txt(path, dict(self.params))
        self.status.showMessage(f'Exported: {path}', 5000)

    def _export_stl(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export STL', 'moth_hull.stl', 'STL files (*.stl)')
        if not path:
            return
        export_stl(path, dict(self.params))
        self.status.showMessage(f'Exported: {path}', 5000)

    def _export_step(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        try:
            import cadquery  # noqa
        except ImportError:
            QMessageBox.warning(self, 'Missing library',
                                'cadquery not installed.\n\nRun:  pip install cadquery')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export STEP', 'moth_hull.step', 'STEP files (*.step *.stp)')
        if not path:
            return
        self.status.showMessage('Generating STEP — please wait...', 0)
        QApplication.processEvents()
        export_step(path, dict(self.params))
        self.status.showMessage(f'Exported: {path}', 5000)

    def _print_config(self):
        p = self.params
        dw = [int(p[f'p{i}_dw']) for i in range(1, 6)]
        dz = [int(p[f'p{i}_dz']) for i in range(1, 6)]
        kw = [int(p[f'p{i}_kw']) for i in range(1, 6)]
        lines = [
            '',
            '# ================================================================',
            '# Paste this block into config.py to restore this design.',
            '# Replace lines  _PT = [  ...  through  DEFAULTS[f\'p{_i}_kw\'] = ...',
            '# ================================================================',
            '_PT = [',
        ]
        for i in range(1, 6):
            x  = int(p[f'p{i}_x'])
            hb = int(p[f'p{i}_hb'])
            d  = int(p[f'p{i}_d'])
            lines.append(f'    ({x:5d}, {hb:4d}, {d:4d}),   # Pt {i}')
        lines.append(']')
        lines.append(
            f"DEFAULTS = dict(transom_half_beam={p['transom_half_beam']:.1f},"
            f" transom_draft={p['transom_draft']:.1f},"
        )
        lines.append(
            f"                transom_sheer={p['transom_sheer']:.1f},"
            f" transom_keel_w={p['transom_keel_w']:.1f},"
        )
        lines.append(
            f"                bow_draft={p['bow_draft']:.1f},"
            f" bow_sheer={p['bow_sheer']:.1f})"
        )
        lines.append('for _i, (_px, _pb, _pd) in enumerate(_PT, 1):')
        lines.append('    DEFAULTS[f\'p{_i}_x\']  = float(_px)')
        lines.append('    DEFAULTS[f\'p{_i}_hb\'] = float(_pb)')
        lines.append('    DEFAULTS[f\'p{_i}_d\']  = float(_pd)')
        lines.append(f'    DEFAULTS[f\'p{{_i}}_dw\'] = {dw}[_i-1]')
        lines.append(f'    DEFAULTS[f\'p{{_i}}_dz\'] = {dz}[_i-1]')
        lines.append(f'    DEFAULTS[f\'p{{_i}}_kw\'] = {kw}[_i-1]')
        lines.append('# ================================================================')
        lines.append('')
        print('\n'.join(lines))

    # ── Redraw ────────────────────────────────────────────────

    def _redraw(self):
        p = self.params
        t_hb    = p['transom_half_beam']
        t_draft = p['transom_draft']
        b_draft = p['bow_draft']
        ctrl_x, beam_hb, keel_d, deck_hb_c, sheer_z_c, keel_w_c, order, sx, shb = build_ctrl(p)

        x = np.linspace(0, LWL, 500)
        beam_arr  = beam_eval(x, ctrl_x, beam_hb)
        depth_arr = lagrange(x, ctrl_x, keel_d,
                             clip_min=0.0, clip_max=float(MAX_DEPTH))
        sheer_arr = lagrange(x, ctrl_x, sheer_z_c, clip_min=0.0)
        kz = -depth_arr

        # ── Profile ─────────────────────────────────────────
        self._p_fill.setData(np.concatenate([x, x[::-1]]),
                             np.concatenate([kz, sheer_arr[::-1]]))
        self._p_keel.setData(x, kz)
        self._p_sheer.setData(x, sheer_arr)
        t_sheer = p['transom_sheer']
        self._p_transom.setData([LWL, LWL], [-t_draft, t_sheer])
        ctrl_d = lagrange(sx, ctrl_x, keel_d, clip_min=0.0, clip_max=float(MAX_DEPTH))
        self._p_ctrl.setData(x=list(sx), y=list(-ctrl_d))
        self._p_bow_stem.setData([0.0, 0.0], [-b_draft, float(sheer_arr[0])])
        self._p_bow_keel.setData([0.0], [-b_draft])

        # ── Plan view ───────────────────────────────────────
        self._pl_fill.setData(np.concatenate([x, x[::-1]]),
                              np.concatenate([beam_arr, (-beam_arr)[::-1]]))
        self._pl_stbd.setData(x, beam_arr)
        self._pl_port.setData(x, -beam_arr)
        self._pl_transom.setData([LWL, LWL], [-t_hb, t_hb])
        self._pl_ctrl_s.setData(x=list(sx), y=list(shb))
        self._pl_ctrl_p.setData(x=list(sx), y=list(-shb))
        for i, (cx, cy, lbl) in enumerate(zip(sx, shb, self._pl_labels)):
            lbl.setText(f'P{order[i]+1}')
            lbl.setPos(cx, cy + 14)

        # Waterlines
        x_s         = np.linspace(0, LWL, 100)
        hb_s        = beam_eval(x_s, ctrl_x, beam_hb)
        depth_s     = lagrange(x_s, ctrl_x, keel_d,    clip_min=0.0, clip_max=float(MAX_DEPTH))
        dhb_s       = lagrange(x_s, ctrl_x, deck_hb_c, clip_min=0.0)
        sheer_s_arr = lagrange(x_s, ctrl_x, sheer_z_c, clip_min=0.0)
        keelw_s     = lagrange(x_s, ctrl_x, keel_w_c,  clip_min=0.0)
        z_levels    = np.linspace(-MAX_DEPTH * 0.9, -MAX_DEPTH * 0.1, N_WL)
        for ji, zl in enumerate(z_levels):
            px, py = [], []
            for si, xi in enumerate(x_s):
                ys, zs = cross_section(hb_s[si], depth_s[si], dhb_s[si],
                                       sheer_s_arr[si], keelw_s[si], 50)
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
        sec_fracs = (np.arange(N_SEC) + 0.5) / N_SEC
        sec_xs    = sec_fracs * LWL
        sec_hb    = beam_eval(sec_xs, ctrl_x, beam_hb)
        sec_depth = lagrange(sec_xs, ctrl_x, keel_d,    clip_min=0.0, clip_max=float(MAX_DEPTH))
        sec_dhb   = lagrange(sec_xs, ctrl_x, deck_hb_c, clip_min=0.0)
        sec_dz    = lagrange(sec_xs, ctrl_x, sheer_z_c, clip_min=0.0)
        sec_kw    = lagrange(sec_xs, ctrl_x, keel_w_c,  clip_min=0.0)
        for i in range(N_SEC):
            ys, zs = cross_section(sec_hb[i], sec_depth[i], sec_dhb[i],
                                   sec_dz[i], sec_kw[i], 40)
            if sec_fracs[i] <= 0.5:
                self._b_sections[i].setData(ys, zs)
            else:
                self._b_sections[i].setData(-ys, zs)

        t_kw = p['transom_keel_w']
        ty_tr, tz_tr = cross_section(t_hb, t_draft, t_hb, t_sheer, t_kw, 40)
        self._b_transom.setData(-ty_tr, tz_tr)

        # ── Displacement waterline ──────────────────────────
        dz = find_disp_waterline(TARGET_DISP_L, ctrl_x, beam_hb, ctrl_x, keel_d)
        self._p_disp_wl.setPos(dz)
        max_vol = displaced_volume(0.0, ctrl_x, beam_hb, ctrl_x, keel_d)
        if max_vol < TARGET_DISP_L:
            lbl_txt = f'{TARGET_DISP_L} L  (hull only {max_vol:.0f} L — too small)'
        else:
            lbl_txt = f'{TARGET_DISP_L} L  (z={dz:.1f} mm)'
        self._p_disp_lbl.setText(lbl_txt)
        self._p_disp_lbl.setPos(LWL * 0.6, dz)

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
