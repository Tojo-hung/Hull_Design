"""
visualize_sections.py
Shows 3 control points per station in front view (Y-Z plane)
with tangent constraints matching Onshape sketch behaviour:
  - CP1 keel shoulder: horizontal tangent (continuous with flat keel)
  - CP2 max beam:      horizontal tangent (handle parallel to beam line)
  - CP3 deck edge:     vertical tangent   (handle perpendicular to deck line)

The flat keel (centreline → keel shoulder) is a separate straight line.
The spline runs from CP1 → CP2 → CP3 with clamped tangents.

Run: python visualize_sections.py
Saves: section_control_points_detail.png  and  section_all_stations.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pathlib import Path
import json

# ── Load best design ────────────────────────────────────────────────
data = json.loads(Path('best_fine_mesh.json').read_text())

# ── 3 control points per station ─────────────────────────────────────
def station_control_points(hb, depth, keel_w, hb_z, deck_hb, deck_z):
    """
    CP1  keel shoulder  — end of flat keel  (y=keel_w,  z=−depth)
    CP2  max beam       — widest point       (y=hb,      z=hb_z)
    CP3  deck edge      — top of section     (y=deck_hb, z=deck_z)
    """
    cp1 = (keel_w,  -depth)
    cp2 = (hb,       hb_z)
    cp3 = (deck_hb,  deck_z)
    return [cp1, cp2, cp3], keel_w, depth

# ── Spline with clamped tangents ──────────────────────────────────────
def spline_segment(pts, dy_start, dz_start, dy_end, dz_end, n=400):
    pts = np.array(pts, dtype=float)
    y, z = pts[:, 0], pts[:, 1]
    ds = np.sqrt(np.diff(y)**2 + np.diff(z)**2)
    t  = np.concatenate([[0], np.cumsum(ds)])
    L  = t[-1]
    t  = t / L
    def scale(dy, dz):
        mag = np.sqrt(dy**2 + dz**2) + 1e-12
        return dy / mag * L, dz / mag * L
    dys, dzs = scale(dy_start, dz_start)
    dye, dze = scale(dy_end,   dz_end)
    cs_y = CubicSpline(t, y, bc_type=((1, dys), (1, dye)))
    cs_z = CubicSpline(t, z, bc_type=((1, dzs), (1, dze)))
    te = np.linspace(0, 1, n)
    return cs_y(te), cs_z(te)

def hull_spline(cps, n=400):
    """Single spline CP1→CP2→CP3 with:
       horizontal tangent at CP1 (parallel to beam / flat keel)
       horizontal tangent at CP2 (max beam, parallel to beam line)
       vertical   tangent at CP3 (perpendicular to deck line)
    """
    return spline_segment(
        cps,
        dy_start=1, dz_start=0,   # CP1: horizontal (continuous with flat keel)
        dy_end=0,   dz_end=1,     # CP3: vertical   (perpendicular to deck)
        n=n)

# ── Tangent handles ───────────────────────────────────────────────────
def tangent_handles(cps):
    cp1, cp2, cp3 = [np.array(c) for c in cps]
    h = max(cp2[0] * 0.13, 10)
    return [
        (cp1[0], cp1[1],  h, 0),           # CP1 horizontal →
        (cp3[0], cp3[1],  0, -h),          # CP3 vertical ↓
        (cp3[0], cp3[1],  0,  h),          # CP3 vertical ↑
    ]

# ── Station data ─────────────────────────────────────────────────────
stations = {
    'Transom  (x = 0)': dict(
        hb=data['transom_half_beam'], depth=data['transom_draft'],
        keel_w=data['transom_keel_w'], hb_z=0,
        deck_hb=data['transom_half_beam'], deck_z=data['transom_sheer']),
    f'P1  (x = {data["p1_x"]:.0f} mm)': dict(
        hb=data['p1_hb'], depth=data['p1_d'],
        keel_w=data['p1_kw'], hb_z=data['p1_hz'],
        deck_hb=data['p1_dw'], deck_z=data['p1_dz']),
    f'P2  (x = {data["p2_x"]:.0f} mm)': dict(
        hb=data['p2_hb'], depth=data['p2_d'],
        keel_w=data['p2_kw'], hb_z=data['p2_hz'],
        deck_hb=data['p2_dw'], deck_z=data['p2_dz']),
    f'P3  midship  (x = {data["p3_x"]:.0f} mm)': dict(
        hb=data['p3_hb'], depth=data['p3_d'],
        keel_w=data['p3_kw'], hb_z=data['p3_hz'],
        deck_hb=data['p3_dw'], deck_z=data['p3_dz']),
    f'P4  (x = {data["p4_x"]:.0f} mm)': dict(
        hb=data['p4_hb'], depth=data['p4_d'],
        keel_w=data['p4_kw'], hb_z=data['p4_hz'],
        deck_hb=data['p4_dw'], deck_z=data['p4_dz']),
    f'P5  (x = {data["p5_x"]:.0f} mm)': dict(
        hb=data['p5_hb'], depth=data['p5_d'],
        keel_w=data['p5_kw'], hb_z=data['p5_hz'],
        deck_hb=data['p5_dw'], deck_z=data['p5_dz']),
}

CP_COLORS = ['#e67e00', '#2ecc71', '#9b59b6']
CP_NAMES  = ['CP1  keel shoulder', 'CP2  max beam', 'CP3  deck edge']

HANDLE_COLOR = '#888888'
HANDLE_DOT   = '#cccccc'

# ── Draw a single section ─────────────────────────────────────────────
def draw_section(ax, params, title, annotate=False):
    cps, keel_w, depth = station_control_points(**params)
    y_sp, z_sp = hull_spline(cps)

    # Filled hull shape (mirrored)
    keel_y = np.array([0, keel_w])
    keel_z = np.array([-depth, -depth])
    full_y = np.concatenate([keel_y, y_sp])
    full_z = np.concatenate([keel_z, z_sp])
    ax.fill_betweenx(full_z,  full_y, -full_y, alpha=0.10, color='#3498db', zorder=1)

    # Flat keel (centreline → keel shoulder)
    ax.plot([ 0,  keel_w], [-depth, -depth], color='#1a5276', lw=2.0, zorder=2)
    ax.plot([ 0, -keel_w], [-depth, -depth], color='#1a5276', lw=2.0, zorder=2)

    # Spline curve (both sides)
    ax.plot( y_sp, z_sp, color='#1a5276', lw=2.0, zorder=2)
    ax.plot(-y_sp, z_sp, color='#1a5276', lw=2.0, zorder=2)

    # Reference lines
    ax.axvline(0, color='#bbbbbb', lw=0.8, linestyle='--', zorder=0)
    ax.axhline(0, color='#3498db', lw=0.8, linestyle='--', alpha=0.5, zorder=0)

    # Beam dimension line at CP2
    cp2 = cps[1]
    ax.annotate('', xy=(cp2[0], cp2[1] - 6), xytext=(-cp2[0], cp2[1] - 6),
                arrowprops=dict(arrowstyle='<->', color='#aaaaaa', lw=0.8))

    # Tangent handles
    for (hx, hz, ddx, ddz) in tangent_handles(cps):
        ax.plot([hx - ddx, hx + ddx], [hz - ddz, hz + ddz],
                color=HANDLE_COLOR, lw=1.2, zorder=3)
        ax.scatter([hx - ddx, hx + ddx], [hz - ddz, hz + ddz],
                   s=22, color=HANDLE_DOT, edgecolors=HANDLE_COLOR, lw=1.0, zorder=4)

    # Control points
    for i, (cp, col) in enumerate(zip(cps, CP_COLORS)):
        ax.scatter( cp[0], cp[1], color=col, s=60, zorder=6, edgecolors='white', lw=1.5)
        ax.scatter(-cp[0], cp[1], color=col, s=30, zorder=6, alpha=0.35,
                   edgecolors='white', lw=0.8)
        if annotate:
            offsets = [(18, 8), (-22, 8), (-22, -14)]
            ox, oz = offsets[i]
            ax.annotate(
                f'{CP_NAMES[i]}\ny = {cp[0]:.0f}  z = {cp[1]:.0f}',
                xy=(cp[0], cp[1]),
                xytext=(cp[0] + ox, cp[1] + oz),
                fontsize=7.5, color='#1a1a2e', va='center',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.28', fc='white',
                          ec=col, lw=1.5, alpha=0.95))

    ax.set_title(title, fontsize=9.5, color='#1a1a2e', pad=5)
    ax.set_facecolor('#f8f9fb')
    ax.set_aspect('equal')
    ax.set_xlabel('Y  (mm)', fontsize=8, color='#555')
    ax.set_ylabel('Z  (mm)', fontsize=8, color='#555')
    ax.tick_params(labelsize=7, colors='#555')
    for sp in ax.spines.values():
        sp.set_edgecolor('#dddddd')
    ax.grid(True, color='#ebebeb', lw=0.5)

# ════════════════════════════════════════════════════════════════════
# Figure 1: annotated midship + tangent rule guide
# ════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 9), facecolor='#ffffff')

p3_key  = f'P3  midship  (x = {data["p3_x"]:.0f} mm)'
p3_data = stations[p3_key]
draw_section(axes1[0], p3_data, p3_key + '\n(annotated)', annotate=True)
axes1[0].set_xlim(-260, 380)

# Guide panel
ax_g = axes1[1]
ax_g.set_facecolor('#f8f9fb')
ax_g.axis('off')
ax_g.set_title('Tangent rules — how to constrain each control point in CAD',
               fontsize=11, color='#1a1a2e', pad=8)

rules = [
    (CP_COLORS[0], 'CP1  keel shoulder  (y = keel_w,  z = −depth)',
     '→  Constrain: z = −depth  (same level as flat keel)\n'
     '→  Tangent handle: HORIZONTAL  (parallel to flat keel / beam line)\n'
     '→  Spline departs horizontally — continuous with the flat keel'),
    (CP_COLORS[1], 'CP2  max beam  (y = hb,  z = hb_z)',
     '→  Constrain: y = hb,  z = hb_z\n'
     '→  Tangent handle: HORIZONTAL  (parallel to beam line)\n'
     '→  Spline passes through here horizontally — true widest point'),
    (CP_COLORS[2], 'CP3  deck edge  (y = deck_hb,  z = deck_z)',
     '→  Constrain: y = deck_hb,  z = deck_z\n'
     '→  Tangent handle: VERTICAL  (perpendicular to deck line)\n'
     '→  Spline arrives straight up into the deck edge'),
]

y_pos = 0.97
for col, title, desc in rules:
    rect_h = 0.165
    ax_g.add_patch(plt.Rectangle(
        (0.01, y_pos - rect_h), 0.97, rect_h - 0.01,
        transform=ax_g.transAxes,
        facecolor='white', edgecolor=col, lw=2.0, zorder=1))
    ax_g.text(0.04, y_pos - 0.022, f'● {title}',
              transform=ax_g.transAxes,
              fontsize=9.5, fontweight='bold', color=col, va='top')
    ax_g.text(0.05, y_pos - 0.052, desc,
              transform=ax_g.transAxes,
              fontsize=8, color='#333333', va='top', fontfamily='monospace')
    y_pos -= rect_h + 0.005

fig1.suptitle('Hull Section Control Points  —  best_fine_mesh.json  |  Front view (Y–Z)',
              fontsize=12, color='#1a1a2e', y=0.99)
fig1.tight_layout(rect=[0, 0, 1, 0.98])
fig1.savefig('section_control_points_detail.png', dpi=150,
             bbox_inches='tight', facecolor='#ffffff')
print('Saved -> section_control_points_detail.png')

# ════════════════════════════════════════════════════════════════════
# Figure 2: all 6 stations
# ════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), facecolor='#ffffff')
axes2 = axes2.flatten()

for ax, (title, params) in zip(axes2, stations.items()):
    draw_section(ax, params, title, annotate=False)
    cps, _, _ = station_control_points(**params)
    coords = '   '.join([f'CP{i+1}({c[0]:.0f}, {c[1]:.0f})'
                         for i, c in enumerate(cps)])
    ax.set_xlabel(coords, fontsize=5.5, color='#666666')

fig2.suptitle('All Stations — Control Points  (best_fine_mesh.json)\n'
              'Horizontal handles at keel & max beam  |  Vertical handle at deck edge',
              fontsize=12, color='#1a1a2e')
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig('section_all_stations.png', dpi=150,
             bbox_inches='tight', facecolor='#ffffff')
print('Saved -> section_all_stations.png')
