"""
Hull from Measurements
======================
Generate a Moth hull from physical measurements taken off a real boat.

Workflow
--------
1. Run this script once — it creates moth_measurements.csv as a template.
2. Fill in the template with your real measurements (cm).
3. Run the script again — it reads your data, shows a preview, and exports
   measured_stations.csv + measured_waterlines.csv for Rhino/FreeCAD lofting.

Measurement columns (all in cm)
--------------------------------
station_x_cm      : distance from bow along centreline
half_beam_wl_cm   : half-beam at the waterline (0 at bow/stern)
depth_cm          : depth from waterline down to hull bottom (positive = below)
half_beam_deck_cm : half-beam at the deck edge / gunwale
freeboard_cm      : height from waterline up to deck (positive = above)

Orange dots on the plots show your raw measured points so you can check
the spline fit looks sensible before exporting.

Requirements: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import CubicSpline
import os
import sys

MEASUREMENTS_CSV  = "moth_measurements.csv"
OUT_STATIONS      = "measured_stations.csv"
OUT_WATERLINES    = "measured_waterlines.csv"
OUT_IMAGE         = "measured_hull.png"

# ─────────────────────────────────────────────────────────────
# TEMPLATE  (written on first run if no CSV exists)
# ─────────────────────────────────────────────────────────────

TEMPLATE = """\
# Moth Hull Measurements
# ---------------------
# Replace the placeholder numbers with measurements from your boat.
# All values in centimetres (cm).
#
# station_x_cm      - distance from bow tip along centreline
# half_beam_wl_cm   - half-beam at the waterline (0 at bow and stern)
# depth_cm          - depth from waterline to hull bottom (positive = below water)
# half_beam_deck_cm - half-beam at the gunwale / deck edge
# freeboard_cm      - height from waterline to deck top (positive = above water)
#
# Add or remove rows as needed — measure wherever is convenient.
# Keep rows sorted by station_x_cm (bow = 0, stern = largest value).
#
station_x_cm,half_beam_wl_cm,depth_cm,half_beam_deck_cm,freeboard_cm
0,0,0,1,24
20,4,4,8,22
40,8,8,13,20
70,12,12,16,18
100,15,14,18,17
140,17,15,20,16
170,17.2,15.5,20.3,15.5
200,17,15,20,16
240,14,13,18,17
280,9,9,14,18
310,4,5,9,20
335,0,0,2,22
"""


def write_template(path):
    with open(path, 'w') as f:
        f.write(TEMPLATE)
    print(f"Template created: {path}")
    print("Fill in your measurements, save the file, then run this script again.")


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

def load_measurements(path):
    """Read the CSV, skipping # comment lines. Returns dict of numpy arrays."""
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append(line)

    if len(rows) < 2:
        raise ValueError(f"Not enough data in {path} — need a header row + at least one measurement row.")

    header = [h.strip() for h in rows[0].split(',')]
    required = ['station_x_cm', 'half_beam_wl_cm', 'depth_cm',
                 'half_beam_deck_cm', 'freeboard_cm']
    for col in required:
        if col not in header:
            raise ValueError(f"Missing column '{col}' in {path}")

    data = {col: [] for col in required}
    for row in rows[1:]:
        vals = row.split(',')
        for col, val in zip(header, vals):
            if col in required:
                data[col].append(float(val.strip()))

    arrays = {k: np.array(v) for k, v in data.items()}

    # Sort by x in case rows are out of order
    order = np.argsort(arrays['station_x_cm'])
    return {k: v[order] for k, v in arrays.items()}


# ─────────────────────────────────────────────────────────────
# SPLINES
# ─────────────────────────────────────────────────────────────

def fit_splines(meas):
    """Cubic spline through each measured quantity vs. station x."""
    x = meas['station_x_cm']
    return {
        'half_beam_wl':   CubicSpline(x, meas['half_beam_wl_cm'],   bc_type='clamped'),
        'depth':          CubicSpline(x, meas['depth_cm'],           bc_type='clamped'),
        'half_beam_deck': CubicSpline(x, meas['half_beam_deck_cm'],  bc_type='clamped'),
        'freeboard':      CubicSpline(x, meas['freeboard_cm'],       bc_type='clamped'),
    }


# ─────────────────────────────────────────────────────────────
# CROSS-SECTION
# ─────────────────────────────────────────────────────────────

def cross_section(x_cm, splines, n_pts=40):
    """
    Build one cross-section at x_cm (starboard side, keel to deck).

    Below waterline  : quarter-ellipse from keel (0, -depth) to waterline (half_beam_wl, 0).
    Above waterline  : straight topsides from (half_beam_wl, 0) to (half_beam_deck, freeboard).

    Returns y_cm, z_cm  (z=0 is waterline, positive up, negative down).
    """
    hb_wl  = float(max(splines['half_beam_wl'](x_cm),   0.0))
    depth  = float(max(splines['depth'](x_cm),           0.0))
    hb_dk  = float(max(splines['half_beam_deck'](x_cm),  hb_wl))
    fb     = float(max(splines['freeboard'](x_cm),        0.0))

    # Below waterline — quarter ellipse
    t_sub = np.linspace(0, np.pi / 2, n_pts)
    y_sub = hb_wl * np.sin(t_sub)
    z_sub = -depth * np.cos(t_sub)      # starts at (0, -depth), ends at (hb_wl, 0)

    # Above waterline — straight topsides
    n_top = max(n_pts // 5, 4)
    t_top = np.linspace(0, 1, n_top)
    y_top = hb_wl + (hb_dk - hb_wl) * t_top
    z_top = fb * t_top

    # Concatenate, dropping duplicate waterline point at the join
    y = np.concatenate([y_sub, y_top[1:]])
    z = np.concatenate([z_sub, z_top[1:]])
    return y, z


# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────

def export_stations(splines, x_range, filepath, n_stations=21, n_pts=40):
    """
    Export full cross-section curves (port to starboard) for surface lofting.
    Format: station index, x (cm), y (cm), z (cm)
    """
    xs = np.linspace(x_range[0], x_range[1], n_stations)
    with open(filepath, 'w') as f:
        f.write("station,x_cm,y_cm,z_cm\n")
        for i, x in enumerate(xs):
            y_s, z_s = cross_section(x, splines, n_pts)
            # Mirror to get port side, reversed so curve runs port-deck -> keel -> stbd-deck
            y_p = -y_s[::-1]
            z_p =  z_s[::-1]
            y_full = np.concatenate([y_p, y_s[1:]])
            z_full = np.concatenate([z_p, z_s[1:]])
            for y, z in zip(y_full, z_full):
                f.write(f"{i},{x:.2f},{y:.3f},{z:.3f}\n")
    print(f"Exported {n_stations} stations -> {filepath}")


def export_waterlines(splines, x_range, filepath, n_wl=7, n_x=150):
    """
    Export waterline curves at evenly spaced heights.
    Format: waterline index, z (cm), x (cm), y (cm)
    Starboard side bow->stern, then port side stern->bow (closed loop).
    """
    x_sample = np.linspace(x_range[0], x_range[1], n_x)
    depths    = np.array([float(splines['depth'](x)) for x in x_sample])
    max_depth = float(np.max(depths))
    z_levels  = np.linspace(-max_depth * 0.95, 0.0, n_wl)

    with open(filepath, 'w') as f:
        f.write("waterline,z_cm,x_cm,y_cm\n")
        for wi, z_level in enumerate(z_levels):
            pts = []
            for x in x_sample:
                y_s, z_s = cross_section(x, splines, 60)
                for k in range(len(z_s) - 1):
                    if (z_s[k] - z_level) * (z_s[k + 1] - z_level) <= 0:
                        frac = (z_level - z_s[k]) / (z_s[k + 1] - z_s[k] + 1e-12)
                        y_i  = y_s[k] + frac * (y_s[k + 1] - y_s[k])
                        pts.append((x, float(y_i)))
                        break
            if not pts:
                continue
            for x, y in pts:
                f.write(f"{wi},{z_level:.3f},{x:.2f},{y:.3f}\n")
            for x, y in reversed(pts):
                f.write(f"{wi},{z_level:.3f},{x:.2f},{-y:.3f}\n")
    print(f"Exported {n_wl} waterlines -> {filepath}")


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_hull(meas, splines, save_path=None):
    x_min = float(meas['station_x_cm'][0])
    x_max = float(meas['station_x_cm'][-1])
    xs    = np.linspace(x_min, x_max, 300)

    fig = plt.figure(figsize=(16, 11), facecolor='#0a0e17')
    fig.suptitle(
        f'Hull from Measurements  |  '
        f'LOA = {x_max:.0f} cm  '
        f'Max beam = {2 * float(np.max(meas["half_beam_deck_cm"])):.0f} cm  '
        f'Max depth = {float(np.max(meas["depth_cm"])):.0f} cm',
        fontsize=13, color='#c8d6e5', fontweight='bold', y=0.98
    )

    gs    = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28,
                              left=0.07, right=0.96, top=0.93, bottom=0.06)
    hull_c = '#1e90ff'
    acc    = '#00d4aa'
    txt_c  = '#8899aa'
    grid_c = '#1a2235'
    dot_c  = '#ffaa33'   # measured data points

    # ── Profile ───────────────────────────────────────────────
    ax_prof = fig.add_subplot(gs[0, :], facecolor='#0d1220')
    ax_prof.set_title('Profile (side view)', color='#c8d6e5', fontsize=11)

    sheer_z = splines['freeboard'](xs)
    keel_z  = -splines['depth'](xs)

    ax_prof.fill_between(xs, keel_z, sheer_z, alpha=0.15, color=hull_c)
    ax_prof.plot(xs, keel_z,  color='#ff6b6b', lw=2,   label='Keel')
    ax_prof.plot(xs, sheer_z, color=acc,        lw=2,   label='Sheer')
    ax_prof.axhline(0, color='#4488cc', lw=1, ls='--', alpha=0.6, label='Waterline')

    # Show raw measured points so the user can verify the spline fit
    ax_prof.scatter(meas['station_x_cm'],  meas['freeboard_cm'],
                    color=dot_c, s=30, zorder=5, label='Measured points')
    ax_prof.scatter(meas['station_x_cm'], -meas['depth_cm'],
                    color=dot_c, s=30, zorder=5)

    ax_prof.set_xlabel('X from bow (cm)', color=txt_c, fontsize=9)
    ax_prof.set_ylabel('Z (cm)',          color=txt_c, fontsize=9)
    ax_prof.legend(fontsize=8, facecolor='#0d1220', edgecolor=grid_c, labelcolor=txt_c)
    ax_prof.tick_params(colors=txt_c, labelsize=8)
    ax_prof.grid(True, color=grid_c, lw=0.5)
    ax_prof.spines[:].set_color(grid_c)

    # ── Body plan ─────────────────────────────────────────────
    ax_body = fig.add_subplot(gs[1, 0], facecolor='#0d1220')
    ax_body.set_title('Body plan (cross-sections)', color='#c8d6e5', fontsize=11)

    n_show  = 9
    xs_show = np.linspace(x_min + (x_max - x_min) * 0.05,
                          x_max - (x_max - x_min) * 0.05, n_show)
    cmap = plt.cm.viridis
    for i, x in enumerate(xs_show):
        y, z = cross_section(x, splines, 40)
        c = cmap(i / (n_show - 1))
        ax_body.plot( y, z, color=c, lw=1.4, alpha=0.9)
        ax_body.plot(-y, z, color=c, lw=1.4, alpha=0.9)

    ax_body.axhline(0, color='#4488cc', lw=0.8, ls='--', alpha=0.5, label='Waterline')
    ax_body.set_xlabel('Y - half breadth (cm)', color=txt_c, fontsize=9)
    ax_body.set_ylabel('Z (cm)',                 color=txt_c, fontsize=9)
    ax_body.set_aspect('equal')
    ax_body.legend(fontsize=8, facecolor='#0d1220', edgecolor=grid_c, labelcolor=txt_c)
    ax_body.tick_params(colors=txt_c, labelsize=8)
    ax_body.grid(True, color=grid_c, lw=0.5)
    ax_body.spines[:].set_color(grid_c)

    # ── Plan view (waterlines) ────────────────────────────────
    ax_plan = fig.add_subplot(gs[1, 1], facecolor='#0d1220')
    ax_plan.set_title('Plan view (waterlines)', color='#c8d6e5', fontsize=11)

    max_depth = float(np.max(splines['depth'](xs)))
    z_levels  = np.linspace(-max_depth * 0.9, 0.0, 7)
    x_sample  = np.linspace(x_min, x_max, 150)
    cmap2     = plt.cm.cool

    for j, z_level in enumerate(z_levels):
        pts_x, pts_y = [], []
        for x in x_sample:
            y_s, z_s = cross_section(x, splines, 60)
            for k in range(len(z_s) - 1):
                if (z_s[k] - z_level) * (z_s[k + 1] - z_level) <= 0:
                    frac = (z_level - z_s[k]) / (z_s[k + 1] - z_s[k] + 1e-12)
                    y_i  = y_s[k] + frac * (y_s[k + 1] - y_s[k])
                    pts_x.append(x)
                    pts_y.append(float(y_i))
                    break
        if pts_x:
            c = cmap2(j / max(len(z_levels) - 1, 1))
            ax_plan.plot(pts_x,  pts_y,             color=c, lw=1.2, alpha=0.8,
                         label=f'z={z_level:.1f}cm')
            ax_plan.plot(pts_x, [-y for y in pts_y], color=c, lw=1.2, alpha=0.8)

    # Deck edge spline
    deck_y = splines['half_beam_deck'](xs)
    ax_plan.plot(xs,  deck_y, color='#ff6b6b', lw=1.5, ls='--', alpha=0.7, label='Deck edge')
    ax_plan.plot(xs, -deck_y, color='#ff6b6b', lw=1.5, ls='--', alpha=0.7)

    # Measured WL beam points
    ax_plan.scatter(meas['station_x_cm'],  meas['half_beam_wl_cm'],
                    color=dot_c, s=25, zorder=5, label='Measured WL beam')
    ax_plan.scatter(meas['station_x_cm'], -meas['half_beam_wl_cm'],
                    color=dot_c, s=25, zorder=5)

    ax_plan.set_xlabel('X from bow (cm)', color=txt_c, fontsize=9)
    ax_plan.set_ylabel('Y (cm)',          color=txt_c, fontsize=9)
    ax_plan.set_aspect('equal')
    ax_plan.legend(fontsize=6, facecolor='#0d1220', edgecolor=grid_c,
                   labelcolor=txt_c, ncol=2, loc='upper left')
    ax_plan.tick_params(colors=txt_c, labelsize=8)
    ax_plan.grid(True, color=grid_c, lw=0.5)
    ax_plan.spines[:].set_color(grid_c)

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0a0e17', bbox_inches='tight')
        print(f"Saved -> {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # First run: create the template CSV and stop
    if not os.path.exists(MEASUREMENTS_CSV):
        print(f"No measurement file found.")
        write_template(MEASUREMENTS_CSV)
        sys.exit(0)

    print(f"Loading: {MEASUREMENTS_CSV}")
    meas = load_measurements(MEASUREMENTS_CSV)
    n    = len(meas['station_x_cm'])
    x0   = meas['station_x_cm'][0]
    x1   = meas['station_x_cm'][-1]
    print(f"  {n} stations  |  x = {x0:.0f} to {x1:.0f} cm")

    splines  = fit_splines(meas)
    x_range  = (x0, x1)

    export_stations(splines,   x_range, OUT_STATIONS,   n_stations=21, n_pts=40)
    export_waterlines(splines, x_range, OUT_WATERLINES, n_wl=7, n_x=150)

    plot_hull(meas, splines, save_path=OUT_IMAGE)
