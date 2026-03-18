"""
Parametric Foiling Sailboat Hull Generator
==========================================
Generates and visualizes a high-performance foiling sailboat hull
using NURBS-inspired parametric curves (Bézier approximation).

Parameters control: waterline length, beam, draft, bow entry angle,
transom width, rocker, freeboard, and chine sharpness.

Outputs an interactive 3D visualization with cross-sections,
waterlines, and buttock lines — the classic naval architecture views.

Requirements: numpy, matplotlib
Usage: python parametric_hull.py
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────
# PARAMETRIC HULL DEFINITION
# ─────────────────────────────────────────────────────────────

class ParametricHull:
    """
    Defines a foiling sailboat hull via parametric curves.
    
    The hull is built from:
      - A keel line (rocker profile) in the XZ plane
      - A deck/sheer line 
      - Cross-section shapes at each station that blend
        between round-bilge (forward) and hard-chine (aft)
      - Bow entry angle and transom shape
    
    Coordinate system:
      X = longitudinal (0 = bow, LWL = stern)
      Y = transverse (0 = centerline, positive = starboard)
      Z = vertical (0 = waterline, positive = up)
    """
    
    def __init__(self, params=None):
        self.params = params or self.default_params()
        
    @staticmethod
    def default_params():
        """Default parameters for a ~6m high-performance foiling skiff."""
        return {
            'LWL': 6.0,           # Waterline length (m)
            'BWL': 1.8,           # Max beam at waterline (m)
            'Tc': 0.25,           # Canoe body draft (m)
            'bow_entry': 18.0,    # Bow half-entry angle (degrees)
            'transom_ratio': 0.55, # Transom beam / max beam
            'rocker': 0.08,       # Rocker height at stern (m, upward)
            'freeboard_bow': 0.70, # Freeboard at bow (m)
            'freeboard_mid': 0.45, # Freeboard at midships (m)
            'freeboard_stern': 0.50,# Freeboard at stern (m)
            'chine_blend': 0.6,   # 0=round bilge, 1=hard chine
            'max_beam_pos': 0.55, # Position of max beam (fraction of LWL)
            'deadrise_bow': 25.0, # Deadrise angle at bow (degrees)
            'deadrise_mid': 8.0,  # Deadrise angle at midships (degrees)
            'deadrise_stern': 5.0,# Deadrise angle at stern (degrees)
            'flare_bow': 20.0,    # Topsides flare at bow (degrees)
            'flare_stern': 5.0,   # Topsides flare at stern (degrees)
            'bow_overhang': 0.3,  # Bow deck overhang past DWL (m)
        }

    @staticmethod
    def moth_params():
        """
        Parameters matched to the 'Hungry Beaver' International Moth.
        Source: Beaver & Zseleczky, CSYS 2009 — full-scale tow tests.

        Hydrostatics at 240 lb design displacement (even keel):
          LWL = 11.0 ft   BWL = 13.6 in   T = 6.2 in
          CB  = 0.598     CP  = 0.661     CX = 0.904
          CWP = 0.734     LCB/LWL = 0.537
          Wetted surface = 16.85 sq ft

        All dimensions converted to metres.
        """
        return {
            'LWL': 3.353,          # 11.0 ft
            'BWL': 0.345,          # 13.6 in  — very narrow hull
            'Tc': 0.157,           # 6.2 in draft
            'bow_entry': 7.0,      # Fine bow entry (deg half-angle)
            'transom_ratio': 0.28, # Pinched Moth transom
            'rocker': 0.045,       # Moderate rocker kicked up at stern
            'freeboard_bow': 0.25, # ~10 in
            'freeboard_mid': 0.15, # ~6 in
            'freeboard_stern': 0.18,# ~7 in
            # CX=0.904 → sections are fuller than a semicircle;
            # a small chine_blend with low deadrise mid approximates a U-section.
            'chine_blend': 0.15,   # Slight hard-chine influence for fuller sections
            'max_beam_pos': 0.53,  # Just forward of LCB (paper: LCB/LWL=0.537)
            'deadrise_bow': 20.0,  # V at bow
            'deadrise_mid': 4.0,   # Near-vertical sides amidships → high CX
            'deadrise_stern': 2.0, # Flat aft
            'flare_bow': 12.0,     # Modest bow flare
            'flare_stern': 2.0,    # Minimal stern flare
            'bow_overhang': 0.0,   # No significant overhang
        }

    @staticmethod
    def optimized_racer_params():
        """
        Parameters mimicking the optimized racing hull construction plan.
        Features: very fine bow entry with pronounced bow overhang,
        pure round bilge, full U-shaped midship sections, narrow transom,
        and high prismatic coefficient — characteristic of an IOR-era or
        modern offshore performance racer (~9.5 m LWL).

        Distinctive shape qualities from the reference plan:
          - Near-needle bow (8° half-entry) with ~1.1 m bow overhang
          - Max beam at ~50% LWL; transom ~22% of max beam
          - Near-zero deadrise amidships for maximum section fullness
          - Low sheer at midships, sweeping high at bow and stern
          - Pure round bilge (chine_blend ≈ 0)
        """
        return {
            'LWL': 9.5,            # ~31 ft waterline length
            'BWL': 2.1,            # Moderate-narrow beam (L/B ≈ 4.5)
            'Tc': 0.52,            # Moderate canoe-body draft
            'bow_entry': 8.0,      # Very fine bow entry (deg half-angle)
            'transom_ratio': 0.22, # Narrow, pinched transom
            'rocker': 0.07,        # Gentle rocker, kicked up at stern
            'freeboard_bow': 1.05, # High bow freeboard
            'freeboard_mid': 0.52, # Low hollow sheer at midships
            'freeboard_stern': 0.72,# Moderate stern freeboard
            'chine_blend': 0.04,   # Essentially pure round bilge
            'max_beam_pos': 0.50,  # Max beam exactly at midships
            'deadrise_bow': 22.0,  # Moderately V'd bow sections
            'deadrise_mid': 2.5,   # Near-flat bottom — full sections
            'deadrise_stern': 1.5, # Flat aft underbody
            'flare_bow': 18.0,     # Pronounced bow flare (spray deflection)
            'flare_stern': 4.0,    # Minimal stern flare
            'bow_overhang': 1.10,  # 1.1 m bow overhang past DWL
        }
    
    def _bezier_curve(self, control_points, n_pts=50):
        """Evaluate a cubic Bézier curve."""
        t = np.linspace(0, 1, n_pts)
        cp = np.array(control_points)
        n = len(cp) - 1
        curve = np.zeros((n_pts, cp.shape[1]))
        for i, p in enumerate(cp):
            # Bernstein polynomial
            binom = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
            basis = binom * (t ** i) * ((1 - t) ** (n - i))
            curve += np.outer(basis, p)
        return curve
    
    def keel_line(self, n_pts=100):
        """
        Generate the keel (rocker) profile.
        Returns X, Z coordinates of the keel from bow to stern.
        Uses a cubic Bézier for smooth curvature.
        """
        p = self.params
        LWL = p['LWL']
        Tc = p['Tc']
        rocker = p['rocker']
        
        # Control points for keel profile [X, Z]
        cp = [
            [0.0, -Tc * 0.3],              # Bow (shallower)
            [LWL * 0.25, -Tc],             # Forward deepest
            [LWL * 0.65, -Tc * 0.95],      # Aft of midships
            [LWL, -Tc + rocker],            # Stern (kicked up)
        ]
        return self._bezier_curve(cp, n_pts)
    
    def waterline_shape(self, n_pts=100):
        """
        Generate the Design Waterline (DWL) — plan view half-breadth.
        Returns X, Y coordinates (starboard side).
        """
        p = self.params
        LWL = p['LWL']
        BWL_half = p['BWL'] / 2
        entry_rad = np.radians(p['bow_entry'])
        transom_half = BWL_half * p['transom_ratio']
        max_pos = p['max_beam_pos']
        
        # Forward section: bow to max beam
        cp_fwd = [
            [0.0, 0.0],                                    # Bow point
            [LWL * max_pos * 0.35, LWL * max_pos * 0.35 * np.tan(entry_rad)],
            [LWL * max_pos * 0.7, BWL_half * 0.85],
            [LWL * max_pos, BWL_half],                     # Max beam
        ]
        fwd = self._bezier_curve(cp_fwd, n_pts // 2)
        
        # Aft section: max beam to transom
        cp_aft = [
            [LWL * max_pos, BWL_half],                     # Max beam
            [LWL * (max_pos + (1 - max_pos) * 0.4), BWL_half * 0.98],
            [LWL * 0.88, transom_half * 1.1],
            [LWL, transom_half],                           # Transom
        ]
        aft = self._bezier_curve(cp_aft, n_pts // 2)
        
        return np.vstack([fwd, aft[1:]])
    
    def sheer_line(self, n_pts=100):
        """
        Generate the sheer (deck edge) profile in XZ.
        If bow_overhang > 0, the sheer starts forward of x=0 (the DWL bow),
        modelling the classic overhanging bow where the deck extends ahead
        of the waterline.  The tip height is set low (~30% of bow freeboard)
        so the overhang sweeps gracefully down to a fine point.
        """
        p = self.params
        LWL = p['LWL']
        oh = p.get('bow_overhang', 0.0)

        if oh > 0:
            # 5-point Bézier: overhang tip → DWL station → mid → midships → stern
            cp = [
                [-oh,          p['freeboard_bow'] * 0.28],   # bow tip (low)
                [-oh * 0.25,   p['freeboard_bow'] * 0.95],   # peak just aft of tip
                [LWL * 0.22,   p['freeboard_bow'] * 0.82],
                [LWL * 0.58,   p['freeboard_mid']],
                [LWL,          p['freeboard_stern']],
            ]
        else:
            cp = [
                [0.0,          p['freeboard_bow']],
                [LWL * 0.3,    p['freeboard_bow'] * 0.85],
                [LWL * 0.6,    p['freeboard_mid']],
                [LWL,          p['freeboard_stern']],
            ]
        return self._bezier_curve(cp, n_pts)
    
    def deck_halfbreadth(self, n_pts=100):
        """
        Deck edge in plan view (wider than waterline due to flare).
        The overhang section tapers to zero beam at the bow tip.
        """
        p = self.params
        LWL = p['LWL']
        oh = p.get('bow_overhang', 0.0)
        wl = self.waterline_shape(n_pts)
        sheer = self.sheer_line(n_pts)
        keel = self.keel_line(n_pts)

        # Deck beam at waterline stations (flare-expanded)
        x_frac = wl[:, 0] / LWL
        flare = np.radians(p['flare_bow'] * (1 - x_frac) + p['flare_stern'] * x_frac)
        sheer_z = np.interp(wl[:, 0], sheer[:, 0], sheer[:, 1])
        keel_z = np.interp(wl[:, 0], keel[:, 0], keel[:, 1])
        freeboard_height = sheer_z - keel_z
        deck_y = wl[:, 1] + freeboard_height * np.tan(flare) * 0.3

        if oh <= 0:
            return np.column_stack([wl[:, 0], deck_y])

        # Overhang section: x from -oh to 0, beam tapers from 0 to deck_y[0]
        n_oh = max(8, n_pts // 8)
        x_oh = np.linspace(-oh, 0.0, n_oh, endpoint=False)
        # Taper via a smooth curve (sin²) so the tip is truly needle-fine
        taper = np.sin(np.linspace(0, np.pi / 2, n_oh)) ** 2
        y_oh = deck_y[0] * taper

        x_full = np.concatenate([x_oh, wl[:, 0]])
        y_full = np.concatenate([y_oh, deck_y])
        return np.column_stack([x_full, y_full])
    
    def cross_section(self, x_pos, n_pts=30):
        """
        Generate a single cross-section at longitudinal position x_pos.
        Returns Y, Z coordinates (starboard side, keel to deck).
        
        Blends between round-bilge and hard-chine based on chine_blend
        and longitudinal position.
        """
        p = self.params
        LWL = p['LWL']
        x_frac = x_pos / LWL
        
        # Interpolate local parameters
        wl = self.waterline_shape(200)
        halfbeam = np.interp(x_pos, wl[:, 0], wl[:, 1])
        
        keel_profile = self.keel_line(200)
        keel_z = np.interp(x_pos, keel_profile[:, 0], keel_profile[:, 1])
        
        sheer_profile = self.sheer_line(200)
        sheer_z = np.interp(x_pos, sheer_profile[:, 0], sheer_profile[:, 1])
        
        deck_hb = self.deck_halfbreadth(200)
        deck_y = np.interp(x_pos, deck_hb[:, 0], deck_hb[:, 1])
        
        # Local deadrise angle
        deadrise = np.radians(
            p['deadrise_bow'] * (1 - x_frac)**2 +
            p['deadrise_mid'] * 2 * x_frac * (1 - x_frac) +
            p['deadrise_stern'] * x_frac**2
        )
        
        # Local chine sharpness (more chine toward stern)
        chine = p['chine_blend'] * x_frac
        
        # Build section from keel upward
        t = np.linspace(0, 1, n_pts)
        
        # Round bilge component (elliptical)
        y_round = halfbeam * np.sin(t * np.pi / 2)
        z_round = keel_z + (0 - keel_z) * (1 - np.cos(t * np.pi / 2))
        
        # Hard chine component (V-bottom + vertical sides)
        chine_y = halfbeam * 0.85  # chine location
        chine_z = keel_z + chine_y * np.tan(deadrise)
        
        y_chine = np.where(t < 0.5,
                          chine_y * (t / 0.5),
                          chine_y + (halfbeam - chine_y) * ((t - 0.5) / 0.5))
        z_chine = np.where(t < 0.5,
                          keel_z + y_chine * np.tan(deadrise),
                          chine_z + (0 - chine_z) * ((t - 0.5) / 0.5))
        
        # Blend round and chine
        y_wl = (1 - chine) * y_round + chine * y_chine
        z_wl = (1 - chine) * z_round + chine * z_chine
        
        # Add topsides (waterline to deck)
        t_top = np.linspace(0, 1, n_pts // 3)
        y_top = halfbeam + (deck_y - halfbeam) * t_top
        z_top = 0 + (sheer_z - 0) * t_top
        
        y_full = np.concatenate([y_wl, y_top[1:]])
        z_full = np.concatenate([z_wl, z_top[1:]])
        
        return y_full, z_full
    
    def generate_surface(self, n_stations=25, n_section_pts=30):
        """
        Generate the full 3D hull surface as a mesh.
        Returns X, Y, Z arrays (n_stations × n_section_pts).
        """
        p = self.params
        LWL = p['LWL']
        
        # Station positions (denser at bow and stern)
        t = np.linspace(0, 1, n_stations)
        x_stations = LWL * (0.5 - 0.5 * np.cos(np.pi * t))  # cosine spacing
        
        # Avoid exact 0 (degenerate bow point)
        x_stations[0] = LWL * 0.005
        
        Y = np.zeros((n_stations, n_section_pts + n_section_pts // 3 - 1))
        Z = np.zeros_like(Y)
        X = np.zeros_like(Y)
        
        for i, x in enumerate(x_stations):
            y, z = self.cross_section(x, n_section_pts)
            X[i, :] = x
            Y[i, :] = y
            Z[i, :] = z
        
        return X, Y, Z


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_hull_comprehensive(hull, save_path=None):
    """
    Create a comprehensive 4-panel hull visualization:
    1. 3D perspective view (both sides)
    2. Body plan (cross-sections)
    3. Profile (side view with keel and sheer)
    4. Half-breadth plan (waterlines from above)
    """
    p = hull.params
    LWL = p['LWL']
    
    fig = plt.figure(figsize=(18, 14), facecolor='#0a0e17')
    oh = p.get('bow_overhang', 0.0)
    fig.suptitle(
        f'Optimized Racing Hull  —  '
        f'LWL={LWL:.2f}m  BWL={p["BWL"]:.2f}m  '
        f'T={p["Tc"]:.3f}m  Entry={p["bow_entry"]:.0f}°  OH={oh:.2f}m',
        fontsize=14, color='#c8d6e5', fontweight='bold', y=0.98
    )
    
    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.25,
                           left=0.06, right=0.96, top=0.93, bottom=0.05)
    
    # Colors
    hull_color = '#1e90ff'
    hull_color_port = '#1874cc'
    grid_color = '#1a2235'
    text_color = '#8899aa'
    accent = '#00d4aa'
    
    # Generate surface
    X, Y, Z = hull.generate_surface(n_stations=30, n_section_pts=30)
    
    # ── Panel 1: 3D Perspective ──
    ax3d = fig.add_subplot(gs[0, 0], projection='3d', facecolor='#0a0e17')
    ax3d.set_title('3D perspective', color='#c8d6e5', fontsize=11, pad=10)
    
    # Plot starboard side
    ax3d.plot_surface(X, Y, Z, alpha=0.7, color=hull_color,
                      edgecolor=hull_color, linewidth=0.15, shade=True)
    # Plot port side (mirror)
    ax3d.plot_surface(X, -Y, Z, alpha=0.55, color=hull_color_port,
                      edgecolor=hull_color_port, linewidth=0.15, shade=True)
    
    # Waterplane
    wl = hull.waterline_shape(100)
    ax3d.plot(wl[:, 0], wl[:, 1], 0, color=accent, linewidth=1.5, alpha=0.8)
    ax3d.plot(wl[:, 0], -wl[:, 1], 0, color=accent, linewidth=1.5, alpha=0.8)
    
    # Keel line
    keel = hull.keel_line(100)
    ax3d.plot(keel[:, 0], np.zeros(100), keel[:, 1],
              color='#ff6b6b', linewidth=1.5, alpha=0.9)
    
    ax3d.set_xlabel('X (m)', color=text_color, fontsize=9)
    ax3d.set_ylabel('Y (m)', color=text_color, fontsize=9)
    ax3d.set_zlabel('Z (m)', color=text_color, fontsize=9)
    ax3d.view_init(elev=20, azim=-55)
    ax3d.set_box_aspect([LWL, p['BWL'] * 1.5, (p['freeboard_bow'] + p['Tc']) * 3])
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.tick_params(colors=text_color, labelsize=7)
    ax3d.grid(True, alpha=0.15)
    
    # ── Panel 2: Body Plan (cross-sections) ──
    ax_body = fig.add_subplot(gs[0, 1], facecolor='#0d1220')
    ax_body.set_title('Body plan (sections)', color='#c8d6e5', fontsize=11)
    
    n_show = 11
    t_vals = np.linspace(0, 1, n_show)
    x_positions = LWL * (0.5 - 0.5 * np.cos(np.pi * t_vals))
    x_positions[0] = LWL * 0.01
    
    cmap = plt.cm.viridis
    for i, x in enumerate(x_positions):
        y, z = hull.cross_section(x, 40)
        color = cmap(i / (n_show - 1))
        label = f'Stn {i} (x={x:.2f}m)'
        ax_body.plot(y, z, color=color, linewidth=1.2, alpha=0.85, label=label)
        ax_body.plot(-y, z, color=color, linewidth=1.2, alpha=0.85)
    
    ax_body.axhline(y=0, color=accent, linewidth=0.8, linestyle='--', alpha=0.5, label='DWL')
    ax_body.set_xlabel('Y — half breadth (m)', color=text_color, fontsize=9)
    ax_body.set_ylabel('Z (m)', color=text_color, fontsize=9)
    ax_body.set_aspect('equal')
    ax_body.legend(fontsize=6, loc='upper right', facecolor='#0d1220',
                   edgecolor='#1a2235', labelcolor=text_color, ncol=2)
    ax_body.tick_params(colors=text_color, labelsize=8)
    ax_body.grid(True, color=grid_color, linewidth=0.5)
    ax_body.spines[:].set_color('#1a2235')
    
    # ── Panel 3: Profile (side view) ──
    ax_prof = fig.add_subplot(gs[1, 0], facecolor='#0d1220')
    ax_prof.set_title('Profile (side view)', color='#c8d6e5', fontsize=11)

    oh = p.get('bow_overhang', 0.0)
    keel = hull.keel_line(200)
    sheer = hull.sheer_line(200)

    # Draw the hull envelope — shade between keel (below DWL) and sheer
    # The sheer may extend further forward than the keel (overhang), so
    # clip the fill to the overlapping x range.
    x_fill_min = max(keel[0, 0], sheer[0, 0])  # = 0 when overhang present
    keel_z_at_fill = np.interp(
        np.linspace(x_fill_min, LWL, 300), keel[:, 0], keel[:, 1])
    sheer_z_at_fill = np.interp(
        np.linspace(x_fill_min, LWL, 300), sheer[:, 0], sheer[:, 1])
    ax_prof.fill_between(
        np.linspace(x_fill_min, LWL, 300),
        keel_z_at_fill, sheer_z_at_fill,
        alpha=0.15, color=hull_color)

    ax_prof.plot(keel[:, 0], keel[:, 1], color='#ff6b6b', linewidth=2, label='Keel (rocker)')
    ax_prof.plot(sheer[:, 0], sheer[:, 1], color=accent, linewidth=2, label='Sheer line')

    # Draw bow overhang closing line (straight line from tip down to DWL stem)
    if oh > 0:
        tip_x = sheer[0, 0]   # = -oh
        tip_z = sheer[0, 1]
        ax_prof.plot([tip_x, 0.0], [tip_z, 0.0],
                     color=accent, linewidth=1.5, linestyle=':', alpha=0.7)

    ax_prof.axhline(y=0, color='#4488cc', linewidth=1, linestyle='--', alpha=0.6, label='DWL')

    # Station lines
    for i, x in enumerate(x_positions):
        ax_prof.axvline(x=x, color='#334455', linewidth=0.5, alpha=0.4)
    
    ax_prof.set_xlabel('X — longitudinal (m)', color=text_color, fontsize=9)
    ax_prof.set_ylabel('Z (m)', color=text_color, fontsize=9)
    ax_prof.set_aspect('equal')
    ax_prof.legend(fontsize=8, loc='upper right', facecolor='#0d1220',
                   edgecolor='#1a2235', labelcolor=text_color)
    ax_prof.tick_params(colors=text_color, labelsize=8)
    ax_prof.grid(True, color=grid_color, linewidth=0.5)
    ax_prof.spines[:].set_color('#1a2235')
    
    # ── Panel 4: Half-Breadth Plan (waterlines from above) ──
    ax_plan = fig.add_subplot(gs[1, 1], facecolor='#0d1220')
    ax_plan.set_title('Half-breadth plan (waterlines)', color='#c8d6e5', fontsize=11)
    
    # Plot waterlines at different heights
    z_levels = np.linspace(-p['Tc'] * 0.8, 0.1, 7)
    cmap2 = plt.cm.cool
    
    for j, z_level in enumerate(z_levels):
        y_at_z = []
        x_at_z = []
        for x in np.linspace(LWL * 0.01, LWL * 0.995, 150):
            y_sec, z_sec = hull.cross_section(x, 60)
            # Find y where z crosses z_level
            for k in range(len(z_sec) - 1):
                if (z_sec[k] - z_level) * (z_sec[k+1] - z_level) <= 0:
                    # Linear interpolation
                    frac = (z_level - z_sec[k]) / (z_sec[k+1] - z_sec[k] + 1e-12)
                    y_interp = y_sec[k] + frac * (y_sec[k+1] - y_sec[k])
                    y_at_z.append(y_interp)
                    x_at_z.append(x)
                    break
        
        if x_at_z:
            color = cmap2(j / max(len(z_levels) - 1, 1))
            lbl = f'WL z={z_level:.2f}m'
            ax_plan.plot(x_at_z, y_at_z, color=color, linewidth=1.2, alpha=0.8, label=lbl)
            ax_plan.plot(x_at_z, [-y for y in y_at_z], color=color,
                        linewidth=1.2, alpha=0.8)
    
    # DWL
    wl = hull.waterline_shape(200)
    ax_plan.plot(wl[:, 0], wl[:, 1], color=accent, linewidth=2, label='DWL')
    ax_plan.plot(wl[:, 0], -wl[:, 1], color=accent, linewidth=2)
    
    # Deck edge
    deck = hull.deck_halfbreadth(200)
    ax_plan.plot(deck[:, 0], deck[:, 1], color='#ff6b6b', linewidth=1.5,
                linestyle='--', alpha=0.7, label='Deck edge')
    ax_plan.plot(deck[:, 0], -deck[:, 1], color='#ff6b6b', linewidth=1.5,
                linestyle='--', alpha=0.7)
    
    ax_plan.set_xlabel('X — longitudinal (m)', color=text_color, fontsize=9)
    ax_plan.set_ylabel('Y — half breadth (m)', color=text_color, fontsize=9)
    ax_plan.set_aspect('equal')
    ax_plan.legend(fontsize=6, loc='upper left', facecolor='#0d1220',
                   edgecolor='#1a2235', labelcolor=text_color, ncol=2)
    ax_plan.tick_params(colors=text_color, labelsize=8)
    ax_plan.grid(True, color=grid_color, linewidth=0.5)
    ax_plan.spines[:].set_color('#1a2235')
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0a0e17', bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def export_stations_csv(hull, filepath, n_stations=21, n_pts=40):
    """
    Export full cross-section curves to CSV for CAD surface lofting.

    Each station is a single open curve running:
        port deck-edge → keel (centreline) → starboard deck-edge
    This is the format expected by Rhino Loft, FreeCAD, and Fusion 360
    when building a surface from section curves.

    Columns: station index, longitudinal position (m), Y (m), Z (m).
    Positive Y = starboard.  Z = 0 is the design waterline, negative = below.
    """
    p = hull.params
    LWL = p['LWL']

    # Cosine-spaced stations — denser at bow and stern for accuracy
    t_vals = np.linspace(0, 1, n_stations)
    x_positions = LWL * (0.5 - 0.5 * np.cos(np.pi * t_vals))
    x_positions[0] = LWL * 0.005
    x_positions[-1] = LWL * 0.995

    with open(filepath, 'w') as f:
        f.write("station,x_m,y_m,z_m\n")
        for i, x in enumerate(x_positions):
            y_stbd, z_stbd = hull.cross_section(x, n_pts)

            # Port side = mirror of starboard, reversed so the curve
            # flows continuously: port-deck → keel → stbd-deck
            y_port = -y_stbd[::-1]
            z_port = z_stbd[::-1]

            # Concatenate, dropping the duplicate keel point at the join
            y_full = np.concatenate([y_port, y_stbd[1:]])
            z_full = np.concatenate([z_port, z_stbd[1:]])

            for j in range(len(y_full)):
                f.write(f"{i},{x:.4f},{y_full[j]:.4f},{z_full[j]:.4f}\n")

    print(f"Exported {n_stations} stations -> {filepath}")


def export_waterlines_csv(hull, filepath, n_waterlines=8, n_pts=120):
    """
    Export waterline curves at evenly spaced heights for CAD.

    Waterlines run from starboard bow to starboard stern, then mirror
    back along the port side, forming closed plan-view curves.

    Columns: waterline index, Z height (m), X (m), Y (m).
    """
    p = hull.params
    LWL = p['LWL']

    # Z levels from just below keel to just above DWL
    z_levels = np.linspace(-p['Tc'] * 0.97, p['Tc'] * 0.08, n_waterlines)
    x_sample = np.linspace(LWL * 0.005, LWL * 0.995, n_pts)

    with open(filepath, 'w') as f:
        f.write("waterline,z_m,x_m,y_m\n")
        for wi, z_level in enumerate(z_levels):
            stbd_pts = []
            for x in x_sample:
                y_sec, z_sec = hull.cross_section(x, 60)
                for k in range(len(z_sec) - 1):
                    if (z_sec[k] - z_level) * (z_sec[k + 1] - z_level) <= 0:
                        frac = (z_level - z_sec[k]) / (z_sec[k + 1] - z_sec[k] + 1e-12)
                        y_i = y_sec[k] + frac * (y_sec[k + 1] - y_sec[k])
                        stbd_pts.append((x, y_i))
                        break

            if not stbd_pts:
                continue

            # Starboard side bow→stern
            for x, y in stbd_pts:
                f.write(f"{wi},{z_level:.4f},{x:.4f},{y:.4f}\n")
            # Port side stern→bow (closes the waterplane loop)
            for x, y in reversed(stbd_pts):
                f.write(f"{wi},{z_level:.4f},{x:.4f},{-y:.4f}\n")

    print(f"Exported {n_waterlines} waterlines  -> {filepath}")


def export_sections_dxf(hull, folder='dxf_sections', n_stations=11, n_pts=60):
    """
    Export each cross-section as an individual DXF file for Onshape sketch import.

    Onshape workflow
    ----------------
    For each station DXF:
      1. In Onshape, create a new Sketch on the plane perpendicular to the
         hull centreline at that station's X position.
         (Right-click the Front plane → Offset plane by the X value shown
          in the filename, e.g. station_05_x1234mm.dxf → offset 1234 mm.)
      2. Inside the sketch: Insert > DXF/DWG, select the file.
      3. Align the DXF origin to the sketch origin (DWL centreline).
      4. Repeat for every station, then use the Loft tool across all sketches.

    The DXF coordinate system
    -------------------------
      X (DXF) = Y (hull) = half-breadth, positive starboard
      Y (DXF) = Z (hull) = vertical,     0 = design waterline, up = positive

    Each file contains one open polyline running port-deck → keel → stbd-deck,
    plus a dashed DWL reference line and a dot at the centreline keel point.
    Units: millimetres (Onshape default on DXF import).
    """
    import ezdxf
    import os

    os.makedirs(folder, exist_ok=True)
    p = hull.params
    LWL = p['LWL']

    t_vals = np.linspace(0, 1, n_stations)
    x_positions = LWL * (0.5 - 0.5 * np.cos(np.pi * t_vals))
    x_positions[0]  = LWL * 0.005
    x_positions[-1] = LWL * 0.995

    for i, x in enumerate(x_positions):
        y_stbd, z_stbd = hull.cross_section(x, n_pts)

        # Full section: port-deck → keel → stbd-deck (mm)
        y_port = -y_stbd[::-1]
        z_port =  z_stbd[::-1]
        y_full = np.concatenate([y_port, y_stbd[1:]]) * 1000.0
        z_full = np.concatenate([z_port, z_stbd[1:]]) * 1000.0

        doc = ezdxf.new('R2010')
        doc.units = 4          # millimetres
        msp = doc.modelspace()

        # Section curve — solid line
        pts = [(float(y), float(z)) for y, z in zip(y_full, z_full)]
        msp.add_lwpolyline(pts, dxfattribs={'layer': 'SECTION', 'color': 1})

        # DWL reference line — dashed, spanning full beam
        beam_mm = float(y_stbd[-1]) * 1000.0
        msp.add_line(
            (-beam_mm * 1.05, 0.0), (beam_mm * 1.05, 0.0),
            dxfattribs={'layer': 'DWL', 'color': 5, 'linetype': 'DASHED'},
        )

        # Centreline
        msp.add_line(
            (0.0, float(z_full[len(z_full)//2]) - 5),
            (0.0, float(z_full[len(z_full)//2]) + 5),
            dxfattribs={'layer': 'CL', 'color': 3},
        )

        x_mm = int(round(x * 1000))
        fname = os.path.join(folder, f'station_{i:02d}_x{x_mm:04d}mm.dxf')
        doc.saveas(fname)

    print(f"Exported {n_stations} section DXFs -> {folder}/")
    print("  Filename encodes X offset, e.g. station_05_x1234mm.dxf")
    print("  -> create an offset sketch plane 1234 mm from the transom.")


def export_lines_plan_dxf(hull, filepath='lines_plan.dxf',
                           n_stations=11, n_waterlines=6, n_pts=80):
    """
    Export the complete lines plan as a single DXF — the three classic views
    arranged side by side as a reference drawing.

    Layout (all in mm, drawn in model space)
    -----------------------------------------
      Left panel   : Body plan  (cross-sections, Y vs Z)
      Centre panel : Profile    (keel & sheer, X vs Z)
      Right panel  : Half-breadth plan (waterlines, X vs Y from above)

    The three panels are offset horizontally by 'gap' so they don't overlap.
    This file is for visual reference only, not for direct Onshape import.
    Use export_sections_dxf() for the per-station files Onshape needs.

    Layers
    ------
      SECTION   : cross-section curves (body plan)
      KEEL      : keel / rocker line
      SHEER     : sheer line
      WATERLINE : waterlines (profile buttock crossing + half-breadth)
      DWL       : design waterline
      CL        : centreline / base reference
      GRID      : panel borders and labels
    """
    import ezdxf

    p = hull.params
    LWL  = p['LWL']
    BWL  = p['BWL']
    Tc   = p['Tc']

    # Scale to mm
    S = 1000.0
    lwl = LWL * S
    bwl = BWL * S
    tc  = Tc  * S

    gap = lwl * 0.12   # horizontal gap between panels

    # Panel origins (bottom-left corner reference)
    # Body plan: centred around Y=0, Z goes from -tc to freeboard
    body_ox = 0.0
    # Profile: to the right of body plan
    prof_ox = bwl + gap
    # Half-breadth: to the right of profile
    plan_ox = prof_ox + lwl + gap

    doc = ezdxf.new('R2010')
    doc.units = 4   # mm
    msp = doc.modelspace()

    # Station X positions (cosine-spaced)
    t_vals = np.linspace(0, 1, n_stations)
    x_pos  = LWL * (0.5 - 0.5 * np.cos(np.pi * t_vals))
    x_pos[0]  = LWL * 0.005
    x_pos[-1] = LWL * 0.995

    # ── Body plan ──────────────────────────────────────────────
    for x in x_pos:
        y_s, z_s = hull.cross_section(x, n_pts)
        y_p = -y_s[::-1];  z_p = z_s[::-1]
        y_f = np.concatenate([y_p, y_s[1:]]) * S
        z_f = np.concatenate([z_p, z_s[1:]]) * S
        pts = [(body_ox + float(y), float(z)) for y, z in zip(y_f, z_f)]
        msp.add_lwpolyline(pts, dxfattribs={'layer': 'SECTION', 'color': 1})

    # DWL and centreline for body plan
    msp.add_line((body_ox - bwl * 0.55, 0), (body_ox + bwl * 0.55, 0),
                 dxfattribs={'layer': 'DWL', 'color': 5, 'linetype': 'DASHED'})
    msp.add_line((body_ox, -tc * 1.1), (body_ox, p['freeboard_bow'] * S * 1.1),
                 dxfattribs={'layer': 'CL', 'color': 3, 'linetype': 'CENTER'})

    # ── Profile (side view) ─────────────────────────────────────
    keel  = hull.keel_line(200)
    sheer = hull.sheer_line(200)

    keel_pts  = [(prof_ox + float(pt[0]) * S, float(pt[1]) * S) for pt in keel]
    sheer_pts = [(prof_ox + float(pt[0]) * S, float(pt[1]) * S) for pt in sheer]

    msp.add_lwpolyline(keel_pts,  dxfattribs={'layer': 'KEEL',  'color': 2})
    msp.add_lwpolyline(sheer_pts, dxfattribs={'layer': 'SHEER', 'color': 6})

    # DWL across profile
    msp.add_line((prof_ox, 0), (prof_ox + lwl, 0),
                 dxfattribs={'layer': 'DWL', 'color': 5, 'linetype': 'DASHED'})

    # Station tick marks on profile
    for x in x_pos:
        kz = float(np.interp(x, keel[:, 0], keel[:, 1])) * S
        sz = float(np.interp(x, sheer[:, 0], sheer[:, 1])) * S
        msp.add_line((prof_ox + x * S, kz), (prof_ox + x * S, sz),
                     dxfattribs={'layer': 'GRID', 'color': 8})

    # ── Half-breadth plan (top view) ────────────────────────────
    z_levels = np.linspace(-Tc * 0.95, Tc * 0.05, n_waterlines)
    x_sample = np.linspace(LWL * 0.005, LWL * 0.995, 120)

    for z_lev in z_levels:
        stbd = []
        for x in x_sample:
            y_sec, z_sec = hull.cross_section(x, 60)
            for k in range(len(z_sec) - 1):
                if (z_sec[k] - z_lev) * (z_sec[k + 1] - z_lev) <= 0:
                    frac = (z_lev - z_sec[k]) / (z_sec[k + 1] - z_sec[k] + 1e-12)
                    y_i  = y_sec[k] + frac * (y_sec[k + 1] - y_sec[k])
                    stbd.append((x, y_i))
                    break
        if not stbd:
            continue
        # Starboard side
        pts_s = [(plan_ox + x * S,  y * S) for x, y in stbd]
        # Port side (mirror, drawn symmetrically below centreline)
        pts_p = [(plan_ox + x * S, -y * S) for x, y in reversed(stbd)]
        color = 4 if abs(z_lev) < 0.001 else 1
        layer = 'DWL' if abs(z_lev) < 0.001 else 'WATERLINE'
        msp.add_lwpolyline(pts_s, dxfattribs={'layer': layer, 'color': color})
        msp.add_lwpolyline(pts_p, dxfattribs={'layer': layer, 'color': color})

    # Centreline across half-breadth plan
    msp.add_line((plan_ox, -bwl * 0.55 * S), (plan_ox, bwl * 0.55 * S),
                 dxfattribs={'layer': 'CL', 'color': 3, 'linetype': 'CENTER'})

    doc.saveas(filepath)
    print(f"Exported lines plan DXF -> {filepath}")
    print("  Layers: SECTION  KEEL  SHEER  WATERLINE  DWL  CL  GRID")


def print_hydrostatics(hull):
    """Compute and print basic hydrostatic properties."""
    p = hull.params
    LWL = p['LWL']
    
    # Approximate displaced volume using trapezoidal integration
    n_stations = 50
    t_vals = np.linspace(0, 1, n_stations)
    x_positions = LWL * (0.5 - 0.5 * np.cos(np.pi * t_vals))
    x_positions[0] = LWL * 0.01
    
    areas = []
    waterplane_widths = []
    
    for x in x_positions:
        y_sec, z_sec = hull.cross_section(x, 60)
        # Area below waterline (z <= 0)
        area = 0
        wp_y = 0
        for k in range(len(z_sec) - 1):
            if z_sec[k] <= 0 and z_sec[k+1] <= 0:
                # Both below waterline — full trapezoid
                dz = abs(z_sec[k+1] - z_sec[k])
                area += 0.5 * (abs(y_sec[k]) + abs(y_sec[k+1])) * dz
                wp_y = max(wp_y, y_sec[k], y_sec[k+1])
            elif z_sec[k] <= 0 and z_sec[k+1] > 0:
                # Crosses waterline
                frac = (0 - z_sec[k]) / (z_sec[k+1] - z_sec[k] + 1e-12)
                y_wl = y_sec[k] + frac * (y_sec[k+1] - y_sec[k])
                wp_y = max(wp_y, y_wl)
                break
        
        # Simpson-friendly section area (approximate as half-ellipse)
        keel_z = z_sec[0]
        section_area = 0.5 * np.pi * wp_y * abs(keel_z) * 0.7  # ~70% of ellipse for typical section
        areas.append(section_area * 2)  # both sides
        waterplane_widths.append(wp_y * 2)
    
    # Trapezoidal integration for volume
    volume = np.trapezoid(areas, x_positions)
    displacement = volume * 1025  # kg (salt water)

    # Waterplane area
    wp_area = np.trapezoid(waterplane_widths, x_positions)
    
    # Prismatic coefficient
    max_area = max(areas)
    Cp = volume / (max_area * LWL) if max_area > 0 else 0
    
    # Block coefficient
    Cb = volume / (LWL * p['BWL'] * p['Tc']) if p['Tc'] > 0 else 0
    
    print("\n" + "=" * 55)
    print("  HYDROSTATIC ESTIMATES")
    print("=" * 55)
    print(f"  Waterline length (LWL):    {LWL:.2f} m")
    print(f"  Beam at waterline (BWL):   {p['BWL']:.2f} m")
    print(f"  Canoe body draft (Tc):     {p['Tc']:.3f} m")
    print(f"  Displaced volume:          {volume:.3f} m^3")
    print(f"  Displacement (salt):       {displacement:.0f} kg")
    print(f"  Waterplane area:           {wp_area:.2f} m^2")
    print(f"  Prismatic coeff (Cp):      {Cp:.3f}")
    print(f"  Block coeff (Cb):          {Cb:.3f}")
    print(f"  Beam/Length ratio:         {p['BWL']/LWL:.3f}")
    print(f"  Length/Disp ratio:         {LWL / volume**(1/3):.2f}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────
# MAIN — run the generator
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 56)
    print("  Optimized Racing Hull — Construction Plan Match")
    print("=" * 56)
    print("  Fine-entry round-bilge racer, LWL~9.5m, BWL~2.1m")
    print("  Bow overhang 1.1m  |  Transom ratio 0.22  |  Cp~0.62\n")

    # Build hull matching the optimized construction plan
    hull = ParametricHull(ParametricHull.optimized_racer_params())

    # Print hydrostatic summary
    print_hydrostatics(hull)

    # ── CAD export ──────────────────────────────────────────────
    export_stations_csv(hull, 'hull_stations.csv', n_stations=21, n_pts=40)
    export_waterlines_csv(hull, 'hull_waterlines.csv', n_waterlines=8, n_pts=120)

    # DXF exports — for direct Onshape sketch import
    export_sections_dxf(hull, folder='dxf_sections', n_stations=11, n_pts=60)
    export_lines_plan_dxf(hull, filepath='lines_plan.dxf',
                          n_stations=11, n_waterlines=6, n_pts=80)

    print("\n  --- Onshape workflow (DXF route) ---")
    print("  1. Open lines_plan.dxf in any viewer to see the full lines plan.")
    print("  2. For each file in dxf_sections/:")
    print("       station_05_x1234mm.dxf  ->  X offset = 1234 mm from transom")
    print("       a. Sketch > offset the Right plane by that X value")
    print("       b. Inside the sketch: Insert > DXF/DWG, pick the file")
    print("       c. Confirm origin = DWL / centreline intersection")
    print("  3. Repeat for all stations, then Loft through all sketches.")

    # Generate comprehensive visualization
    fig = plot_hull_comprehensive(hull, save_path='hull_design.png')
