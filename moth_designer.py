"""
Moth Hull Designer
==================
Interactive tool for exploring Moth hull shapes.
LWL = 3355 mm — fixed by the International Moth class rule.

Sliders
-------
  Max half-beam (mm)  100 – 280   how wide the hull is at its widest point
  Bow rocker (mm)       0 – 100   keel rise from deepest point up to bow
  Stern rocker (mm)     0 – 120   keel rise from deepest point up to stern
  Bow radius          0.5 – 3.0   bow plan-view sharpness  (low=full, high=fine)
  Stern radius        0.5 – 3.0   stern plan-view sharpness (low=full, high=fine)

Requirements: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ─── Fixed class-rule / structural constants ──────────────────
LWL       = 3355   # mm — Moth class rule maximum
MAX_DEPTH = 140    # mm — canoe body draft (fixed)
FREEBOARD = 195    # mm — sheer height above waterline (fixed)
DEEP_POS  = 0.38   # fraction of LWL where keel is deepest (bow=0, stern=1)
BEAM_POS  = 0.50   # fraction of LWL where beam is maximum

# ─── Default slider values ────────────────────────────────────
DEFAULTS = dict(
    half_beam    = 175,   # mm
    bow_rocker   =  32,   # mm
    stern_rocker =  58,   # mm
    bow_radius   = 2.0,   # dimensionless shape exponent
    stern_radius = 1.8,
)

# ─────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────

def keel_rise(x, bow_rocker, stern_rocker):
    """
    How far the keel is above its deepest point at position x (mm).
    Uses a parabolic arc on each side of the deepest point so the
    minimum is smooth and the derivative is zero at x = DEEP_POS * LWL.
    """
    x  = np.asarray(x, dtype=float)
    xd = DEEP_POS * LWL
    fwd = bow_rocker   * (1.0 - np.clip(x, 0, xd) / xd) ** 2
    aft = stern_rocker * (np.clip(x - xd, 0, None) / (LWL - xd)) ** 2
    return np.where(x <= xd, fwd, aft)


def half_beam_wl(x, max_hb, bow_r, stern_r):
    """
    Half-beam at the waterline at position x (mm).
    Power-law taper on each side of the maximum-beam station.
      bow_r / stern_r > 1  → fine entry/exit
      bow_r / stern_r < 1  → full entry/exit
    """
    x     = np.asarray(x, dtype=float)
    x_max = BEAM_POS * LWL
    fwd = max_hb * (np.clip(x,       0, x_max)     / x_max)          ** bow_r
    aft = max_hb * (np.clip(LWL - x, 0, LWL - x_max) / (LWL - x_max)) ** stern_r
    return np.where(x <= x_max, fwd, aft)


def cross_section(x, max_hb, bow_rocker, stern_rocker, bow_r, stern_r, n=40):
    """
    One cross-section at longitudinal position x.
    Below waterline : quarter-ellipse (round bilge).
    Above waterline : vertical topsides (Moth style).
    Returns y (mm), z (mm) for the starboard side, keel → deck.
    z = 0 is the waterline; negative = submerged.
    """
    hb    = float(half_beam_wl(np.array([float(x)]), max_hb, bow_r, stern_r)[0])
    depth = float(MAX_DEPTH - keel_rise(np.array([float(x)]), bow_rocker, stern_rocker)[0])
    depth = max(depth, 0.0)

    t     = np.linspace(0, np.pi / 2, n)
    y_sub = hb * np.sin(t)
    z_sub = -depth * np.cos(t)

    n_top = max(n // 5, 3)
    y_top = np.full(n_top, hb)
    z_top = np.linspace(0.0, FREEBOARD, n_top)

    return (np.concatenate([y_sub, y_top[1:]]),
            np.concatenate([z_sub, z_top[1:]]))


# ─────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────

C = dict(
    bg     = '#0a0e17',
    panel  = '#0d1220',
    grid   = '#1a2235',
    text   = '#8899aa',
    hull   = '#1e90ff',
    accent = '#00d4aa',
    keel   = '#ff6b6b',
    wl     = '#4488cc',
)


def _style(ax):
    ax.set_facecolor(C['panel'])
    ax.tick_params(colors=C['text'], labelsize=7)
    ax.grid(True, color=C['grid'], lw=0.5)
    ax.spines[:].set_color(C['grid'])


def redraw(axes, title_txt, p):
    ax_prof, ax_plan, ax_body = axes
    max_hb = p['half_beam']
    brock  = p['bow_rocker']
    srock  = p['stern_rocker']
    brad   = p['bow_radius']
    srad   = p['stern_radius']

    x = np.linspace(0, LWL, 500)

    # ── Profile ───────────────────────────────────────────────
    ax_prof.cla()
    _style(ax_prof)

    kz  = -(MAX_DEPTH - keel_rise(x, brock, srock))
    shz = np.full_like(x, FREEBOARD)

    ax_prof.fill_between(x, kz, shz, alpha=0.15, color=C['hull'])
    ax_prof.plot(x, kz,  color=C['keel'],   lw=2,   label='Keel')
    ax_prof.plot(x, shz, color=C['accent'], lw=2,   label='Sheer')
    ax_prof.axhline(0,   color=C['wl'],     lw=1,   ls='--', alpha=0.7, label='Waterline')

    ax_prof.set_title('Profile', color='#c8d6e5', fontsize=10, pad=5)
    ax_prof.set_xlabel('X from bow (mm)', color=C['text'], fontsize=8)
    ax_prof.set_ylabel('Z (mm)',          color=C['text'], fontsize=8)
    ax_prof.legend(fontsize=7, facecolor=C['panel'], edgecolor=C['grid'], labelcolor=C['text'])
    ax_prof.set_xlim(-80, LWL + 80)

    # ── Plan view ─────────────────────────────────────────────
    ax_plan.cla()
    _style(ax_plan)

    hb_arr = half_beam_wl(x, max_hb, brad, srad)
    ax_plan.fill_between(x,  hb_arr, -hb_arr, alpha=0.15, color=C['hull'])
    ax_plan.plot(x,  hb_arr, color=C['accent'], lw=2)
    ax_plan.plot(x, -hb_arr, color=C['accent'], lw=2)

    # Waterlines at several submerged heights
    x_s      = np.linspace(0, LWL, 100)
    z_levels = np.linspace(-MAX_DEPTH * 0.9, -MAX_DEPTH * 0.1, 6)
    cmap_wl  = plt.cm.cool
    for ji, zl in enumerate(z_levels):
        px, py = [], []
        for xi in x_s:
            ys, zs = cross_section(xi, max_hb, brock, srock, brad, srad, 50)
            for k in range(len(zs) - 1):
                if (zs[k] - zl) * (zs[k + 1] - zl) <= 0:
                    f = (zl - zs[k]) / (zs[k + 1] - zs[k] + 1e-12)
                    px.append(xi)
                    py.append(float(ys[k] + f * (ys[k + 1] - ys[k])))
                    break
        if px:
            c = cmap_wl(ji / max(len(z_levels) - 1, 1))
            ax_plan.plot(px,  py,             color=c, lw=0.9, alpha=0.75)
            ax_plan.plot(px, [-y for y in py], color=c, lw=0.9, alpha=0.75)

    ax_plan.set_title('Plan view (waterlines)', color='#c8d6e5', fontsize=10, pad=5)
    ax_plan.set_xlabel('X from bow (mm)', color=C['text'], fontsize=8)
    ax_plan.set_ylabel('Y (mm)',          color=C['text'], fontsize=8)
    ax_plan.set_aspect('equal')
    ax_plan.set_xlim(-80, LWL + 80)

    # ── Body plan ─────────────────────────────────────────────
    ax_body.cla()
    _style(ax_body)

    # Forebody sections on right, afterbody on left (conventional)
    n_sec = 10
    cmap_sec = plt.cm.viridis
    for i in range(n_sec):
        frac = (i + 0.5) / n_sec
        xi   = frac * LWL
        ys, zs = cross_section(xi, max_hb, brock, srock, brad, srad, 40)
        c = cmap_sec(frac)
        if frac <= 0.5:
            ax_body.plot( ys, zs, color=c, lw=1.4, alpha=0.9)  # forebody right
        else:
            ax_body.plot(-ys, zs, color=c, lw=1.4, alpha=0.9)  # afterbody left

    ax_body.axhline(0,  color=C['wl'],   lw=0.8, ls='--', alpha=0.5)
    ax_body.axvline(0,  color=C['grid'], lw=0.6, alpha=0.6)
    ax_body.set_title('Body plan  (fwd right  |  aft left)', color='#c8d6e5', fontsize=10, pad=5)
    ax_body.set_xlabel('Y (mm)', color=C['text'], fontsize=8)
    ax_body.set_ylabel('Z (mm)', color=C['text'], fontsize=8)
    ax_body.set_aspect('equal')

    # Update info line at top
    title_txt.set_text(
        f'Moth Hull Designer  |  LWL = {LWL} mm (fixed)  |  '
        f'Total beam = {2 * max_hb:.0f} mm   '
        f'Bow rocker = {brock:.0f} mm   '
        f'Stern rocker = {srock:.0f} mm'
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(17, 11), facecolor=C['bg'])

    title_txt = fig.text(
        0.5, 0.975, '', ha='center', va='top',
        color='#c8d6e5', fontsize=11, fontweight='bold'
    )

    # ── Axes layout ───────────────────────────────────────────
    #   [left, bottom, width, height]  (all 0–1 in figure coords)
    ax_prof = fig.add_axes([0.05, 0.62, 0.90, 0.32])
    ax_plan = fig.add_axes([0.05, 0.36, 0.42, 0.21])
    ax_body = fig.add_axes([0.54, 0.34, 0.42, 0.24])

    # ── Sliders ───────────────────────────────────────────────
    slider_defs = [
        # (key,           label,                  vmin, vmax,  valinit)
        ('half_beam',    'Max half-beam (mm)',    100,  280,   DEFAULTS['half_beam']),
        ('bow_rocker',   'Bow rocker (mm)',          0,  100,   DEFAULTS['bow_rocker']),
        ('stern_rocker', 'Stern rocker (mm)',         0,  120,   DEFAULTS['stern_rocker']),
        ('bow_radius',   'Bow radius',             0.5,  3.0,   DEFAULTS['bow_radius']),
        ('stern_radius', 'Stern radius',           0.5,  3.0,   DEFAULTS['stern_radius']),
    ]

    SL_L, SL_W, SL_H, SL_GAP = 0.20, 0.65, 0.022, 0.007
    SL_BOT_TOP = 0.295   # bottom of the topmost slider

    sliders = {}
    for i, (key, label, vmin, vmax, vdefault) in enumerate(slider_defs):
        bot   = SL_BOT_TOP - i * (SL_H + SL_GAP)
        ax_sl = fig.add_axes([SL_L, bot, SL_W, SL_H], facecolor='#1a2235')
        sl    = Slider(ax_sl, label, vmin, vmax, valinit=vdefault,
                       color='#1e90ff', track_color='#0d1220')
        sl.label.set_color('#c8d6e5')
        sl.label.set_fontsize(8)
        sl.valtext.set_color(C['accent'])
        sl.valtext.set_fontsize(8)
        sliders[key] = sl

    # ── Reset button ──────────────────────────────────────────
    ax_btn = fig.add_axes([0.875, 0.015, 0.08, 0.028])
    btn    = Button(ax_btn, 'Reset', color='#1a2235', hovercolor='#334455')
    btn.label.set_color('#c8d6e5')
    btn.label.set_fontsize(8)

    # ── Hint text ─────────────────────────────────────────────
    fig.text(
        0.05, 0.015,
        'Bow/Stern radius:  low value = full/round entry   |   high value = fine/sharp entry',
        color=C['text'], fontsize=7.5, va='bottom'
    )

    axes = (ax_prof, ax_plan, ax_body)

    def get_params():
        return {k: float(sl.val) for k, sl in sliders.items()}

    def update(_):
        redraw(axes, title_txt, get_params())
        fig.canvas.draw_idle()

    def reset(_):
        for key, sl in sliders.items():
            sl.set_val(DEFAULTS[key])

    for sl in sliders.values():
        sl.on_changed(update)
    btn.on_clicked(reset)

    redraw(axes, title_txt, DEFAULTS)
    plt.show()


if __name__ == '__main__':
    main()
