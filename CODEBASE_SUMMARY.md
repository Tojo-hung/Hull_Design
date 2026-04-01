# Moth Hull Designer — Codebase Summary

This document describes the desktop application architecture and geometry system
for use when onboarding a new Claude instance to work on this codebase.

---

## Project Structure

```
Hull_Design/
├── run.py                      # Entry point: python run.py
├── hull_design.json            # Current working design parameters
├── best_hull.json              # Best CFD-optimised design
├── hull_optimizer.py           # CMA-ES CFD optimizer (runs in WSL)
├── openfoam_template/          # OpenFOAM interFoam case template
├── moth_designer/              # Desktop app (PyQt5)
│   ├── app.py                  # Main window, UI, event handling
│   ├── geometry.py             # All geometry + hydrostatics (pure numpy)
│   ├── config.py               # Constants and parameter defaults
│   └── exports.py              # STL, DXF, TXT export
└── hull_web/                   # Web app (Streamlit) — separate, do not modify
    ├── streamlit_app.py
    ├── geometry.py             # Copy of geometry (standalone, no package imports)
    └── config.py
```

---

## Parameter System

All hull geometry is controlled by a flat Python dict. Keys follow this naming convention:

```python
params = {
    # Transom (stern end, x = LWL = 3355 mm)
    'transom_half_beam': 195.0,  # mm — half width at transom
    'transom_draft':      65.0,  # mm — depth at transom
    'transom_sheer':     200.0,  # mm — deck height at transom
    'transom_keel_w':    100.0,  # mm — flat keel width at transom

    # Bow (x = 0)
    'bow_draft':   75.0,  # mm — depth at bow
    'bow_sheer':  150.0,  # mm — deck height at bow

    # 4 internal control points P1–P4 (desktop app / optimizer)
    # P1 is near bow, P4 is near stern
    'p1_x':   335.0,   # mm — longitudinal position from bow
    'p1_hb':   90.0,   # mm — half-beam (max width of cross-section)
    'p1_d':   110.0,   # mm — draft (depth below waterline)
    'p1_dz':  150.0,   # mm — deck height
    'p1_kw':    5.0,   # mm — flat keel width
    'p1_hz':    0.0,   # mm — z-height of max-beam point (0 = waterline)

    # p2_, p3_, p4_ follow same pattern...
}
```

**Important:** The web app (`hull_web/`) has 5 control points (P1–P5).
The desktop app and CFD optimizer use only P1–P4 (P5 was removed to reduce
parameter count for optimisation). Do not conflate the two.

Parameters are saved/loaded as JSON (`hull_design.json`, `best_hull.json`).
Defaults are defined in `moth_designer/config.py`.

---

## Geometry Pipeline (`moth_designer/geometry.py`)

### Step 1: `build_ctrl(params)` — Build longitudinal control arrays

Takes the parameter dict and builds 1D arrays along the hull length (x-axis).
Adds bow (x=0) and transom (x=LWL) endpoints to the 4 internal stations.
Sorts stations by x position.

Returns:
```python
ctrl_x,    # x positions: [0, p1_x, p2_x, p3_x, p4_x, LWL]
beam_hb,   # half-beam at each station
keel_d,    # draft at each station
deck_hb,   # deck half-beam (= beam_hb)
sheer_z,   # deck height at each station
keel_w,    # flat keel width at each station
hb_z,      # z-height of max-beam point at each station
order,     # sort indices
sx, shb    # sorted station x and half-beam arrays
```

### Step 2: `lagrange(x_eval, ctrl_x, ctrl_y)` — Longitudinal interpolation

**Not actually Lagrange** — uses **PCHIP** (Piecewise Cubic Hermite Interpolating
Polynomial) via `scipy.interpolate.PchipInterpolator`. Named "lagrange" for
historic compatibility. Do not rename.

PCHIP is monotone-preserving and locally-influenced — no oscillation between
control points. Used to interpolate all hull properties (beam, draft, deck
height, keel width, hz) at any x position along the hull.

### Step 3: `cross_section(hb, depth, ...)` — Cross-section shape

Generates the (y, z) points of a half cross-section (starboard side) at a
given station. Two implementations:

**`cross_section()`** — quarter-ellipse bilge + straight topsides:
```
y_bilge = keel_w + (hb - keel_w) × sin(t)      t: 0 → π/2
z_bilge = hb_z  + (-depth - hb_z) × cos(t)

y_top = linear from hb to deck_hb
z_top = linear from hb_z to deck_z
```
- Flat keel prepended if keel_w > 0: horizontal line from (0, -depth) to (keel_w, -depth)
- This is used for STL generation and hydrostatics

**`cross_section_spline()`** — cubic spline bilge + vertical topsides:
- Uses CubicSpline with dy/dt=0 at keel (symmetry) and dy/dt=0 at max beam
- Topsides are strictly vertical (x=hb from hb_z to deck_z)
- Used for display in the app (smoother visual)

### Step 4: Hydrostatics

**`displaced_volume(z_wl, ctrl_x, beam_hb, keel_d, ...)`**
Numerically integrates cross-section areas along the hull at waterline z_wl.
Uses Simpson's rule over N_X stations.

**`find_disp_waterline(target_L, ...)`**
Bisection search to find the waterline z that gives target displacement (130L).
Used to position the hull correctly in the OpenFOAM domain (shifts hull so
130L waterline lands at z=0).

**`hydrostatic_coefficients(...)`**
Returns Cb, Cp, Cm, LCB, wetted area, waterplane area.

---

## Cross-Section Anatomy

```
z (up)
│
deck_z ──────────────●  deck edge (y=deck_hb)
                    /│
                   / │  straight topsides
                  /  │
hb_z   ──────────●   │  max beam point (y=hb)
                  \  │
                   \ │  quarter-ellipse bilge
                    \│
-depth ─────────────●──────── y (outboard)
       centreline  keel_w    hb
       (y=0)       (flat keel end)
```

Parameters control:
- `hb` — how wide the hull is (max beam)
- `depth` — how deep the keel is
- `keel_w` — how wide the flat keel bottom is (0 = pointed keel)
- `hb_z` — whether the widest point is at or above the waterline (0 = at waterline)
- `deck_hb` / `deck_z` — deck edge position

---

## Desktop App (`moth_designer/app.py`)

**Main window class:** `HullDesigner(QMainWindow)`

**Layout:** Two-panel — left input panel (fixed 310px), right plot panel (stretchy).

**Input panel sections:**
- CONTROL POINTS — x, half-beam, depth for P1–P4
- DECK & KEEL — deck height, keel width, hz offset for P1–P4
- BOW & TRANSOM — bow/transom specific params
- Buttons: Save, Load Config, Print Config, 3D View, Export

**Plot panel:** Two pyqtgraph plots stacked vertically:
- Top: Body plan (cross-sections, front view, Y-Z plane)
- Bottom: Profile / plan view (side and top view, X-Z and X-Y planes)

**Update flow:**
1. User edits a field → `returnPressed` or `editingFinished` fires
2. `_make_updater(key, vmin, vmax)` clamps value and updates `self.params[key]`
3. `_redraw()` is called — recomputes all geometry and redraws plots
4. Config is auto-saved to `hull_design.json` on every change

**`_redraw()`** calls `build_ctrl()` then evaluates geometry at N_SEC cross-section
stations and N_WL waterline heights for the body plan, plus a dense longitudinal
mesh for the profile view.

**3D view:** `Hull3DView` — separate QMainWindow using pyqtgraph.opengl (GLViewWidget).
Known OpenGL issue on some Windows drivers: `glClearColor` error on startup.
Fixed by moving `setBackgroundColor` to after `setCentralWidget`.

---

## CFD Optimizer (`hull_optimizer.py`)

- Runs in WSL: `source ~/hull-env/bin/activate && python3 hull_optimizer.py`
- Uses CMA-ES (covariance matrix adaptation evolution strategy)
- Two-stage: coarse mesh (fast, ~2 min/eval) → fine mesh (accurate, ~5-8 min/eval)
- Imports geometry from `moth_designer.geometry` to build STL
- Shifts hull so 130L displacement waterline is at z=0 in OpenFOAM domain
- Search space: 25 parameters (P1–P4 station params + transom + bow)
- Results saved to `best_hull.json` when a new best is found
- Logs to `optimization_log.csv`

**OpenFOAM setup:**
- Solver: interFoam (two-phase VoF, water + air free surface)
- Speed: 7 knots = 3.601 m/s
- 8 cores, MPI with `--use-hwthread-cpus`
- Template at `openfoam_template/`

---

## Key Constants (`moth_designer/config.py`)

```python
LWL       = 3355   # mm — Moth class rule waterline length (fixed)
MAX_DEPTH = 350    # mm — maximum allowed draft
FREEBOARD = 100    # mm — default deck height above waterline
TARGET_DISP_L = 130  # litres — target displacement
```

---

## File I/O

- `hull_design.json` — auto-saved on every parameter change (desktop app)
- `best_hull.json` — written by optimizer when new best drag found
- On startup, desktop app loads `hull_design.json`; optimizer prefers `best_hull.json`
- JSON keys match parameter dict keys exactly (except transom params in web app
  which use a `_KEY_MAP` for short session_state keys)

---

## Common Gotchas

1. **`lagrange()` is actually PCHIP** — the function name is misleading, do not change it.
2. **`deck_hb` always equals `beam_hb`** — set in `build_ctrl()` line 149. The deck width
   tracks the max beam width; there is no independent deck beam parameter.
3. **P5 exists in web app only** — desktop app and optimizer use `range(1, 5)`.
   The web app uses `range(1, 6)`. Do not add P5 back to geometry.py.
4. **`hull_web/geometry.py` is a copy** — it is standalone (no package imports).
   Changes to `moth_designer/geometry.py` must be manually mirrored if needed.
5. **`build_ctrl()` returns `raw_dw`** — this was removed from the desktop app's
   parameter set. `deck_hb` is now derived as `beam_hb.copy()` inside `build_ctrl`.

---

## Updates (April 2026)

### STL Export Geometry (`moth_designer/geometry.py`)

Significant fixes were made to the `build_3d_mesh` function to resolve artifacts in exported STL files, particularly concerning the transom and bow.

**1. Welded Mesh Construction:**
   - The mesh generation for the transom and bow caps was refactored to reuse existing vertices from the hull skin.
   - This "welds" the caps to the hull, creating a single, clean, manifold mesh and fixing the "shell" artifacts caused by disconnected surfaces.
   - The winding order of faces on the bow and transom caps was corrected to ensure surface normals point outwards consistently.

**3. Bow Shape:**
   - All code that artificially added a minimum radius to the bow (a 1mm flat face) has been removed from `build_3d_mesh`.
   - The bow is now generated with a true sharp point, as defined by the control parameters. The `cross_section_spline` function was also tweaked to ensure it produces a straight vertical line at the bow stem.
