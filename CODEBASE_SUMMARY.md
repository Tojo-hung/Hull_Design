# Hull_Design Codebase Summary

This repository contains a parametric Moth hull design tool, a separate web
prototype of that designer, and an OpenFOAM-based CFD optimization and
validation pipeline.

The codebase is easiest to understand as three layers:

1. Design and geometry generation
2. Visualization and export
3. CFD case generation, optimization, and validation

## Repository Layout

```text
Hull_Design/
|- run.py                     # Desktop app entry point
|- CODEBASE_SUMMARY.md        # This summary
|- hull_design.json           # Current working desktop design
|- best_hull.json             # Best saved optimizer result
|- hull_optimizer.py          # Optuna + OpenFOAM optimization workflow
|- physics_validator.py       # CFD verification and validation suite
|- generate_trial_json.py     # Rebuild a JSON config from an optimization CSV row
|- test_mesh.py               # cfMesh smoke test
|- test_interfoam.py          # Short interFoam smoke test
|- moth_designer/             # Main desktop application package
|  |- __init__.py
|  |- app.py                  # PyQt5 desktop UI
|  |- config.py               # Desktop constants and default parameters
|  |- geometry.py             # Core geometry, hydrostatics, 3D mesh
|  |- exports.py              # TXT, STL, STEP, DXF exporters
|- hull_web/                  # Standalone Streamlit prototype
|  |- streamlit_app.py
|  |- config.py
|  |- geometry.py
|  |- requirements.txt
|- openfoam_template/         # Source OpenFOAM case template used by optimizer
|- openfoam_runs/             # Generated or archived CFD case directories
|- cfd_results/               # Archived CFD result directories from earlier runs
|- bow_test.stl               # Sample/generated mesh artifact
|- optuna_study_fast.db       # Existing Optuna study database
|- __pycache__/               # Python bytecode cache
```

## High-Level Data Flow

Desktop and optimizer path:

```text
parameter dict
-> build_ctrl()
-> longitudinal interpolation with lagrange() / PCHIP
-> cross-section generation
-> previews, hydrostatics, exports, or 3D mesh
-> optional OpenFOAM case generation and drag evaluation
```

Web path:

```text
Streamlit widgets
-> standalone hull_web geometry copy
-> Plotly plots, metrics, and STL download
```

## Core Parameter Model

The project uses a flat parameter dictionary rather than classes.

Desktop package and optimizer keys:

- Bow and stern:
  - `bow_draft`, `bow_sheer`
  - `transom_half_beam`, `transom_draft`, `transom_sheer`,
    `transom_keel_w`, `transom_hb_z`
- Four internal control points:
  - `p{i}_x`
  - `p{i}_hb`
  - `p{i}_d`
  - `p{i}_dz`
  - `p{i}_kw`
  - `p{i}_hz`
  - for `i = 1..4`

Important desktop/optimizer notes:

- The desktop geometry model has 4 internal stations, not 5.
- Desktop deck half-beam is not an independent parameter. In
  `moth_designer.geometry.build_ctrl()`, `deck_hb` is derived from `beam_hb`.
- `bow_half_beam` is a hidden desktop parameter supported by
  `moth_designer.geometry.build_ctrl()`. The desktop defaults now set it to
  `1.0` mm even though the desktop UI still does not expose it.
- `bow_mesh_half_beam` is a separate hidden mesh-only override used to avoid a
  zero-area first mesh strip at the bow.

Web app keys differ:

- The Streamlit app uses 5 internal points (`p1` through `p5`).
- It keeps an independent deck half-beam per station with `p{i}_dw`.
- Its defaults are not identical to the desktop defaults.

JSON behavior:

- `hull_design.json` is the main working design loaded by the desktop app.
- `best_hull.json` stores an optimizer result.
- Loaders merge known keys onto defaults and ignore unknown keys.
- Example: the current `best_hull.json` contains `ends_draft`, but the desktop
  app ignores it because it is not a known default key.

## Desktop Package: `moth_designer/`

### `run.py`

Tiny launcher:

- Imports `main` from `moth_designer.app`
- Starts the desktop application

### `moth_designer/config.py`

Defines desktop constants and the built-in default hull:

- `LWL = 3355`
- `MAX_DEPTH = 350`
- `FREEBOARD = 200`
- `TARGET_DISP_L = 130`
- plot color palettes for sections and waterlines
- `DEFAULTS` for the 4-point desktop design

This file is the canonical default parameter source for:

- the desktop app
- the optimizer base-parameter merge
- validation tools

### `moth_designer/geometry.py`

This is the core of the project. It is pure numerical geometry with no GUI code.

Key functions:

- `lagrange(...)`
  - Despite the name, this is PCHIP interpolation via
    `scipy.interpolate.PchipInterpolator`.
  - It is used for all longitudinal interpolation of beam, draft, sheer,
    keel width, and beam-height offset.

- `cross_section(...)`
  - Builds a quarter-ellipse bilge plus straight topsides.
  - Supports a flat keel and a `hb_z` max-beam height offset.

- `cross_section_spline(...)`
  - Desktop-preferred section generator.
  - Uses a cubic spline bilge and a vertical topside.
  - Falls back to a bounded easing curve if the spline overshoots on very fine
    bow sections.

- `build_ctrl(params)`
  - Converts the flat parameter dict into sorted control arrays.
  - Adds bow and transom endpoints.
  - Returns 12 values, including duplicated sheer arrays kept for existing
    callers.

- `beam_eval(...)`
  - Convenience wrapper for beam interpolation.

- `_section_area_half_below(...)`
  - Shoelace-area helper used by hydrostatics.

- `section_area_full(...)`
  - Computes full immersed section area at a given waterline.

- `displaced_volume(...)`
  - Integrates section areas along the hull and returns litres.

- `find_disp_waterline(...)`
  - Bisection search for the waterline that produces a target displacement.

- `hydrostatic_coefficients(...)`
  - Computes `Cb`, `Cp`, `Cm`, `Cw`, plus waterline, draft, beam, and areas.

- `block_coefficient(...)`
  - Alias kept for backward compatibility.

- `build_3d_mesh(...)`
  - Builds the triangulated hull mesh used by:
    - the desktop 3D view
    - STL export
    - CFD geometry generation
  - Creates starboard and port skins, the deck strip, the transom cap, and the
    bow cap.
  - Uses a tiny rectangular starter section at the bow so the first mesh strip
    stays robust.

Geometry-specific gotchas:

- `lagrange()` is PCHIP, not polynomial Lagrange interpolation.
- Desktop `deck_hb` always tracks hull half-beam.
- `transom_hb_z` is supported by the desktop geometry and UI.
- The desktop freeboard constant is 200 mm, not 100 mm.

### `moth_designer/exports.py`

Export layer built on top of the geometry functions.

Exports provided:

- `export_txt(...)`
  - Writes per-station XYZ points for CAD import workflows such as SolidWorks.

- `export_stl(...)`
  - Writes a binary STL from the triangulated mesh.

- `export_step(...)`
  - Builds spline section wires and lofts them into a STEP solid.
  - Requires `cadquery`.

- `export_dxf_sections(...)`
  - Writes per-station or combined section DXFs.
  - Intended for Onshape or other sketch-based workflows.

- `export_dxf_lines_plan(...)`
  - Writes a full reference lines-plan DXF with body plan, profile, and
    half-breadth view laid out side by side.

### `moth_designer/app.py`

Main desktop UI built with PyQt5 and pyqtgraph.

Top-level responsibilities:

- load the working parameter set from `hull_design.json`
- present editable controls
- redraw profile, plan, and body-plan views
- open a separate 3D OpenGL window
- export files
- show hydrostatic coefficients

Main pieces:

- `load_config()` / `save_config()`
  - Read and write `hull_design.json`.

- theme helpers
  - Two themes are supported: dark and light.
  - The app starts in dark mode.

- `Hull3DView`
  - Separate OpenGL-backed window using `pyqtgraph.opengl`.
  - Reuses the same mesh generator as exports and CFD.
  - Also overlays the target displacement waterline when it is below the deck.

- `MothDesigner`
  - Main window class.
  - Left panel: control-point, deck/keel, transom, and bow inputs.
  - Right panel: profile, plan view, and body plan.

Desktop UI features worth knowing:

- Inputs are split into:
  - control point X / half-beam / depth
  - deck height / keel width / beam-height offset
  - transom settings
  - bow settings
- A shared "Hull draft" field updates both bow and transom draft together.
- Buttons include:
  - 3D view
  - save/load config
  - hydrostatic coefficients dialog
  - TXT / STL / STEP / DXF exports
  - reset
  - measurement mode
  - profile and plan reference image overlay
  - theme toggle

Plot layout:

- profile view
- plan view / waterlines
- body plan

Redraw behavior:

- Every field update triggers `_safe_redraw()`.
- The app does not auto-save on every edit.
- Saving is explicit through `Save Config`.

Other important app details:

- Measurement mode shows distance from bow and stern in the profile view.
- Reference images can be overlaid onto the profile and plan plots for tracing.
- The status bar reports max beam, max depth, transom size, total volume, and
  the target-displacement waterline.

## Web Prototype: `hull_web/`

This directory is a standalone Streamlit prototype, not a thin wrapper over the
desktop package.

### `hull_web/config.py`

Defines a separate set of constants and defaults for the web app.

Notable differences from desktop:

- `FREEBOARD = 100`
- 5 control points instead of 4
- includes `p{i}_dw` deck half-beam defaults
- different default hull shape

### `hull_web/geometry.py`

Standalone geometry copy for Streamlit use.

Key differences from desktop geometry:

- imports `config` directly and is intentionally package-independent
- uses 5 internal stations
- keeps independent `deck_hb`
- uses `cross_section()` rather than the desktop spline version for its section
  and hydrostatics path
- `build_3d_mesh()` is not identical to the desktop mesh builder

The web geometry file is duplicated logic, not a shared import from
`moth_designer.geometry`, so desktop changes do not automatically carry over.

### `hull_web/streamlit_app.py`

Standalone Streamlit UI with Plotly rendering.

Responsibilities:

- config upload/download through Streamlit session state
- editable 5-row control-point table
- bow/transom/LWL/displacement controls in the sidebar
- profile, body-plan, and plan-view plots
- 3D Plotly mesh view
- STL download generated in memory
- hydrostatic coefficient calculation on demand

Implementation notes:

- The app allows runtime `LWL` changes and mutates `geometry.LWL` so the
  geometry functions use the edited length.
- It also allows runtime target displacement changes, unlike the desktop app.

### `hull_web/requirements.txt`

Minimal web dependency list:

- `streamlit`
- `numpy`
- `scipy`
- `plotly`

## CFD Optimization: `hull_optimizer.py`

This is the current optimization engine.

Important correction from older docs:

- It uses Optuna TPE, not CMA-ES.

### What it does

- loads a base hull from `hull_design.json` or `best_hull.json`
- applies fixed parameters and Optuna trial suggestions
- prunes obviously invalid hulls with fast geometry checks
- builds a combined ASCII STL for hull plus domain box
- creates an OpenFOAM case from the template
- runs cfMesh and interFoam
- extracts drag from `force.dat`
- logs every trial
- saves the best parameter set

### Search-space model

The optimizer currently varies 20 design variables:

- four station X positions
- `p2`, `p3`, and `p4` half-beam / draft / keel-width / beam-height offset
- `p1` half-beam / draft / keel-width
- `bow_draft`

It keeps several parameters fixed, including:

- transom half-beam
- transom draft
- transom sheer
- all station deck heights
- `p1_hz = 0`

### Constraint pruning

`check_constraints(...)` rejects a trial before CFD if it violates:

- minimum station draft floor relative to bow/transom draft
- rocker range: 60 mm to 130 mm
- prismatic coefficient range: 0.55 to 0.85

`check_draft_angle(...)` exists as a helper but is not currently called by the
Optuna objective.

### Geometry and case generation

`build_geometry_stl(...)` is a key bridge function:

- calls desktop geometry to build the hull mesh
- finds the 130 L displacement waterline
- shifts the hull so that waterline becomes `z = 0`
- emits a single ASCII STL containing named solids for:
  - `hull`
  - `inlet`
  - `outlet`
  - `front`
  - `back`
  - `bottom`
  - `atmosphere`

That single-file, named-solid STL is what cfMesh uses to create boundary
patches.

### OpenFOAM execution helpers

Important functions:

- `find_openfoam_exe(...)`
  - resolves OpenFOAM utilities from PATH, env vars, local builds, or standard
    install locations

- `setup_case(...)`
  - copies `openfoam_template/`
  - writes `constant/triSurface/geometry.stl`
  - patches inlet velocity in `0/U`

- `apply_fast_mode(...)`
  - replaces `meshDict` with a much coarser mesh
  - shortens the run to 0.5 s

- `_run_foam_utility(...)`
  - common subprocess wrapper with log capture and error reporting

- `run_cfmesh_pipeline(...)`
  - runs `surfaceFeatureEdges`
  - rewrites `system/meshDict` with dynamic refinement boxes
  - runs `cartesianMesh`
  - runs `checkMesh`
  - patches final boundary types

- `run_case(...)`
  - full CFD pipeline for a case

- `run_lts_simulation(...)`
  - alternate local-time-stepping solve path used mainly by validation scripts

- `extract_drag(...)`
  - averages the last 20 percent of force samples

### CLI behavior

Supported flags:

- `--speed`
- `--n-trials`
- `--resume`
- `--eval-best`
- `--fast`
- `--n-procs`
- `--tag`

Current behavior note:

- Resume is effectively controlled by whether the selected study database
  already exists. The parsed `--resume` flag is not used to change logic.

### Generated optimizer outputs

Depending on mode and tag, the optimizer writes:

- `optimization_log*.csv`
- `optuna_study*.db`
- `best_hull*.json`
- `optimization_runs/YYYY-MM-DD_NN/` case directories

## Validation and Utility Scripts

### `physics_validator.py`

Validation suite for the CFD pipeline. It imports shared helpers from
`hull_optimizer.py` so it uses the same geometry and case-generation path.

Available tests:

- `archimedes`
  - buoyancy vs theoretical displacement weight
- `grid`
  - three-level grid convergence study
- `steady`
  - drag stabilization over a longer run
- `sweep`
  - parameter sweep to check for a smooth drag landscape
- `ittc`
  - flat-plate friction validation against the ITTC-1957 line

### `generate_trial_json.py`

Small but useful utility:

- reads an optimization CSV
- extracts a selected trial row
- merges it onto a base parameter set
- writes a usable JSON hull config

This is the easiest way to turn a trial number from a log into a design file
you can load back into the desktop app.

### `test_mesh.py`

Quick cfMesh smoke test:

- loads params
- builds the combined hull/domain STL
- sets up a case
- runs `cartesianMesh`
- runs `checkMesh`

### `test_interfoam.py`

Short solver smoke test:

- assumes `test_mesh.py` has already generated a mesh
- cleans time directories
- patches end time to 0.5 s
- runs `setFields`, `decomposePar`, `interFoam`, `reconstructPar`
- prints a quick drag estimate if force output exists

## OpenFOAM Template: `openfoam_template/`

This is the source case template copied by `setup_case()`.

Main contents:

- `0/`
  - initial fields such as velocity, pressure, turbulence, and phase fraction
- `constant/`
  - gravity and transport / turbulence properties
- `system/`
  - solver controls, discretization, decomposition, meshing, and setFields

Important current template notes:

- `system/controlDict`
  - uses `interFoam`
  - defines force, hull-pressure, and residual outputs

- `system/setFieldsDict`
  - initializes water below `z = 0`

- `system/meshDict`
  - exists as a baseline template
  - is usually overwritten dynamically by `hull_optimizer.run_cfmesh_pipeline()`

- `Allrun`
  - older manual helper script for a blockMesh/snappyHexMesh style workflow
  - not the same pipeline currently used by `hull_optimizer.py`

## Generated and Archived Directories

These directories are present in the repo but are not core source code:

- `openfoam_runs/`
  - generated or archived run directories
- `cfd_results/`
  - archived CFD outputs from earlier experiments

These are useful as reference data, but the current code path creates fresh
cases from `openfoam_template/` rather than editing these directories in place.

Also note that many archived run folders appear to come from an older
blockMesh/snappyHexMesh workflow, while the current optimizer uses cfMesh with
a single combined STL.

## Common Workflows

Desktop design:

```text
python run.py
```

Web prototype:

```text
cd hull_web
streamlit run streamlit_app.py
```

Optimizer:

```text
python3 hull_optimizer.py --speed 3.601 --n-trials 150
python3 hull_optimizer.py --fast --tag laptop_test
python3 hull_optimizer.py --eval-best
```

Validation:

```text
python3 physics_validator.py --test archimedes
python3 physics_validator.py --test grid
```

## Current-State Gotchas

- The existing codebase is split between a 4-point desktop model and a 5-point
  web model.
- Desktop and web defaults are different.
- Desktop freeboard is 200 mm; web freeboard is 100 mm.
- Desktop geometry derives deck width from hull beam; web geometry keeps deck
  width independent.
- `lagrange()` means PCHIP everywhere.
- The desktop app redraws live but only saves when the user clicks `Save Config`.
- `best_hull.json` may contain extra keys that the desktop app silently ignores.
- `build_ctrl()` returns duplicate sheer arrays for legacy callers.
- The optimizer docs inside older notes may still refer to CMA-ES; the current
  code uses Optuna TPE.
- `openfoam_template/Allrun` documents an older manual pipeline and should not
  be treated as the source of truth for the optimizer path.

## Recommended Mental Model

If you are onboarding to this repo, treat these as the three primary files:

1. `moth_designer/geometry.py` for hull math
2. `moth_designer/app.py` for the desktop UX
3. `hull_optimizer.py` for the CFD pipeline

Everything else is either:

- a UI/export layer built on top of that geometry
- a standalone web copy of the same idea
- validation or glue code around the OpenFOAM workflow
