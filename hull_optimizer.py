"""
hull_optimizer.py  —  Optuna TPE hull optimisation via OpenFOAM + cfMesh
========================================================================
Run from WSL:
    cd ~/Hull_Design
    source ~/hull-env/bin/activate
    python3 hull_optimizer.py --speed 3.601 --n-trials 150

Each trial = one full cfMesh + interFoam run.
Results append to optimization_log.csv after every trial.
Best hull is saved to best_hull.json (load it in the desktop app
via Save/Load config or by copying over hull_design.json).

Resume a previous run:
    python3 hull_optimizer.py --resume

Evaluate best_hull.json once:
    python3 hull_optimizer.py --eval-best
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
TEMPLATE_DIR = ROOT / 'openfoam_template'
WORK_DIR     = Path.home() / 'openfoam_runs'
LOG_FILE     = ROOT / 'optimization_log.csv'
BEST_FILE    = ROOT / 'best_hull.json'
CONFIG_FILE  = ROOT / 'hull_design.json'
STUDY_DB     = ROOT / 'optuna_study.db'
STUDY_NAME   = 'moth_hull'

N_CORES = 8   # set to your CPU core count (check with: nproc)
OF      = 'source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null'

sys.path.insert(0, str(ROOT))
from moth_designer.config   import DEFAULTS, LWL, TARGET_DISP_L
from moth_designer.geometry import (build_ctrl, build_3d_mesh, beam_eval,
                                    find_disp_waterline)

# ══════════════════════════════════════════════════════════════
# SEARCH SPACE — edit bounds to suit your study
# ══════════════════════════════════════════════════════════════
SEARCH_SPACE = {
    # Midship
    'p3_hb':  (160, 220),   # midship half-beam (mm)
    'p3_d':   (130, 195),   # midship draft
    'p3_kw':  (80,  150),   # midship keel width
    'p3_x':   (1800, 2100), # midship position

    # Forward section
    'p2_hb':  (150, 230),   # forward half-beam
    'p2_d':   (140, 210),   # forward draft
    'p2_kw':  (20,  100),   # forward keel width
    'p2_x':   (700, 1300),  # forward section position

    # Aft section
    'p4_hb':  (165, 240),   # aft half-beam
    'p4_d':   (130, 200),   # aft draft
    'p4_kw':  (60,  160),   # aft keel width
    'p4_x':   (2150, 2600), # aft position

    # Bow section
    'p1_hb':  (60,  140),   # bow half-beam
    'p1_d':   (70,  150),   # bow draft
    'p1_kw':  (0,   30),    # bow keel width
    'p1_x':   (200, 500),   # bow section position

    # Transom
    'transom_half_beam': (100, 250),
    'transom_draft':     (40,  130),

    # Bow entry
    'bow_draft': (30, 120),

    # Beam height (controls V vs U section shape)
    'p1_hz':  (0,  100),
    'p2_hz':  (0,  150),
    'p3_hz':  (0,  80),
    'p4_hz':  (0,  100),
}
# ══════════════════════════════════════════════════════════════


def load_base_params():
    """Load hull params, preferring best_hull.json over hull_design.json."""
    src = BEST_FILE if BEST_FILE.exists() else CONFIG_FILE
    if src.exists():
        data   = json.loads(src.read_text())
        merged = {k: float(v) for k, v in DEFAULTS.items()}
        merged.update({k: float(v) for k, v in data.items() if k in merged})
        print(f'Base params loaded from: {src.name}')
        return merged
    return {k: float(v) for k, v in DEFAULTS.items()}


def check_draft_angle(params, min_angle_deg=1.0):
    """Check STL normals satisfy a minimum mould draft angle (pull direction = +Z).

    A face passes if its normal's Z-component >= sin(min_angle_deg).
    Returns True if all faces pass, False otherwise.
    """
    verts, faces = build_3d_mesh(params, N_X=60, N_T=24)  # low-res — just for check
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float64)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = (nlen[:, 0] > 0)
    normals[valid] /= nlen[valid]
    min_dot = np.sin(np.radians(min_angle_deg))
    if np.any(normals[valid, 2] < min_dot):
        print(f'  Draft angle < {min_angle_deg}° detected — invalid geometry')
        return False
    return True


def build_geometry_stl(params, N_X=250, N_T=100):
    """Generate combined ASCII STL (hull + domain box) for cfMesh.

    OpenFOAM 2406 cfMesh requires a single top-level surfaceFile.
    This function combines all geometry into one ASCII STL with named
    solid groups — cfMesh uses the solid name as the boundary patch name.

    Solids produced:
        hull        — hull surface (~100k triangles)
        inlet       — x = -1.0 m face
        outlet      — x =  5.0 m face
        front       — y = -1.5 m face
        back        — y =  1.5 m face
        bottom      — z = -0.8 m face
        atmosphere  — z =  0.5 m face

    The hull is shifted so the 130L waterline lands at z=0.
    All coordinates in metres.
    """
    # ── Hull ──────────────────────────────────────────────────
    ctrl_x, beam_hb, keel_d, deck_hb_c, sheer_z_c, keel_w_c, hb_z_c, _, _, _ = \
        build_ctrl(params)
    dz_wl = find_disp_waterline(TARGET_DISP_L, ctrl_x, beam_hb,
                                 ctrl_x, keel_d,
                                 deck_x=ctrl_x, deck_hb_arr=deck_hb_c,
                                 sheer_z_arr=sheer_z_c, keel_w_arr=keel_w_c)
    print(f'  130 L waterline: z = {dz_wl:.1f} mm  (shifted to z=0 in domain)')

    verts, faces = build_3d_mesh(params, N_X=N_X, N_T=N_T)
    verts[:, 2] -= dz_wl
    verts *= 0.001                  # mm → metres
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    normals = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1.0
    normals /= nlen

    lines = ['solid hull\n']
    for i in range(len(faces)):
        n = normals[i]
        p = [verts[faces[i, j]] for j in range(3)]
        lines.append(
            f'  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n'
            f'    outer loop\n'
            f'      vertex {p[0][0]:.6e} {p[0][1]:.6e} {p[0][2]:.6e}\n'
            f'      vertex {p[1][0]:.6e} {p[1][1]:.6e} {p[1][2]:.6e}\n'
            f'      vertex {p[2][0]:.6e} {p[2][1]:.6e} {p[2][2]:.6e}\n'
            f'    endloop\n'
            f'  endfacet\n'
        )
    lines.append('endsolid hull\n')

    # ── Domain bounding box ────────────────────────────────────
    xn, xx = -1.0,  5.0
    yn, yx = -1.5,  1.5
    zn, zx = -0.8,  0.5

    # (name, tri1, tri2) — winding order gives inward-pointing normals
    box_tris = [
        ('inlet',
         [(xn,yn,zn),(xn,yx,zn),(xn,yx,zx)], [(xn,yn,zn),(xn,yx,zx),(xn,yn,zx)]),
        ('outlet',
         [(xx,yx,zn),(xx,yn,zn),(xx,yn,zx)], [(xx,yx,zn),(xx,yn,zx),(xx,yx,zx)]),
        ('front',
         [(xn,yn,zx),(xx,yn,zx),(xx,yn,zn)], [(xn,yn,zx),(xx,yn,zn),(xn,yn,zn)]),
        ('back',
         [(xn,yx,zn),(xx,yx,zn),(xx,yx,zx)], [(xn,yx,zn),(xx,yx,zx),(xn,yx,zx)]),
        ('bottom',
         [(xn,yn,zn),(xx,yn,zn),(xx,yx,zn)], [(xn,yn,zn),(xx,yx,zn),(xn,yx,zn)]),
        ('atmosphere',
         [(xn,yx,zx),(xx,yx,zx),(xx,yn,zx)], [(xn,yx,zx),(xx,yn,zx),(xn,yn,zx)]),
    ]

    def fmt_tri(p0, p1, p2):
        n = np.cross(np.subtract(p1, p0), np.subtract(p2, p0)).astype(float)
        n /= np.linalg.norm(n)
        return (f'  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n'
                f'    outer loop\n'
                f'      vertex {p0[0]} {p0[1]} {p0[2]}\n'
                f'      vertex {p1[0]} {p1[1]} {p1[2]}\n'
                f'      vertex {p2[0]} {p2[1]} {p2[2]}\n'
                f'    endloop\n'
                f'  endfacet\n')

    for name, t1, t2 in box_tris:
        lines.append(f'solid {name}\n')
        lines.append(fmt_tri(*t1))
        lines.append(fmt_tri(*t2))
        lines.append(f'endsolid {name}\n')

    return ''.join(lines)


def setup_case(run_dir, geometry_stl, speed_ms):
    """Copy template → run_dir, write combined geometry STL, patch inlet speed."""
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    shutil.copytree(TEMPLATE_DIR, run_dir,
                    copy_function=shutil.copy,
                    ignore_dangling_symlinks=True)

    tri_dir = run_dir / 'constant' / 'triSurface'
    tri_dir.mkdir(parents=True, exist_ok=True)

    # Single combined ASCII STL — hull + domain walls (all named solids)
    (tri_dir / 'geometry.stl').write_text(geometry_stl)

    # Patch velocity in 0/U
    u_file = run_dir / '0' / 'U'
    txt = u_file.read_text()
    txt = re.sub(
        r'uniform\s*\(\s*[\d.eE+\-]+\s+0\s+0\s*\)',
        f'uniform ({speed_ms:.4f} 0 0)',
        txt,
    )
    u_file.write_text(txt)


def run_case(run_dir):
    """cartesianMesh → setFields → decomposePar → interFoam → reconstructPar."""
    steps = [
        ('cartesianMesh', 'cartesianMesh'),
        ('setFields',     'setFields'),
        ('decomposePar',  f'decomposePar -force'),
        ('interFoam',     f'mpirun --use-hwthread-cpus -np {N_CORES} interFoam -parallel'),
        ('reconstructPar','reconstructPar -latestTime'),
    ]

    for label, cmd in steps:
        t0   = time.time()
        full = f'{OF}; {cmd} > log.{label} 2>&1'
        ret  = subprocess.run(['bash', '--norc', '--noprofile', '-c', full], cwd=run_dir)
        elapsed = time.time() - t0
        status  = 'ok' if ret.returncode == 0 else 'FAILED'
        print(f'  {label:20s} {status}  ({elapsed:.0f}s)')
        if ret.returncode != 0:
            log = run_dir / f'log.{label}'
            if log.exists():
                print('\n'.join(log.read_text().splitlines()[-20:]))
            raise RuntimeError(f'{label} failed')


def extract_drag(run_dir):
    """Parse postProcessing/forces/*/force*.dat, return mean drag (N)
    averaged over the last 20% of simulation time (quasi-steady state).
    Drag = pressure_x + viscous_x
    """
    candidates = sorted((run_dir / 'postProcessing').rglob('force*.dat'))
    if not candidates:
        raise FileNotFoundError('No force.dat found')

    rows = []
    for line in candidates[-1].read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        nums = []
        for tok in line.split():
            try:
                nums.append(float(tok.strip('()')))
            except ValueError:
                pass
        if len(nums) >= 7:
            rows.append(nums)

    if not rows:
        raise ValueError('force.dat is empty or unparseable')

    n_avg = max(1, len(rows) // 5)
    drag_values = [abs(r[1] + r[4]) for r in rows[-n_avg:]]
    return float(np.mean(drag_values))


# ── Optuna objective ────────────────────────────────────────────
def make_objective(base_params, speed_ms, log_rows):
    import optuna

    def objective(trial):
        # Suggest parameter values from search space
        params = {**base_params}
        for name, (lo, hi) in SEARCH_SPACE.items():
            params[name] = trial.suggest_float(name, lo, hi)

        n       = trial.number + 1
        run_dir = WORK_DIR / f'run_{n:04d}'

        print(f'\n{"="*55}  Trial {n}  |  {speed_ms:.2f} m/s ({speed_ms*1.944:.1f} kn)')
        for name in SEARCH_SPACE:
            print(f'  {name:20s} = {params[name]:.1f}')

        # Check manufacturing constraint before running CFD
        if not check_draft_angle(params):
            print('  Pruning: draft angle constraint violated')
            raise optuna.TrialPruned()

        drag, status = 1e6, 'error'
        try:
            geom = build_geometry_stl(params)
            setup_case(run_dir, geom, speed_ms)
            run_case(run_dir)
            drag   = extract_drag(run_dir)
            status = 'ok'
            print(f'  -> Drag = {drag:.3f} N')
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f'  -> ERROR: {e}')
            status = str(e)
            raise optuna.TrialPruned()

        row = {'n': n, 'drag_N': round(drag, 4),
               'status': status, 'speed_ms': speed_ms}
        row.update({k: round(params[k], 2) for k in SEARCH_SPACE})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(LOG_FILE, index=False)

        return drag

    return objective


# ── Main ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed',    type=float, default=3.601,
                    help='Boat speed m/s (default 3.601 ≈ 7.0 knots)')
    ap.add_argument('--n-trials', type=int,   default=150,
                    help='Number of Optuna trials (default 150)')
    ap.add_argument('--resume',   action='store_true',
                    help='Resume from existing Optuna study database')
    ap.add_argument('--eval-best', action='store_true',
                    help='Run a single evaluation on best_hull.json and exit')
    args = ap.parse_args()

    try:
        import optuna
    except ImportError:
        sys.exit('pip install optuna')

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    base     = load_base_params()
    log_rows = pd.read_csv(LOG_FILE).to_dict('records') if LOG_FILE.exists() else []

    # ── Single eval of best_hull.json ───────────────────────────
    if args.eval_best:
        print('\nRunning single evaluation on best config...')
        params  = {**base}
        run_dir = WORK_DIR / 'run_eval_best'
        try:
            geom = build_geometry_stl(params)
            setup_case(run_dir, geom, args.speed)
            run_case(run_dir)
            drag = extract_drag(run_dir)
            print(f'\nDrag for best config: {drag:.3f} N')
        except Exception as e:
            print(f'ERROR: {e}')
        return

    # ── Optuna study ─────────────────────────────────────────────
    storage    = f'sqlite:///{STUDY_DB}'
    load_exist = args.resume and STUDY_DB.exists()

    study = optuna.create_study(
        study_name  = STUDY_NAME,
        storage     = storage,
        load_if_exists = load_exist,
        direction   = 'minimize',
        sampler     = optuna.samplers.TPESampler(seed=42),
    )

    n_done = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE])
    print(f'\n{"="*55}')
    print(f'Optuna TPE  |  budget={args.n_trials} trials  |  completed so far={n_done}')
    print(f'Study DB    : {STUDY_DB}')
    print(f'Params      : {list(SEARCH_SPACE.keys())}\n')

    study.optimize(
        make_objective(base, args.speed, log_rows),
        n_trials = args.n_trials,
    )

    # ── Save best ────────────────────────────────────────────────
    best_trial  = study.best_trial
    best_params = {**base, **best_trial.params}
    BEST_FILE.write_text(json.dumps(
        {k: round(float(v), 4) for k, v in best_params.items()}, indent=2))

    print(f'\n{"="*55}')
    print(f'Best drag   : {best_trial.value:.3f} N  (trial #{best_trial.number + 1})')
    for k, v in best_trial.params.items():
        print(f'  {k:20s} = {v:.1f}')
    print(f'\nBest config -> {BEST_FILE}')
    print(f'Full log    -> {LOG_FILE}')
    print(f'Study DB    -> {STUDY_DB}')
    print('\nTo use: copy best_hull.json over hull_design.json and reopen the app.')


if __name__ == '__main__':
    main()