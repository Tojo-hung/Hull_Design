"""
hull_optimizer.py  —  Bayesian hull optimisation via OpenFOAM
=============================================================
Run from WSL:
    cd /mnt/c/Users/xrace/OneDrive/Documents/GitHub/Hull_Design
    source ~/hull-env/bin/activate
    python3 hull_optimizer.py --speed 3.0 --n-calls 30

Each call = one full mesh + simpleFoam run (~5-15 min each).
Results append to optimization_log.csv after every evaluation.
Best hull is saved to best_hull.json (load it in the desktop app
via Save/Load config or by copying over hull_design.json).
"""

import argparse
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
TEMPLATE_DIR  = ROOT / 'openfoam_template'
WORK_DIR      = Path.home() / 'openfoam_runs'   # Linux fs — avoids NTFS permission errors
LOG_FILE      = ROOT / 'optimization_log.csv'
BEST_FILE     = ROOT / 'best_hull.json'
CONFIG_FILE   = ROOT / 'hull_design.json'

sys.path.insert(0, str(ROOT))
from moth_designer.config   import DEFAULTS, LWL, MAX_DEPTH, TARGET_DISP_L
from moth_designer.geometry import (build_ctrl, build_3d_mesh, beam_eval, lagrange,
                                    find_disp_waterline)

# ══════════════════════════════════════════════════════════════
# SEARCH SPACE — edit bounds to suit your study
# Leave a param out to keep it fixed at the saved/default value.
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

    # Stern section
    'p5_hb':  (150, 230),   # stern half-beam
    'p5_d':   (90,  170),   # stern draft
    'p5_kw':  (40,  150),   # stern keel width
    'p5_x':   (2700, 3100), # stern section position

    # Transom
    'transom_half_beam': (100, 250), # transom width
    'transom_draft':     (40,  130), # transom depth

    # Bow entry
    'bow_draft': (30, 120),          # bow entry draft

    # Beam height (z of max beam point — controls V vs U section shape)
    'p1_hz':  (0,  100),  # bow section beam height
    'p2_hz':  (0,  150),  # forward beam height
    'p3_hz':  (0,  80),   # midship beam height
    'p4_hz':  (0,  100),  # aft beam height
    'p5_hz':  (0,  80),   # stern beam height
}
# ══════════════════════════════════════════════════════════════


def load_base_params():
    # Prefer best_hull.json (optimizer output) over hull_design.json
    src = BEST_FILE if BEST_FILE.exists() else CONFIG_FILE
    if src.exists():
        data   = json.loads(src.read_text())
        merged = {k: float(v) for k, v in DEFAULTS.items()}
        merged.update({k: float(v) for k, v in data.items() if k in merged})
        print(f'Base params loaded from: {src.name}')
        return merged
    return {k: float(v) for k, v in DEFAULTS.items()}


def build_stl_bytes(params, N_X=80, N_T=32):
    """Generate binary STL bytes from hull params.

    The hull is shifted vertically so the 130 L displacement waterline
    lands exactly at z=0 in the OpenFOAM domain (= the slip free-surface).
    """
    ctrl_x, beam_hb, keel_d, deck_hb_c, sheer_z_c, keel_w_c, hb_z_c, _, _, _ = \
        build_ctrl(params)
    dz_wl = find_disp_waterline(TARGET_DISP_L, ctrl_x, beam_hb,
                                 ctrl_x, keel_d,
                                 deck_x=ctrl_x, deck_hb_arr=deck_hb_c,
                                 sheer_z_arr=sheer_z_c, keel_w_arr=keel_w_c)
    print(f'  130 L waterline: z = {dz_wl:.1f} mm  (shifted to z=0 in domain)')

    verts, faces = build_3d_mesh(params, N_X=N_X, N_T=N_T)
    # Shift hull so the displacement waterline is at z=0
    verts[:, 2] -= dz_wl   # z column (mm), will be scaled *0.001 by snappy
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1.0
    normals /= nlen
    buf = io.BytesIO()
    buf.write(b'Moth hull optimiser' + b' ' * 61)
    buf.write(struct.pack('<I', len(faces)))
    for i in range(len(faces)):
        buf.write(normals[i].tobytes())
        buf.write(verts[faces[i, 0]].tobytes())
        buf.write(verts[faces[i, 1]].tobytes())
        buf.write(verts[faces[i, 2]].tobytes())
        buf.write(b'\x00\x00')
    return buf.getvalue()


def setup_case(run_dir, stl_bytes, speed_ms, fidelity='high'):
    """Copy template → run_dir, write hull STL, patch inlet speed and mesh fidelity.

    fidelity='low'       → coarse mesh (level 1 1), 2 BL layers, endTime 5   (~1-2 min)
    fidelity='high'      → finer mesh  (level 1 2), 3 BL layers, endTime 8   (~5-8 min)
    fidelity='very_high' → finest mesh (level 2 3), 5 BL layers, endTime 12  (~20-40 min)
    """
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    shutil.copytree(TEMPLATE_DIR, run_dir,
                    copy_function=shutil.copy,
                    ignore_dangling_symlinks=True)

    stl_path = run_dir / 'constant' / 'triSurface' / 'hull.stl'
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    stl_path.write_bytes(stl_bytes)

    # Patch velocity in 0/U
    u_file = run_dir / '0' / 'U'
    txt = u_file.read_text()
    txt = re.sub(
        r'uniform\s*\(\s*[\d.eE+\-]+\s+0\s+0\s*\)',
        f'uniform ({speed_ms:.4f} 0 0)',
        txt,
    )
    u_file.write_text(txt)

    if fidelity == 'low':
        # Coarse mesh: faster but less accurate — for broad exploration
        snappy = run_dir / 'system' / 'snappyHexMeshDict'
        txt = snappy.read_text()
        txt = re.sub(r'level\s*\(\s*\d+\s+\d+\s*\)', 'level (1 1)', txt)
        txt = re.sub(r'nSurfaceLayers\s+\d+', 'nSurfaceLayers 2', txt)
        snappy.write_text(txt)

        ctrl = run_dir / 'system' / 'controlDict'
        txt = ctrl.read_text()
        txt = re.sub(r'endTime\s+[\d.]+\s*;', 'endTime 5;', txt)
        txt = re.sub(r'writeInterval\s+[\d.]+\s*;', 'writeInterval 5;', txt)
        ctrl.write_text(txt)

    elif fidelity == 'very_high':
        # Finest mesh: high-accuracy single evaluation
        snappy = run_dir / 'system' / 'snappyHexMeshDict'
        txt = snappy.read_text()
        txt = re.sub(r'level\s*\(\s*\d+\s+\d+\s*\)', 'level (2 3)', txt)
        txt = re.sub(r'nSurfaceLayers\s+\d+', 'nSurfaceLayers 5', txt)
        snappy.write_text(txt)

        ctrl = run_dir / 'system' / 'controlDict'
        txt = ctrl.read_text()
        txt = re.sub(r'endTime\s+[\d.]+\s*;', 'endTime 12;', txt)
        txt = re.sub(r'writeInterval\s+[\d.]+\s*;', 'writeInterval 12;', txt)
        ctrl.write_text(txt)


N_CORES = 16   # set to your CPU core count (check with: nproc)

def run_case(run_dir):
    """surfaceFeatureExtract → blockMesh → snappyHexMesh → setFields → interFoam (parallel)."""
    OF  = 'source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null'

    steps = [
        ('surfaceFeatureExtract',              f'surfaceFeatureExtract'),
        ('blockMesh',                          f'blockMesh'),
        ('snappyHexMesh',                      f'snappyHexMesh -overwrite'),
        ('setFields',                          f'setFields'),
        ('decomposePar',  f'decomposePar -force'),
        ('interFoam',     f'mpirun --use-hwthread-cpus -np {N_CORES} interFoam -parallel'),
    ]

    for label, cmd in steps:
        t0  = time.time()
        full = f'{OF}; {cmd} > log.{label} 2>&1'
        ret  = subprocess.run(['bash', '--norc', '--noprofile', '-c', full], cwd=run_dir)
        elapsed = time.time() - t0
        status  = 'ok' if ret.returncode == 0 else 'FAILED'
        print(f'  {label:25s} {status}  ({elapsed:.0f}s)')
        if ret.returncode != 0:
            log = run_dir / f'log.{label}'
            if log.exists():
                lines = log.read_text().splitlines()
                print('\n'.join(lines[-20:]))
            raise RuntimeError(f'{label} failed')


def extract_drag(run_dir):
    """
    Parse postProcessing/forces/*/force.dat and return mean drag (N)
    averaged over the last 20% of simulation time (quasi-steady state).
    Columns: time  (Fp_x Fp_y Fp_z)  (Fv_x Fv_y Fv_z)  (Fw_x Fw_y Fw_z)
    Drag = Fp_x + Fv_x
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

    # Average over last 20% of time steps for quasi-steady result
    n_avg = max(1, len(rows) // 5)
    drag_values = [abs(r[1] + r[4]) for r in rows[-n_avg:]]
    return float(np.mean(drag_values))


# ── Objective ──────────────────────────────────────────────────
_n = [0]

def objective(x, param_names, base_params, speed_ms, log_rows, fidelity='high'):
    _n[0] += 1
    n      = _n[0]
    params = {**base_params, **dict(zip(param_names, map(float, x)))}
    run_dir = WORK_DIR / f'run_{n:04d}_{fidelity}'

    print(f'\n{"="*55}  Eval {n} [{fidelity}]  |  {speed_ms:.2f} m/s ({speed_ms*1.944:.1f} kn)')
    for k, v in zip(param_names, x):
        print(f'  {k:12s} = {v:.1f}')

    drag, status = 1e6, 'error'
    try:
        stl  = build_stl_bytes(params)
        setup_case(run_dir, stl, speed_ms, fidelity=fidelity)
        run_case(run_dir)
        drag   = extract_drag(run_dir)
        status = 'ok'
        print(f'  -> Drag = {drag:.3f} N')
    except Exception as e:
        print(f'  -> ERROR: {e}')
        status = str(e)

    row = {'n': n, 'drag_N': round(drag, 4),
           'status': status, 'speed_ms': speed_ms, 'fidelity': fidelity}
    row.update(dict(zip(param_names, [round(float(v), 2) for v in x])))
    log_rows.append(row)
    pd.DataFrame(log_rows).to_csv(LOG_FILE, index=False)
    return drag


# ── Main ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed',    type=float, default=3.601,
                    help='Boat speed m/s (default 3.601 ≈ 7.0 knots)')
    ap.add_argument('--n-low',    type=int,   default=40,
                    help='Stage 1: coarse-mesh evaluations (default 40)')
    ap.add_argument('--n-high',   type=int,   default=20,
                    help='Stage 2: fine-mesh evaluations (default 20)')
    ap.add_argument('--top-k',    type=int,   default=5,
                    help='Top-K coarse candidates to seed fine-mesh stage (default 5)')
    ap.add_argument('--n-init',   type=int,   default=16,
                    help='CMA-ES population size (default 16)')
    ap.add_argument('--sigma0',   type=float, default=0.25,
                    help='CMA-ES initial step size as fraction of range (default 0.25)')
    ap.add_argument('--resume',   action='store_true',
                    help='Resume from existing optimization_log.csv')
    ap.add_argument('--stage',    type=int,   default=0,
                    help='Start at stage 1 or 2 directly (default 0 = run both)')
    ap.add_argument('--eval-best', action='store_true',
                    help='Run a single fine-mesh evaluation on best_hull.json and exit')
    args = ap.parse_args()

    # ── Single fine-mesh eval of best_hull.json ─────────────────
    if args.eval_best:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        base  = load_base_params()
        names = list(SEARCH_SPACE.keys())
        x     = [base.get(n, SEARCH_SPACE[n][0]) for n in names]
        log_rows = []
        if LOG_FILE.exists():
            log_rows = pd.read_csv(LOG_FILE).to_dict('records')
            _n[0]    = int(pd.read_csv(LOG_FILE)['n'].max())
        drag = objective(x, names, base, args.speed, log_rows, fidelity='very_high')
        print(f'\nVery-fine-mesh drag for best config: {drag:.3f} N')
        return

    try:
        import cma
    except ImportError:
        sys.exit('pip install cma')

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    base   = load_base_params()
    names  = list(SEARCH_SPACE.keys())
    lows   = np.array([SEARCH_SPACE[n][0] for n in names], dtype=float)
    highs  = np.array([SEARCH_SPACE[n][1] for n in names], dtype=float)
    ranges = highs - lows

    log_rows = []
    best_x   = None
    best_f   = 1e6

    if args.resume and LOG_FILE.exists():
        prev = pd.read_csv(LOG_FILE)
        ok   = prev[prev['status'] == 'ok']
        if not ok.empty:
            log_rows = prev.to_dict('records')
            _n[0]    = int(prev['n'].max())
            best_row = ok.loc[ok['drag_N'].idxmin()]
            best_f   = float(best_row['drag_N'])
            best_x   = best_row[names].values.astype(float).tolist()
            print(f'Resuming from {len(ok)} previous runs. Best so far: {best_f:.3f} N')

    def denorm(x_norm):
        return np.clip(lows + np.array(x_norm) * ranges, lows, highs)

    def run_cma(fidelity, n_evals, x0_norm, sigma0, label):
        """Run CMA-ES at given fidelity. Returns (best_x, best_f, all results)."""
        print(f'\n{"="*55}')
        print(f'STAGE: {label}  |  fidelity={fidelity}  |  budget={n_evals} evals')
        print(f'Params: {names}\n')

        nonlocal log_rows
        stage_best_x, stage_best_f = x0_norm, 1e6

        def obj(x_norm):
            nonlocal stage_best_x, stage_best_f
            x = denorm(x_norm)
            f = objective(x, names, base, args.speed, log_rows, fidelity=fidelity)
            if f < stage_best_f:
                stage_best_f = f
                stage_best_x = list(x_norm)
            return f

        opts = cma.CMAOptions()
        opts['bounds']    = [[0.0] * len(names), [1.0] * len(names)]
        opts['maxfevals'] = n_evals
        opts['popsize']   = args.n_init
        opts['tolx']      = 1e-4
        opts['tolfun']    = 0.5
        opts['verbose']   = -9
        opts['seed']      = 42

        es = cma.CMAEvolutionStrategy(x0_norm, sigma0, opts)
        while not es.stop():
            solutions = es.ask()
            fitnesses = [obj(x) for x in solutions]
            es.tell(solutions, fitnesses)
            print(f'  Gen best: {min(fitnesses):.3f} N  |  Stage best: {stage_best_f:.3f} N')

        return stage_best_x, stage_best_f

    # ── Stage 1: coarse mesh — broad exploration ────────────────
    if args.stage in (0, 1):
        # Start from hull_design.json values (clipped to search bounds)
        base_x = np.clip([base.get(n, (lows[i]+highs[i])/2)
                          for i, n in enumerate(names)], lows, highs)
        x0 = ((base_x - lows) / ranges).tolist() if best_x is None else \
             ((np.array(best_x) - lows) / ranges).tolist()

        s1_best_norm, s1_best_f = run_cma(
            fidelity='low',
            n_evals=args.n_low,
            x0_norm=x0,
            sigma0=args.sigma0,
            label='Stage 1 — coarse mesh',
        )

        # Pick top-K from coarse log to seed fine-mesh stage
        df = pd.read_csv(LOG_FILE)
        ok_low = df[(df['status'] == 'ok') & (df['fidelity'] == 'low')]
        top_k  = ok_low.nsmallest(args.top_k, 'drag_N')
        print(f'\nTop {args.top_k} coarse candidates (drag N):')
        print(top_k[['drag_N'] + names].to_string(index=False))

        # Best coarse candidate becomes the fine-mesh starting point
        best_coarse_row = top_k.iloc[0]
        s1_best_norm = ((best_coarse_row[names].values.astype(float) - lows) / ranges).tolist()
        print(f'\nBest coarse drag: {s1_best_f:.3f} N  →  seeding fine-mesh stage')

    # ── Stage 2: fine mesh — refine around best ─────────────────
    if args.stage in (0, 2):
        if args.stage == 2:
            # Jump straight to stage 2: start from best_hull.json
            s1_best_norm = ((np.array(best_x or [0.5]*len(names)) - lows) / ranges).tolist()

        s2_best_norm, s2_best_f = run_cma(
            fidelity='high',
            n_evals=args.n_high,
            x0_norm=s1_best_norm,
            sigma0=0.1,        # tighter search around best coarse point
            label='Stage 2 — fine mesh',
        )

        best_x = denorm(s2_best_norm).tolist()
        best_f = s2_best_f

    # ── Save best ───────────────────────────────────────────────
    best = {**base, **dict(zip(names, map(float, best_x)))}
    BEST_FILE.write_text(json.dumps(
        {k: round(float(v), 4) for k, v in best.items()}, indent=2))

    print(f'\n{"="*55}')
    print(f'Best drag: {best_f:.3f} N')
    for k, v in zip(names, best_x):
        print(f'  {k:12s} = {v:.1f}')
    print(f'\nBest config -> {BEST_FILE}')
    print(f'Full log    -> {LOG_FILE}')
    print('\nTo use: copy best_hull.json over hull_design.json and reopen the app.')


if __name__ == '__main__':
    main()
