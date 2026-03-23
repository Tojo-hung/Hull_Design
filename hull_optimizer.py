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
    'p3_hb':  (140, 230),   # midship half-beam    (current 180)
    'p3_d':   (120, 200),   # midship draft        (current 160)
    'p3_kw':  (40,  130),   # midship keel width   (current 80)
    'p2_hb':  (100, 210),   # forward half-beam    (current 150)
    'p2_d':   (110, 200),   # forward draft        (current 155)
    'p4_hb':  (150, 250),   # aft half-beam        (current 195)
    'p4_d':   (120, 200),   # aft draft            (current 160)
    'p3_x':   (1500, 2000), # midship position     (current 1700)
}
# ══════════════════════════════════════════════════════════════


def load_base_params():
    if CONFIG_FILE.exists():
        data   = json.loads(CONFIG_FILE.read_text())
        merged = {k: float(v) for k, v in DEFAULTS.items()}
        merged.update({k: float(v) for k, v in data.items() if k in merged})
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


def setup_case(run_dir, stl_bytes, speed_ms):
    """Copy template → run_dir, write hull STL, patch inlet speed."""
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    # Use copy (not copy2) to avoid permission errors on Windows NTFS via WSL
    errors = []
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

def objective(x, param_names, base_params, speed_ms, log_rows):
    _n[0] += 1
    n      = _n[0]
    params = {**base_params, **dict(zip(param_names, map(float, x)))}
    run_dir = WORK_DIR / f'run_{n:04d}'

    print(f'\n{"="*55}  Eval {n}  |  {speed_ms:.2f} m/s ({speed_ms*1.944:.1f} kn)')
    for k, v in zip(param_names, x):
        print(f'  {k:12s} = {v:.1f}')

    drag, status = 1e6, 'error'
    try:
        stl  = build_stl_bytes(params)
        setup_case(run_dir, stl, speed_ms)
        run_case(run_dir)
        drag   = extract_drag(run_dir)
        status = 'ok'
        print(f'  -> Drag = {drag:.3f} N')
    except Exception as e:
        print(f'  -> ERROR: {e}')
        status = str(e)

    row = {'n': n, 'drag_N': round(drag, 4),
           'status': status, 'speed_ms': speed_ms}
    row.update(dict(zip(param_names, [round(float(v), 2) for v in x])))
    log_rows.append(row)
    pd.DataFrame(log_rows).to_csv(LOG_FILE, index=False)
    return drag


# ── Main ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed',   type=float, default=3.601,
                    help='Boat speed m/s (default 3.601 ≈ 7.0 knots)')
    ap.add_argument('--n-calls', type=int,   default=30,
                    help='Total evaluations (default 30)')
    ap.add_argument('--n-init',  type=int,   default=8,
                    help='Random initial samples (default 8)')
    ap.add_argument('--resume',  action='store_true',
                    help='Resume from existing optimization_log.csv')
    args = ap.parse_args()

    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        sys.exit('pip install scikit-optimize')

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    base      = load_base_params()
    names     = list(SEARCH_SPACE.keys())
    space     = [Real(lo, hi, name=n) for n, (lo, hi) in SEARCH_SPACE.items()]
    log_rows  = []
    x0 = y0  = None

    if args.resume and LOG_FILE.exists():
        prev = pd.read_csv(LOG_FILE)
        ok   = prev[prev['status'] == 'ok']
        if not ok.empty:
            x0       = ok[names].values.tolist()
            y0       = ok['drag_N'].tolist()
            log_rows = prev.to_dict('records')
            _n[0]    = int(prev['n'].max())
            print(f'Resuming from {len(ok)} previous runs.')

    print(f'\nOptimising {len(names)} parameters  |  {args.n_calls} total evals')
    print(f'Target speed: {args.speed:.2f} m/s  ({args.speed*1.944:.1f} knots)')
    print(f'Params: {names}\n')

    res = gp_minimize(
        func=lambda x: objective(x, names, base, args.speed, log_rows),
        dimensions=space,
        n_calls=args.n_calls,
        n_initial_points=args.n_init,
        x0=x0, y0=y0,
        acq_func='EI',
        noise=1e-3,
        random_state=42,
    )

    best = {**base, **dict(zip(names, map(float, res.x)))}
    BEST_FILE.write_text(json.dumps(
        {k: round(float(v), 4) for k, v in best.items()}, indent=2))

    print(f'\n{"="*55}')
    print(f'Best drag: {res.fun:.3f} N')
    for k, v in zip(names, res.x):
        print(f'  {k:12s} = {v:.1f}')
    print(f'\nBest config -> {BEST_FILE}')
    print(f'Full log    -> {LOG_FILE}')
    print('\nTo use: copy best_hull.json over hull_design.json and reopen the app.')


if __name__ == '__main__':
    main()
