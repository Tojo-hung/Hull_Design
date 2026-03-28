"""
test_mesh.py — Quick cfMesh smoke test
Generates geometry, sets up the case, and runs cartesianMesh only.
No interFoam — just verifies the STL generation and meshing pipeline.

Usage:
    source ~/hull-env/bin/activate
    python test_mesh.py
"""
import json, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT         = Path(__file__).parent
TEMPLATE_DIR = ROOT / 'openfoam_template'
RUN_DIR      = Path.home() / 'openfoam_runs' / 'test_mesh'
CONFIG_FILE  = ROOT / 'hull_design.json'
OF           = 'source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null'

sys.path.insert(0, str(ROOT))
from moth_designer.config   import DEFAULTS
from hull_optimizer import build_geometry_stl, setup_case


def load_params():
    params = {k: float(v) for k, v in DEFAULTS.items()}
    if CONFIG_FILE.exists():
        data = json.loads(CONFIG_FILE.read_text())
        params.update({k: float(v) for k, v in data.items() if k in params})
    return params


def run_step(label, cmd):
    t0  = time.time()
    ret = subprocess.run(
        ['bash', '--norc', '--noprofile', '-c', f'{OF}; {cmd} > log.{label} 2>&1'],
        cwd=RUN_DIR
    )
    elapsed = time.time() - t0
    status  = 'OK' if ret.returncode == 0 else 'FAILED'
    print(f'  {label:20s} {status}  ({elapsed:.0f}s)')
    if ret.returncode != 0:
        log = RUN_DIR / f'log.{label}'
        if log.exists():
            print('\n'.join(log.read_text().splitlines()[-30:]))
        sys.exit(1)


if __name__ == '__main__':
    RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
    print('=== cfMesh Smoke Test ===')
    print(f'Run dir: {RUN_DIR}\n')

    print('Generating geometry STL (hull + domain box)...')
    params = load_params()
    geom   = build_geometry_stl(params, N_X=80, N_T=32)   # low-res for speed
    chars  = len(geom)
    tris   = geom.count('endfacet')
    print(f'  {tris} triangles  ({chars//1024} KB)\n')

    print('Setting up case...')
    setup_case(RUN_DIR, geom, speed_ms=3.601)
    print('  Done\n')

    print('Running cartesianMesh...')
    run_step('cartesianMesh', 'cartesianMesh')

    print('\n=== PASSED — mesh generated successfully ===')
    print(f'Inspect mesh: paraFoam -case {RUN_DIR}')
    print(f'Check quality: source /usr/lib/openfoam/openfoam2406/etc/bashrc && checkMesh -case {RUN_DIR}')