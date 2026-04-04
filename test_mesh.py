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
from hull_optimizer import build_geometry_stl, setup_case, find_openfoam_exe


def load_params():
    params = {k: float(v) for k, v in DEFAULTS.items()}
    if CONFIG_FILE.exists():
        data = json.loads(CONFIG_FILE.read_text())
        changed = False
        for k, v in data.items():
            if k in params:
                new_val = float(v)
                if params[k] != new_val:
                    if not changed:
                        print("  Parameters changed from defaults:")
                        changed = True
                    print(f"    {k:15s}: {params[k]:.1f} -> {new_val:.1f}")
                params[k] = new_val
    return params


def run_step(label, cmd):
    exe_path = find_openfoam_exe(cmd.split()[0])
    run_cmd = cmd.replace(cmd.split()[0], exe_path)
    from pathlib import Path
    lib_dir = Path(exe_path).parent.parent / 'lib'
    of_env = 'source $(ls -1 /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | tail -n 1) 2>/dev/null'
    t0  = time.time()
    print(f'  Starting {label}...')
    process = subprocess.Popen(
        ['bash', '--norc', '--noprofile', '-c', f'{of_env}; export LD_LIBRARY_PATH="{lib_dir}:$LD_LIBRARY_PATH"; {run_cmd} > log.{label} 2>&1'],
        cwd=RUN_DIR
    )
    while True:
        try:
            retcode = process.wait(timeout=60.0)
            break
        except subprocess.TimeoutExpired:
            elapsed_min = (time.time() - t0) / 60.0
            print(f'    [{label}] Still running... ({elapsed_min:.1f} min elapsed)')

    elapsed = time.time() - t0
    status  = 'OK' if retcode == 0 else 'FAILED'
    print(f'  {label:20s} {status}  ({elapsed:.0f}s)')
    if retcode != 0:
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
    setup_case(geom, speed_ms=3.601, run_dir=RUN_DIR)
    print('  Done\n')

    print('Running cartesianMesh...')
    run_step('cartesianMesh', 'cartesianMesh')

    print('\nChecking mesh quality...')
    run_step('checkMesh', 'checkMesh')

    print('\n=== PASSED — mesh generated successfully ===')
    print(f'Inspect mesh: paraFoam -case {RUN_DIR}')
    print(f'Check quality: source /usr/lib/openfoam/openfoam2406/etc/bashrc && checkMesh -case {RUN_DIR}')