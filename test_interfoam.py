"""
test_interfoam.py — Short interFoam pipeline test
Reuses the mesh already in ~/openfoam_runs/test_mesh (from test_mesh.py).
Runs: setFields → decomposePar → interFoam (0.5 s) → reconstructPar

Usage:
    source ~/hull-env/bin/activate
    python test_interfoam.py
"""
import re, shutil, subprocess, sys, time
from pathlib import Path

RUN_DIR  = Path.home() / 'openfoam_runs' / 'test_mesh'
OF       = 'source $(ls -1 /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | tail -n 1) 2>/dev/null'
N_CORES  = 8
END_TIME = 0.5   # seconds — just enough to see solver start and forces stabilise


def run_step(label, cmd, cwd=RUN_DIR):
    t0  = time.time()
    ret = subprocess.run(
        ['bash', '--norc', '--noprofile', '-c', f'{OF}; {cmd} > log.{label} 2>&1'],
        cwd=cwd,
    )
    elapsed = time.time() - t0
    status  = 'OK' if ret.returncode == 0 else 'FAILED'
    print(f'  {label:20s} {status}  ({elapsed:.0f}s)')
    if ret.returncode != 0:
        log = RUN_DIR / f'log.{label}'
        if log.exists():
            print('\n'.join(log.read_text().splitlines()[-40:]))
        sys.exit(1)


def patch_end_time(end_time):
    """Set endTime and top-level writeInterval in controlDict.
    Only patches lines before the 'functions' block to avoid touching
    function-object writeInterval entries (which require integers).
    """
    ctrl = RUN_DIR / 'system' / 'controlDict'
    txt  = ctrl.read_text()
    # Split at 'functions' block — only patch the header section
    if 'functions' in txt:
        header, rest = txt.split('functions', 1)
    else:
        header, rest = txt, ''
    header = re.sub(r'(endTime\s+)[\d.]+(\s*;)', f'\\g<1>{end_time}\\2', header)
    header = re.sub(r'(writeInterval\s+)[\d.]+(\s*;)', f'\\g<1>{end_time}\\2', header)
    ctrl.write_text(header + ('functions' + rest if rest else ''))
    print(f'  controlDict: endTime={end_time}, writeInterval={end_time}')


if __name__ == '__main__':
    if not (RUN_DIR / 'constant' / 'polyMesh').exists():
        print('ERROR: No mesh found. Run test_mesh.py first.')
        sys.exit(1)

    print('=== interFoam Pipeline Test ===')
    print(f'Case dir : {RUN_DIR}')
    print(f'End time : {END_TIME} s\n')

    # Clean any previous time directories (keep 0/)
    for d in RUN_DIR.iterdir():
        try:
            t = float(d.name)
            if t > 0:
                shutil.rmtree(d)
        except ValueError:
            pass
    for proc in RUN_DIR.glob('processor*'):
        shutil.rmtree(proc)

    print('Patching controlDict...')
    patch_end_time(END_TIME)

    # Recreate ParaView marker file (may have been wiped)
    (RUN_DIR / 'test_mesh.foam').touch()

    print('\nRunning pipeline...')
    run_step('setFields',      'setFields')
    run_step('decomposePar',   'decomposePar -force')
    run_step('interFoam',      f'mpirun --use-hwthread-cpus -np {N_CORES} interFoam -parallel')
    run_step('reconstructPar', 'reconstructPar -latestTime')

    # Quick drag readout
    force_files = sorted((RUN_DIR / 'postProcessing').rglob('force*.dat')) if \
                  (RUN_DIR / 'postProcessing').exists() else []
    if force_files:
        lines = force_files[-1].read_text().splitlines()
        data  = [l for l in lines if not l.startswith('#') and l.strip()]
        if data:
            last = data[-1].split()
            # columns: time  (fp_x fp_y fp_z)  (fv_x fv_y fv_z)  (pm_x ...)
            try:
                drag = float(last[1]) + float(last[4])   # pressure_x + viscous_x
                print(f'\n  Drag at t={last[0]} s: {drag:.2f} N')
            except (IndexError, ValueError):
                pass

    print('\n=== PASSED ===')
    print(f'View results : paraFoam -case {RUN_DIR}')
