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
import datetime
import json
import os
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
LOG_FILE     = ROOT / 'optimization_log.csv'
BEST_FILE    = ROOT / 'best_hull.json'
CONFIG_FILE  = ROOT / 'hull_design.json'
STUDY_DB     = ROOT / 'optuna_study.db'
STUDY_NAME   = 'moth_hull'

DEFAULT_N_PROCS = 16
FAST_END_TIME = 0.5
OF      = 'source $(ls -1 /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | tail -n 1) 2>/dev/null'

DOMAIN_BOUNDS = {
    'xn': -1.0, 'xx': 5.0,  # m
    'yn': -1.5, 'yx': 1.5,  # m
    'zn': -0.8, 'zx': 0.5,  # m
}

sys.path.insert(0, str(ROOT))
from moth_designer.config   import DEFAULTS, LWL, TARGET_DISP_L
from moth_designer.geometry import (build_ctrl, build_3d_mesh, beam_eval,
                                    find_disp_waterline, hydrostatic_coefficients,
                                    lagrange)

# ══════════════════════════════════════════════════════════════
# SEARCH SPACE — edit bounds to suit your study
# ══════════════════════════════════════════════════════════════
# Longitudinal anchor positions (mm from bow)
X_BASE = {
    'p1_x': 335,    # 10% LWL — bow entry
    'p2_x': 1000,   # 30% LWL — forward shoulder
    'p3_x': 1700,   # 51% LWL — max section
    'p4_x': 2500,   # 75% LWL — aft shoulder
}

FIXED_PARAMS = {
    'transom_half_beam': 195.0,
    'transom_draft': 65.0,
    'transom_sheer': 200.0,
    'p1_dz': 200.0,
    'p2_dz': 200.0,
    'p3_dz': 200.0,
    'p4_dz': 200.0,
}

SEARCH_SPACE = {
    # Station X positions
    'p1_x':  (X_BASE['p1_x'] - 200, X_BASE['p1_x'] + 200),
    'p2_x':  (X_BASE['p2_x'] - 200, X_BASE['p2_x'] + 200),
    'p3_x':  (X_BASE['p3_x'] - 200, X_BASE['p3_x'] + 200),
    'p4_x':  (X_BASE['p4_x'] - 200, X_BASE['p4_x'] + 200),

    # Midship (p3)
    'p3_hb':  (140, 270),   # half-beam (mm)
    'p3_d':   (110, 215),   # draft
    'p3_kw':  (20,  150),   # keel width
    'p3_hz':  (-100, 100),  # beam height offset (V vs U shape)

    # Forward (p2)
    'p2_hb':  (130, 250),   # half-beam
    'p2_d':   (120, 230),   # draft
    'p2_kw':  (5,    70),   # keel width
    'p2_hz':  (-100, 100),  # beam height offset

    # Aft (p4)
    'p4_hb':  (145, 260),   # half-beam
    'p4_d':   (110, 220),   # draft
    'p4_kw':  (20,  150),   # keel width
    'p4_hz':  (-100, 100),  # beam height offset

    # Bow (p1) — no hz (too narrow to matter)
    'p1_hb':  (40,  160),   # half-beam
    'p1_d':   (50,  170),   # draft
    'p1_kw':  (5,    70),   # keel width
    'bow_draft': (10, 140),
}
# ══════════════════════════════════════════════════════════════

def find_openfoam_exe(executable_name):
    # 1. System PATH
    path = shutil.which(executable_name)
    if path:
        return path

    # 2. Active Environment Variables
    for env_var in ['FOAM_USER_APPBIN', 'FOAM_APPBIN']:
        env_path = os.environ.get(env_var)
        if env_path:
            candidate = Path(env_path) / executable_name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())

    # 3. Local User Compilations
    home = Path.home()
    for candidate in home.glob(f"OpenFOAM/*/platforms/*/bin/{executable_name}"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    # 4. Global System Installations
    for candidate in Path('/usr/lib/openfoam').glob(f"*/platforms/*/bin/{executable_name}"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    for candidate in Path('/opt').glob(f"openfoam*/platforms/*/bin/{executable_name}"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    raise FileNotFoundError(f"Could not find OpenFOAM executable: {executable_name}")


def load_base_params():
    """Load hull params, preferring hull_design.json over best_hull.json."""
    src = CONFIG_FILE if CONFIG_FILE.exists() else BEST_FILE
    if src.exists():
        data   = json.loads(src.read_text())
        merged = {k: float(v) for k, v in DEFAULTS.items()}
        merged.update({k: float(v) for k, v in data.items() if k in merged})
        merged.update(FIXED_PARAMS)
        print(f'Base params loaded from: {src.name}')
        return merged
    merged = {k: float(v) for k, v in DEFAULTS.items()}
    merged.update(FIXED_PARAMS)
    return merged


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


CP_MIN, CP_MAX       = 0.55, 0.85
ROCKER_MIN, ROCKER_MAX = 60.0, 130.0   # mm


def check_constraints(params):
    """Check prismatic coefficient and rocker before running CFD.
    Returns (ok: bool, reason: str).

    Rocker = max rise of the keel above the straight chord line that joins
    the lowest bow point to the lowest stern point.
    """
    ctrl_x, beam_hb, keel_d, deck_hb, sheer_z, keel_w, hb_z, *_ = build_ctrl(params)
    bow_d   = float(params['bow_draft'])
    stern_d = float(params['transom_draft'])
    draft_floor = max(bow_d, stern_d)

    # ── Station draft floor ───────────────────────────────────
    for name in ('p1_d', 'p2_d', 'p3_d', 'p4_d'):
        draft = float(params[name])
        if draft < draft_floor:
            return False, f'{name}={draft:.1f} mm  (need >= {draft_floor:.1f} mm from bow/transom draft)'

    # ── Rocker ────────────────────────────────────────────────
    x_arr   = np.linspace(0.0, float(LWL), 300)
    depths  = lagrange(x_arr, ctrl_x, keel_d, clip_min=0.0)
    chord   = bow_d + (stern_d - bow_d) * (x_arr / float(LWL))
    rocker  = float(np.max(depths - chord))
    if not (ROCKER_MIN <= rocker <= ROCKER_MAX):
        return False, f'rocker={rocker:.1f} mm  (need {ROCKER_MIN}–{ROCKER_MAX})'

    # ── Prismatic coefficient ──────────────────────────────────
    hs = hydrostatic_coefficients(
        TARGET_DISP_L, ctrl_x, beam_hb, ctrl_x, keel_d,
        deck_x=ctrl_x, deck_hb_arr=deck_hb,
        sheer_z_arr=sheer_z, keel_w_arr=keel_w,
        hb_z_arr=hb_z,
    )
    cp = hs['Cp']
    if np.isnan(cp) or not (CP_MIN <= cp <= CP_MAX):
        return False, f'Cp={cp:.3f}  (need {CP_MIN}–{CP_MAX})'

    return True, f'ok  Cp={cp:.3f}  rocker={rocker:.1f} mm'


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
    ctrl_x, beam_hb, keel_d, deck_hb_c, sheer_z_c, keel_w_c, hb_z_c, *_ = \
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
    xn, xx, yn, yx, zn, zx = [DOMAIN_BOUNDS[k] for k in ['xn', 'xx', 'yn', 'yx', 'zn', 'zx']]
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


FAST_MESH_DICT = """\
FoamFile { version 2.0; format ascii; class dictionary; object meshDict; }
surfaceFile     "constant/triSurface/geometry.stl";
maxCellSize     0.40;
boundaryCellSize                    0.12;
boundaryCellSizeRefinementThickness 0.24;
localRefinement { hull { cellSize 0.07; } }
meshQualitySettings { maxNonOrthogonality 70; maxSkewness 6; minTetQuality -1e30; }
"""


def apply_fast_mode(run_dir, end_time=FAST_END_TIME):
    """Laptop-friendly coarse mesh + short sim for quick ranking. Target ~30k cells."""
    (run_dir / 'system' / 'meshDict').write_text(FAST_MESH_DICT)
    ctrl = run_dir / 'system' / 'controlDict'
    txt  = ctrl.read_text()
    header, rest = txt.split('functions', 1)
    header = re.sub(r'(endTime\s+)[\d.]+(\s*;)',      f'\\g<1>{end_time}\\2', header)
    header = re.sub(r'(writeInterval\s+)[\d.]+(\s*;)', f'\\g<1>{end_time}\\2', header)
    ctrl.write_text(header + 'functions' + rest)


def create_daily_run_dir():
    """Creates a sequentially numbered, date-based run directory."""
    project_root = Path(__file__).resolve().parent
    base_dir = project_root / 'optimization_runs'
    base_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    
    existing_dirs = [d for d in base_dir.glob(f'{today_str}_*') if d.is_dir()]
    next_num = 1
    for d in existing_dirs:
        try:
            num = int(d.name.split('_')[-1])
            if num >= next_num:
                next_num = num + 1
        except ValueError:
            pass
            
    run_dir = base_dir / f'{today_str}_{next_num:02d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def setup_case(geometry_stl, speed_ms, run_dir=None):
    """Create daily run dir (or use provided), copy template, write STL, patch inlet speed."""
    if run_dir is None:
        run_dir = create_daily_run_dir()
    else:
        run_dir = Path(run_dir)
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / 'case.foam').touch()

    shutil.copytree(TEMPLATE_DIR, run_dir,
                    copy_function=shutil.copy,
                    ignore_dangling_symlinks=True,
                    dirs_exist_ok=True)

    tri_dir = run_dir / 'constant' / 'triSurface'
    tri_dir.mkdir(parents=True, exist_ok=True)

    # 1. STL COMBINING STEP (Fixing the Ghost Hull)
    # The geometry_stl string natively contains BOTH the boat hull and the 
    # bounding box domain explicitly unified into a single ASCII STL file.
    # cfMesh requires them in the same file to properly recognize the hull patch!
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
    return run_dir


def modify_control_dict(case_dir, settings):
    """Programmatically changes settings in system/controlDict."""
    cd_path = Path(case_dir) / 'system' / 'controlDict'
    content = cd_path.read_text()

    if 'functions' in content:
        header, rest = content.split('functions', 1)
    else:
        header, rest = content, ''

    for key, value in settings.items():
        header, count = re.subn(rf"({key}\s+)\S+;", rf"\g<1>{value};", header)
        if count > 0:
            print(f"  controlDict: Set {key} to {value}")
    cd_path.write_text(header + ('functions' + rest if rest else ''))


def _run_foam_utility(label, cmd, cwd, check_log_for=None):
    """Helper to run an OpenFOAM utility, log output, and check for success."""
    cmds = cmd.split()
    exe_name = cmds[0]

    if exe_name == 'mpirun':
        mpirun_path = shutil.which('mpirun')
        if not mpirun_path:
            raise FileNotFoundError("mpirun not found. Is OpenMPI installed?")
        solver_name = next((arg for arg in cmds if 'Foam' in arg), None)
        if not solver_name:
            raise ValueError(f"Could not find OpenFOAM solver in mpirun command: {cmd}")
        solver_path = find_openfoam_exe(solver_name)
        run_cmd = cmd.replace('mpirun', mpirun_path).replace(solver_name, str(solver_path))
        lib_dir = Path(solver_path).parent.parent / 'lib'
    else:
        exe_path = find_openfoam_exe(exe_name)
        run_cmd = cmd.replace(exe_name, str(exe_path))
        lib_dir = Path(exe_path).parent.parent / 'lib'

    log_path = cwd / f'log.{label}'
    full_cmd = f'{OF}; export LD_LIBRARY_PATH="{lib_dir}:$LD_LIBRARY_PATH"; {run_cmd} > {log_path} 2>&1'

    t0 = time.time()
    print(f'  Starting {label}...')
    process = subprocess.Popen(['bash', '--norc', '--noprofile', '-c', full_cmd], cwd=cwd)

    while True:
        try:
            retcode = process.wait(timeout=60.0)
            break
        except subprocess.TimeoutExpired:
            elapsed_min = (time.time() - t0) / 60.0
            print(f'    [{label}] Still running... ({elapsed_min:.1f} min elapsed)')

    elapsed = time.time() - t0
    status = 'ok' if retcode == 0 else 'FAILED'
    print(f'  {label:25s} {status}  ({elapsed:.1f}s)')

    log_content = log_path.read_text() if log_path.exists() else ""

    if retcode != 0:
        print('\n'.join(log_content.splitlines()[-20:]))
        raise RuntimeError(f'{label} failed. See log: {log_path}')

    if check_log_for and check_log_for not in log_content:
        print('\n'.join(log_content.splitlines()[-20:]))
        raise RuntimeError(f"'{check_log_for}' not found in log for {label}. See log: {log_path}")


def patch_boundary_conditions(case_dir):
    boundary_file = Path(case_dir) / 'constant' / 'polyMesh' / 'boundary'
    if not boundary_file.is_file():
        return
    print('\n--- Patching Boundary Conditions ---')
    content = boundary_file.read_text()
    patch_types = {'inlet': 'patch', 'outlet': 'patch', 'atmosphere': 'patch',
                   'bottom': 'wall', 'front': 'wall', 'back': 'wall', 'hull': 'wall'}
    for patch_name, patch_type in patch_types.items():
        pattern = re.compile(rf"({patch_name}\s+{{(?:(?!^\s*}}).)*?type\s+)\w+", re.DOTALL | re.MULTILINE)
        content, count = pattern.subn(rf"\g<1>{patch_type}", content)
        if count > 0: print(f"  {patch_name:12s} -> type {patch_type}")
    boundary_file.write_text(content)
    print('--- Boundary Patching Finished ---')


def set_decompose_subdomains(case_dir, n_procs):
    decompose_path = Path(case_dir) / 'system' / 'decomposeParDict'
    if not decompose_path.is_file():
        return
    content = decompose_path.read_text()
    content, count = re.subn(
        r'(numberOfSubdomains\s+)\d+(\s*;)',
        rf'\g<1>{n_procs}\2',
        content,
    )
    if count > 0:
        decompose_path.write_text(content)
        print(f'  decomposeParDict: Set numberOfSubdomains to {n_procs}')


def run_cfmesh_pipeline(case_dir):
    print('\n--- Starting Professional cfMesh Pipeline ---')
    case_dir = Path(case_dir)
    tri_surface_dir = case_dir / 'constant' / 'triSurface'
    stl_path = tri_surface_dir / 'geometry.stl'
    fms_path = tri_surface_dir / 'geometry.fms'
    if not stl_path.exists(): raise FileNotFoundError(f"STL file not found at {stl_path}")
    _run_foam_utility('surfaceFeatureEdges', f'surfaceFeatureEdges -angle 15 {stl_path.name} {fms_path.name}', cwd=tri_surface_dir)
    mesh_dict_path = case_dir / 'system' / 'meshDict'

    cx = (DOMAIN_BOUNDS['xx'] + DOMAIN_BOUNDS['xn']) / 2.0
    cy = (DOMAIN_BOUNDS['yx'] + DOMAIN_BOUNDS['yn']) / 2.0
    cz = 0.0
    lx = DOMAIN_BOUNDS['xx'] - DOMAIN_BOUNDS['xn']
    ly = DOMAIN_BOUNDS['yx'] - DOMAIN_BOUNDS['yn']
    lz = 0.07
    
    # Dynamically scale the nested wake box dimensions (converting mm to m)
    boat_l = LWL / 1000.0
    boat_beam = (DEFAULTS['p3_hb'] * 2.0) / 1000.0
    boat_draft = DEFAULTS['p3_d'] / 1000.0
    
    inner_lx = boat_l * 1.5
    inner_ly = boat_beam * 2.0
    inner_lz = boat_draft * 4.0

    outer_lx = boat_l * 3.0
    outer_ly = boat_beam * 4.0
    outer_lz = boat_draft * 8.0

    # Shift the boxes longitudinally so they center on the boat instead of the bow
    # This pushes the back half of the boxes out to cover the stern and the wake
    box_cx = boat_l / 2.0

    md_text = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      meshDict;
}}

surfaceFile     "constant/triSurface/{fms_path.name}";

// Global Coarse Grid
maxCellSize     0.40;

localRefinement
{{
    // Far-Field Fix: Force boundary patches to remain coarse
    "(inlet|outlet|atmosphere|bottom|front|back)"
    {{
        cellSize    0.40;
    }}
    
    // Hull Surface & Grading
    hull
    {{
        cellSize                    0.02;
        additionalRefinementLevels  3;
    }}
}}

objectRefinements
{{
    // Waterline Refinement
    waterline
    {{
        type        box;
        cellSize    0.02;
        centre      ({cx} {cy} {cz});
        lengthX     {lx};
        lengthY     {ly};
        lengthZ     {lz};
    }}
    
    // Inner Wake Box
    wakeBoxInner
    {{
        type        box;
        cellSize    0.05;
        centre      ({box_cx} 0 0);
        lengthX     {inner_lx};
        lengthY     {inner_ly};
        lengthZ     {inner_lz};
    }}

    // Outer Wake Box
    wakeBoxOuter
    {{
        type        box;
        cellSize    0.15;
        centre      ({box_cx} 0 0);
        lengthX     {outer_lx};
        lengthY     {outer_ly};
        lengthZ     {outer_lz};
    }}
}}

// Mesh Quality Settings for cfMesh
meshQualitySettings
{{
    maxNonOrthogonality     70;
    maxSkewness             8;
    minTetQuality           -1e30;
}}
"""
    mesh_dict_path.write_text(md_text)
    _run_foam_utility('cartesianMesh', 'cartesianMesh', cwd=case_dir)

    try:
        _run_foam_utility('checkMesh', 'checkMesh -latestTime', cwd=case_dir)
    except RuntimeError:
        print('  [Warning] checkMesh reported imperfect quality (e.g., highly skewed faces). Proceeding anyway.')
        
    (case_dir / f'{case_dir.name}.foam').touch()
    patch_boundary_conditions(case_dir)
    print('--- cfMesh Pipeline Finished ---')


def run_case(run_dir, n_procs=DEFAULT_N_PROCS):
    """Run the full CFD simulation for a given case directory."""
    run_cfmesh_pipeline(run_dir)
    print('\n--- Starting interFoam Simulation ---')
    _run_foam_utility('setFields', 'setFields', cwd=run_dir)
    set_decompose_subdomains(run_dir, n_procs)
    _run_foam_utility('decomposePar', 'decomposePar -force', cwd=run_dir)
    _run_foam_utility('interFoam', f'mpirun --use-hwthread-cpus -np {n_procs} interFoam -parallel', cwd=run_dir)
    _run_foam_utility('reconstructPar', 'reconstructPar -latestTime', cwd=run_dir)
    print('--- interFoam Simulation Finished ---\n')


def run_lts_simulation(case_dir, n_cores=DEFAULT_N_PROCS):
    """Run steady-state Local Time Stepping (LTS) simulation for faster optimization."""
    case_dir = Path(case_dir)
    run_cfmesh_pipeline(case_dir)
    print('\n--- Starting interFoam (LTS mode) Simulation ---')

    # 1. Modify controlDict settings for LTS
    modify_control_dict(case_dir, {
        'application': 'interFoam',
        'endTime': '2000',
        'deltaT': '1',
        'writeControl': 'timeStep',
        'writeInterval': '2000'
    })

    # 2. Modify fvSchemes to use localEuler for ddtSchemes
    fv_schemes_path = case_dir / 'system' / 'fvSchemes'
    if fv_schemes_path.exists():
        content = fv_schemes_path.read_text()
        content, count = re.subn(
            r'(ddtSchemes\s*\{[^}]*?default\s+)[^;]+(;)',
            r'\g<1>localEuler\2',
            content
        )
        if count > 0:
            print("  fvSchemes: Set ddtSchemes default to localEuler")
        fv_schemes_path.write_text(content)

    # 3. Execution Pipeline
    _run_foam_utility('setFields', 'setFields', cwd=case_dir)
    set_decompose_subdomains(case_dir, n_cores)
    _run_foam_utility('decomposePar', 'decomposePar -force', cwd=case_dir)

    print(f'  Starting interFoam (LTS mode) on {n_cores} cores (streaming output to terminal)...')
    mpirun_path = shutil.which('mpirun')
    if not mpirun_path:
        raise FileNotFoundError("mpirun not found. Is OpenMPI installed?")
    
    solver_path = find_openfoam_exe('interFoam')
    lib_dir = Path(solver_path).parent.parent / 'lib'
    
    run_cmd = f'{mpirun_path} --use-hwthread-cpus -np {n_cores} {solver_path} -parallel'
    full_cmd = f'{OF}; export LD_LIBRARY_PATH="{lib_dir}:$LD_LIBRARY_PATH"; {run_cmd}'
    
    try:
        subprocess.check_call(['bash', '--norc', '--noprofile', '-c', full_cmd], cwd=case_dir)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'interFoam (LTS mode) failed with exit code {e.returncode}')

    try:
        _run_foam_utility('reconstructPar', 'reconstructPar -latestTime', cwd=case_dir)
    except RuntimeError:
        print("  reconstructPar failed (likely no write times). Skipping visual reconstruction.")
    print('--- interFoam (LTS mode) Simulation Finished ---\n')


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
def make_objective(base_params, speed_ms, log_rows, fast=False,
                   log_file=None, n_procs=DEFAULT_N_PROCS):
    import optuna
    _log_file = log_file or LOG_FILE

    def objective(trial):
        # Suggest parameter values from search space
        params = {**base_params}
        draft_names = {'p1_d', 'p2_d', 'p3_d', 'p4_d'}
        for name, (lo, hi) in SEARCH_SPACE.items():
            if name == 'bow_draft' or name in draft_names:
                continue
            params[name] = trial.suggest_float(name, lo, hi)

        bow_lo, bow_hi = SEARCH_SPACE['bow_draft']
        params['bow_draft'] = trial.suggest_float('bow_draft', bow_lo, bow_hi)

        draft_floor = max(float(params['bow_draft']), float(params['transom_draft']))
        for name in ('p1_d', 'p2_d', 'p3_d', 'p4_d'):
            lo, hi = SEARCH_SPACE[name]
            params[name] = trial.suggest_float(name, max(lo, draft_floor), hi)

        params.update(FIXED_PARAMS)

        # Fixed bow params
        params['p1_hz'] = 0.0

        n       = trial.number + 1

        print(f'\n{"="*55}  Trial {n}  |  {speed_ms:.2f} m/s ({speed_ms*1.944:.1f} kn)')
        for name in SEARCH_SPACE:
            print(f'  {name:20s} = {params[name]:.1f}')

        # Check constraints before running CFD — log and prune if failed
        ok, reason = check_constraints(params)
        print(f'  Constraints: {reason}')
        if not ok:
            row = {'n': n, 'drag_N': None, 'status': f'pruned: {reason}', 'speed_ms': speed_ms}
            row.update({k: round(params[k], 2) for k in SEARCH_SPACE})
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(_log_file, index=False)
            raise optuna.TrialPruned()

        drag, status = 1e6, 'error'
        try:
            stl_nx = 60 if fast else 250
            stl_nt = 24 if fast else 100
            geom = build_geometry_stl(params, N_X=stl_nx, N_T=stl_nt)
            run_dir = setup_case(geom, speed_ms)
            if fast:
                apply_fast_mode(run_dir, end_time=FAST_END_TIME)
            run_case(run_dir, n_procs=n_procs)
            drag   = extract_drag(run_dir)
            status = 'ok'
            print(f'  -> Drag = {drag:.3f} N')
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f'  -> ERROR: {e}')
            status = f'failed: {e}'
            row = {'n': n, 'drag_N': None, 'status': status, 'speed_ms': speed_ms}
            row.update({k: round(params[k], 2) for k in SEARCH_SPACE})
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(_log_file, index=False)
            raise optuna.TrialPruned()

        row = {'n': n, 'drag_N': round(drag, 4),
               'status': status, 'speed_ms': speed_ms}
        row.update({k: round(params[k], 2) for k in SEARCH_SPACE})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(_log_file, index=False)

        return drag

    return objective


def get_study_progress(study, optuna):
    counts = {state.name.lower(): 0 for state in optuna.trial.TrialState}
    for trial in study.trials:
        counts[trial.state.name.lower()] += 1
    counts['total'] = len(study.trials)
    return counts


def get_log_progress(log_rows):
    counts = {'ok': 0, 'pruned': 0, 'failed': 0, 'other': 0, 'total': len(log_rows)}
    for row in log_rows:
        status = str(row.get('status', '')).strip().lower()
        if status == 'ok':
            counts['ok'] += 1
        elif status.startswith('pruned:'):
            counts['pruned'] += 1
        elif status.startswith('failed:'):
            counts['failed'] += 1
        else:
            counts['other'] += 1
    return counts


# ── Main ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed',    type=float, default=3.601,
                    help='Boat speed m/s (default 3.601 ≈ 7.0 knots)')
    ap.add_argument('--n-trials', type=int,   default=150,
                    help='Target number of completed Optuna trials / interFoam runs (default 150)')
    ap.add_argument('--resume',    action='store_true',
                    help='Resume from existing Optuna study database')
    ap.add_argument('--eval-best', action='store_true',
                    help='Run a single evaluation on best_hull.json and exit')
    ap.add_argument('--fast',      action='store_true',
                    help='Low-fidelity mode: coarse mesh + 0.5s sim for quick exploration')
    ap.add_argument('--n-procs',   type=int, default=DEFAULT_N_PROCS,
                    help='MPI ranks for interFoam (default 16 for 8C/16T laptop)')
    ap.add_argument('--tag',       type=str, default='',
                    help='Optional suffix for study/log/work dirs so a run starts cleanly')
    args = ap.parse_args()

    try:
        import optuna
    except ImportError:
        sys.exit('pip install optuna')

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tag_suffix = f'_{args.tag}' if args.tag else ''
    study_name = ('moth_hull_fast' if args.fast else STUDY_NAME) + tag_suffix
    study_db   = ROOT / ((f'optuna_study_fast{tag_suffix}.db')      if args.fast else f'optuna_study{tag_suffix}.db')
    log_file   = ROOT / ((f'optimization_log_fast{tag_suffix}.csv') if args.fast else f'optimization_log{tag_suffix}.csv')
    best_file  = ROOT / (f'best_hull{tag_suffix}.json' if args.tag else BEST_FILE.name)

    base     = load_base_params()
    log_rows = pd.read_csv(log_file).to_dict('records') if log_file.exists() else []

    # ── Single eval of best_hull.json ───────────────────────────
    if args.eval_best:
        print('\nRunning single evaluation on best config...')
        params  = {**base}
        try:
            geom = build_geometry_stl(params)
            run_dir = setup_case(geom, args.speed)
            run_case(run_dir, n_procs=args.n_procs)
            drag = extract_drag(run_dir)
            print(f'\nDrag for best config: {drag:.3f} N')
        except Exception as e:
            print(f'ERROR: {e}')
        return

    # ── Optuna study ─────────────────────────────────────────────
    storage    = f'sqlite:///{study_db}'
    load_exist = study_db.exists()

    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        load_if_exists = load_exist,
        direction      = 'minimize',
        sampler        = optuna.samplers.TPESampler(seed=42),
    )

    progress = get_study_progress(study, optuna)
    log_progress = get_log_progress(log_rows)
    print(f'\n{"="*55}')
    print(f'Optuna TPE  |  target completed trials={args.n_trials}  |  completed so far={progress["complete"]}')
    print(f'Study DB    : {study_db}')
    print(f'Mode        : {f"FAST (coarse mesh, {FAST_END_TIME:.1f}s sim)" if args.fast else "full fidelity"}')
    print(f'MPI ranks   : {args.n_procs}')
    print(f'Tracked log : total={log_progress["total"]}  pruned={log_progress["pruned"]}  failed={log_progress["failed"]}')
    print(f'Params      : {list(SEARCH_SPACE.keys())}\n')

    objective = make_objective(base, args.speed, log_rows, fast=args.fast,
                               log_file=log_file, n_procs=args.n_procs)
    while progress['complete'] < args.n_trials:
        remaining = args.n_trials - progress['complete']
        print(f'Running up to {remaining} more Optuna attempts to reach {args.n_trials} completed trials...')
        study.optimize(objective, n_trials=remaining)
        progress = get_study_progress(study, optuna)
        log_progress = get_log_progress(log_rows)
        print(f'Progress    : attempts={log_progress["total"]}  complete={progress["complete"]}  pruned={log_progress["pruned"]}  failed={log_progress["failed"]}')

    if progress['complete'] == 0:
        sys.exit('No completed trials are available yet, so there is no best hull to save.')

    # ── Save best ────────────────────────────────────────────────
    best_trial  = study.best_trial
    best_params = {**base, **best_trial.params}
    best_file.write_text(json.dumps(
        {k: round(float(v), 4) for k, v in best_params.items()}, indent=2))

    print(f'\n{"="*55}')
    print(f'Best drag   : {best_trial.value:.3f} N  (trial #{best_trial.number + 1})')
    print(f'Completed   : {progress["complete"]} / {args.n_trials}')
    print(f'Pruned      : {log_progress["pruned"]}')
    print(f'Failed      : {log_progress["failed"]}')
    print(f'Total tries : {log_progress["total"]}')
    for k, v in best_trial.params.items():
        print(f'  {k:20s} = {v:.1f}')
    print(f'\nBest config -> {best_file}')
    print(f'Full log    -> {log_file}')
    print(f'Study DB    -> {study_db}')
    print('\nTo use: load best_hull.json in the desktop app or copy values into hull_design.json.')


if __name__ == '__main__':
    main()
