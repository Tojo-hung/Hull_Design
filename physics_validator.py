"""
physics_validator.py - Verification & Validation Suite for the Hull CFD Pipeline.

This script runs a series of benchmark tests to ensure the simulation setup is
physically and numerically sound before running the main optimization. It imports
logic from the main application scripts to ensure consistency.

Usage:
    source ~/hull-env/bin/activate
    python3 physics_validator.py --test <test_name>

Available Tests:
    - archimedes: Checks if buoyant force matches displacement weight (at U=0). comand: python3 physics_validator.py --test archimedes
    - grid:       Runs a 3-level grid convergence study on drag.
    - steady:     Checks if the drag force stabilizes over a long run.
    - sweep:      Sweeps a design parameter to check for a smooth drag landscape.
    - ittc:       Validates skin friction against the theoretical ITTC-1957 line.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path
import numpy as np

# --- Setup Project Environment -----------------------------------------------
# This ensures that the script can find the other project modules.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Import Project-Specific Logic -------------------------------------------
from hull_optimizer import (
    build_geometry_stl,
    setup_case,
    run_cfmesh_pipeline,
    _run_foam_utility,
    modify_control_dict,
    DOMAIN_BOUNDS
)
from moth_designer.config import DEFAULTS, TARGET_DISP_L, LWL

# --- Global Constants --------------------------------------------------------
RUNS_DIR = Path.home() / 'openfoam_runs' / 'validation'
N_CORES = 8

# Physical constants for water at 20°C
RHO = 998.2      # kg/m^3
NU = 1.004e-6    # m^2/s
G = 9.81         # m/s^2
U_SIM = 5.14     # Default simulation velocity, m/s (~10 knots)

# --- Helper Functions --------------------------------------------------------

def run_simulation_pipeline(run_dir, end_time=None):
    """A local runner for the full CFD pipeline: mesh, then solve."""
    print(f"\n--- Running Full Case: {run_dir.name} ---")
    run_cfmesh_pipeline(run_dir)

    if end_time:
        modify_control_dict(run_dir, {'endTime': end_time})

    _run_foam_utility('setFields', 'setFields', cwd=run_dir)
    _run_foam_utility('decomposePar', 'decomposePar -force', cwd=run_dir)
    _run_foam_utility('interFoam', f'mpirun -np {N_CORES} interFoam -parallel', cwd=run_dir)
    _run_foam_utility('reconstructPar', 'reconstructPar -latestTime', cwd=run_dir)
    print(f"--- Case Finished: {run_dir.name} ---")

def parse_forces(run_dir, time_average_from=0.8):
    """Parses force.dat, returning mean forces and full history."""
    force_file = run_dir / 'postProcessing' / 'forces' / '0' / 'force.dat'
    if not force_file.exists():
        raise FileNotFoundError(f"force.dat not found in {run_dir}")

    lines = force_file.read_text().splitlines()
    data = np.loadtxt([line for line in lines if not line.startswith('#')])
    if data.ndim == 1: data = data.reshape(1, -1) # Handle single-line files

    max_time = data[-1, 0]
    avg_start_time = max_time * time_average_from
    avg_idx = np.argmax(data[:, 0] >= avg_start_time)

    mean_forces = np.mean(data[avg_idx:, :], axis=0)
    force_dict = {
        'pressure_x': mean_forces[1], 'pressure_y': mean_forces[2], 'pressure_z': mean_forces[3],
        'viscous_x': mean_forces[4], 'viscous_y': mean_forces[5], 'viscous_z': mean_forces[6],
    }
    force_dict['drag'] = force_dict['pressure_x'] + force_dict['viscous_x']
    force_dict['lift'] = force_dict['pressure_z'] + force_dict['viscous_z']
    return force_dict, data

# --- Test Implementations ----------------------------------------------------

def run_archimedes_check():
    """Verifies that buoyant force on a static hull equals its displacement weight."""
    print("\n" + "="*70 + "\n--- V&V Test 1: Archimedes Buoyancy Check ---\n" + "="*70)
    run_dir = RUNS_DIR / 'archimedes_check'

    print("\n[1/4] Setting up case with default parameters...")

    geom = build_geometry_stl(DEFAULTS)
    setup_case(run_dir, geom, speed_ms=0.0)

    print("\n[2/4] Modifying case for static buoyancy test (U=0)...")
    modify_control_dict(run_dir, {'endTime': 5})
    u_file = run_dir / '0' / 'U'
    content = u_file.read_text()
    content = re.sub(r'internalField\s+uniform\s+\(.*\)', 'internalField   uniform (0 0 0)', content)
    content = re.sub(r'inletValue\s+uniform\s+\(.*\)', 'inletValue      uniform (0 0 0)', content)
    u_file.write_text(content)
    print("  Set velocity field to 0 m/s.")

    print("\n[3/4] Running simulation...")
    run_simulation_pipeline(run_dir)

    print("\n[4/4] Analyzing results...")
    forces, _ = parse_forces(run_dir)
    simulated_force = forces.get('pressure_z', 0.0)
    theoretical_force = TARGET_DISP_L * G
    error = 100 * (simulated_force - theoretical_force) / theoretical_force

    print("\n--- Archimedes Check Results ---")
    print(f"  Target Displacement: {TARGET_DISP_L} kg")
    print(f"  Theoretical Buoyant Force (Z): {theoretical_force:.2f} N")
    print(f"  Simulated Buoyant Force (Z): {simulated_force:.2f} N")
    print(f"  Percentage Error: {error:.2f}%")
    print("\n  PASS: Buoyant force is within 2% of theoretical value." if abs(error) < 2.0 else
          "\n  FAIL: Buoyant force differs significantly from theoretical value.")
    print("="*70 + "\n")

def run_grid_convergence():
    """Runs the same simulation on 3 different mesh densities."""
    print("\n" + "="*70 + "\n--- V&V Test 2: Grid Convergence Study ---\n" + "="*70)
    mesh_levels = {'coarse': 1.25, 'medium': 1.0, 'fine': 0.80}
    results = []

    for i, (name, scale) in enumerate(mesh_levels.items()):
        run_dir = RUNS_DIR / f'grid_study_{name}'
        print(f"\n--- [{i+1}/{len(mesh_levels)}] Starting mesh level: {name.upper()} (scale={scale}) ---")
        geom = build_geometry_stl(DEFAULTS)
        setup_case(run_dir, geom, speed_ms=U_SIM)

        mesh_dict_path = run_dir / 'system' / 'meshDict'
        content = mesh_dict_path.read_text()
        content = re.sub(r'((?:maxCell|boundaryCell|cellSize|maxFirstLayerThickness)\w*\s+)(\S+);',
                         lambda m: f"{m.group(1)}{float(m.group(2))*scale:.4f};", content)
        mesh_dict_path.write_text(content)

        run_simulation_pipeline(run_dir)
        forces, _ = parse_forces(run_dir)
        checkmesh_log = (run_dir / 'log.checkMesh').read_text()
        cell_count = int(re.search(r'cells:\s+(\d+)', checkmesh_log).group(1))
        results.append({'level': name, 'drag': forces['drag'], 'cells': cell_count})

    print("\n--- Grid Convergence Results ---")
    print(f"{'Level':<10} {'Cell Count':<15} {'Drag (N)':<10}")
    print("-"*40)
    for res in results:
        print(f"{res['level']:<10} {res['cells']:,d}{'':<5} {res['drag']:.3f}")
    print("="*70 + "\n")

def run_steady_state_check():
    """Runs a long simulation to find when drag force stabilizes."""
    print("\n" + "="*70 + "\n--- V&V Test 3: Steady-State Extraction Test ---\n" + "="*70)
    run_dir = RUNS_DIR / 'steady_state_test'
    geom = build_geometry_stl(DEFAULTS)
    setup_case(run_dir, geom, speed_ms=U_SIM)
    run_simulation_pipeline(run_dir, end_time=20.0)

    _, history = parse_forces(run_dir)
    times, drags = history[:, 0], history[:, 1] + history[:, 4]
    avg_idx = np.argmax(times >= times[-1] * 0.8)
    stable_drags = drags[avg_idx:]
    mean_drag, std_dev = np.mean(stable_drags), np.std(stable_drags)
    variation = 100 * (std_dev / abs(mean_drag)) if mean_drag != 0 else 0

    print("\n--- Steady-State Analysis (last 20% of run) ---")
    print(f"  Mean Drag: {mean_drag:.3f} N")
    print(f"  Coefficient of Variation: {variation:.2f}%")
    print("\n  PASS: Drag is stable (variation < 1%)." if variation < 1.0 else
          "\n  WARN: Drag may not be fully stable (variation >= 1%).")
    print("="*70 + "\n")

def run_parameter_sweep():
    """Sweeps a single design parameter to see its effect on drag."""
    print("\n" + "="*70 + "\n--- V&V Test 4: Parameter Sweep (Max Beam) ---\n" + "="*70)
    param, base_val = 'p3_hb', DEFAULTS['p3_hb']
    results = []

    for i, value in enumerate(np.linspace(base_val * 0.9, base_val * 1.1, 5)):
        run_dir = RUNS_DIR / f'param_sweep_{i+1}'
        print(f"\n--- [{i+1}/5] Running sweep for {param} = {value:.1f} mm ---")
        params = DEFAULTS.copy(); params[param] = value
        geom = build_geometry_stl(params)
        setup_case(run_dir, geom, speed_ms=U_SIM)
        run_simulation_pipeline(run_dir)
        forces, _ = parse_forces(run_dir)
        results.append({'param_val': value, 'drag': forces['drag']})

    print(f"\n--- Parameter Sweep Results ---\n{param:<15} {'Drag (N)':<10}\n" + "-"*30)
    for res in results:
        print(f"{res['param_val']:<15.1f} {res['drag']:.3f}")
    print("="*70 + "\n")

def run_ittc_friction_check():
    """Simulates a flat plate to validate skin friction against the ITTC line."""
    print("\n" + "="*70 + "\n--- V&V Test 5: ITTC-1957 Skin Friction Check ---\n" + "="*70)
    run_dir, L, W = RUNS_DIR / 'ittc_check', LWL / 1000.0, 0.5

    # Generate a simple flat plate STL
    pxn, pxx, pyn, pyx, pzn, pzx = 0, L, -W/2, W/2, -0.001, 0.001
    v = [(pxn,pyn,pzn), (pxx,pyn,pzn), (pxx,pyx,pzn), (pxn,pyx,pzn),
         (pxn,pyn,pzx), (pxx,pyn,pzx), (pxx,pyx,pzx), (pxn,pyx,pzx)]
    faces = [(0,1,2,3), (7,6,5,4), (0,4,5,1), (1,5,6,2), (2,6,7,3), (3,7,4,0)]
    normals = [(0,0,-1), (0,0,1), (0,-1,0), (1,0,0), (0,1,0), (-1,0,0)]
    geom_lines = ["solid plate\n"]
    for i, face in enumerate(faces):
        for tri in [(face[0], face[1], face[2]), (face[0], face[2], face[3])]:
            geom_lines.append(f"  facet normal {' '.join(map(str, normals[i]))}\n    outer loop\n")
            for idx in tri: geom_lines.append(f"      vertex {' '.join(map(str, v[idx]))}\n")
            geom_lines.append("    endloop\n  endfacet\n")
    geom_lines.append("endsolid plate\n")

    setup_case(run_dir, "".join(geom_lines), speed_ms=U_SIM)
    mesh_dict_path = run_dir / 'system' / 'meshDict'
    content = mesh_dict_path.read_text().replace('hull', 'plate')
    mesh_dict_path.write_text(content)

    run_simulation_pipeline(run_dir)
    forces, _ = parse_forces(run_dir)
    viscous_drag = forces['viscous_x']

    wetted_area = L * W * 2
    Re = U_SIM * L / NU
    Cf_ittc = 0.075 / ((np.log10(Re) - 2)**2)
    Cf_sim = viscous_drag / (0.5 * RHO * (U_SIM**2) * wetted_area)
    error = 100 * (Cf_sim - Cf_ittc) / Cf_ittc

    print("\n--- ITTC Friction Check Results ---")
    print(f"  Reynolds Number (Re): {Re:.2e}")
    print(f"  ITTC-1957 Friction Coeff (Cf): {Cf_ittc:.5f}")
    print(f"  Simulated Friction Coeff (Cf): {Cf_sim:.5f}")
    print(f"  Percentage Error: {error:.2f}%")
    print("\n  PASS: Simulated friction is within 5% of ITTC-1957 line." if abs(error) < 5.0 else
          "\n  FAIL: Simulated friction differs significantly from theory.")
    print("="*70 + "\n")

# --- Main CLI Router ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Physics Validation Suite for the Hull CFD Pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    test_choices = {
        'archimedes': run_archimedes_check,
        'grid': run_grid_convergence,
        'steady': run_steady_state_check,
        'sweep': run_parameter_sweep,
        'ittc': run_ittc_friction_check,
    }

    parser.add_argument(
        '--test',
        required=True,
        choices=test_choices.keys(),
        help="Select the validation test to run:\n" +
             "  archimedes: Checks if buoyant force matches displacement weight.\n" +
             "  grid:       Runs a 3-level grid convergence study.\n" +
             "  steady:     Checks if the drag force stabilizes over time.\n" +
             "  sweep:      Sweeps a parameter to check for a smooth drag landscape.\n" +
             "  ittc:       Validates skin friction against the ITTC-1957 formula."
    )

    args = parser.parse_args()
    RUNS_DIR.mkdir(exist_ok=True)
    selected_test_func = test_choices[args.test]
    
    try:
        selected_test_func()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n--- A test step failed ---")
        print(f"ERROR: {e}")
        print("Check the log files in the corresponding run directory for details.")
        sys.exit(1)