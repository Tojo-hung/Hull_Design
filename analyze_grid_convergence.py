#!/usr/bin/env python3
"""
analyze_grid_convergence.py - Plotting tool for the Grid Convergence V&V Test.

This script automatically finds the most recent grid convergence runs,
extracts the cell counts and drag histories, and plots the results.

Usage:
    source ~/hull-env/bin/activate
    python3 analyze_grid_convergence.py
"""

import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent

def get_latest_runs():
    """Finds all valid simulation directories from the most recent grid convergence batch."""
    batch_dirs = []
    for folder_name in ['optimization_runs', 'validation_runs']:
        d_path = ROOT / folder_name
        if d_path.exists():
            batch_dirs.extend([d for d in d_path.iterdir() if d.is_dir() and d.name.startswith('grid_batch_')])
            
    if not batch_dirs:
        return []
        
    # Sort by name descending (newest first)
    batch_dirs.sort(key=lambda d: d.name, reverse=True)
    latest_batch = batch_dirs[0]
    print(f"Reading from batch: {latest_batch.parent.name}/{latest_batch.name}")
    
    valid_runs = []
    for d in latest_batch.iterdir():
        has_mesh = (d / 'log.checkMesh').exists() or (d / 'log.cartesianMesh').exists()
        if has_mesh and list(d.rglob('force*.dat')):
            valid_runs.append(d)
            
    return valid_runs

def parse_run(run_dir):
    """Extracts cell count and drag history from a run directory."""
    # 1. Parse cell count from mesh logs
    cell_count = 0
    for log_name in ['log.checkMesh', 'log.cartesianMesh']:
        log_file = run_dir / log_name
        if log_file.exists():
            match = re.search(r'(?:Number of cells|cells)[\s:]+(\d+)', log_file.read_text(), re.IGNORECASE)
            if match:
                cell_count = int(match.group(1))
                break
    
    # 2. Parse force data (handling both LTS and Transient phases)
    def get_time(p):
        try:
            return float(p.parent.name)
        except ValueError:
            return -1.0
            
    force_files = sorted((run_dir / 'postProcessing').rglob('force*.dat'), key=get_time)
    if not force_files:
        return None
    
    rows_dict = {}
    for f in force_files:
        for line in f.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            nums = []
            for tok in line.split():
                try:
                    nums.append(float(tok.strip('()')))
                except ValueError:
                    pass
            if len(nums) >= 7:
                # Deduplicate overlaps between multiple force files using the exact simulation time
                time_val = nums[0]
                rows_dict[time_val] = nums
            
    if not rows_dict:
        return None
        
    sorted_times = sorted(rows_dict.keys())
    data = np.array([rows_dict[t] for t in sorted_times])
    # Use sequential indices to prevent LTS (dt=1) vs Transient (dt=0.001) x-axis squashing
    iterations = np.arange(1, len(data) + 1)
    drag_pressure = np.abs(data[:, 1])
    drag_viscous = np.abs(data[:, 4])
    drag = drag_pressure + drag_viscous  # Absolute Total Drag
    
    # Average the final 10% (ignoring the first 90%) to get the converged drag
    avg_idx = int(len(drag) * 0.9)
    final_drag = np.mean(drag[avg_idx:])
    final_pressure = np.mean(drag_pressure[avg_idx:])
    final_viscous = np.mean(drag_viscous[avg_idx:])
    
    # Calculate Standard Error of the Mean (SEM)
    drag_err = np.std(drag[avg_idx:]) / np.sqrt(len(drag[avg_idx:]))
    
    return {
        'dir': run_dir.name,
        'cells': cell_count,
        'iterations': iterations,
        'drag_history': drag,
        'final_drag': final_drag,
        'final_pressure': final_pressure,
        'final_viscous': final_viscous,
        'drag_err': drag_err
    }

def main():
    print("Scanning for the most recent batch of grid convergence runs...")
    runs = get_latest_runs()
        
    if not runs:
        print("No valid runs found yet. Waiting for OpenFOAM to write data...")
        return
        
    print(f"Found {len(runs)} run(s) actively writing force data.")
    
    results = sorted([parse_run(r) for r in runs if parse_run(r)], key=lambda x: x['cells'])
    
    if not results:
        print("Data is not fully written yet. Please wait...")
        return
        
    if len(results) < 2:
        print(f"\nNote: Only {len(results)} mesh level has completed so far. The grid independence curve (right plot) will only show a single point until the next mesh finishes.")
    
    # --- Calculate Grid Convergence Index (GCI) ---
    for i in range(len(results)):
        if i == 0:
            results[i]['gci'] = float('nan')
        else:
            n_fine = results[i]['cells']
            n_coarse = results[i-1]['cells']
            f_fine = results[i]['final_drag']
            f_coarse = results[i-1]['final_drag']
            
            # 3D volumetric refinement ratio
            r = (n_fine / n_coarse)**(1/3)
            # Assuming 2nd order spatial accuracy
            p = 2.0 
            # Factor of safety (1.25 for >=3 grids)
            fs = 1.25 
            
            e_a = abs(f_coarse - f_fine) / f_fine
            results[i]['gci'] = (fs * e_a / ((r**p) - 1.0)) * 100.0

    plt.figure(figsize=(14, 6))
    
    # --- Plot 1: Drag History (Iteration Convergence) ---
    plt.subplot(1, 2, 1)
    for res in results:
        # Skip the first 5 points to keep the y-axis scale readable
        plt.plot(res['iterations'][5:], res['drag_history'][5:], marker='.', label=f"{res['cells']:,} cells")
    plt.title('Solver Convergence History')
    plt.xlabel('Iteration / Time Step')
    plt.ylabel('Drag Force (N)')
    plt.grid(True)
    plt.legend()
    
    # --- Plot 2: Grid Independence (Cell Count Convergence) ---
    plt.subplot(1, 2, 2)
    cells = [res['cells'] for res in results]
    drags = [res['final_drag'] for res in results]
    pressures = [res['final_pressure'] for res in results]
    viscous = [res['final_viscous'] for res in results]
    errs = [res['drag_err'] for res in results]
    
    # Plot Total, Pressure, and Viscous Drag
    plt.errorbar(cells, drags, yerr=errs, marker='o', markersize=8, linestyle='-', linewidth=2, color='#d62728', capsize=5, label='Total Drag')
    plt.plot(cells, pressures, marker='s', markersize=6, linestyle='--', color='#1f77b4', label='Pressure Drag')
    plt.plot(cells, viscous, marker='^', markersize=6, linestyle='-.', color='#2ca02c', label='Viscous Drag')
    
    for i, res in enumerate(results):
        plt.annotate(f"{res['final_drag']:.2f}±{res['drag_err']:.2f} N", (res['cells'], res['final_drag']),
                     textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('Grid Convergence Study')
    plt.xlabel('Mesh Cell Count')
    plt.ylabel('Converged Drag Force (N)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = ROOT / 'grid_convergence_plot.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot successfully generated and saved to:\n{save_path}")
    
    # --- Print Terminal Table ---
    print(f"\n--- Detailed Analysis Results ---")
    print(f"{'Level':<10} {'Cell Count':<15} {'Total Drag':<12} {'Pressure':<12} {'Viscous':<12} {'Std Error':<12} {'GCI'}")
    print("-" * 85)
    for res in results:
        err_str = f"±{res['drag_err']:.3f}"
        gci_str = f"{res['gci']:.2f}%" if not np.isnan(res['gci']) else "N/A"
        print(f"{res['dir']:<10} {res['cells']:<15,d} {res['final_drag']:<12.3f} {res['final_pressure']:<12.3f} {res['final_viscous']:<12.3f} {err_str:<12} {gci_str}")
    print("=" * 85 + "\n")
    
    plt.show()

if __name__ == '__main__':
    main()