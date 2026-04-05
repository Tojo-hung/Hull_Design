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
RUNS_DIR = ROOT / 'optimization_runs'

def get_latest_runs(n=3):
    """Finds the most recent valid simulation directories."""
    if not RUNS_DIR.exists():
        print(f"Error: {RUNS_DIR} not found.")
        return []
    
    # Sort directories by modification time (newest first)
    dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    
    valid_runs = []
    for d in dirs:
        if (d / 'log.checkMesh').exists() and list(d.rglob('force*.dat')):
            valid_runs.append(d)
        if len(valid_runs) == n:
            break
            
    # Return chronologically (oldest of the 3 first -> coarse, medium, fine)
    return valid_runs[::-1]

def parse_run(run_dir):
    """Extracts cell count and drag history from a run directory."""
    # 1. Parse cell count from checkMesh log
    checkmesh_log = (run_dir / 'log.checkMesh').read_text()
    match = re.search(r'cells:\s+(\d+)', checkmesh_log)
    cell_count = int(match.group(1)) if match else 0
    
    # 2. Parse force data from the LTS run
    force_files = sorted((run_dir / 'postProcessing').rglob('force*.dat'))
    if not force_files:
        return None
    
    rows = []
    for line in force_files[-1].read_text().splitlines():
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
        return None
        
    data = np.array(rows)
    iterations = data[:, 0]
    drag = np.abs(data[:, 1] + data[:, 4])  # Absolute Drag (Pressure X + Viscous X)
    
    # Average the final 20% to get the formal converged drag
    avg_idx = int(len(drag) * 0.8)
    final_drag = np.mean(drag[avg_idx:])
    
    return {
        'dir': run_dir.name,
        'cells': cell_count,
        'iterations': iterations,
        'drag_history': drag,
        'final_drag': final_drag
    }

def main():
    print("Scanning for the 3 most recent grid convergence runs...")
    runs = get_latest_runs(3)
    if len(runs) < 3:
        print("Warning: Could not find 3 valid runs. Did you run the grid convergence test?")
        
    results = sorted([parse_run(r) for r in runs if parse_run(r)], key=lambda x: x['cells'])
    
    plt.figure(figsize=(14, 6))
    
    # --- Plot 1: Drag History (Iteration Convergence) ---
    plt.subplot(1, 2, 1)
    for res in results:
        plt.plot(res['iterations'], res['drag_history'], marker='.', label=f"{res['cells']:,} cells")
    plt.title('LTS Solver Convergence History')
    plt.xlabel('Iteration')
    plt.ylabel('Drag Force (N)')
    plt.grid(True)
    plt.legend()
    
    # --- Plot 2: Grid Independence (Cell Count Convergence) ---
    plt.subplot(1, 2, 2)
    cells = [res['cells'] for res in results]
    drags = [res['final_drag'] for res in results]
    
    plt.plot(cells, drags, marker='o', markersize=8, linestyle='-', linewidth=2, color='#d62728')
    
    for i, res in enumerate(results):
        plt.annotate(f"{res['final_drag']:.2f} N", (res['cells'], res['final_drag']),
                     textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('Grid Convergence Study')
    plt.xlabel('Mesh Cell Count')
    plt.ylabel('Converged Drag Force (N)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = ROOT / 'grid_convergence_plot.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot successfully generated and saved to:\n{save_path}")
    plt.show()

if __name__ == '__main__':
    main()