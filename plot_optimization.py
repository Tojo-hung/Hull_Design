"""
plot_optimization.py — visualize hull optimizer progress
Run from WSL:   python3 plot_optimization.py
Run on Windows: python plot_optimization.py
Saves plots to optimization_plots.png in the same directory.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LOG_FILE  = Path(__file__).parent / 'optimization_log.csv'
OUT_FILE  = Path(__file__).parent / 'optimization_plots.png'

if not LOG_FILE.exists():
    sys.exit(f'No log file found at {LOG_FILE}')

df = pd.read_csv(LOG_FILE)
ok = df[df['status'] == 'ok'].copy()

if ok.empty:
    sys.exit('No successful evaluations yet.')

params = [c for c in ok.columns
          if c not in ('n', 'drag_N', 'status', 'speed_ms', 'fidelity')]

has_fidelity = 'fidelity' in ok.columns

# ── Layout ──────────────────────────────────────────────────────
n_params = len(params)
n_cols   = 4
n_rows   = (n_params + n_cols - 1) // n_cols

fig = plt.figure(figsize=(18, 5 + 3 * n_rows), facecolor='#ffffff')
gs  = gridspec.GridSpec(2 + n_rows, n_cols,
                        figure=fig, hspace=0.55, wspace=0.35)

def ax_style(ax, title):
    ax.set_facecolor('#f5f7fa')
    ax.set_title(title, color='#1a1a2e', fontsize=9, pad=4)
    ax.tick_params(colors='#444444', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
    ax.xaxis.label.set_color('#444444')
    ax.yaxis.label.set_color('#444444')

# ── 1. Drag vs evaluation (full width) ──────────────────────────
ax_drag = fig.add_subplot(gs[0, :])
ax_style(ax_drag, 'Drag vs Evaluation')

if has_fidelity:
    for fid, color, label in [('low', '#1a7abf', 'Coarse mesh'),
                               ('high', '#d94f4f', 'Fine mesh')]:
        sub = ok[ok['fidelity'] == fid]
        if not sub.empty:
            ax_drag.scatter(sub['n'], sub['drag_N'], c=color, s=20,
                            alpha=0.7, label=label, zorder=3)
else:
    ax_drag.scatter(ok['n'], ok['drag_N'], c='#1a7abf', s=20,
                    alpha=0.7, zorder=3)

# Running minimum line
running_min = ok['drag_N'].cummin()
ax_drag.plot(ok['n'], running_min, color='#e67e00', linewidth=1.5,
             label=f'Best: {ok["drag_N"].min():.2f} N', zorder=4)

ax_drag.set_xlabel('Evaluation #')
ax_drag.set_ylabel('Drag (N)')
ax_drag.legend(fontsize=8, facecolor='#ffffff', edgecolor='#cccccc')
ax_drag.grid(True, color='#dddddd', linewidth=0.5)

# ── 2. Drag distribution histogram ──────────────────────────────
ax_hist = fig.add_subplot(gs[1, :2])
ax_style(ax_hist, 'Drag Distribution')
ax_hist.hist(ok['drag_N'], bins=20, color='#1a7abf', alpha=0.7,
             edgecolor='#ffffff')
ax_hist.axvline(ok['drag_N'].min(), color='#e67e00', linewidth=1.5,
                label=f'Best: {ok["drag_N"].min():.2f} N')
ax_hist.set_xlabel('Drag (N)')
ax_hist.set_ylabel('Count')
ax_hist.legend(fontsize=8, facecolor='#ffffff', edgecolor='#cccccc')
ax_hist.grid(True, color='#dddddd', linewidth=0.5)

# ── 3. Best params bar chart ─────────────────────────────────────
ax_bar = fig.add_subplot(gs[1, 2:])
ax_style(ax_bar, 'Best Hull Parameters (normalised to search range)')
best_row = ok.loc[ok['drag_N'].idxmin()]
from hull_optimizer import SEARCH_SPACE
norm_vals = [(best_row[p] - SEARCH_SPACE[p][0]) /
             (SEARCH_SPACE[p][1] - SEARCH_SPACE[p][0])
             for p in params if p in SEARCH_SPACE]
p_labels  = [p for p in params if p in SEARCH_SPACE]
colors    = ['#1a7abf' if v < 0.3 or v > 0.7 else '#e67e00'
             for v in norm_vals]
bars = ax_bar.barh(p_labels, norm_vals, color=colors, edgecolor='#ffffff')
ax_bar.axvline(0.5, color='#999999', linewidth=0.8, linestyle='--')
ax_bar.set_xlim(0, 1)
ax_bar.set_xlabel('Position in search range  (0=min, 1=max)')
ax_bar.grid(True, color='#dddddd', linewidth=0.5, axis='x')

# ── 4. Parameter scatter plots ───────────────────────────────────
for idx, p in enumerate(params):
    if p not in ok.columns:
        continue
    row = 2 + idx // n_cols
    col = idx % n_cols
    ax = fig.add_subplot(gs[row, col])
    ax_style(ax, p)
    sc = ax.scatter(ok['n'], ok[p], c=ok['drag_N'],
                    cmap='coolwarm_r', s=15, alpha=0.8,
                    vmin=ok['drag_N'].quantile(0.1),
                    vmax=ok['drag_N'].quantile(0.9))
    # Mark best
    ax.scatter(best_row['n'], best_row[p], c='#e67e00',
               s=60, marker='*', zorder=5)
    ax.set_xlabel('Eval #', fontsize=7)
    ax.set_ylabel('mm', fontsize=7)
    ax.grid(True, color='#dddddd', linewidth=0.4)

fig.suptitle(
    f'Hull Optimisation  |  {len(ok)} evals  |  '
    f'Best drag: {ok["drag_N"].min():.2f} N  (eval {ok["drag_N"].idxmin()+1})',
    color='#1a1a2e', fontsize=12, y=0.995)

plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight',
            facecolor='#ffffff')
print(f'Saved -> {OUT_FILE}')
