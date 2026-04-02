"""
Create a hull JSON file from a selected trial in an optimization log CSV.

Examples:
    python3 generate_trial_json.py \
        --csv optimization_log_fast_laptop30_20260402_constraints.csv \
        --trial 11

    python3 generate_trial_json.py \
        --csv optimization_log_fast_laptop30_20260402_constraints.csv \
        --trial 11 \
        --output hull_trial_11.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from hull_optimizer import BEST_FILE, CONFIG_FILE, FIXED_PARAMS
from moth_designer.config import DEFAULTS


ROOT = Path(__file__).parent


def load_base_params(base_file=None):
    params = {k: float(v) for k, v in DEFAULTS.items()}

    if base_file:
        candidates = [Path(base_file)]
    else:
        candidates = [CONFIG_FILE, BEST_FILE]

    for candidate in candidates:
        if candidate.exists():
            data = json.loads(candidate.read_text())
            params.update({k: float(v) for k, v in data.items() if k in params})
            break

    params.update({k: float(v) for k, v in FIXED_PARAMS.items()})
    return params


def read_trial_row(csv_path, trial_number):
    with Path(csv_path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                if int(row['n']) == trial_number:
                    return row
            except (KeyError, TypeError, ValueError):
                continue
    raise ValueError(f'Trial {trial_number} not found in {csv_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True,
                        help='Optimization log CSV to read from')
    parser.add_argument('--trial', type=int, required=True,
                        help='Trial number from the CSV column "n"')
    parser.add_argument('--output', type=str, default='',
                        help='Output JSON path (default: trial_<n>.json beside the CSV)')
    parser.add_argument('--base', type=str, default='',
                        help='Optional base JSON to merge onto before applying the trial row')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f'CSV not found: {csv_path}')

    row = read_trial_row(csv_path, args.trial)
    params = load_base_params(args.base or None)

    ignored = {'n', 'drag_N', 'status', 'speed_ms'}
    for key, value in row.items():
        if key in ignored or value in ('', None):
            continue
        if key not in params:
            continue
        params[key] = float(value)

    if not args.output:
        stem = csv_path.stem
        output_path = csv_path.with_name(f'{stem}_trial_{args.trial}.json')
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path

    output_path.write_text(
        json.dumps({k: round(float(v), 4) for k, v in params.items()}, indent=2) + '\n'
    )

    print(f'Wrote {output_path}')
    print(f'Trial   : {args.trial}')
    print(f'Status  : {row.get("status", "")}')
    print(f'Drag (N): {row.get("drag_N", "")}')


if __name__ == '__main__':
    main()
