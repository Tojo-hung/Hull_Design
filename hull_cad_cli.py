#!/usr/bin/env python3
"""
hull_cad_cli.py  —  CAD proof-path entry point.

Run from the repo root (WSL or Linux):

    # Use the default hull parameters from moth_designer/config.py:
    python hull_cad_cli.py --defaults --output-dir ./cad_output

    # Use an existing hull config JSON:
    python hull_cad_cli.py --config hull_design.json --output-dir ./cad_output
    python hull_cad_cli.py --config best_hull.json   --output-dir ./cad_output

    # Compare CAD STL with Python triangulator STL:
    python hull_cad_cli.py --defaults --compare --output-dir ./cad_output

Outputs (in --output-dir):
    hull.step         — watertight STEP solid (for Onshape / FreeCAD / GMSH)
    hull_raw.stl      — binary STL in mm, design coordinate system
    hull_cfd.stl      — ASCII STL in m, waterline at z=0 (for cfMesh)
    report.txt        — validation summary

The existing Python geometry path is NOT modified or replaced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Default domain bounds (must match hull_optimizer.py DOMAIN_BOUNDS) ────────
_DEFAULT_DOMAIN_BOUNDS = {
    "xn": -1.0, "xx": 5.0,
    "yn": -1.5, "yx": 1.5,
    "zn": -0.8, "zx": 0.5,
}


def _load_params(args) -> dict:
    """Return the flat hull parameter dict from CLI arguments."""
    if args.defaults:
        from moth_designer.config import DEFAULTS
        return dict(DEFAULTS)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(config_path.read_text())

    # Merge with DEFAULTS so missing keys get sensible values
    from moth_designer.config import DEFAULTS
    merged = {k: float(v) for k, v in DEFAULTS.items()}
    merged.update({k: float(v) for k, v in raw.items() if k in merged})
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAD proof-path: generate STEP + STL from hull parameters",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--defaults",
        action="store_true",
        help="Use the default hull from moth_designer/config.py",
    )
    source.add_argument(
        "--config",
        metavar="FILE",
        help="Path to a hull JSON config file (hull_design.json or best_hull.json)",
    )

    parser.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        default="./cad_output",
        help="Output directory (default: ./cad_output)",
    )
    parser.add_argument(
        "--n-stations",
        type=int,
        default=40,
        metavar="N",
        help="Number of loft stations (default: 40)",
    )
    parser.add_argument(
        "--n-pts",
        type=int,
        default=48,
        metavar="N",
        help="Section profile points per half-section (default: 48)",
    )
    parser.add_argument(
        "--linear-deflection",
        type=float,
        default=0.5,
        metavar="MM",
        help="STL tessellation linear deflection in mm (default: 0.5)",
    )
    parser.add_argument(
        "--no-cfd-stl",
        action="store_true",
        help="Skip CFD STL generation (STEP and raw STL only)",
    )
    parser.add_argument(
        "--no-domain-box",
        action="store_true",
        help="Omit domain bounding-box faces from CFD STL",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also generate the Python triangulator STL and compare",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg)

    # ── Import CAD builder ────────────────────────────────────────────────────
    try:
        from hull_cad.builder import (
            build_hull_solid,
            export_step,
            export_stl,
            export_cfd_stl,
            compare_with_python_stl,
            describe,
        )
    except ImportError as exc:
        print(f"ERROR: cannot import hull_cad.builder: {exc}", file=sys.stderr)
        print(
            "  Install cadquery:  conda install -c cadquery cadquery",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = _load_params(args)
    source_name = "defaults" if args.defaults else Path(args.config).name
    log(f"\nCAD proof path — source: {source_name}")
    log(f"Output directory: {output_dir.resolve()}")
    log(f"Stations: {args.n_stations}  |  Section pts: {args.n_pts}")

    # ── Build solid ───────────────────────────────────────────────────────────
    log("\n[1/4] Building hull solid...")
    t0 = time.perf_counter()
    try:
        result = build_hull_solid(
            params,
            n_stations=args.n_stations,
            n_pts=args.n_pts,
        )
    except Exception as exc:
        print(f"\nERROR: build_hull_solid failed: {exc}", file=sys.stderr)
        sys.exit(1)
    dt_build = time.perf_counter() - t0

    log(f"   Done in {dt_build:.1f}s")
    log(describe(result))

    # ── Export STEP ───────────────────────────────────────────────────────────
    log("\n[2/4] Exporting STEP...")
    t0 = time.perf_counter()
    step_path = output_dir / "hull.step"
    try:
        export_step(result, step_path)
        log(f"   Written: {step_path}  ({step_path.stat().st_size / 1024:.0f} kB)")
    except Exception as exc:
        print(f"  WARNING: STEP export failed: {exc}", file=sys.stderr)
    log(f"   Done in {time.perf_counter() - t0:.1f}s")

    # ── Export raw STL (mm, design CS) ────────────────────────────────────────
    log("\n[3/4] Exporting raw STL (mm, design CS)...")
    t0 = time.perf_counter()
    raw_stl_path = output_dir / "hull_raw.stl"
    try:
        export_stl(
            result,
            raw_stl_path,
            linear_deflection_mm=args.linear_deflection,
        )
        log(f"   Written: {raw_stl_path}  ({raw_stl_path.stat().st_size / 1024:.0f} kB)")
    except Exception as exc:
        print(f"  WARNING: raw STL export failed: {exc}", file=sys.stderr)
    log(f"   Done in {time.perf_counter() - t0:.1f}s")

    # ── Export CFD STL ────────────────────────────────────────────────────────
    if not args.no_cfd_stl:
        log("\n[4/4] Exporting CFD STL (m, waterline at z=0)...")
        t0 = time.perf_counter()
        cfd_stl_path = output_dir / "hull_cfd.stl"
        domain = None if args.no_domain_box else _DEFAULT_DOMAIN_BOUNDS
        try:
            export_cfd_stl(
                result,
                params,
                cfd_stl_path,
                domain_bounds=domain,
                linear_deflection_mm=args.linear_deflection,
            )
            log(f"   Written: {cfd_stl_path}  ({cfd_stl_path.stat().st_size / 1024:.0f} kB)")
        except Exception as exc:
            print(f"  WARNING: CFD STL export failed: {exc}", file=sys.stderr)
        log(f"   Done in {time.perf_counter() - t0:.1f}s")
    else:
        log("\n[4/4] CFD STL skipped (--no-cfd-stl)")

    # ── Optional Python comparison ────────────────────────────────────────────
    if args.compare:
        log("\n[+] Comparing with Python triangulator...")
        cfd_stl_path_for_compare = output_dir / "hull_cfd.stl"
        try:
            cmp = compare_with_python_stl(
                result,
                params,
                cad_stl_path=cfd_stl_path_for_compare,
                python_stl_path=output_dir / "python_geometry.stl",
                domain_bounds=_DEFAULT_DOMAIN_BOUNDS,
            )
            log(f"   CAD triangles   : {cmp.cad_n_triangles:,}")
            log(f"   Python triangles: {cmp.python_n_triangles:,}")
            log(f"   CAD file size   : {cmp.cad_file_bytes / 1024:.0f} kB")
            log(f"   Python file size: {cmp.python_file_bytes / 1024:.0f} kB")
            log(f"   Bbox match      : {cmp.bbox_match}")
            log(f"   CAD bbox (m)    : x=[{cmp.cad_bbox[0]:.3f}, {cmp.cad_bbox[1]:.3f}]  "
                f"y=[{cmp.cad_bbox[2]:.3f}, {cmp.cad_bbox[3]:.3f}]  "
                f"z=[{cmp.cad_bbox[4]:.3f}, {cmp.cad_bbox[5]:.3f}]")
            log(f"   Py  bbox (m)    : x=[{cmp.python_bbox[0]:.3f}, {cmp.python_bbox[1]:.3f}]  "
                f"y=[{cmp.python_bbox[2]:.3f}, {cmp.python_bbox[3]:.3f}]  "
                f"z=[{cmp.python_bbox[4]:.3f}, {cmp.python_bbox[5]:.3f}]")
        except Exception as exc:
            log(f"   Comparison failed: {exc}")

    # ── Write report ──────────────────────────────────────────────────────────
    report_lines = [
        "CAD proof-path report",
        f"Source: {source_name}",
        "",
        describe(result),
        "",
        f"Build time: {dt_build:.2f}s",
        f"STEP: {step_path}",
        f"Raw STL: {raw_stl_path}",
    ]
    if not args.no_cfd_stl:
        report_lines.append(f"CFD STL: {output_dir / 'hull_cfd.stl'}")

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    log(f"\nReport: {report_path}")

    # ── Exit status ───────────────────────────────────────────────────────────
    if not result.is_valid:
        print("\nWARNING: solid did not pass BRepCheck_Analyzer", file=sys.stderr)
        sys.exit(2)           # exit 2 = built but invalid (not a crash)

    log("\nDone.")


if __name__ == "__main__":
    main()
