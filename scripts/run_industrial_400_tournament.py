#!/usr/bin/env python3
"""
Industrial 400 SAT Competition Tournament Runner: 4-Way Head-to-Head
====================================================================
Rigorous, publication-grade tournament evaluating:
  1. Kissat Stock 4.0.1 (Armin Biere official baseline)
  2. CaDiCaL Stock 2.1.3 (Armin Biere official baseline)
  3. M3.5 Version 2 (Microsecond AVX2 SIMD guidance on MiniSat 2.2)
  4. Folder 52 / NeuroCDCL (Continuous Causal Implication Flow on Kissat 4.0.1)

Evaluates Across:
  - 400 Official SAT Competition Industrial Benchmarks (UNSEEN_400_MANIFEST.csv)
  - SAT Competition PAR-2 scoring (Penalized Average Runtime = 2x timeout on cutoff)
  - Incremental CSV logging and real-time console telemetry.
"""

import os
import sys
import re
import time
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent

# Solver Binary Paths (Windows / POSIX dual support)
EXT = ".exe" if os.name == "nt" else ""
SOLVERS = {
    "Kissat_Stock": str(BASE_DIR / "solvers/kissat_stock" / f"kissat{EXT}"),
    "CaDiCaL_Stock": str(BASE_DIR / "solvers/cadical_stock" / f"cadical{EXT}"),
    "M3_5_V2": str(BASE_DIR / "solvers/m3_5_v2" / f"m3_5_v2{EXT}"),
    "NeuroCDCL_Folder52": str(BASE_DIR / "solvers/neuro_cdcl_f52" / f"neuro_cdcl{EXT}"),
}

MANIFEST_DEFAULT = BASE_DIR / "data/industrial_400/UNSEEN_400_MANIFEST.csv"
OUTPUT_CSV_DEFAULT = BASE_DIR / "results/authoritative_tables/industrial_400_tournament_results.csv"

def parse_solver_output(solver_name: str, stdout: str, stderr: str, elapsed_wall: float, retcode: int) -> dict:
    """Robust unified telemetry parser for Kissat, CaDiCaL, and M3.5 V2."""
    status = "UNKNOWN"
    decisions = 0
    conflicts = 0
    props = 0
    cpu_time = elapsed_wall

    text = stdout + "\n" + stderr

    # 1. Determine Status
    if "s SATISFIABLE" in text or "Status:         SATISFIABLE" in text or retcode == 10:
        status = "SAT"
    elif "s UNSATISFIABLE" in text or "Status:         UNSATISFIABLE" in text or retcode == 20:
        status = "UNSAT"
    elif "INDETERMINATE" in text or retcode == 0:
        status = "UNKNOWN"

    # 2. Solver-specific parsing
    if "Kissat" in solver_name or "NeuroCDCL" in solver_name:
        for line in text.splitlines():
            line_s = line.strip()
            if "decisions:" in line_s:
                parts = line_s.split()
                try: decisions = int(parts[parts.index("decisions:") + 1])
                except (ValueError, IndexError): pass
            elif "conflicts:" in line_s:
                parts = line_s.split()
                try: conflicts = int(parts[parts.index("conflicts:") + 1])
                except (ValueError, IndexError): pass
            elif "propagations:" in line_s:
                parts = line_s.split()
                try: props = int(parts[parts.index("propagations:") + 1])
                except (ValueError, IndexError): pass
            elif "process-time:" in line_s:
                parts = line_s.split()
                try:
                    time_str = parts[parts.index("process-time:") + 2].replace("seconds", "").strip()
                    cpu_time = float(time_str)
                except (ValueError, IndexError): pass

    elif "CaDiCaL" in solver_name:
        for line in text.splitlines():
            line_s = line.strip()
            if "decisions:" in line_s:
                parts = line_s.split()
                try: decisions = int(parts[parts.index("decisions:") + 1])
                except (ValueError, IndexError): pass
            elif "conflicts:" in line_s:
                parts = line_s.split()
                try: conflicts = int(parts[parts.index("conflicts:") + 1])
                except (ValueError, IndexError): pass
            elif "propagations:" in line_s:
                parts = line_s.split()
                try: props = int(parts[parts.index("propagations:") + 1])
                except (ValueError, IndexError): pass
            elif "total process time since initialization:" in line_s:
                parts = line_s.split()
                try:
                    idx = parts.index("initialization:")
                    cpu_time = float(parts[idx + 1])
                except (ValueError, IndexError): pass

    elif "M3_5" in solver_name:
        for line in text.splitlines():
            line_s = line.strip()
            if line_s.startswith("Decisions:"):
                try: decisions = int(line_s.split(":")[1].strip())
                except ValueError: pass
            elif line_s.startswith("Conflicts:"):
                try: conflicts = int(line_s.split(":")[1].strip())
                except ValueError: pass
            elif line_s.startswith("Propagations:"):
                try: props = int(line_s.split(":")[1].strip())
                except ValueError: pass
            elif "TOTAL LATENCY:" in line_s:
                try:
                    ms_val = float(line_s.split("(")[1].split("ms")[0].strip())
                    cpu_time = ms_val / 1000.0
                except (ValueError, IndexError): pass

    return {
        "status": status,
        "decisions": decisions,
        "conflicts": conflicts,
        "propagations": props,
        "cpu_time": cpu_time,
        "wall_time": elapsed_wall
    }

def run_instance(solver_name: str, exe_path: str, cnf_path: str, timeout_sec: float) -> dict:
    """Executes a single solver on a single CNF instance under strict process isolation."""
    if not os.path.exists(exe_path):
        return {
            "solver": solver_name,
            "instance": Path(cnf_path).name,
            "status": "MISSING_BINARY",
            "decisions": -1,
            "conflicts": -1,
            "propagations": -1,
            "time": timeout_sec,
            "par2": 2.0 * timeout_sec
        }

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [exe_path, str(cnf_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            errors="replace"
        )
        elapsed_wall = time.perf_counter() - t0
        metrics = parse_solver_output(solver_name, proc.stdout, proc.stderr, elapsed_wall, proc.returncode)
        
        is_solved = metrics["status"] in ["SAT", "UNSAT"]
        par2 = metrics["cpu_time"] if is_solved else (2.0 * timeout_sec)

        return {
            "solver": solver_name,
            "instance": Path(cnf_path).name,
            "status": metrics["status"],
            "decisions": metrics["decisions"],
            "conflicts": metrics["conflicts"],
            "propagations": metrics["propagations"],
            "time": metrics["cpu_time"],
            "par2": par2
        }

    except subprocess.TimeoutExpired:
        return {
            "solver": solver_name,
            "instance": Path(cnf_path).name,
            "status": "TIMEOUT",
            "decisions": -1,
            "conflicts": -1,
            "propagations": -1,
            "time": timeout_sec,
            "par2": 2.0 * timeout_sec
        }
    except Exception as e:
        return {
            "solver": solver_name,
            "instance": Path(cnf_path).name,
            "status": f"ERROR_{type(e).__name__}",
            "decisions": -1,
            "conflicts": -1,
            "propagations": -1,
            "time": timeout_sec,
            "par2": 2.0 * timeout_sec
        }

def print_summary_table(df_results: pd.DataFrame):
    """Computes and prints the publication-style PAR-2 and decision summary."""
    print("\n" + "=" * 90)
    print(" 4-WAY INDUSTRIAL TOURNAMENT: OFFICIAL SAT COMPETITION SUMMARY")
    print("=" * 90)
    
    summary = []
    for s_name in SOLVERS.keys():
        sdf = df_results[df_results["solver"] == s_name]
        if sdf.empty:
            continue
        total = len(sdf)
        sat = len(sdf[sdf["status"] == "SAT"])
        unsat = len(sdf[sdf["status"] == "UNSAT"])
        solved = sat + unsat
        timeouts = len(sdf[sdf["status"] == "TIMEOUT"])
        mean_par2 = sdf["par2"].mean()
        
        solved_df = sdf[sdf["status"].isin(["SAT", "UNSAT"])]
        mean_dec = solved_df["decisions"].mean() if not solved_df.empty else 0.0
        mean_time = solved_df["time"].mean() if not solved_df.empty else 0.0

        summary.append({
            "Solver Architecture": s_name,
            "Solved (SAT/UNS)": f"{solved}/{total} ({sat}/{unsat})",
            "Timeouts": timeouts,
            "Solved Mean Dec": f"{mean_dec:,.0f}" if mean_dec > 0 else "N/A",
            "Solved Mean Time": f"{mean_time:.3f}s" if mean_time > 0 else "N/A",
            "PAR-2 Score": f"{mean_par2:.2f}s"
        })

    sum_df = pd.DataFrame(summary)
    print(sum_df.to_string(index=False))
    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Run 4-Way Industrial 400 SAT Tournament (Kissat Stock vs CaDiCaL Stock vs M3.5 V2 vs Folder 52)"
    )
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT),
                        help="Path to UNSEEN_400_MANIFEST.csv")
    parser.add_argument("--bench-dir", type=str, default="",
                        help="Path to directory containing benchmark CNF files (optional, searches manifest relative paths or local fallback)")
    parser.add_argument("--timeout", type=float, default=1200.0,
                        help="Per-instance timeout cutoff in seconds (SAT Competition standard is 1200.0s)")
    parser.add_argument("--limit", type=int, default=400,
                        help="Limit execution to first N instances (useful for quick verification)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Number of concurrent solver workers (default: 1 for single-core pinning)")
    parser.add_argument("--output-csv", type=str, default=str(OUTPUT_CSV_DEFAULT),
                        help="Target CSV path to write tournament results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify solver binaries, manifest, and print schedule without launching runs")
    args = parser.parse_args()

    print("=" * 90)
    print(" NEURO-CDCL RESEARCH PLATFORM: 4-WAY INDUSTRIAL SAT TOURNAMENT")
    print("=" * 90)
    print(f" Cutoff Timeout:      {args.timeout:.1f}s (PAR-2 Penalty: {2.0 * args.timeout:.1f}s)")
    print(f" Concurrency Workers: {args.max_workers} physical core(s)")
    print(f" Output Results CSV:  {args.output_csv}")
    print("-" * 90)
    print(" Solvers Verified:")
    for name, path in SOLVERS.items():
        exists = os.path.exists(path)
        status_str = f"FOUND ({path})" if exists else f"MISSING ({path})"
        print(f"   [{'OK' if exists else 'FAIL'}] {name:<20}: {status_str}")

    # Check manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"\n[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    df_manifest = pd.read_csv(manifest_path)
    print(f" Loaded manifest:     {manifest_path.name} ({len(df_manifest)} instances indexed)")

    # Resolve benchmark paths
    bench_candidates = []
    bench_dir = Path(args.bench_dir) if args.bench_dir else None

    for _, row in df_manifest.iterrows():
        fname = row["filename"]
        target_path = None
        
        # Check explicit bench_dir
        if bench_dir and (bench_dir / fname).exists():
            target_path = bench_dir / fname
        # Check relative_path from manifest
        elif "relative_path" in row and (BASE_DIR / row["relative_path"]).exists():
            target_path = BASE_DIR / row["relative_path"]
        # Check standard data search paths
        elif (BASE_DIR / "data/industrial_400" / fname).exists():
            target_path = BASE_DIR / "data/industrial_400" / fname

        if target_path:
            bench_candidates.append(target_path)
        else:
            # File not yet downloaded or in external storage
            bench_candidates.append(fname)

    resolved_count = sum(1 for b in bench_candidates if isinstance(b, Path) and b.exists())
    print(f" Benchmark Files:     {resolved_count} / {len(df_manifest)} present on local storage")

    if args.limit and args.limit < len(bench_candidates):
        bench_candidates = bench_candidates[:args.limit]
        print(f" Evaluation Scope:    Limited to first {args.limit} instances (--limit {args.limit})")

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] Solvers, manifest, and paths verified. Ready to execute.")
        print("To launch the real tournament, re-run without --dry-run.")
        print("=" * 90)
        return

    # Execution Loop
    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    results = []

    print("-" * 90)
    print(f" Starting tournament on {len(bench_candidates)} benchmarks across 4 architectures...")
    print("-" * 90)

    for idx, cnf_item in enumerate(bench_candidates, 1):
        if not isinstance(cnf_item, Path) or not cnf_item.exists():
            fname = cnf_item.name if isinstance(cnf_item, Path) else str(cnf_item)
            print(f"[{idx}/{len(bench_candidates)}] {fname:<45} -> SKIPPED (File not found on local disk)")
            continue

        cnf_name = cnf_item.name
        print(f"[{idx}/{len(bench_candidates)}] Running: {cnf_name}...")

        # Run all 4 solvers on this instance
        for s_name, s_exe in SOLVERS.items():
            res = run_instance(s_name, s_exe, str(cnf_item), args.timeout)
            results.append(res)
            print(f"    -> {s_name:<20}: {res['status']:<8} in {res['time']:.3f}s (Dec: {res['decisions']}, PAR2: {res['par2']:.1f}s)")

        # Save progress incrementally
        df_cur = pd.DataFrame(results)
        df_cur.to_csv(args.output_csv, index=False)

    # Final summary
    if results:
        df_final = pd.DataFrame(results)
        df_final.to_csv(args.output_csv, index=False)
        print_summary_table(df_final)
        print(f" Results written to: {args.output_csv}")
    else:
        print("\nNo instances were executed (ensure benchmark files are provided via --bench-dir).")

if __name__ == "__main__":
    main()
