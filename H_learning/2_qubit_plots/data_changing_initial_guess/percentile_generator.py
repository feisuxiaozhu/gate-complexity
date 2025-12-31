#!/usr/bin/env python3
# Example:
# python percentile_generator.py --dir . --glob "l2_error_*.csv" --column l2_error_unfiltered --shots 29 --eps 0.0001
#
# Expected filename pattern:
# l2_error_nu5_eps0.0001_shots29_scaling1.csv
# l2_error_nu5_eps0.0001_shots29_scaling50.csv

# python ./percentile_generator.py --dir . --glob "l2_error_*.csv" --column l2_error_unfiltered --shots 29 --scaling_min 1 --scaling_max 50 --eps 0.0001

import argparse, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

FILENAME_RE = re.compile(
    r"(?:^|.*[/\\])"
    r"l2_error_"
    r"nu((?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)_"
    r"eps([0-9.eE+-]+)_"
    r"shots(\d+)_"
    r"scaling(\d+)\.csv$",
    re.IGNORECASE
)

SHOTS_PREFERENCE = [25, 27, 29, 31, 33]

def parse_name(path: str):
    """Extract (nu, eps, shots, scaling) from filename or return None."""
    m = FILENAME_RE.search(path)
    if not m:
        return None
    nu = float(m.group(1))
    eps = float(m.group(2))
    shots = int(m.group(3))
    scaling = int(m.group(4))
    return nu, eps, shots, scaling

def pick_numeric_series(df: pd.DataFrame, force_col: Optional[str]) -> Tuple[str, np.ndarray]:
    """Choose a numeric column to summarize (prefers names with 'l2' or 'error')."""
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    if force_col is not None:
        if force_col not in num.columns:
            raise ValueError(f"Requested column '{force_col}' not found. Available numeric columns: {list(num.columns)}")
        arr = num[force_col].dropna().to_numpy()
        if arr.size == 0:
            raise ValueError(f"Column '{force_col}' has no finite values.")
        return force_col, arr

    # auto pick
    candidates = [c for c in num.columns if ("l2" in c.lower() or "error" in c.lower())]
    col = candidates[0] if candidates else (num.columns[0] if len(num.columns) > 0 else None)
    if col is None:
        raise ValueError("No numeric columns found.")
    arr = num[col].dropna().to_numpy()
    if arr.size == 0:
        raise ValueError(f"Column '{col}' has no finite values.")
    return col, arr

def percentiles(arr: np.ndarray):
    """Return p25, p35, median, p65, p75."""
    p25 = float(np.percentile(arr, 25))
    p35 = float(np.percentile(arr, 35))
    med = float(np.percentile(arr, 50))
    p65 = float(np.percentile(arr, 65))
    p75 = float(np.percentile(arr, 75))
    return p25, p35, med, p65, p75

def f3e(x: Optional[float]) -> str:
    """Format float in scientific 3 sig fig; None or NaN -> 'null'."""
    if x is None:
        return "null"
    try:
        if not np.isfinite(x):
            return "null"
    except Exception:
        return "null"
    return f"{x:.3e}"

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate p25/p35/median/p65/p75 by nu and scaling from filenames l2_error_nu*_eps*_shots*_scaling*.csv"
    )
    ap.add_argument("--dir", default="data", help="Directory to scan (default: data)")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern (used recursively, default: *.csv)")
    ap.add_argument("--column", default=None, help="Force a specific numeric column name")
    ap.add_argument("--shots", type=int, default=25, help="Preferred shots if multiple exist (default: 25)")
    ap.add_argument("--scaling_min", type=int, default=1, help="Only include scaling >= this (default: 1)")
    ap.add_argument("--scaling_max", type=int, default=10**9, help="Only include scaling <= this (default: very large)")
    ap.add_argument("--eps", type=float, default=None, help="If set, only use files with this eps value")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    search_root = (base / args.dir).resolve()
    paths = sorted(search_root.rglob(args.glob))

    print(f"[info] Searching in: {search_root}")
    print(f"[info] Pattern: {args.glob}")
    print(f"[info] Found {len(paths)} file(s).")
    for p in paths[:6]:
        print(f"[info]  {p.name}")
    if not paths:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    # Group: (nu, eps, scaling) -> shots -> path
    files_by_key: Dict[Tuple[float, float, int], Dict[int, str]] = defaultdict(dict)
    eps_values_by_nu: Dict[float, List[float]] = defaultdict(list)

    for p in paths:
        meta = parse_name(str(p))
        if meta is None:
            continue
        nu, eps, shots, scaling = meta

        if scaling < args.scaling_min or scaling > args.scaling_max:
            continue
        if args.eps is not None and float(eps) != float(args.eps):
            continue

        files_by_key[(nu, eps, scaling)][shots] = str(p)
        eps_values_by_nu[nu].append(float(eps))

    if not files_by_key:
        print("No files matching l2_error_nu*_eps*_shots*_scaling*.csv pattern (after filters).", file=sys.stderr)
        sys.exit(1)

    # If eps is not given, pick the most common eps per nu
    chosen_eps_by_nu: Dict[float, float] = {}
    if args.eps is not None:
        for nu in eps_values_by_nu.keys():
            chosen_eps_by_nu[nu] = float(args.eps)
    else:
        for nu, eps_list in eps_values_by_nu.items():
            vals, counts = np.unique(np.array(eps_list, dtype=float), return_counts=True)
            chosen_eps_by_nu[nu] = float(vals[int(np.argmax(counts))])

    shots_pref = [args.shots] + [s for s in SHOTS_PREFERENCE if s != args.shots]

    # stats_by_nu: nu -> scaling -> dict(p25 p35 median p65 p75)
    stats_by_nu: Dict[float, Dict[int, Dict[str, float]]] = defaultdict(dict)

    for (nu, eps, scaling), by_shots in files_by_key.items():
        if float(eps) != float(chosen_eps_by_nu.get(nu, eps)):
            continue

        # pick a file with preferred shots
        chosen_path = None
        for s in shots_pref:
            if s in by_shots:
                chosen_path = by_shots[s]
                break
        if chosen_path is None:
            s_any = next(iter(sorted(by_shots)))
            chosen_path = by_shots[s_any]

        try:
            df = pd.read_csv(chosen_path, header=0)
            col, arr = pick_numeric_series(df, args.column)
            p25, p35, med, p65, p75 = percentiles(arr)
            stats_by_nu[nu][scaling] = {
                "p25": p25, "p35": p35, "median": med, "p65": p65, "p75": p75
            }
        except Exception as e:
            print(f"ERROR processing {chosen_path}: {e}", file=sys.stderr)

    if not stats_by_nu:
        print("No statistics computed. Check filename parsing and filters.", file=sys.stderr)
        sys.exit(1)

    # Print: { nu: { scaling: {median,p25,p35,p65,p75}, ... }, ... }
    print("{")
    for nu in sorted(stats_by_nu.keys(), key=float):
        scalings_present = sorted(stats_by_nu[nu].keys())
        print(f"  {nu:g}: {{")
        for s in scalings_present:
            item = stats_by_nu[nu][s]
            print(f"    {s}: {{")
            print(f"      \"median\": {f3e(item['median'])},")
            print(f"      \"p25\":    {f3e(item['p25'])},")
            print(f"      \"p35\":    {f3e(item['p35'])},")
            print(f"      \"p65\":    {f3e(item['p65'])},")
            print(f"      \"p75\":    {f3e(item['p75'])},")
            print(f"    }},")
        print(f"  }},")
    print("}")

if __name__ == "__main__":
    main()
