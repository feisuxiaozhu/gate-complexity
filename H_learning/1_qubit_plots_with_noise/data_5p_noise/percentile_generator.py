#!/usr/bin/env python3
# python percentile_generator.py --dir . --glob "l2_error_*.csv" --column l2_error_unfiltered --shots 25

import argparse, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

# Accept float nu, decimal/scientific eps, int shots, and Ttotal number or 'nan'.
# Matches both "..._Ttotal<...>.csv" and "..._T_total<...>.csv".
FILENAME_RE = re.compile(
    r"nu((?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)_"      # nu
    r"eps([0-9.eE+-]+)_"                                   # eps
    r"shots(\d+)_"                                         # shots
    r"T_?total([0-9.eE+-]+|nan)\.csv$",                    # Ttotal or T_total
    re.IGNORECASE
)

# Fixed eps order (positions correspond to these)
EPS_ORDER = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# Preferred shots if multiple exist for a (nu, eps)
SHOTS_PREFERENCE = [25, 27, 29, 31, 33]

def parse_name(path: str):
    """Extract (nu, eps, shots, T_total) from filename or return None if no match."""
    m = FILENAME_RE.search(os.path.basename(path))
    if not m:
        return None
    nu    = float(m.group(1))
    eps   = float(m.group(2))
    shots = int(m.group(3))
    tstr  = m.group(4).lower()
    ttot  = np.nan if tstr == "nan" else float(tstr)
    return nu, eps, shots, ttot

def pick_numeric_series(df: pd.DataFrame, force_col: Optional[str]) -> Tuple[str, np.ndarray]:
    """Choose a numeric column to summarize (prefers names with 'l2' or 'error')."""
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
    if num.shape[1] == 0:
        raise ValueError("No numeric columns after dropping NaN/Inf.")
    if force_col:
        if force_col not in num.columns:
            raise ValueError(f"Requested column '{force_col}' not found among numeric columns: {list(num.columns)}")
        col = force_col
    else:
        pref = [c for c in num.columns if ("l2" in c.lower() or "error" in c.lower())]
        col = pref[0] if pref else num.columns[0]
    return col, num[col].to_numpy()

def percentiles(arr: np.ndarray):
    """Return p25, p35, median, p65, p75."""
    p25 = float(np.percentile(arr, 25))
    p35 = float(np.percentile(arr, 35))
    med = float(np.percentile(arr, 50))
    p65 = float(np.percentile(arr, 65))
    p75 = float(np.percentile(arr, 75))
    return p25, p35, med, p65, p75

def f3e(x: Optional[float]) -> str:
    """Format float in scientific 3-sig-fig; NaN/None -> 'null'."""
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
        description="Aggregate T_total and p25/p35/median/p65/p75 by nu, ordered by eps=[1e-2..1e-6]."
    )
    ap.add_argument("--dir", default="data", help="Directory to scan (default: data)")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern (used recursively, default: *.csv)")
    ap.add_argument("--column", default=None, help="Force a specific numeric column name")
    ap.add_argument("--shots", type=int, default=25, help="Preferred shots if multiple exist (default: 25)")
    args = ap.parse_args()

    # Build absolute search root relative to this script
    base = Path(__file__).resolve().parent
    search_root = (base / args.dir).resolve()

    # Recursive search
    paths = sorted(search_root.rglob(args.glob))

    # Debug info
    print(f"[info] Searching in: {search_root}")
    print(f"[info] Pattern: {args.glob}")
    print(f"[info] Found {len(paths)} file(s).")
    for p in paths[:6]:
        print(f"[info]  {p.name}")
    if not paths:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    # Group: (nu, eps) -> shots -> (path, T_total)
    files_by_key: Dict[Tuple[float, float], Dict[int, Tuple[str, float]]] = defaultdict(dict)
    for p in paths:
        meta = parse_name(str(p))
        if meta is None:
            continue
        nu, eps, shots, ttot = meta
        files_by_key[(nu, eps)][shots] = (str(p), ttot)

    if not files_by_key:
        print("No files matching nu/eps/shots/Ttotal pattern.", file=sys.stderr)
        sys.exit(1)

    # Build preference list starting with requested shots, then the rest
    shots_pref = [args.shots] + [s for s in SHOTS_PREFERENCE if s != args.shots]

    # For each (nu, eps), pick preferred shots file
    chosen: Dict[float, Dict[float, Tuple[str, float]]] = defaultdict(dict)
    for (nu, eps), by_shots in files_by_key.items():
        chosen_item = None
        for s in shots_pref:
            if s in by_shots:
                chosen_item = by_shots[s]
                break
        if chosen_item is None:
            # fallback to smallest shots key
            s_any = next(iter(sorted(by_shots)))
            chosen_item = by_shots[s_any]
        chosen[nu][eps] = chosen_item  # (path, T_total)

    # Compute stats
    # stats_by_nu: nu -> eps -> dict(T_total, p25, p35, median, p65, p75)
    stats_by_nu: Dict[float, Dict[float, Dict[str, float]]] = defaultdict(dict)

    for nu in sorted(chosen.keys(), key=float):
        for eps, (path, ttot) in chosen[nu].items():
            try:
                df = pd.read_csv(path, header=0)
                col, arr = pick_numeric_series(df, args.column)
                p25, p35, med, p65, p75 = percentiles(arr)
                stats_by_nu[nu][eps] = {
                    "T_total": ttot,
                    "p25": p25, "p35": p35, "median": med, "p65": p65, "p75": p75
                }
            except Exception as e:
                print(f"ERROR processing {path}: {e}", file=sys.stderr)

    # Print in requested dict format
    print("{")
    for nu in sorted(stats_by_nu.keys(), key=float):
        T_list:   List[Optional[float]] = []
        med_list: List[Optional[float]] = []
        p25_list: List[Optional[float]] = []
        p35_list: List[Optional[float]] = []
        p65_list: List[Optional[float]] = []
        p75_list: List[Optional[float]] = []

        for e in EPS_ORDER:
            item = stats_by_nu[nu].get(e)
            if item is None:
                T_list.append(None); p25_list.append(None); p35_list.append(None)
                med_list.append(None); p65_list.append(None); p75_list.append(None)
            else:
                T_list.append(item["T_total"])
                med_list.append(item["median"])
                p25_list.append(item["p25"]); p35_list.append(item["p35"])
                p65_list.append(item["p65"]); p75_list.append(item["p75"])

        print(f"  {nu:g}: {{")
        print(f"        \"T_total\": [{','.join(f3e(x) for x in T_list)}], ")
        print(f"        \"median\":  [{','.join(f3e(x) for x in med_list)}], ")
        print(f"        \"p25\":     [{','.join(f3e(x) for x in p25_list)}], ")
        print(f"        \"p35\":     [{','.join(f3e(x) for x in p35_list)}], ")
        print(f"        \"p65\":     [{','.join(f3e(x) for x in p65_list)}], ")
        print(f"        \"p75\":     [{','.join(f3e(x) for x in p75_list)}], ")
        print(f"    }},")
    print("}")

if __name__ == "__main__":
    main()
