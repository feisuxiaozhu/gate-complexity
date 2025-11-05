# python percentile_generator.py --dir data --glob "l2_error_*.csv" --column l2_error_unfiltered --shots 25
# thank chat!
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

# Example filename: l2_error_nu3_eps1e-05_shots31.csv

FILENAME_RE = re.compile(
    r"nu([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)_eps([0-9.eE+-]+)_shots(\d+)\.csv$",
    re.IGNORECASE
)

# Fixed eps order for output
EPS_ORDER = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# Preference order when multiple shots exist for the same (nu, eps)
SHOTS_PREFERENCE = [25, 27, 29, 31, 33]


def parse_name(path):
    m = FILENAME_RE.search(os.path.basename(path))
    if not m:
        return None
    nu   = float(m.group(1))   # <-- was int(...)
    eps  = float(m.group(2))
    shots = int(m.group(3))
    return nu, eps, shots

def pick_numeric_series(df: pd.DataFrame, force_col: Optional[str]) -> Tuple[str, np.ndarray]:
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
    if num.shape[1] == 0:
        raise ValueError("No numeric columns after dropping NaN/Inf.")
    if force_col:
        if force_col not in num.columns:
            raise ValueError(f"Requested column '{force_col}' not in numeric columns: {list(num.columns)}")
        col = force_col
    else:
        pref = [c for c in num.columns if ("l2" in c.lower() or "error" in c.lower())]
        col = pref[0] if pref else num.columns[0]
    return col, num[col].to_numpy()

def percentiles(arr: np.ndarray):
    # return p25, p35, median, p65, p75
    p25 = float(np.percentile(arr, 25))
    p35 = float(np.percentile(arr, 35))
    med = float(np.percentile(arr, 50))
    p65 = float(np.percentile(arr, 65))
    p75 = float(np.percentile(arr, 75))
    return p25, p35, med, p65, p75

def f3e(x: Optional[float]) -> str:
    return "null" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x:.3e}"

def main():
    ap = argparse.ArgumentParser(description="Aggregate p25/median/p75 by nu, ordered by eps=[1e-2..1e-6].")
    ap.add_argument("--dir", default="data", help="Directory to scan (default: data)")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern (default: *.csv)")
    ap.add_argument("--column", default=None, help="Force a specific numeric column name")
    ap.add_argument("--shots", type=int, default=25, help="Preferred shots if multiple exist (default: 25)")
    args = ap.parse_args()

    # Build preference list starting with requested shots, then the rest
    shots_pref = [args.shots] + [s for s in SHOTS_PREFERENCE if s != args.shots]

    paths = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not paths:
        print("No CSV files found.", file=sys.stderr); sys.exit(1)

    # Group files by (nu, eps) -> {shots: path}
    files_by_key: Dict[Tuple[int, float], Dict[int, str]] = defaultdict(dict)
    for p in paths:
        meta = parse_name(p)
        if meta is None:
            continue
        nu, eps, shots = meta
        files_by_key[(nu, eps)][shots] = p

    if not files_by_key:
        print("No files matching nu/eps/shots pattern.", file=sys.stderr); sys.exit(1)

    # For each (nu, eps), pick the preferred shots file
    chosen: Dict[int, Dict[float, str]] = defaultdict(dict)
    for (nu, eps), by_shots in files_by_key.items():
        chosen_path = None
        for s in shots_pref:
            if s in by_shots:
                chosen_path = by_shots[s]
                break
        if chosen_path is None:
            # fallback to any available shots
            s_any = next(iter(sorted(by_shots)))
            chosen_path = by_shots[s_any]
        chosen[nu][eps] = chosen_path

    # Compute percentiles and structure output
    # map: nu -> eps -> (p25, median, p75)
    # stats_by_nu: Dict[int, Dict[float, Tuple[float, float, float]]] = defaultdict(dict)
    stats_by_nu: Dict[float, Dict[float, Tuple[float, float, float, float, float]]] = defaultdict(dict)

    for nu in sorted(chosen.keys()):
        for eps, path in chosen[nu].items():
            try:
                df = pd.read_csv(path, header=0)  # skip header row only
                col, arr = pick_numeric_series(df, args.column)
                p25, p35, med, p65, p75 = percentiles(arr)
                stats_by_nu[nu][eps] = (p25, p35, med, p65, p75)
            except Exception as e:
                print(f"ERROR processing {path}: {e}", file=sys.stderr)

    # Print in your requested format
    print("{")
    for nu in sorted(stats_by_nu.keys()):
        med_list: List[Optional[float]] = []
        p25_list: List[Optional[float]] = []
        p75_list: List[Optional[float]] = []
        p35_list: List[Optional[float]] = []
        p65_list: List[Optional[float]] = []
        for e in EPS_ORDER:
            triplet = stats_by_nu[nu].get(e)
            if triplet is None:
                p25_list.append(None); med_list.append(None); p75_list.append(None); p35_list.append(None); p65_list.append(None)
            else:
                p25, p35, med, p65, p75 = triplet
                p25_list.append(p25); med_list.append(med); p75_list.append(p75); p35_list.append(p35); p65_list.append(p65)

        print(f"  {nu}: {{")
        print(f"        \"median\": [{','.join(f3e(x) for x in med_list)}], ")
        print(f"        \"p25\": [{','.join(f3e(x) for x in p25_list)}], ")
        print(f"        \"p35\": [{','.join(f3e(x) for x in p35_list)}], ")
        print(f"        \"p65\": [{','.join(f3e(x) for x in p65_list)}], ")
        print(f"        \"p75\": [{','.join(f3e(x) for x in p75_list)}], ")
        print(f"    }},")
    print("}")

if __name__ == "__main__":
    main()
