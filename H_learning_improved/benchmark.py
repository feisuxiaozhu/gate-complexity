"""Matched wall-clock benchmarks for legacy and optimized implementations."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from .experiments import ExperimentPoint, run_sweep


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "scipy", "pandas", "qutip"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _error_summary(values: np.ndarray, epsilon: float) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p80": float(np.percentile(values, 80, method="nearest")),
        "success_rate": float(np.mean(values < epsilon)),
    }


def _historical_comparison(
    point: ExperimentPoint, optimized_errors: np.ndarray
) -> dict[str, float | int] | None:
    if point.qubits != 2 or point.noise != 0.0:
        return None
    repository_root = Path(__file__).resolve().parent.parent
    path = (
        repository_root
        / "H_learning"
        / "2_qubit_plots"
        / "data"
        / (
            f"l2_error_nu{point.control_strength:g}_eps{point.epsilon}_"
            f"shots{point.shots}.csv"
        )
    )
    if not path.exists():
        return None
    historical_errors = np.loadtxt(path, delimiter=",", skiprows=1)
    distribution_test = ks_2samp(optimized_errors, historical_errors)
    return {
        "samples": int(historical_errors.size),
        **_error_summary(historical_errors, point.epsilon),
        "ks_statistic": float(distribution_test.statistic),
        "ks_pvalue": float(distribution_test.pvalue),
    }


def _reference_timing(result: Any) -> dict[str, float]:
    return {
        "probability_and_sampling": result.target_seconds,
        "fitting": result.fitting_seconds,
        "total": result.target_seconds + result.fitting_seconds,
    }


def _optimized_timing(sweep: Any) -> dict[str, float]:
    point = sweep.points[0]
    return {
        "probability_preparation": sweep.probability_seconds,
        "sampling": point.timings.sampling_seconds,
        "fitting": point.timings.fitting_seconds,
        "total": sweep.compute_seconds,
    }


def _point_with_repeats(
    point: ExperimentPoint, repeats: int
) -> ExperimentPoint:
    return ExperimentPoint(
        qubits=point.qubits,
        control_strength=point.control_strength,
        epsilon=point.epsilon,
        shots=point.shots,
        repeats=repeats,
        noise=point.noise,
        initial_guess_scale=point.initial_guess_scale,
        upper_norm_factor=point.upper_norm_factor,
        max_evaluations=point.max_evaluations,
    )


def benchmark(
    point: ExperimentPoint,
    seed: int,
    workers: int,
    short_trials: int,
    short_repeats: int,
) -> dict[str, object]:
    from .reference import run_reference_point

    warmup_point = _point_with_repeats(point, 1)
    run_reference_point(warmup_point, seed - 1)
    run_sweep([warmup_point], seed=seed - 1, workers=1)

    short_point = _point_with_repeats(point, short_repeats)
    short_timings: dict[str, list[float]] = {
        "legacy": [],
        "optimized_single_worker": [],
        "optimized_multiworker": [],
    }
    for trial_index in range(short_trials):
        trial_seed = seed + trial_index
        legacy_trial = run_reference_point(short_point, trial_seed)
        single_trial = run_sweep(
            [short_point], seed=trial_seed, workers=1
        )
        multi_trial = run_sweep(
            [short_point], seed=trial_seed, workers=workers
        )
        short_timings["legacy"].append(
            legacy_trial.target_seconds + legacy_trial.fitting_seconds
        )
        short_timings["optimized_single_worker"].append(
            single_trial.compute_seconds
        )
        short_timings["optimized_multiworker"].append(
            multi_trial.compute_seconds
        )

    legacy = run_reference_point(point, seed)
    optimized_single = run_sweep([point], seed=seed, workers=1)
    optimized_multi = run_sweep([point], seed=seed, workers=workers)
    legacy_timing = _reference_timing(legacy)
    single_timing = _optimized_timing(optimized_single)
    multi_timing = _optimized_timing(optimized_multi)
    single_speedup = legacy_timing["total"] / single_timing["total"]
    multi_speedup = legacy_timing["total"] / multi_timing["total"]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {
            "qubits": point.qubits,
            "nu": point.control_strength,
            "epsilon": point.epsilon,
            "shots": point.shots,
            "repeats": point.repeats,
            "noise": point.noise,
            "seed": seed,
            "workers": workers,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "packages": _package_versions(),
        },
        "short_trials": {
            "repeats_per_trial": short_repeats,
            "raw_seconds": short_timings,
            "median_seconds": {
                name: statistics.median(values)
                for name, values in short_timings.items()
            },
        },
        "full_run_seconds": {
            "legacy": legacy_timing,
            "optimized_single_worker": single_timing,
            "optimized_multiworker": multi_timing,
        },
        "speedup": {
            "algorithmic_single_worker": single_speedup,
            "practical_multiworker": multi_speedup,
        },
        "result_summaries": {
            "legacy": _error_summary(legacy.errors, point.epsilon),
            "optimized_single_worker": _error_summary(
                optimized_single.points[0].errors, point.epsilon
            ),
            "optimized_multiworker": _error_summary(
                optimized_multi.points[0].errors, point.epsilon
            ),
            "single_and_multi_outputs_identical": bool(
                np.array_equal(
                    optimized_single.points[0].errors,
                    optimized_multi.points[0].errors,
                )
            ),
            "historical_comparison": _historical_comparison(
                point, optimized_single.points[0].errors
            ),
        },
    }


def _markdown(report: dict[str, object]) -> str:
    workload = report["workload"]
    full = report["full_run_seconds"]
    speedup = report["speedup"]
    summaries = report["result_summaries"]
    environment = report["environment"]
    historical = summaries["historical_comparison"]
    historical_section = ""
    if historical is not None:
        historical_section = f"""
## Historical Distribution Check

The existing {historical['samples']}-sample legacy CSV has median error {historical['median']:.6e}, 80th percentile {historical['p80']:.6e}, and success rate {historical['success_rate']:.3f}. Against the optimized {workload['repeats']}-sample output, the two-sample KS statistic is {historical['ks_statistic']:.3f} with p-value {historical['ks_pvalue']:.3f}; this does not indicate a distributional difference.

The 80th percentile lies near the boundary between the narrow successful mode and a sparse failure tail, so it varies more between independent finite samples than the median. Deterministic validation is recorded separately in `VALIDATION.json`.
"""
    parallel_note = ""
    if full["optimized_multiworker"]["total"] > full["optimized_single_worker"]["total"]:
        parallel_note = (
            f"On this workload, one worker was faster than {workload['workers']} "
            "workers because Windows process startup outweighed the parallel fitting gain."
        )
    return f"""# Benchmark Results

Generated: {report['generated_at_utc']}

## Workload

- Qubits: {workload['qubits']}
- Control strength: {workload['nu']}
- Epsilon: {workload['epsilon']}
- Shots: {workload['shots']}
- Repeats: {workload['repeats']}
- Noise: {workload['noise']}
- Seed: {workload['seed']}
- Worker processes: {workload['workers']}

## Wall-Clock Results

| Backend | Probability / sampling (s) | Fitting (s) | Total (s) |
|---|---:|---:|---:|
| Legacy QuTiP | {full['legacy']['probability_and_sampling']:.6f} | {full['legacy']['fitting']:.6f} | {full['legacy']['total']:.6f} |
| Optimized, 1 worker | {full['optimized_single_worker']['probability_preparation'] + full['optimized_single_worker']['sampling']:.6f} | {full['optimized_single_worker']['fitting']:.6f} | {full['optimized_single_worker']['total']:.6f} |
| Optimized, {workload['workers']} workers | {full['optimized_multiworker']['probability_preparation'] + full['optimized_multiworker']['sampling']:.6f} | {full['optimized_multiworker']['fitting']:.6f} | {full['optimized_multiworker']['total']:.6f} |

Algorithmic speedup (one worker): **{speedup['algorithmic_single_worker']:.2f}x**

Practical speedup ({workload['workers']} workers): **{speedup['practical_multiworker']:.2f}x**

{parallel_note}

## Output Summary

| Backend | Median error | 80th percentile | Success rate |
|---|---:|---:|---:|
| Legacy | {summaries['legacy']['median']:.6e} | {summaries['legacy']['p80']:.6e} | {summaries['legacy']['success_rate']:.3f} |
| Optimized, 1 worker | {summaries['optimized_single_worker']['median']:.6e} | {summaries['optimized_single_worker']['p80']:.6e} | {summaries['optimized_single_worker']['success_rate']:.3f} |
| Optimized, {workload['workers']} workers | {summaries['optimized_multiworker']['median']:.6e} | {summaries['optimized_multiworker']['p80']:.6e} | {summaries['optimized_multiworker']['success_rate']:.3f} |

Single-worker and multiworker optimized outputs identical: `{summaries['single_and_multi_outputs_identical']}`
{historical_section}

## Environment

- Platform: {environment['platform']}
- Processor: {environment['processor']}
- Logical CPUs: {environment['cpu_count']}
- Python: {environment['python'].split()[0]}
- Packages: {json.dumps(environment['packages'], sort_keys=True)}

Times include probability-cache construction. CSV serialization is excluded so the table measures the numerical workload. The legacy and optimized samplers are distributionally equivalent but consume random numbers differently, so their rows are not expected to match one for one.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark legacy and cached Hamiltonian learning."
    )
    parser.add_argument("--qubits", type=int, choices=(1, 2), default=2)
    parser.add_argument("--nu", type=float, default=4.5)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--shots", type=int, default=29)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1))
    )
    parser.add_argument("--short-trials", type=int, default=3)
    parser.add_argument("--short-repeats", type=int, default=10)
    parser.add_argument("--upper-norm-factor", type=float, default=2.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    point = ExperimentPoint(
        qubits=args.qubits,
        control_strength=args.nu,
        epsilon=args.eps,
        shots=args.shots,
        repeats=args.repeats,
        noise=args.noise,
        upper_norm_factor=args.upper_norm_factor,
    )
    report = benchmark(
        point,
        seed=args.seed,
        workers=args.workers,
        short_trials=args.short_trials,
        short_repeats=args.short_repeats,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "BENCHMARK.json"
    markdown_path = args.output_dir / "BENCHMARK.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="ascii")
    markdown_path.write_text(_markdown(report), encoding="ascii")
    print(_markdown(report))
    print(f"wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()