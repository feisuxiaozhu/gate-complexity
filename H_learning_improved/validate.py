"""Validate optimized primitives and output distributions against the legacy path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qutip import Qobj, expect
from scipy.stats import ks_2samp

from .experiments import (
    ExperimentPoint,
    base_initial_guess,
    model_for_qubits,
    run_sweep,
    true_coefficients,
)
from .reference import reference_fit_targets, run_reference_point
from .rfe import build_probability_table
from .solver import fit_gap_targets, gaps_and_jacobian


def _direct_probability_error(
    point: ExperimentPoint,
) -> tuple[float, float, int]:
    model = model_for_qubits(point.qubits)
    truth = true_coefficients(point.qubits)
    upper = point.control_strength + point.upper_norm_factor * model.trace_norm(
        truth
    )
    table = build_probability_table(
        model,
        truth,
        point.control_strength,
        upper,
        point.epsilon,
    )
    hamiltonians = model.controlled_hamiltonians(
        truth, point.control_strength
    )
    maximum_error = 0.0
    for config_index in range(model.configuration_count):
        hamiltonian = Qobj(hamiltonians[config_index])
        state = Qobj(model.states[config_index])
        observable_c = Qobj(model.observables_c[config_index])
        observable_s = Qobj(model.observables_s[config_index])
        for time_index, evolution_time in enumerate(table.schedule.times):
            evolved = (-1j * hamiltonian * evolution_time).expm() * state
            direct_c = (1.0 + float(expect(observable_c, evolved))) / 2.0
            direct_s = (1.0 + float(expect(observable_s, evolved))) / 2.0
            maximum_error = max(
                maximum_error,
                abs(
                    direct_c
                    - table.probabilities_c[config_index, time_index]
                ),
                abs(
                    direct_s
                    - table.probabilities_s[config_index, time_index]
                ),
            )
    return (
        maximum_error,
        float(table.schedule.times[-1]),
        table.schedule.iteration_count,
    )


def _jacobian_error(point: ExperimentPoint) -> float:
    model = model_for_qubits(point.qubits)
    truth = true_coefficients(point.qubits)
    _, analytic = gaps_and_jacobian(
        model, truth, point.control_strength
    )
    numerical = np.empty_like(analytic)
    for coefficient_index in range(model.coefficient_count):
        step = 1e-6 * max(1.0, abs(truth[coefficient_index]))
        high = truth.copy()
        low = truth.copy()
        high[coefficient_index] += step
        low[coefficient_index] -= step
        numerical[:, coefficient_index] = (
            model.spectral_gaps(high, point.control_strength)
            - model.spectral_gaps(low, point.control_strength)
        ) / (2.0 * step)
    return float(np.max(np.abs(analytic - numerical)))


def _fixed_target_solver_error(point: ExperimentPoint) -> dict[str, float]:
    model = model_for_qubits(point.qubits)
    truth = true_coefficients(point.qubits)
    exact_targets = model.spectral_gaps(truth, point.control_strength)
    perturbation = 0.05 * point.epsilon * np.sin(
        np.arange(model.configuration_count, dtype=np.float64)
    )
    targets = exact_targets + perturbation
    optimized = fit_gap_targets(
        model,
        targets,
        point.control_strength,
        base_initial_guess(point.qubits),
        max_evaluations=point.max_evaluations,
    ).coefficients
    legacy = reference_fit_targets(model, targets, point)
    return {
        "coefficient_difference": float(np.linalg.norm(optimized - legacy)),
        "optimized_error_from_truth": float(np.linalg.norm(optimized - truth)),
        "legacy_error_from_truth": float(np.linalg.norm(legacy - truth)),
    }


def _distribution_summary(values: np.ndarray, epsilon: float) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p80": float(np.percentile(values, 80, method="nearest")),
        "success_rate": float(np.mean(values < epsilon)),
    }


def _historical_summary(point: ExperimentPoint) -> dict[str, float] | None:
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
    values = pd.read_csv(path)["l2_error_unfiltered"].to_numpy()
    result = _distribution_summary(values, point.epsilon)
    result["samples"] = int(values.size)
    return result


def validate(point: ExperimentPoint, seed: int) -> dict[str, object]:
    probability_error, maximum_time, iterations = _direct_probability_error(
        point
    )
    probability_tolerance = max(1e-12, 5e-15 * maximum_time)
    jacobian_error = _jacobian_error(point)
    solver = _fixed_target_solver_error(point)

    reference = run_reference_point(point, seed)
    optimized_sweep = run_sweep([point], seed=seed, workers=1)
    optimized = optimized_sweep.points[0]
    distribution_test = ks_2samp(reference.errors, optimized.errors)
    deterministic_passed = (
        probability_error <= probability_tolerance
        and jacobian_error <= 2e-8
        and solver["coefficient_difference"] <= 1e-7
    )
    return {
        "passed": deterministic_passed,
        "parameters": {
            "qubits": point.qubits,
            "nu": point.control_strength,
            "epsilon": point.epsilon,
            "shots": point.shots,
            "repeats": point.repeats,
            "noise": point.noise,
            "seed": seed,
        },
        "probabilities": {
            "maximum_absolute_error": probability_error,
            "tolerance": probability_tolerance,
            "maximum_evolution_time": maximum_time,
            "iterations": iterations,
        },
        "jacobian": {
            "maximum_absolute_error": jacobian_error,
            "tolerance": 2e-8,
        },
        "fixed_target_solver": solver,
        "distributions": {
            "legacy": _distribution_summary(reference.errors, point.epsilon),
            "optimized": _distribution_summary(
                optimized.errors, point.epsilon
            ),
            "ks_statistic": float(distribution_test.statistic),
            "ks_pvalue": float(distribution_test.pvalue),
            "historical": _historical_summary(point),
        },
        "timings_seconds": {
            "legacy": reference.target_seconds + reference.fitting_seconds,
            "optimized": optimized_sweep.compute_seconds,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare optimized and legacy Hamiltonian learning."
    )
    parser.add_argument("--qubits", type=int, choices=(1, 2), default=2)
    parser.add_argument("--nu", type=float, default=4.5)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--shots", type=int, default=29)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--upper-norm-factor", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
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
    report = validate(point, args.seed)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()