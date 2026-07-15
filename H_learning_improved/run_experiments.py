"""Command-line entry point for optimized experiment sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import (
    ExperimentPoint,
    run_sweep,
    write_sweep_outputs,
)

DEFAULT_EPSILONS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
DEFAULT_SHOTS = [25, 27, 29, 31, 33]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cached Hamiltonian-learning experiments."
    )
    parser.add_argument("--qubits", type=int, choices=(1, 2), default=2)
    parser.add_argument("--nu", type=float, nargs="+", default=None)
    parser.add_argument("--eps", type=float, nargs="+", default=None)
    parser.add_argument("--shots", type=int, nargs="+", default=None)
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--initial-guess-scale", type=float, nargs="+", default=[1.0]
    )
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--upper-norm-factor", type=float, default=2.0)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--no-write", action="store_true")
    return parser


def _paired_precision_arguments(
    epsilons: list[float] | None, shots: list[int] | None
) -> tuple[list[float], list[int]]:
    epsilon_values = DEFAULT_EPSILONS if epsilons is None else epsilons
    shot_values = DEFAULT_SHOTS if shots is None else shots
    if len(epsilon_values) != len(shot_values):
        raise ValueError("--eps and --shots must contain the same number of values")
    return epsilon_values, shot_values


def main() -> None:
    args = _parser().parse_args()
    epsilon_values, shot_values = _paired_precision_arguments(
        args.eps, args.shots
    )
    control_strengths = args.nu
    if control_strengths is None:
        control_strengths = [1.7 if args.qubits == 1 else 4.5]

    points = [
        ExperimentPoint(
            qubits=args.qubits,
            control_strength=control_strength,
            epsilon=epsilon,
            shots=shot_count,
            repeats=args.repeats,
            noise=noise,
            initial_guess_scale=initial_guess_scale,
            upper_norm_factor=args.upper_norm_factor,
            max_evaluations=args.max_evaluations,
        )
        for control_strength in control_strengths
        for noise in args.noise
        for initial_guess_scale in args.initial_guess_scale
        for epsilon, shot_count in zip(epsilon_values, shot_values)
    ]
    sweep = run_sweep(points, seed=args.seed, workers=args.workers)
    for result in sweep.points:
        summary = result.summary()
        print(
            f"qubits={result.point.qubits} nu={result.point.control_strength:g} "
            f"eps={result.point.epsilon:.3e} shots={result.point.shots} "
            f"noise={result.point.noise:g} median={summary['median']:.3e} "
            f"p80={summary['p80']:.3e} success={summary['success_rate']:.3f}"
        )
    print(
        f"probability preparation: {sweep.probability_seconds:.6f} s; "
        f"total compute: {sweep.compute_seconds:.6f} s"
    )
    if not args.no_write:
        paths = write_sweep_outputs(sweep, args.output_dir)
        print(f"wrote {len(paths)} files under {args.output_dir}")


if __name__ == "__main__":
    main()