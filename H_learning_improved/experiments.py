"""Reproducible experiment sweeps and legacy-compatible output."""

from __future__ import annotations

import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .models import ExperimentModel, one_qubit_model, two_qubit_model
from .rfe import (
    ProbabilityTable,
    build_probability_table,
    modeled_total_time,
    simulate_gap_estimates,
)
from .solver import FitResult, fit_gap_targets

FloatArray = NDArray[np.float64]

ONE_QUBIT_TRUE = np.array([0.1, 0.5, 0.3], dtype=np.float64)
ONE_QUBIT_INITIAL = np.array([0.09, 0.51, 0.29], dtype=np.float64)
TWO_QUBIT_TRUE = np.array(
    [
        0.1,
        0.2,
        0.3,
        0.5,
        0.6,
        0.3,
        0.2,
        0.1,
        0.1,
        0.2,
        0.1,
        0.1,
        0.3,
        0.22,
        0.15,
    ],
    dtype=np.float64,
)
TWO_QUBIT_INITIAL = np.array(
    [
        0.11,
        0.21,
        0.32,
        0.51,
        0.63,
        0.31,
        0.22,
        0.11,
        0.11,
        0.22,
        0.11,
        0.11,
        0.33,
        0.22,
        0.15,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class ExperimentPoint:
    qubits: int
    control_strength: float
    epsilon: float
    shots: int
    repeats: int
    noise: float = 0.0
    initial_guess_scale: float = 1.0
    upper_norm_factor: float = 2.0
    max_evaluations: int | None = None

    def __post_init__(self) -> None:
        if self.qubits not in (1, 2):
            raise ValueError("qubits must be 1 or 2")
        if self.control_strength <= 0:
            raise ValueError("control_strength must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.shots <= 0 or self.repeats <= 0:
            raise ValueError("shots and repeats must be positive")
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("noise must be between 0 and 1")
        if self.initial_guess_scale <= 0:
            raise ValueError("initial_guess_scale must be positive")
        if self.upper_norm_factor < 0:
            raise ValueError("upper_norm_factor must be nonnegative")


@dataclass(frozen=True, slots=True)
class PhaseTimings:
    sampling_seconds: float
    fitting_seconds: float


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    point: ExperimentPoint
    seed_entropy: int
    seed_spawn_key: tuple[int, ...]
    upper: float
    modeled_total_time: float
    errors: FloatArray
    estimates: FloatArray
    target_gaps: FloatArray
    fits: tuple[FitResult, ...]
    timings: PhaseTimings

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.errors < self.point.epsilon))

    def summary(self) -> dict[str, float | int]:
        return {
            "repeats": self.point.repeats,
            "success_rate": self.success_rate,
            "mean": float(np.mean(self.errors)),
            "std": float(np.std(self.errors)),
            "p25": float(np.percentile(self.errors, 25)),
            "p35": float(np.percentile(self.errors, 35)),
            "median": float(np.percentile(self.errors, 50)),
            "p65": float(np.percentile(self.errors, 65)),
            "p75": float(np.percentile(self.errors, 75)),
            "p80": float(np.percentile(self.errors, 80, method="nearest")),
            "solver_success_rate": float(
                np.mean([fit.success for fit in self.fits])
            ),
            "mean_function_evaluations": float(
                np.mean([fit.evaluations for fit in self.fits])
            ),
        }


@dataclass(frozen=True, slots=True)
class SweepResult:
    seed: int
    workers: int
    probability_seconds: float
    compute_seconds: float
    points: tuple[ExperimentResult, ...]


def model_for_qubits(qubits: int) -> ExperimentModel:
    if qubits == 1:
        return one_qubit_model()
    if qubits == 2:
        return two_qubit_model()
    raise ValueError("qubits must be 1 or 2")


def true_coefficients(qubits: int) -> FloatArray:
    return (ONE_QUBIT_TRUE if qubits == 1 else TWO_QUBIT_TRUE).copy()


def base_initial_guess(qubits: int) -> FloatArray:
    return (
        ONE_QUBIT_INITIAL if qubits == 1 else TWO_QUBIT_INITIAL
    ).copy()


def scaled_initial_guess(qubits: int, scale: float) -> FloatArray:
    truth = true_coefficients(qubits)
    initial = base_initial_guess(qubits)
    return truth - (truth - initial) * scale


_FIT_CONTEXT: tuple[
    ExperimentModel, float, FloatArray, int | None
] | None = None


def _initialize_fit_worker(
    qubits: int,
    control_strength: float,
    initial_guess: FloatArray,
    max_evaluations: int | None,
) -> None:
    global _FIT_CONTEXT
    _FIT_CONTEXT = (
        model_for_qubits(qubits),
        control_strength,
        initial_guess,
        max_evaluations,
    )


def _fit_worker(targets: FloatArray) -> FitResult:
    if _FIT_CONTEXT is None:
        raise RuntimeError("fit worker was not initialized")
    model, control_strength, initial_guess, max_evaluations = _FIT_CONTEXT
    return fit_gap_targets(
        model,
        targets,
        control_strength,
        initial_guess,
        max_evaluations=max_evaluations,
    )


def fit_target_batch(
    model: ExperimentModel,
    targets: FloatArray,
    point: ExperimentPoint,
    workers: int,
) -> tuple[FitResult, ...]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    initial_guess = scaled_initial_guess(
        point.qubits, point.initial_guess_scale
    )
    if workers == 1:
        return tuple(
            fit_gap_targets(
                model,
                target,
                point.control_strength,
                initial_guess,
                max_evaluations=point.max_evaluations,
            )
            for target in targets
        )

    chunk_size = max(1, point.repeats // (workers * 4))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_fit_worker,
        initargs=(
            point.qubits,
            point.control_strength,
            initial_guess,
            point.max_evaluations,
        ),
    ) as executor:
        return tuple(executor.map(_fit_worker, targets, chunksize=chunk_size))


def run_experiment_point(
    point: ExperimentPoint,
    table: ProbabilityTable,
    seed_sequence: np.random.SeedSequence,
    workers: int = 1,
) -> ExperimentResult:
    model = model_for_qubits(point.qubits)
    truth = true_coefficients(point.qubits)
    sampling_start = perf_counter()
    target_gaps = simulate_gap_estimates(
        table,
        epsilon=point.epsilon,
        shots=point.shots,
        repeats=point.repeats,
        noise=point.noise,
        rng=np.random.default_rng(seed_sequence),
    )
    sampling_seconds = perf_counter() - sampling_start

    fitting_start = perf_counter()
    fits = fit_target_batch(model, target_gaps, point, workers)
    fitting_seconds = perf_counter() - fitting_start
    estimates = np.stack([fit.coefficients for fit in fits])
    errors = np.linalg.norm(estimates - truth[None, :], axis=1)

    return ExperimentResult(
        point=point,
        seed_entropy=int(seed_sequence.entropy),
        seed_spawn_key=tuple(seed_sequence.spawn_key),
        upper=table.schedule.upper,
        modeled_total_time=modeled_total_time(
            table, point.epsilon, point.shots
        ),
        errors=np.asarray(errors, dtype=np.float64),
        estimates=np.asarray(estimates, dtype=np.float64),
        target_gaps=np.asarray(target_gaps, dtype=np.float64),
        fits=fits,
        timings=PhaseTimings(
            sampling_seconds=sampling_seconds,
            fitting_seconds=fitting_seconds,
        ),
    )


def run_sweep(
    points: Iterable[ExperimentPoint],
    seed: int,
    workers: int = 1,
) -> SweepResult:
    point_values = tuple(points)
    if not point_values:
        raise ValueError("at least one experiment point is required")
    qubit_counts = {point.qubits for point in point_values}
    if len(qubit_counts) != 1:
        raise ValueError("a sweep must use one qubit count")
    upper_factors = {point.upper_norm_factor for point in point_values}
    if len(upper_factors) != 1:
        raise ValueError("a sweep must use one upper_norm_factor")
    if workers <= 0:
        raise ValueError("workers must be positive")

    compute_start = perf_counter()
    model = model_for_qubits(point_values[0].qubits)
    truth = true_coefficients(point_values[0].qubits)
    probability_start = perf_counter()
    tables: dict[float, ProbabilityTable] = {}
    for control_strength in sorted(
        {point.control_strength for point in point_values}
    ):
        matching_points = [
            point
            for point in point_values
            if point.control_strength == control_strength
        ]
        finest_epsilon = min(point.epsilon for point in matching_points)
        upper = control_strength + matching_points[0].upper_norm_factor * model.trace_norm(
            truth
        )
        tables[control_strength] = build_probability_table(
            model,
            truth,
            control_strength,
            upper,
            finest_epsilon,
        )
    probability_seconds = perf_counter() - probability_start

    seed_sequences = np.random.SeedSequence(seed).spawn(len(point_values))
    results = tuple(
        run_experiment_point(
            point,
            tables[point.control_strength],
            seed_sequence,
            workers=workers,
        )
        for point, seed_sequence in zip(point_values, seed_sequences)
    )
    return SweepResult(
        seed=seed,
        workers=workers,
        probability_seconds=probability_seconds,
        compute_seconds=perf_counter() - compute_start,
        points=results,
    )


def _format_number(value: float) -> str:
    return f"{value:g}"


def legacy_filename(result: ExperimentResult) -> str:
    point = result.point
    prefix = (
        f"l2_error_nu{_format_number(point.control_strength)}_"
        f"eps{point.epsilon}_shots{point.shots}"
    )
    if point.initial_guess_scale != 1.0:
        return f"{prefix}_scaling{_format_number(point.initial_guess_scale)}.csv"
    if point.qubits == 2 and point.noise == 0.0:
        return f"{prefix}.csv"
    return f"{prefix}_Ttotal{result.modeled_total_time:.3e}.csv"


def _package_versions() -> dict[str, str]:
    packages = {}
    for package in ("numpy", "scipy", "pandas", "qutip"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return packages


def _metadata_for_result(
    result: ExperimentResult, sweep: SweepResult
) -> dict[str, object]:
    point_metadata = asdict(result.point)
    initial = scaled_initial_guess(
        result.point.qubits, result.point.initial_guess_scale
    )
    truth = true_coefficients(result.point.qubits)
    return {
        "parameters": point_metadata,
        "seed": sweep.seed,
        "seed_entropy": result.seed_entropy,
        "seed_spawn_key": result.seed_spawn_key,
        "workers": sweep.workers,
        "upper": result.upper,
        "upper_norm": "trace_norm",
        "modeled_total_time": result.modeled_total_time,
        "modeled_time_convention": "sum(configurations * shots * evolution_time)",
        "initial_guess_distance": float(np.linalg.norm(initial - truth)),
        "summary": result.summary(),
        "timings_seconds": {
            "shared_probability_preparation": sweep.probability_seconds,
            "sampling": result.timings.sampling_seconds,
            "fitting": result.timings.fitting_seconds,
            "sweep_compute_total": sweep.compute_seconds,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "packages": _package_versions(),
        },
    }


def _result_directory(base_directory: Path, result: ExperimentResult) -> Path:
    directory = base_directory
    if result.point.noise != 0.0:
        directory = directory / f"noise_{100.0 * result.point.noise:g}p"
    if result.point.initial_guess_scale != 1.0:
        directory = directory / "initial_guess"
    return directory


def write_sweep_outputs(
    sweep: SweepResult, output_directory: str | Path
) -> tuple[Path, ...]:
    base_directory = Path(output_directory)
    written: list[Path] = []
    for result in sweep.points:
        directory = _result_directory(base_directory, result)
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / legacy_filename(result)
        pd.DataFrame(
            {"l2_error_unfiltered": result.errors}
        ).to_csv(csv_path, index=False)
        metadata_path = csv_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(_metadata_for_result(result, sweep), indent=2),
            encoding="ascii",
        )
        written.extend((csv_path, metadata_path))
    return tuple(written)