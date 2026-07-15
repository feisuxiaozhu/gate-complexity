"""Parameterized legacy-style backend for validation and timing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj, expect
from scipy.optimize import least_squares

from .experiments import (
    ExperimentPoint,
    scaled_initial_guess,
    true_coefficients,
)
from .models import ExperimentModel, one_qubit_model, two_qubit_model

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    errors: FloatArray
    estimates: FloatArray
    target_gaps: FloatArray
    target_seconds: float
    fitting_seconds: float


def _model_for_qubits(qubits: int) -> ExperimentModel:
    return one_qubit_model() if qubits == 1 else two_qubit_model()


def _sample_expectation(
    state: Qobj,
    operator: Qobj,
    shots: int,
    noise: float,
    rng: np.random.Generator,
) -> float:
    expectation = float(expect(operator, state))
    probability_plus = (1.0 + expectation) / 2.0
    outcomes = rng.random(shots) < probability_plus
    results = 2.0 * outcomes.astype(np.float64) - 1.0
    noise_mask = rng.random(shots) < noise
    results[noise_mask] *= -1.0
    return float(results.mean())


def reference_gap_estimate(
    state: Qobj,
    hamiltonian: Qobj,
    observable_c: Qobj,
    observable_s: Qobj,
    upper: float,
    epsilon: float,
    shots: int,
    noise: float,
    rng: np.random.Generator,
) -> float:
    low = 0.0
    high = upper
    while high - low > epsilon:
        evolution_time = np.pi / (high - low)
        evolved_state = (-1j * hamiltonian * evolution_time).expm() * state
        mean_c = _sample_expectation(
            evolved_state, observable_c, shots, noise, rng
        )
        mean_s = _sample_expectation(
            evolved_state, observable_s, shots, noise, rng
        )
        phase = np.exp(
            -1j * (low + high) * np.pi / (2.0 * (high - low))
        )
        keep_high = np.imag((mean_c + 1j * mean_s) * phase) > 0
        if keep_high:
            low = (2.0 * low + high) / 3.0
        else:
            high = (low + 2.0 * high) / 3.0
    return 0.5 * (low + high)


def reference_target_gaps(
    point: ExperimentPoint,
    seed: int,
) -> FloatArray:
    model = _model_for_qubits(point.qubits)
    truth = true_coefficients(point.qubits)
    upper = point.control_strength + point.upper_norm_factor * model.trace_norm(
        truth
    )
    rng = np.random.default_rng(seed)
    targets = np.empty(
        (point.repeats, model.configuration_count), dtype=np.float64
    )
    for repeat_index in range(point.repeats):
        for config_index in range(model.configuration_count):
            hamiltonian = Qobj(
                model.hamiltonian(truth)
                - point.control_strength * model.controls[config_index]
            )
            targets[repeat_index, config_index] = reference_gap_estimate(
                Qobj(model.states[config_index]),
                hamiltonian,
                Qobj(model.observables_c[config_index]),
                Qobj(model.observables_s[config_index]),
                upper,
                point.epsilon,
                point.shots,
                point.noise,
                rng,
            )
    return targets


def reference_fit_targets(
    model: ExperimentModel,
    targets: FloatArray,
    point: ExperimentPoint,
) -> FloatArray:
    initial = scaled_initial_guess(
        point.qubits, point.initial_guess_scale
    )

    def residuals(coefficients: FloatArray) -> FloatArray:
        values = []
        for config_index, target in enumerate(targets):
            # This deliberately follows the original code and rebuilds H_0
            # inside every configuration and finite-difference evaluation.
            hamiltonian = Qobj(model.hamiltonian(coefficients)) - (
                point.control_strength * Qobj(model.controls[config_index])
            )
            eigenvalues = np.sort(hamiltonian.eigenenergies())
            values.append(eigenvalues[1] - eigenvalues[0] - target)
        return np.asarray(values, dtype=np.float64)

    result = least_squares(
        residuals, initial, max_nfev=point.max_evaluations
    )
    return np.asarray(result.x, dtype=np.float64)


def run_reference_point(
    point: ExperimentPoint,
    seed: int,
) -> ReferenceResult:
    target_start = perf_counter()
    targets = reference_target_gaps(point, seed)
    target_seconds = perf_counter() - target_start
    model = _model_for_qubits(point.qubits)
    fitting_start = perf_counter()
    estimates = np.stack(
        [reference_fit_targets(model, target, point) for target in targets]
    )
    fitting_seconds = perf_counter() - fitting_start
    truth = true_coefficients(point.qubits)
    errors = np.linalg.norm(estimates - truth[None, :], axis=1)
    return ReferenceResult(
        errors=np.asarray(errors, dtype=np.float64),
        estimates=np.asarray(estimates, dtype=np.float64),
        target_gaps=targets,
        target_seconds=target_seconds,
        fitting_seconds=fitting_seconds,
    )