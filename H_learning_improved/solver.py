"""Fast spectral-gap objectives for Hamiltonian coefficient fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .models import ExperimentModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FitResult:
    coefficients: FloatArray
    cost: float
    optimality: float
    evaluations: int
    jacobian_evaluations: int | None
    status: int
    success: bool
    message: str


def one_qubit_gaps_and_jacobian(
    coefficients: ArrayLike, control_strength: float
) -> tuple[FloatArray, FloatArray]:
    """Return the three controlled gaps and their exact derivatives."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"expected 3 coefficients, got {values.shape}")

    effective = np.broadcast_to(values, (3, 3)).copy()
    # Configurations are beta X, Y, Z; coefficients are ordered Z, X, Y.
    effective[0, 1] -= 0.5 * control_strength
    effective[1, 2] -= 0.5 * control_strength
    effective[2, 0] -= 0.5 * control_strength
    norms = np.linalg.norm(effective, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("gap derivative is undefined at an exact degeneracy")
    return 2.0 * norms, 2.0 * effective / norms[:, None]


def _finite_difference_jacobian(
    model: ExperimentModel,
    coefficients: FloatArray,
    control_strength: float,
) -> FloatArray:
    jacobian = np.empty(
        (model.configuration_count, model.coefficient_count), dtype=np.float64
    )
    for coefficient_index in range(model.coefficient_count):
        step = 1e-6 * max(1.0, abs(coefficients[coefficient_index]))
        high = coefficients.copy()
        low = coefficients.copy()
        high[coefficient_index] += step
        low[coefficient_index] -= step
        jacobian[:, coefficient_index] = (
            model.spectral_gaps(high, control_strength)
            - model.spectral_gaps(low, control_strength)
        ) / (2.0 * step)
    return jacobian


def two_qubit_gaps_and_jacobian(
    model: ExperimentModel,
    coefficients: ArrayLike,
    control_strength: float,
    degeneracy_tolerance: float = 1e-10,
) -> tuple[FloatArray, FloatArray]:
    """Return gaps and the Hellmann-Feynman Jacobian for all configurations."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (model.coefficient_count,):
        raise ValueError(
            f"expected {model.coefficient_count} coefficients, got {values.shape}"
        )

    eigenvalues, eigenvectors = np.linalg.eigh(
        model.controlled_hamiltonians(values, control_strength)
    )
    gaps = np.asarray(eigenvalues[:, 1] - eigenvalues[:, 0], dtype=np.float64)
    if np.any(gaps <= degeneracy_tolerance):
        return gaps, _finite_difference_jacobian(
            model, values, control_strength
        )

    expectations = np.einsum(
        "cdi,pde,cei->cip",
        eigenvectors.conj(),
        model.basis,
        eigenvectors,
        optimize=True,
    ).real
    jacobian = expectations[:, 1, :] - expectations[:, 0, :]
    return gaps, np.asarray(jacobian, dtype=np.float64)


def gaps_and_jacobian(
    model: ExperimentModel,
    coefficients: ArrayLike,
    control_strength: float,
) -> tuple[FloatArray, FloatArray]:
    if model.name == "one_qubit":
        return one_qubit_gaps_and_jacobian(coefficients, control_strength)
    return two_qubit_gaps_and_jacobian(
        model, coefficients, control_strength
    )


class GapObjective:
    """Cache residual and Jacobian work shared by SciPy callbacks."""

    def __init__(
        self,
        model: ExperimentModel,
        targets: ArrayLike,
        control_strength: float,
    ) -> None:
        target_values = np.asarray(targets, dtype=np.float64)
        if target_values.shape != (model.configuration_count,):
            raise ValueError(
                f"expected {model.configuration_count} targets, got {target_values.shape}"
            )
        self.model = model
        self.targets = target_values
        self.control_strength = float(control_strength)
        self._coefficients: FloatArray | None = None
        self._residuals: FloatArray | None = None
        self._jacobian: FloatArray | None = None

    def _evaluate(self, coefficients: ArrayLike) -> None:
        values = np.asarray(coefficients, dtype=np.float64)
        if self._coefficients is not None and np.array_equal(
            values, self._coefficients
        ):
            return
        gaps, jacobian = gaps_and_jacobian(
            self.model, values, self.control_strength
        )
        self._coefficients = values.copy()
        self._residuals = gaps - self.targets
        self._jacobian = jacobian

    def residuals(self, coefficients: ArrayLike) -> FloatArray:
        self._evaluate(coefficients)
        assert self._residuals is not None
        return self._residuals

    def jacobian(self, coefficients: ArrayLike) -> FloatArray:
        self._evaluate(coefficients)
        assert self._jacobian is not None
        return self._jacobian


def fit_gap_targets(
    model: ExperimentModel,
    targets: ArrayLike,
    control_strength: float,
    initial_guess: ArrayLike,
    max_evaluations: int | None = None,
) -> FitResult:
    initial_values = np.asarray(initial_guess, dtype=np.float64)
    if initial_values.shape != (model.coefficient_count,):
        raise ValueError(
            f"expected {model.coefficient_count} initial coefficients, got {initial_values.shape}"
        )
    objective = GapObjective(model, targets, control_strength)
    result = least_squares(
        objective.residuals,
        initial_values,
        jac=objective.jacobian,
        max_nfev=max_evaluations,
    )
    return FitResult(
        coefficients=np.asarray(result.x, dtype=np.float64),
        cost=float(result.cost),
        optimality=float(result.optimality),
        evaluations=int(result.nfev),
        jacobian_evaluations=(
            None if result.njev is None else int(result.njev)
        ),
        status=int(result.status),
        success=bool(result.success),
        message=str(result.message),
    )