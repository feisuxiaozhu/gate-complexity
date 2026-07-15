"""Dense one- and two-qubit experiment definitions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

IDENTITY = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PAULI_BY_BETA = {1: PAULI_X, 2: PAULI_Y, 3: PAULI_Z}
PAULI_BY_LABEL = {
    "I": IDENTITY,
    "X": PAULI_X,
    "Y": PAULI_Y,
    "Z": PAULI_Z,
}

TWO_QUBIT_ORDER = (
    "XI",
    "YI",
    "ZI",
    "IX",
    "IY",
    "IZ",
    "XX",
    "XY",
    "XZ",
    "YX",
    "YY",
    "YZ",
    "ZX",
    "ZY",
    "ZZ",
)


def _readonly(array: ComplexArray) -> ComplexArray:
    array.setflags(write=False)
    return array


def _normalized(vector: ComplexArray) -> ComplexArray:
    return vector / np.linalg.norm(vector)


def single_qubit_eigenstate(state: int, beta: int) -> ComplexArray:
    """Return the legacy eigenstate of Pauli ``beta`` with sign ``(-1)^state``."""
    if state not in (0, 1):
        raise ValueError("state must be 0 or 1")
    if beta == 1:
        return np.array([1, 1 if state == 0 else -1], dtype=np.complex128) / np.sqrt(2)
    if beta == 2:
        return np.array([1, 1j if state == 0 else -1j], dtype=np.complex128) / np.sqrt(2)
    if beta == 3:
        result = np.zeros(2, dtype=np.complex128)
        result[state] = 1
        return result
    raise ValueError("beta must be 1, 2, or 3")


def _operator_on_qubit(operator: ComplexArray, qubit: int) -> ComplexArray:
    if qubit == 0:
        return np.kron(operator, IDENTITY)
    if qubit == 1:
        return np.kron(IDENTITY, operator)
    raise ValueError("qubit must be 0 or 1")


@dataclass(frozen=True, slots=True)
class ExperimentModel:
    """Static matrices for one Hamiltonian-learning experiment family."""

    name: str
    basis_labels: tuple[str, ...]
    basis: ComplexArray
    controls: ComplexArray
    states: ComplexArray
    observables_c: ComplexArray
    observables_s: ComplexArray
    configurations: tuple[tuple[Any, ...], ...]

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[-1])

    @property
    def coefficient_count(self) -> int:
        return int(self.basis.shape[0])

    @property
    def configuration_count(self) -> int:
        return int(self.controls.shape[0])

    def hamiltonian(self, coefficients: ArrayLike) -> ComplexArray:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (self.coefficient_count,):
            raise ValueError(
                f"expected {self.coefficient_count} coefficients, got {values.shape}"
            )
        return np.tensordot(values, self.basis, axes=(0, 0))

    def controlled_hamiltonians(
        self, coefficients: ArrayLike, control_strength: float
    ) -> ComplexArray:
        return self.hamiltonian(coefficients)[None, :, :] - control_strength * self.controls

    def spectral_gaps(
        self, coefficients: ArrayLike, control_strength: float
    ) -> FloatArray:
        eigenvalues = np.linalg.eigvalsh(
            self.controlled_hamiltonians(coefficients, control_strength)
        )
        return np.asarray(eigenvalues[:, 1] - eigenvalues[:, 0], dtype=np.float64)

    def trace_norm(self, coefficients: ArrayLike) -> float:
        eigenvalues = np.linalg.eigvalsh(self.hamiltonian(coefficients))
        return float(np.abs(eigenvalues).sum())


@lru_cache(maxsize=1)
def one_qubit_model() -> ExperimentModel:
    basis = np.stack((PAULI_Z, PAULI_X, PAULI_Y))
    controls: list[ComplexArray] = []
    states: list[ComplexArray] = []
    observables_c: list[ComplexArray] = []
    observables_s: list[ComplexArray] = []
    configurations: list[tuple[int]] = []

    oc_table = {1: PAULI_Z, 2: PAULI_Z, 3: PAULI_X}
    os_table = {1: PAULI_Y, 2: -PAULI_X, 3: -PAULI_Y}

    for beta in (1, 2, 3):
        controls.append(0.5 * PAULI_BY_BETA[beta])
        states.append(
            _normalized(
                single_qubit_eigenstate(1, beta)
                + single_qubit_eigenstate(0, beta)
            )
        )
        observables_c.append(oc_table[beta])
        observables_s.append(os_table[beta])
        configurations.append((beta,))

    return ExperimentModel(
        name="one_qubit",
        basis_labels=("Z", "X", "Y"),
        basis=_readonly(np.stack(basis).astype(np.complex128)),
        controls=_readonly(np.stack(controls).astype(np.complex128)),
        states=_readonly(np.stack(states).astype(np.complex128)),
        observables_c=_readonly(np.stack(observables_c).astype(np.complex128)),
        observables_s=_readonly(np.stack(observables_s).astype(np.complex128)),
        configurations=tuple(configurations),
    )


@lru_cache(maxsize=1)
def two_qubit_model() -> ExperimentModel:
    basis = np.stack(
        [np.kron(PAULI_BY_LABEL[label[0]], PAULI_BY_LABEL[label[1]]) for label in TWO_QUBIT_ORDER]
    )
    controls: list[ComplexArray] = []
    states: list[ComplexArray] = []
    observables_c: list[ComplexArray] = []
    observables_s: list[ComplexArray] = []
    configurations: list[tuple[int, int, int, int, int]] = []

    oc_table = {1: PAULI_Z, 2: PAULI_Z, 3: PAULI_X}
    os_table = {1: PAULI_Y, 2: -PAULI_X, 3: -PAULI_Y}

    for qubit in (0, 1):
        for state_1 in (0, 1):
            for state_2 in (0, 1):
                for beta_1 in (1, 2, 3):
                    for beta_2 in (1, 2, 3):
                        first_control_weight = 0.5 if qubit == 0 else 1.0
                        second_control_weight = 0.5 if qubit == 1 else 1.0
                        control = (
                            first_control_weight
                            * (-1) ** state_1
                            * _operator_on_qubit(PAULI_BY_BETA[beta_1], 0)
                            + second_control_weight
                            * (-1) ** state_2
                            * _operator_on_qubit(PAULI_BY_BETA[beta_2], 1)
                        )

                        first_state = np.kron(
                            single_qubit_eigenstate(state_1, beta_1),
                            single_qubit_eigenstate(state_2, beta_2),
                        )
                        flipped_state = np.kron(
                            single_qubit_eigenstate(
                                1 - state_1 if qubit == 0 else state_1,
                                beta_1,
                            ),
                            single_qubit_eigenstate(
                                1 - state_2 if qubit == 1 else state_2,
                                beta_2,
                            ),
                        )
                        beta = beta_1 if qubit == 0 else beta_2
                        measured_state = state_1 if qubit == 0 else state_2

                        controls.append(control)
                        states.append(_normalized(first_state + flipped_state))
                        observables_c.append(_operator_on_qubit(oc_table[beta], qubit))
                        observables_s.append(
                            (-1) ** measured_state
                            * _operator_on_qubit(os_table[beta], qubit)
                        )
                        configurations.append(
                            (qubit, state_1, state_2, beta_1, beta_2)
                        )

    return ExperimentModel(
        name="two_qubit",
        basis_labels=TWO_QUBIT_ORDER,
        basis=_readonly(np.stack(basis).astype(np.complex128)),
        controls=_readonly(np.stack(controls).astype(np.complex128)),
        states=_readonly(np.stack(states).astype(np.complex128)),
        observables_c=_readonly(np.stack(observables_c).astype(np.complex128)),
        observables_s=_readonly(np.stack(observables_s).astype(np.complex128)),
        configurations=tuple(configurations),
    )