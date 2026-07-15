"""Cached probabilities and vectorized robust frequency estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import ExperimentModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class RFESchedule:
    upper: float
    epsilon: float
    widths: FloatArray
    times: FloatArray

    @property
    def iteration_count(self) -> int:
        return int(self.times.size)

    def prefix_length(self, epsilon: float) -> int:
        if epsilon < self.epsilon:
            raise ValueError(
                f"epsilon {epsilon} is finer than cached epsilon {self.epsilon}"
            )
        return int(np.count_nonzero(self.widths > epsilon))


@dataclass(frozen=True, slots=True)
class ProbabilityTable:
    schedule: RFESchedule
    probabilities_c: FloatArray
    probabilities_s: FloatArray

    @property
    def configuration_count(self) -> int:
        return int(self.probabilities_c.shape[0])

    def probabilities_for_epsilon(
        self, epsilon: float
    ) -> tuple[FloatArray, FloatArray]:
        length = self.schedule.prefix_length(epsilon)
        return self.probabilities_c[:, :length], self.probabilities_s[:, :length]


def build_rfe_schedule(upper: float, epsilon: float) -> RFESchedule:
    if upper <= 0:
        raise ValueError("upper must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    widths: list[float] = []
    width = float(upper)
    while width > epsilon:
        widths.append(width)
        width *= 2.0 / 3.0

    width_array = np.asarray(widths, dtype=np.float64)
    return RFESchedule(
        upper=float(upper),
        epsilon=float(epsilon),
        widths=width_array,
        times=np.pi / width_array,
    )


def build_probability_table(
    model: ExperimentModel,
    coefficients: ArrayLike,
    control_strength: float,
    upper: float,
    epsilon: float,
) -> ProbabilityTable:
    schedule = build_rfe_schedule(upper, epsilon)
    hamiltonians = model.controlled_hamiltonians(coefficients, control_strength)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonians)

    state_in_eigenbasis = np.einsum(
        "cdi,cd->ci", eigenvectors.conj(), model.states, optimize=True
    )
    phases = np.exp(
        -1j * eigenvalues[:, :, None] * schedule.times[None, None, :]
    )
    evolved_states = np.einsum(
        "cdi,cit->cdt",
        eigenvectors,
        state_in_eigenbasis[:, :, None] * phases,
        optimize=True,
    )

    expectations_c = np.einsum(
        "cdt,cde,cet->ct",
        evolved_states.conj(),
        model.observables_c,
        evolved_states,
        optimize=True,
    ).real
    expectations_s = np.einsum(
        "cdt,cde,cet->ct",
        evolved_states.conj(),
        model.observables_s,
        evolved_states,
        optimize=True,
    ).real

    probabilities_c = np.clip((1.0 + expectations_c) / 2.0, 0.0, 1.0)
    probabilities_s = np.clip((1.0 + expectations_s) / 2.0, 0.0, 1.0)
    return ProbabilityTable(
        schedule=schedule,
        probabilities_c=np.asarray(probabilities_c, dtype=np.float64),
        probabilities_s=np.asarray(probabilities_s, dtype=np.float64),
    )


def apply_sign_flip_noise(probabilities: ArrayLike, noise: float) -> FloatArray:
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be between 0 and 1")
    values = np.asarray(probabilities, dtype=np.float64)
    return noise + (1.0 - 2.0 * noise) * values


def gap_estimates_from_counts(
    upper: float,
    counts_c: IntArray,
    counts_s: IntArray,
    shots: int,
) -> FloatArray:
    if shots <= 0:
        raise ValueError("shots must be positive")
    if counts_c.shape != counts_s.shape or counts_c.ndim != 3:
        raise ValueError("counts must both have shape (iterations, repeats, configs)")

    _, repeats, configurations = counts_c.shape
    low = np.zeros((repeats, configurations), dtype=np.float64)
    high = np.full((repeats, configurations), upper, dtype=np.float64)

    for count_c, count_s in zip(counts_c, counts_s):
        mean_c = 2.0 * count_c / shots - 1.0
        mean_s = 2.0 * count_s / shots - 1.0
        phase = np.exp(-1j * (low + high) * np.pi / (2.0 * (high - low)))
        keep_high = np.imag((mean_c + 1j * mean_s) * phase) > 0
        old_low = low
        old_high = high
        low = np.where(keep_high, (2.0 * old_low + old_high) / 3.0, old_low)
        high = np.where(
            keep_high, old_high, (old_low + 2.0 * old_high) / 3.0
        )

    return 0.5 * (low + high)


def simulate_gap_estimates(
    table: ProbabilityTable,
    epsilon: float,
    shots: int,
    repeats: int,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if shots <= 0:
        raise ValueError("shots must be positive")
    if rng is None:
        rng = np.random.default_rng()

    probabilities_c, probabilities_s = table.probabilities_for_epsilon(epsilon)
    probabilities_c = apply_sign_flip_noise(probabilities_c, noise)
    probabilities_s = apply_sign_flip_noise(probabilities_s, noise)
    iterations, configurations = probabilities_c.T.shape

    counts_c = np.empty((iterations, repeats, configurations), dtype=np.int64)
    counts_s = np.empty_like(counts_c)
    for iteration in range(iterations):
        counts_c[iteration] = rng.binomial(
            shots,
            probabilities_c[:, iteration],
            size=(repeats, configurations),
        )
        counts_s[iteration] = rng.binomial(
            shots,
            probabilities_s[:, iteration],
            size=(repeats, configurations),
        )

    return gap_estimates_from_counts(
        table.schedule.upper, counts_c, counts_s, shots
    )


def modeled_total_time(
    table: ProbabilityTable, epsilon: float, shots: int
) -> float:
    length = table.schedule.prefix_length(epsilon)
    return float(
        table.configuration_count * shots * table.schedule.times[:length].sum()
    )