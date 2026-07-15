from __future__ import annotations

import unittest

import numpy as np
from qutip import Qobj, expect

from H_learning_improved.experiments import (
    ONE_QUBIT_TRUE,
    TWO_QUBIT_TRUE,
)
from H_learning_improved.models import one_qubit_model, two_qubit_model
from H_learning_improved.rfe import (
    apply_sign_flip_noise,
    build_probability_table,
    build_rfe_schedule,
    gap_estimates_from_counts,
    simulate_gap_estimates,
)


def _scalar_gap_from_counts(
    upper: float,
    counts_c: np.ndarray,
    counts_s: np.ndarray,
    shots: int,
) -> float:
    low = 0.0
    high = upper
    for count_c, count_s in zip(counts_c, counts_s):
        mean_c = 2.0 * count_c / shots - 1.0
        mean_s = 2.0 * count_s / shots - 1.0
        phase = np.exp(
            -1j * (low + high) * np.pi / (2.0 * (high - low))
        )
        if np.imag((mean_c + 1j * mean_s) * phase) > 0:
            low = (2.0 * low + high) / 3.0
        else:
            high = (low + 2.0 * high) / 3.0
    return 0.5 * (low + high)


class RFEEquivalenceTests(unittest.TestCase):
    def test_schedule_is_branch_independent(self) -> None:
        schedule = build_rfe_schedule(7.0, 1e-5)
        expected_widths = 7.0 * (2.0 / 3.0) ** np.arange(
            schedule.iteration_count
        )
        np.testing.assert_allclose(schedule.widths, expected_widths)
        np.testing.assert_allclose(schedule.times, np.pi / expected_widths)
        self.assertLessEqual(
            schedule.widths[-1] * 2.0 / 3.0, schedule.epsilon
        )

    def test_cached_probabilities_match_direct_qutip(self) -> None:
        cases = (
            (one_qubit_model(), ONE_QUBIT_TRUE, 3.0, 1e-6, 2e-9),
            (two_qubit_model(), TWO_QUBIT_TRUE, 4.5, 1e-4, 2e-10),
        )
        for model, coefficients, control_strength, epsilon, tolerance in cases:
            upper = control_strength + 2.0 * model.trace_norm(coefficients)
            table = build_probability_table(
                model,
                coefficients,
                control_strength,
                upper,
                epsilon,
            )
            hamiltonians = model.controlled_hamiltonians(
                coefficients, control_strength
            )
            maximum_error = 0.0
            for config_index in range(model.configuration_count):
                hamiltonian = Qobj(hamiltonians[config_index])
                state = Qobj(model.states[config_index])
                observable_c = Qobj(model.observables_c[config_index])
                observable_s = Qobj(model.observables_s[config_index])
                for time_index, evolution_time in enumerate(
                    table.schedule.times
                ):
                    evolved = (
                        (-1j * hamiltonian * evolution_time).expm() * state
                    )
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
            self.assertLess(maximum_error, tolerance)

    def test_vectorized_interval_updates_match_scalar_updates(self) -> None:
        rng = np.random.default_rng(8128)
        iterations = 12
        repeats = 7
        configurations = 5
        shots = 29
        counts_c = rng.integers(
            0,
            shots + 1,
            size=(iterations, repeats, configurations),
            dtype=np.int64,
        )
        counts_s = rng.integers(
            0,
            shots + 1,
            size=(iterations, repeats, configurations),
            dtype=np.int64,
        )
        vectorized = gap_estimates_from_counts(
            7.25, counts_c, counts_s, shots
        )
        scalar = np.empty_like(vectorized)
        for repeat_index in range(repeats):
            for config_index in range(configurations):
                scalar[repeat_index, config_index] = _scalar_gap_from_counts(
                    7.25,
                    counts_c[:, repeat_index, config_index],
                    counts_s[:, repeat_index, config_index],
                    shots,
                )
        np.testing.assert_array_equal(vectorized, scalar)

    def test_noise_folding_is_exact(self) -> None:
        probabilities = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        noise = 0.17
        expected = probabilities * (1.0 - noise) + (
            1.0 - probabilities
        ) * noise
        np.testing.assert_allclose(
            apply_sign_flip_noise(probabilities, noise), expected
        )

    def test_sampling_is_reproducible(self) -> None:
        model = one_qubit_model()
        upper = 3.0 + 2.0 * model.trace_norm(ONE_QUBIT_TRUE)
        table = build_probability_table(
            model, ONE_QUBIT_TRUE, 3.0, upper, 1e-3
        )
        first = simulate_gap_estimates(
            table,
            epsilon=1e-3,
            shots=27,
            repeats=10,
            rng=np.random.default_rng(42),
        )
        second = simulate_gap_estimates(
            table,
            epsilon=1e-3,
            shots=27,
            repeats=10,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()