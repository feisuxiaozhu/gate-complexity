from __future__ import annotations

import unittest

import numpy as np

from H_learning_improved.experiments import (
    ONE_QUBIT_INITIAL,
    ONE_QUBIT_TRUE,
    TWO_QUBIT_INITIAL,
    TWO_QUBIT_TRUE,
)
from H_learning_improved.models import one_qubit_model, two_qubit_model
from H_learning_improved.solver import fit_gap_targets, gaps_and_jacobian


class SolverTests(unittest.TestCase):
    def test_analytic_jacobians_match_central_differences(self) -> None:
        cases = (
            (one_qubit_model(), ONE_QUBIT_TRUE, 3.0, 1e-8),
            (two_qubit_model(), TWO_QUBIT_TRUE, 4.5, 2e-8),
        )
        for model, coefficients, control_strength, tolerance in cases:
            _, analytic = gaps_and_jacobian(
                model, coefficients, control_strength
            )
            numerical = np.empty_like(analytic)
            for coefficient_index in range(model.coefficient_count):
                step = 1e-6
                high = coefficients.copy()
                low = coefficients.copy()
                high[coefficient_index] += step
                low[coefficient_index] -= step
                numerical[:, coefficient_index] = (
                    model.spectral_gaps(high, control_strength)
                    - model.spectral_gaps(low, control_strength)
                ) / (2.0 * step)
            np.testing.assert_allclose(
                analytic, numerical, atol=tolerance, rtol=0.0
            )

    def test_noiseless_targets_recover_true_coefficients(self) -> None:
        cases = (
            (
                one_qubit_model(),
                ONE_QUBIT_TRUE,
                ONE_QUBIT_INITIAL,
                3.0,
            ),
            (
                two_qubit_model(),
                TWO_QUBIT_TRUE,
                TWO_QUBIT_INITIAL,
                4.5,
            ),
        )
        for model, truth, initial, control_strength in cases:
            targets = model.spectral_gaps(truth, control_strength)
            result = fit_gap_targets(
                model, targets, control_strength, initial
            )
            self.assertTrue(result.success)
            np.testing.assert_allclose(
                result.coefficients, truth, atol=1e-10, rtol=0.0
            )


if __name__ == "__main__":
    unittest.main()