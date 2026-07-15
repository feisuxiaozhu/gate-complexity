from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np

from H_learning_improved.experiments import (
    ONE_QUBIT_TRUE,
    TWO_QUBIT_TRUE,
)
from H_learning_improved.models import one_qubit_model, two_qubit_model


def _load_legacy_modules():
    legacy_directory = Path(__file__).resolve().parents[2] / "H_learning"
    try:
        importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        matplotlib_module = types.ModuleType("matplotlib")
        matplotlib_module.__path__ = []
        sys.modules["matplotlib"] = matplotlib_module
        sys.modules["matplotlib.pyplot"] = types.ModuleType(
            "matplotlib.pyplot"
        )
    sys.path.insert(0, str(legacy_directory))
    try:
        legacy_rfe = importlib.import_module("RFE")
        legacy_two_qubit = importlib.import_module("RFE_2_qubits")
    finally:
        sys.path.pop(0)
    return legacy_rfe, legacy_two_qubit


class ModelEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy_rfe, cls.legacy_two_qubit = _load_legacy_modules()

    def test_one_qubit_model_matches_legacy(self) -> None:
        model = one_qubit_model()
        legacy_hamiltonian = (
            ONE_QUBIT_TRUE[0] * self.legacy_rfe.sigmaz()
            + ONE_QUBIT_TRUE[1] * self.legacy_rfe.sigmax()
            + ONE_QUBIT_TRUE[2] * self.legacy_rfe.sigmay()
        )
        np.testing.assert_allclose(
            model.hamiltonian(ONE_QUBIT_TRUE),
            legacy_hamiltonian.full(),
            atol=0.0,
            rtol=0.0,
        )
        for config_index, (beta,) in enumerate(model.configurations):
            np.testing.assert_allclose(
                model.controls[config_index],
                (0.5 * self.legacy_rfe.pauli[beta]).full(),
            )
            np.testing.assert_allclose(
                model.states[config_index],
                self.legacy_rfe.def_phi_plus(1, beta).full().ravel(),
            )
            np.testing.assert_allclose(
                model.observables_c[config_index],
                self.legacy_rfe.Oc_table[beta].full(),
            )
            np.testing.assert_allclose(
                model.observables_s[config_index],
                self.legacy_rfe.Os_table[beta].full(),
            )

    def test_two_qubit_model_matches_all_legacy_configurations(self) -> None:
        model = two_qubit_model()
        np.testing.assert_allclose(
            model.hamiltonian(TWO_QUBIT_TRUE),
            self.legacy_two_qubit.H_0(TWO_QUBIT_TRUE).full(),
        )
        self.assertEqual(model.configuration_count, 72)
        for config_index, configuration in enumerate(model.configurations):
            qubit, state_1, state_2, beta_1, beta_2 = configuration
            legacy_control = self.legacy_two_qubit.H_ctrl_func(
                qubit, state_1, state_2, beta_1, beta_2
            )
            legacy_c, legacy_s = self.legacy_two_qubit.Oc_Os_decider(
                qubit, state_1, state_2, beta_1, beta_2
            )
            first_state = self.legacy_two_qubit.two_qubit_eigenstate(
                state_1, state_2, beta_1, beta_2
            )
            if qubit == 0:
                second_state = self.legacy_two_qubit.two_qubit_eigenstate(
                    1 - state_1, state_2, beta_1, beta_2
                )
            else:
                second_state = self.legacy_two_qubit.two_qubit_eigenstate(
                    state_1, 1 - state_2, beta_1, beta_2
                )
            legacy_state = (first_state + second_state).unit()

            np.testing.assert_allclose(
                model.controls[config_index], legacy_control.full()
            )
            np.testing.assert_allclose(
                model.states[config_index], legacy_state.full().ravel()
            )
            np.testing.assert_allclose(
                model.observables_c[config_index], legacy_c.full()
            )
            np.testing.assert_allclose(
                model.observables_s[config_index], legacy_s.full()
            )

    def test_trace_norm_matches_qutip(self) -> None:
        for model, coefficients, legacy_hamiltonian in (
            (
                one_qubit_model(),
                ONE_QUBIT_TRUE,
                ONE_QUBIT_TRUE[0] * self.legacy_rfe.sigmaz()
                + ONE_QUBIT_TRUE[1] * self.legacy_rfe.sigmax()
                + ONE_QUBIT_TRUE[2] * self.legacy_rfe.sigmay(),
            ),
            (
                two_qubit_model(),
                TWO_QUBIT_TRUE,
                self.legacy_two_qubit.H_0(TWO_QUBIT_TRUE),
            ),
        ):
            self.assertAlmostEqual(
                model.trace_norm(coefficients),
                float(legacy_hamiltonian.norm()),
                places=13,
            )


if __name__ == "__main__":
    unittest.main()