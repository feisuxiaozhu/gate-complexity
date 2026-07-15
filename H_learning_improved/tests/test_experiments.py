from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from H_learning_improved.experiments import (
    ExperimentPoint,
    run_sweep,
    write_sweep_outputs,
)


class ExperimentTests(unittest.TestCase):
    def test_seeded_sweep_is_reproducible(self) -> None:
        point = ExperimentPoint(
            qubits=1,
            control_strength=3.0,
            epsilon=1e-3,
            shots=27,
            repeats=5,
            noise=0.1,
        )
        first = run_sweep([point], seed=101, workers=1).points[0]
        second = run_sweep([point], seed=101, workers=1).points[0]
        np.testing.assert_array_equal(first.target_gaps, second.target_gaps)
        np.testing.assert_array_equal(first.estimates, second.estimates)
        np.testing.assert_array_equal(first.errors, second.errors)

    def test_worker_count_does_not_change_results(self) -> None:
        point = ExperimentPoint(
            qubits=2,
            control_strength=4.5,
            epsilon=1e-3,
            shots=27,
            repeats=3,
        )
        serial = run_sweep([point], seed=2026, workers=1).points[0]
        parallel = run_sweep([point], seed=2026, workers=2).points[0]
        np.testing.assert_array_equal(serial.target_gaps, parallel.target_gaps)
        np.testing.assert_array_equal(serial.estimates, parallel.estimates)
        np.testing.assert_array_equal(serial.errors, parallel.errors)

    def test_csv_and_metadata_are_legacy_compatible(self) -> None:
        point = ExperimentPoint(
            qubits=2,
            control_strength=4.5,
            epsilon=1e-3,
            shots=27,
            repeats=3,
        )
        sweep = run_sweep([point], seed=44, workers=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_sweep_outputs(sweep, temporary_directory)
            csv_paths = [path for path in paths if path.suffix == ".csv"]
            json_paths = [path for path in paths if path.suffix == ".json"]
            self.assertEqual(len(csv_paths), 1)
            self.assertEqual(len(json_paths), 1)
            self.assertEqual(
                csv_paths[0].name,
                "l2_error_nu4.5_eps0.001_shots27.csv",
            )
            dataframe = pd.read_csv(csv_paths[0])
            self.assertEqual(list(dataframe.columns), ["l2_error_unfiltered"])
            metadata = json.loads(
                Path(json_paths[0]).read_text(encoding="ascii")
            )
            self.assertEqual(metadata["seed"], 44)
            self.assertEqual(metadata["parameters"]["qubits"], 2)


if __name__ == "__main__":
    unittest.main()