# Optimized Hamiltonian Learning

This directory contains a faster, reproducible implementation of the one- and two-qubit experiments in `H_learning`. The original directory is unchanged and remains the scientific reference.

## Why It Is Faster

The legacy RFE loop recomputes a matrix exponential and measurement probabilities for every Monte Carlo repeat. That work is deterministic and can be reused.

At every RFE iteration, either branch changes the interval width from `w` to `2w/3`. The evolution time

```text
t = pi / w
```

therefore depends only on iteration depth, not on the sampled branch. The optimized implementation:

1. Builds every state, control, and observable once.
2. Diagonalizes each fixed controlled Hamiltonian once.
3. Computes both quadrature probabilities for the complete RFE schedule once.
4. Reuses finer-schedule prefixes for coarser epsilon values.
5. Samples all repeats in batches with exact binomial draws.
6. Updates all repeat and configuration intervals with vectorized NumPy operations.
7. Fits one-qubit gaps in closed form and two-qubit gaps with batched eigensolvers and analytic Hellmann-Feynman Jacobians.

Independent sign-flip noise is folded into the measurement probability exactly:

```text
q_observed = noise + (1 - 2 * noise) * q
```

Sampling `Binomial(shots, q_observed)` has the same distribution as drawing each legacy measurement and then independently flipping its sign.

## Layout

- `models.py`: dense Pauli bases and static experiment configurations.
- `rfe.py`: deterministic schedules, cached probabilities, batched sampling, and vectorized interval updates.
- `solver.py`: optimized spectral gaps, analytic Jacobians, and coefficient fitting.
- `experiments.py`: seeded sweeps, optional multiprocessing, summaries, CSV files, and JSON metadata.
- `reference.py`: parameterized QuTiP implementation used for validation and baseline timing.
- `run_experiments.py`: main experiment command.
- `validate.py`: deterministic and statistical comparison command.
- `benchmark.py`: matched wall-clock benchmark and report generator.
- `tests/`: regression and legacy-equivalence tests.
- `BENCHMARK.md` and `BENCHMARK.json`: measured results for the headline workload.

## Environment

Run commands from the repository root. Install the numerical dependencies into the active environment:

```powershell
python -m pip install -r H_learning_improved/requirements.txt
```

The improved experiment runner does not require Matplotlib. QuTiP is used only by the reference and validation paths.

## Run Experiments

Run the standard two-qubit epsilon and shot sweep:

```powershell
python -m H_learning_improved.run_experiments --qubits 2 --nu 4.5 --repeats 200 --seed 20260714
```

Run a one-qubit sweep:

```powershell
python -m H_learning_improved.run_experiments --qubits 1 --nu 1.7 --repeats 200 --seed 20260714
```

Run selected two-qubit control strengths:

```powershell
python -m H_learning_improved.run_experiments --qubits 2 --nu 3 3.5 4 4.5 5 --repeats 200
```

Run noise variants. Each noise level is written to a separate subdirectory:

```powershell
python -m H_learning_improved.run_experiments --qubits 2 --nu 4.5 --noise 0.03 0.05 --repeats 200
```

Run the changing-initial-guess experiment:

```powershell
python -m H_learning_improved.run_experiments --qubits 2 --nu 5 --eps 1e-4 --shots 29 --initial-guess-scale 1 5 10 20 50 --repeats 200
```

Use worker processes when a sweep is large enough to amortize Windows process startup:

```powershell
python -m H_learning_improved.run_experiments --qubits 2 --nu 3 3.5 4 4.5 5 --workers 4
```

The default is one worker. A single worker is often faster for small runs.

## Output Compatibility

Every CSV contains the legacy column:

```text
l2_error_unfiltered
```

Standard two-qubit filenames retain the form expected by the existing percentile parser:

```text
l2_error_nu4.5_eps0.0001_shots29.csv
```

One-qubit and noisy profiles retain `Ttotal` in the filename. Initial-guess files end at `_scaling<value>.csv`, correcting the mismatch between the old generator and parser. A JSON sidecar records all parameters, seed, modeled time, summaries, solver diagnostics, wall-clock phase timings, worker count, and package versions.

The modeled resource time follows the existing convention:

```text
T_total = sum(configurations * shots * evolution_time)
```

It is not wall-clock time. The legacy `temp_T = 1` offsets are not retained.

## Correctness

Run the complete test suite:

```powershell
python -m unittest discover -s H_learning_improved/tests -v
```

The tests compare all 72 two-qubit configurations directly with the current legacy modules, compare cached probabilities with direct QuTiP exponentials, verify scalar and vectorized interval updates with identical counts, check both analytic Jacobians, recover noiseless coefficients, and verify seed and worker-count reproducibility.

Run an end-to-end validation:

```powershell
python -m H_learning_improved.validate --qubits 2 --nu 4.5 --eps 1e-4 --shots 29 --repeats 20 --seed 20260714 --output H_learning_improved/VALIDATION.json
```

Legacy and optimized samples are not expected to match row by row because binomial sampling consumes random numbers differently. Correctness is established at deterministic boundaries and by comparing output distributions.

## Benchmark

Generate the headline benchmark:

```powershell
python -m H_learning_improved.benchmark --qubits 2 --nu 4.5 --eps 1e-4 --shots 29 --repeats 200 --workers 4
```

The report includes cache construction in optimized wall time and excludes CSV serialization for all backends. It reports single-worker algorithmic speedup separately from optional multiworker speedup. See `BENCHMARK.md` for the measured result on the current machine.