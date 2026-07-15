# Benchmark Results

Generated: 2026-07-15T00:52:10.293292+00:00

## Workload

- Qubits: 2
- Control strength: 4.5
- Epsilon: 0.0001
- Shots: 29
- Repeats: 200
- Noise: 0.0
- Seed: 20260714
- Worker processes: 4

## Wall-Clock Results

| Backend | Probability / sampling (s) | Fitting (s) | Total (s) |
|---|---:|---:|---:|
| Legacy QuTiP | 87.410923 | 142.324849 | 229.735772 |
| Optimized, 1 worker | 0.137887 | 1.032905 | 1.171113 |
| Optimized, 4 workers | 0.150118 | 1.754671 | 1.905276 |

Algorithmic speedup (one worker): **196.17x**

Practical speedup (4 workers): **120.58x**

On this workload, one worker was faster than four workers because Windows process startup outweighed the parallel fitting gain.

## Output Summary

| Backend | Median error | 80th percentile | Success rate |
|---|---:|---:|---:|
| Legacy | 1.096021e-05 | 9.829018e-05 | 0.800 |
| Optimized, 1 worker | 1.086424e-05 | 5.273939e-04 | 0.705 |
| Optimized, 4 workers | 1.086424e-05 | 5.273939e-04 | 0.705 |

Single-worker and multiworker optimized outputs identical: `True`

## Correctness Evidence

The deterministic headline validation in `VALIDATION.json` measured:

- Maximum cached-probability error: `8.027e-11` (tolerance `1.176e-10`).
- Maximum analytic-Jacobian error: `8.361e-09` (tolerance `2.000e-08`).
- Optimized versus legacy fixed-target coefficient difference: `6.206e-13`.
- Twenty-repeat legacy versus optimized KS p-value: `0.336`.

The existing 200-sample legacy CSV has median error `1.056045e-05`, 80th percentile `1.496554e-04`, and success rate `0.770`. Against the optimized 200-sample output, the two-sample KS statistic is `0.090` with p-value `0.394`; this does not indicate a distributional difference.

The 80th percentile lies near the boundary between the narrow successful mode and a sparse failure tail, so it varies more between independent finite samples than the median.

## Environment

- Platform: Windows-11-10.0.26200-SP0
- Processor: ARMv8 (64-bit) Family 8 Model 1 Revision 201, Qualcomm Technologies Inc
- Logical CPUs: 12
- Python: 3.12.10
- Packages: {"numpy": "2.5.1", "pandas": "3.0.3", "qutip": "5.3.0", "scipy": "1.18.0"}

Times include probability-cache construction. CSV serialization is excluded so the table measures the numerical workload. The legacy and optimized samplers are distributionally equivalent but consume random numbers differently, so their rows are not expected to match one for one.
