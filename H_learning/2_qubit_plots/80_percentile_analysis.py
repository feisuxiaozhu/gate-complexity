import numpy as np
import matplotlib.pyplot as plt


# t_total and 80 percentile l2 error 
data = {
    10: {
        "T_total": [1.484e+06, 1.691e+07, 1.284e+08, 1.463e+09, 1.666e+10],
        "l2":      [9.867e-04, 8.825e-05, 4.887e-05, 1.050e-06, 1.663e-05],
    },
    15: {
        "T_total": [1.145e+06, 1.305e+07, 1.486e+08, 1.693e+09, 1.286e+10],
        "l2":      [1.218e-03, 1.111e-04, 1.003e-05, 2.329e-06, 1.311e-07],
    },
    20: {
        "T_total": [1.398e+06, 1.593e+07, 1.210e+08, 1.378e+09, 1.570e+10],
        "l2":      [9.742e-04, 8.840e-05, 1.287e-05, 2.619e-06, 1.966e-07],
    },
    25: {
        "T_total": [1.179e+06, 1.343e+07, 1.530e+08, 1.162e+09, 1.324e+10],
        "l2":      [1.203e-03, 1.049e-04, 1.153e-05, 1.809e-05, 4.383e-06],
    },
}

# t_total and l2 error average when succes
data = {
    10: {
        "T_total": [1.484e+06, 1.691e+07, 1.284e+08, 1.463e+09, 1.666e+10],
        "l2":      [9.329e-04, 8.224e-05, 1.243e-05, 9.266e-07, 7.341e-08],
    },
    15: {
        "T_total": [1.145e+06, 1.305e+07, 1.486e+08, 1.693e+09, 1.286e+10],
        "l2":      [1.183e-03, 9.511e-05, 9.232e-06, 8.751e-07, 1.047e-07],
    },
    20: {
        "T_total": [1.398e+06, 1.593e+07, 1.210e+08, 1.378e+09, 1.570e+10],
        "l2":      [8.748e-04, 8.657e-05, 1.131e-05, 9.338e-07, 9.743e-08],
    },
    25: {
        "T_total": [1.179e+06, 1.343e+07, 1.530e+08, 1.162e+09, 1.324e+10],
        "l2":      [1.063e-03, 9.228e-05, 9.388e-06, 1.027e-06, 9.196e-08],
    },
}

plt.figure(figsize=(7, 5))

slopes = {}

for nu, d in data.items():
    T_total = np.array(d["T_total"], dtype=float)
    l2      = np.array(d["l2"], dtype=float)

    log_T = np.log10(T_total)
    log_inv_error = np.log10(1.0 / l2)

    # Fit: log10(1/error) ≈ slope * log10(T) + intercept
    slope, intercept = np.polyfit(log_T, log_inv_error, 1)
    slopes[nu] = (slope, intercept)

    # Plot data (points with line) and fit (same color, dashed)
    line, = plt.plot(log_T, log_inv_error, 'o-', label=f'ν={nu} data')
    color = line.get_color()
    x_fit = np.linspace(log_T.min(), log_T.max(), 200)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, '--', label=f'ν={nu} fit (s={slope:.3f})', color=color)

plt.xlabel('log10(T_total)')
plt.ylabel('log10(1 / l2_error)')
plt.title('Scaling of error with total time (2 qubit, 80th percentile, shots=25)')
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# Print slopes and intercepts
for nu in sorted(slopes):
    s, c = slopes[nu]
    print(f"nu={nu}: slope={s:.6f}, intercept={c:.6f}")
