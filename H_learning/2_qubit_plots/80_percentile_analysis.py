import numpy as np
import matplotlib.pyplot as plt


# t_total and 80 percentile l2 error 
data = {
    # varying shots
    3: {
        "T_total": [1.690e+06,1.284e+07,1.579e+08,1.932e+09,1.569e+10],
        "l2":      [3.022e-02,3.426e-02,3.146e-02,3.278e-02,3.080e-02],
    },
    # varying shots
    4: {
        "T_total": [1.534e+06,1.166e+07,1.434e+08,1.755e+09,1.424e+10],
        "l2":      [5.389e-03,5.591e-03,5.124e-03,5.585e-03,1.606e-03],
    },
    # varying shots
    5: {
        "T_total": [1.405e+06,1.602e+07,1.314e+08,1.607e+09,1.957e+10],
        "l2":      [1.096e-03, 1.057e-04,8.261e-05, 2.072e-06,1.044e-07],
    },
    # varying shots
    10: {
        "T_total": [1.484e+06, 1.691e+07, 1.284e+08, 1.463e+09, 1.666e+10],
        "l2":      [9.143e-04,8.515e-05,1.115e-05, 9.258e-07, 8.203e-08],
    },
    # 15: {
    #     "T_total": [1.145e+06, 1.305e+07, 1.486e+08, 1.693e+09, 1.286e+10],
    #     "l2":      [1.218e-03, 1.111e-04, 1.003e-05, 2.329e-06, 1.311e-07],
    # },
    # 20: {
    #     "T_total": [1.398e+06, 1.593e+07, 1.210e+08, 1.378e+09, 1.570e+10],
    #     "l2":      [9.742e-04, 8.840e-05, 1.287e-05, 2.619e-06, 1.966e-07],
    # },
    # 25: {
    #     "T_total": [1.179e+06, 1.343e+07, 1.530e+08, 1.162e+09, 1.324e+10],
    #     "l2":      [1.203e-03, 1.049e-04, 1.153e-05, 1.809e-05, 4.383e-06],
    # },
    # varying shots
    25: {
        "T_total": [1.179e+06, 1.343e+07, 1.530e+08, 1.162e+09, 1.324e+10],
        "l2":      [1.208e-03, 1.073e-04,8.242e-06, 1.160e-06,8.742e-08],
    },
    # 30: {
    #     "T_total": [1.529e+06,1.161e+07,1.323e+08,1.507e+09, 1.144e+10],
    #     "l2":      [ 8.949e-04,1.260e-04, 1.181e-05,1.073e-06,2.965e-07],
    # },
}



# t_total and l2 error average when succes
# data = {
#     10: {
#         "T_total": [1.484e+06, 1.691e+07, 1.284e+08, 1.463e+09, 1.666e+10],
#         "l2":      [9.329e-04, 8.224e-05, 1.243e-05, 9.266e-07, 7.341e-08],
#     },
#     15: {
#         "T_total": [1.145e+06, 1.305e+07, 1.486e+08, 1.693e+09, 1.286e+10],
#         "l2":      [1.183e-03, 9.511e-05, 9.232e-06, 8.751e-07, 1.047e-07],
#     },
#     20: {
#         "T_total": [1.398e+06, 1.593e+07, 1.210e+08, 1.378e+09, 1.570e+10],
#         "l2":      [8.748e-04, 8.657e-05, 1.131e-05, 9.338e-07, 9.743e-08],
#     },
#     25: {
#         "T_total": [1.179e+06, 1.343e+07, 1.530e+08, 1.162e+09, 1.324e+10],
#         "l2":      [1.063e-03, 9.228e-05, 9.388e-06, 1.027e-06, 9.196e-08],
#     },
# }

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
    line, = plt.plot(log_T, log_inv_error, 'o-', label=f'ν={nu}')
    color = line.get_color()
    # x_fit = np.linspace(log_T.min(), log_T.max(), 200)
    # y_fit = slope * x_fit + intercept
    # plt.plot(x_fit, y_fit, '--', label=f'ν={nu} fit (s={slope:.3f})', color=color)

plt.xlabel('log10(T_total)')
plt.ylabel('log10(1 / l2_error)')
plt.title('Scaling of error with total time (2 qubit, 80th percentile, varying shots)')
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# Print slopes and intercepts
for nu in sorted(slopes):
    s, c = slopes[nu]
    print(f"nu={nu}: slope={s:.6f}, intercept={c:.6f}")
