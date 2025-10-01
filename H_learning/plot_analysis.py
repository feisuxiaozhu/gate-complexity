import numpy as np
import matplotlib.pyplot as plt

# N_shots = 10, repeat = 200, success rate almost monotonically decreasing, eps start from 1e-3 to 1e-9
T_total = [70508.281088, 803198.675487, 9149000.198948,  69475261.688068, 791366716.835435, 9014161279.198977, 68451287047.073158]
l2_error = [1.791e-04,  1.300e-05, 1.280e-06, 1.747e-07, 1.328e-08,  1.361e-09,2.790e-10]


# N_shots = 13 and varies, repeat = 500, success rate >=90, eps start from eps = 1e-3 to 1e-6, (lambda_1,lambda_2,lambda_3) = (0.1,0.5,0.3)
T_total = [8.745e+03,1.179e+05, 8.953e+05,1.177e+07,1.430e+08]
l2_error=[1.642e-03,1.345e-04, 1.648e-05, 1.404e-06, 1.208e-07]

# N_shots = 13 and varies, repeat = 500, success rate >=90, eps start from eps = 1e-2 to 1e-6, (lambda_1,lambda_2,lambda_3) = (0.1,0.5,0.3)
T_total = [ 2.624e+04,3.537e+05,3.099e+06,3.530e+07 ,4.289e+08]
l2_error = [1.578e-03,1.267e-04,1.541e-05,1.458e-06,1.253e-07]

T_total = np.array(T_total)
l2_error = np.array(l2_error)


# Transform
log_inv_error = np.log10(1 / l2_error)
log_T = np.log10(T_total)

# Linear fit: log(1/error) ≈ slope * log(T) + intercept
coeffs = np.polyfit(log_T, log_inv_error, 1)
slope, intercept = coeffs

# Plot data and fit
plt.figure(figsize=(6, 4))
plt.plot(log_T, log_inv_error, 'o', label='Data')
plt.plot(log_T, slope * log_T + intercept, '-', label=f'Fit: slope={slope:.3f}, c={intercept:.3f}')
plt.xlabel('log(T_total)')
plt.ylabel('log(1 / l2_error)')
plt.title('Scaling of error with total time (1 qubit)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Slope: {slope:.6f}")
print(f"c (intercept): {intercept:.6f}")







