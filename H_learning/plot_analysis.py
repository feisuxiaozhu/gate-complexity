import numpy as np
import matplotlib.pyplot as plt

# N_shots = 10, repeat = 200, success rate almost monotonically decreasing, eps start from 1e-3 to 1e-9
T_total = [70508.281088, 803198.675487, 9149000.198948,  69475261.688068, 791366716.835435, 9014161279.198977, 68451287047.073158]
l2_error = [1.791e-04,  1.300e-05, 1.280e-06, 1.747e-07, 1.328e-08,  1.361e-09,2.790e-10]


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
plt.title('Scaling of error with total time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Slope: {slope:.6f}")
print(f"c (intercept): {intercept:.6f}")







