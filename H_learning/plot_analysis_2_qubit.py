import numpy as np
import matplotlib.pyplot as plt

# N_shots = 25 and varies, repeat = 200, success rate >90, eps start from 1e-3
# np.array([0.1,0.2,0.3, 0.5,0.6,0.3,0.2,0.1,0.1, 0.2,0.1,0.1, 0.3,0.22,0.15])
T_total = [2.213e+05,1.815e+06,2.220e+07,2.704e+08]
l2_error= [9.456e-05,1.001e-05, 8.747e-07,7.505e-08]


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
plt.title('Scaling of error with total time (2 qubit)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Slope: {slope:.6f}")
print(f"c (intercept): {intercept:.6f}")









