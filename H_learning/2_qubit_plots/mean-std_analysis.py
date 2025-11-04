
import numpy as np
import matplotlib.pyplot as plt


# t_total, mean, std, varying shot 
data = {
    # varying shots
    3: {
        "T_total": [1.690e+06,1.386e+07,1.696e+08,2.066e+09,1.670e+10],
        "l2_mean": [2.138e-02,1.865e-02,1.686e-02,1.668e-02,1.505e-02],
        "std": [2.542e-02,1.914e-02,1.986e-02,1.851e-02,1.566e-02],
         "median": [1.379e-02,1.119e-02,8.961e-03,9.818e-03,7.521e-03], 
        "p25": [5.106e-03,5.074e-03,4.869e-03,5.256e-03,4.542e-03], 
        "p75": [2.837e-02,2.893e-02,2.457e-02,2.544e-02,2.622e-02], 
        
    },
    4: {
        "T_total": [1.534e+06,1.259e+07,1.541e+08,1.876e+09,1.516e+10],
        "l2_mean": [8.678e-03,3.194e-03,3.400e-03,4.606e-03,1.943e-03],
        "std": [1.869e-02,7.839e-03,9.130e-03,1.227e-02,5.875e-03],
        "median": [9.896e-04,1.273e-04,1.108e-05,1.105e-06,8.871e-07], 
        "p25": [8.213e-04,1.035e-04,9.425e-06,8.380e-07,1.066e-07], 
        "p75": [6.866e-03,2.365e-03,2.069e-04,7.289e-04,1.316e-04], 
    },
    5: {
        "T_total": [1.405e+06,1.730e+07,1.411e+08,1.718e+09,2.083e+10],
        "l2_mean": [1.830e-03,1.111e-03,4.686e-04,4.456e-04,3.491e-04],
        "std": [4.164e-03,6.013e-03,2.041e-03,3.841e-03,3.073e-03],
         "median": [8.718e-04,7.644e-05,9.896e-06,9.097e-07,7.531e-08],
        "p25": [7.594e-04,6.562e-05,8.535e-06,7.815e-07,6.371e-08],
        "p75": [9.986e-04,9.618e-05,1.252e-05,1.074e-06,8.828e-08],
        
    },
    6: {
        "T_total": [1.296e+06,1.596e+07,1.952e+08,1.585e+09,1.921e+10],
        "l2_mean": [1.490e-03,5.329e-04,9.611e-05,1.886e-04,9.749e-05],
        "std": [4.017e-03,2.253e-03,4.077e-04,1.167e-03,7.788e-04],
         "median": [9.237e-04,8.329e-05,7.233e-06,9.112e-07,7.789e-08],
        "p25": [8.150e-04,7.170e-05,6.382e-06,7.961e-07,6.785e-08],
        "p75": [1.063e-03,9.891e-05,8.211e-06,1.074e-06,9.077e-08],
        
    },
    10: {
        "T_total": [1.484e+06,1.826e+07,1.490e+08,1.814e+09,2.199e+10],
        "l2_mean": [2.026e-03, 1.288e-03,4.069e-04,1.240e-03,9.841e-05],
        "std": [7.585e-03,7.098e-03,2.776e-03,8.232e-03,1.182e-03],
        "median": [7.626e-04,6.832e-05,8.992e-06,7.914e-07,6.764e-08],
        "p25": [6.786e-04,5.948e-05,7.918e-06,6.713e-07,5.994e-08],
        "p75": [9.043e-04,7.951e-05,1.013e-05,9.087e-07,7.812e-08],
    },
    25: {
        "T_total": [1.179e+06,1.451e+07,1.775e+08,1.441e+09,1.747e+10],
        "l2_mean": [3.762e-03,7.491e-04,4.385e-04,2.020e-04,4.266e-04],
        "std": [1.369e-02,4.559e-03,2.345e-03,2.802e-03,5.442e-03],
         "median": [1.014e-03,8.694e-05,7.433e-06,9.672e-07,8.431e-08],
        "p25": [8.930e-04,7.512e-05,6.309e-06,8.249e-07,7.515e-08],
        "p75": [1.161e-03,9.883e-05,8.475e-06,1.096e-06,9.443e-08],
    },
    
}


def y_and_errors(med, p25, p75):
    
    med = np.asarray(med, float)
    p25 = np.asarray(p25, float)
    p75 = np.asarray(p75, float)

    lo = np.minimum(p25, p75)
    hi = np.maximum(p25, p75)

    y = np.log10(1.0 / med)
    
    # reverse way !
    y_upper = np.log10(1.0 / lo) - y      
    y_lower = y - np.log10(1.0 / hi)  
       

    yerr = np.vstack([y_lower, y_upper])
    return y, yerr



plt.figure(figsize=(7, 5))
slopes = {}

for nu, d in data.items():
    T_total = np.asarray(d["T_total"], float)
    med     = np.asarray(d["median"], float)
    p25     = np.asarray(d["p25"], float)
    p75     = np.asarray(d["p75"], float)

   
    log_T = np.log10(T_total)
    y, yerr = y_and_errors(med, p25, p75)
    print(yerr)

    # fit slope on the median-based y
    slope, intercept = np.polyfit(log_T, y, 1)
    slopes[nu] = (slope, intercept)

    # plot series with asymmetric IQR caps as error bars
    line = plt.errorbar(
        log_T, y, yerr=yerr,
        fmt='o-', capsize=3, elinewidth=1.0, alpha=0.9, label=f'ν={nu}'
    )
    color = line[0].get_color()


plt.xlabel('log10(T_total)')
plt.ylabel('log10(1 / l2_error)  (median)')
plt.title('Scaling using median error with 25-75 error bars')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(ncol=2, frameon=False)
plt.tight_layout()
plt.show()




# for nu, d in data.items():
#     T_total = np.array(d["T_total"], dtype=float)
#     l2_mean = np.array(d["l2_mean"], dtype=float)
#     std = np.array(d["std"], dtype=float)

#     log_T = np.log10(T_total)
#     log_inv_error = np.log10(1.0 / l2_mean)
#     yerr = std / (l2_mean * np.sqrt(200)* np.log(10) )
#     # yerr = std/l2_mean

#     # linear fit, don't show it
#     slope, intercept = np.polyfit(log_T, log_inv_error, 1)
#     slopes[nu] = (slope, intercept)

#     plt.errorbar(
#         log_T, # x value
#         log_inv_error, # y value
#         yerr=yerr,
#         fmt='o-',
#         capsize=3,
#         label=f'ν={nu}'
#     )

# plt.xlabel('log10(T_total)')
# plt.ylabel('log10(1 / l2_error)')
# plt.title('Scaling of error with total time (2-qubit, l2_average, varying shots)')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(ncol=2)
# plt.tight_layout()
# plt.show()

# # fit
# for nu in sorted(slopes):
#     s, c = slopes[nu]
#     print(f"nu={nu}: slope={s:.6f}, intercept={c:.6f}")

