import numpy as np
import matplotlib.pyplot as plt


# t_total, mean, std, varying shot 
data = {
  1.8: {
        "T_total": [4.941e+04,6.092e+05,7.454e+06,6.051e+07,7.337e+08],
        "median":  [1.876e-02,5.771e-03,9.750e-04,9.475e-04,2.809e-04],
        "p25":     [2.002e-03,1.664e-04,2.956e-05,5.867e-05,1.113e-04],
        "p35":     [7.138e-03,1.201e-03,4.370e-05,4.978e-04,1.832e-04],
        "p65":     [2.802e-02,8.716e-03,1.695e-03,1.181e-03,9.639e-04],
        "p75":     [3.317e-02,1.043e-02,2.308e-03,1.790e-03,1.693e-03],
    },
  1.9: {
        "T_total": [4.826e+04,5.949e+05,7.279e+06,5.909e+07,7.165e+08],
        "median":  [1.137e-02,1.147e-03,1.174e-04,1.313e-05,1.401e-06],
        "p25":     [5.965e-03,5.829e-04,5.781e-05,5.809e-06,5.887e-07],
        "p35":     [7.680e-03,8.333e-04,8.058e-05,8.683e-06,8.823e-07],
        "p65":     [1.864e-02,1.640e-03,1.854e-04,1.866e-05,2.063e-06],
        "p75":     [2.891e-02,2.036e-03,2.460e-04,3.057e-05,1.595e-05],
    },
  2: {
        "T_total": [4.715e+04,5.813e+05,7.113e+06,8.661e+07,7.001e+08],
        "median":  [5.718e-03,4.187e-04,5.509e-05,5.536e-06,6.309e-07],
        "p25":     [2.946e-03,2.386e-04,2.405e-05,2.342e-06,3.507e-07],
        "p35":     [4.225e-03,3.103e-04,3.184e-05,3.311e-06,4.256e-07],
        "p65":     [7.791e-03,5.660e-04,7.888e-05,8.166e-06,1.069e-06],
        "p75":     [1.068e-02,7.108e-04,1.083e-04,1.149e-05,3.705e-06],
    },
  2.2: {
        "T_total": [6.768e+04,5.558e+05,6.801e+06,8.281e+07,6.694e+08],
        "median":  [2.548e-03,2.759e-04,2.823e-05,2.555e-06,3.488e-07],
        "p25":     [1.526e-03,1.716e-04,1.617e-05,1.457e-06,2.196e-07],
        "p35":     [1.961e-03,2.026e-04,1.937e-05,1.897e-06,2.580e-07],
        "p65":     [3.235e-03,3.997e-04,3.864e-05,3.461e-06,4.800e-07],
        "p75":     [3.664e-03,4.955e-04,4.722e-05,4.474e-06,6.509e-07],
    },
  2.4: {
        "T_total": [6.484e+04,5.325e+05,6.516e+06,7.934e+07,6.413e+08],
        "median":  [1.867e-03,2.515e-04,1.867e-05,1.604e-06,2.076e-07],
        "p25":     [1.163e-03,1.563e-04,1.276e-05,1.018e-06,1.206e-07],
        "p35":     [1.470e-03,1.909e-04,1.540e-05,1.236e-06,1.528e-07],
        "p65":     [2.509e-03,3.106e-04,2.431e-05,2.116e-06,2.934e-07],
        "p75":     [2.851e-03,3.700e-04,2.956e-05,2.480e-06,3.829e-07],
    },
  3: {
        "T_total": [5.759e+04,7.095e+05,5.787e+06,7.047e+07,8.545e+08],
        "median":  [1.461e-03,1.167e-04,1.585e-05,1.444e-06,1.175e-07],
        "p25":     [1.058e-03,8.793e-05,1.092e-05,1.094e-06,8.194e-08],
        "p35":     [1.209e-03,9.865e-05,1.254e-05,1.194e-06,9.616e-08],
        "p65":     [1.756e-03,1.415e-04,2.034e-05,1.757e-06,1.400e-07],
        "p75":     [1.946e-03,1.625e-04,2.400e-05,2.030e-06,1.709e-07],
    },
}

def y_and_errors(med, p35, p65, p25=None, p75=None):
    """Return y plus two asymmetric errorbars:
       - yerr_box:  from p35–p65 (your current bar)
       - yerr_whisk: from p25–p75 (whiskers), if provided
    """
    med = np.asarray(med, float)
    p35 = np.asarray(p35, float)
    p65 = np.asarray(p65, float)

    lo_box = np.minimum(p35, p65)
    hi_box = np.maximum(p35, p65)

    y = np.log10(1.0 / med)
    y_upper_box = np.log10(1.0 / lo_box) - y
    y_lower_box = y - np.log10(1.0 / hi_box)
    yerr_box = np.vstack([y_lower_box, y_upper_box])

    yerr_whisk = None
    if p25 is not None and p75 is not None:
        p25 = np.asarray(p25, float)
        p75 = np.asarray(p75, float)
        lo_w = np.minimum(p25, p75)
        hi_w = np.maximum(p25, p75)
        y_upper_w = np.log10(1.0 / lo_w) - y
        y_lower_w = y - np.log10(1.0 / hi_w)
        yerr_whisk = np.vstack([y_lower_w, y_upper_w])

    return y, yerr_box, yerr_whisk

plt.figure(figsize=(7, 5))
slopes = {}

for nu, d in data.items():
    T_total = np.asarray(d["T_total"], float)
    med     = np.asarray(d["median"], float)
    p35     = np.asarray(d["p35"], float)
    p65     = np.asarray(d["p65"], float)
    p25     = np.asarray(d["p25"], float)
    p75     = np.asarray(d["p75"], float)

    log_T = np.log10(T_total)
    y, yerr_box, yerr_whisk = y_and_errors(med, p35, p65, p25, p75)

    # Fit on median
    slope, intercept = np.polyfit(log_T, y, 1)
    slopes[nu] = (slope, intercept)

    # Main (box) error bar: 35–65%
    line = plt.errorbar(
        log_T, y, yerr=yerr_box,
        fmt='o-', capsize=3, elinewidth=1.4, alpha=0.95, label=f'ν={nu}'
    )
    color = line[0].get_color()

    # Whiskers: 25–75% (no markers/line, just whiskers)
    plt.errorbar(
        log_T, y, yerr=yerr_whisk,
        fmt='none', ecolor=color, elinewidth=0.9, alpha=0.6, capsize=6
    )

plt.xlabel('log10(T_total)')
plt.ylabel('log10(1 / l2_error)  (median)')
plt.title('Median with 35–65% bars and 25–75% whiskers')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(ncol=2, frameon=False)
plt.tight_layout()
plt.show()