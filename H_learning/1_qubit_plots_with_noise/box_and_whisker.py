import numpy as np
import matplotlib.pyplot as plt

# ---------------- data (as given) ----------------
data = {
  1.8: {
        "T_total": [4.941e+04,6.092e+05,7.454e+06,6.051e+07,7.337e+08], 
        "median":  [2.484e-02,4.715e-03,1.301e-03,1.013e-03,3.220e-04], 
        "p25":     [2.664e-03,1.500e-04,3.990e-05,3.216e-04,6.595e-05], 
        "p35":     [1.396e-02,1.030e-03,8.536e-05,6.497e-04,1.957e-04], 
        "p65":     [3.492e-02,7.689e-03,2.439e-03,1.441e-03,1.040e-03],
        "p75":     [5.450e-02,9.271e-03,3.336e-03,2.255e-03,1.956e-03],
    },
  1.9: {
        "T_total": [4.826e+04,5.949e+05,7.279e+06,5.909e+07,7.165e+08],
        "median":  [1.195e-02,1.252e-03,1.374e-04,1.512e-05,1.373e-06],
        "p25":     [6.088e-03,5.551e-04,6.340e-05,7.292e-06,5.934e-07],
        "p35":     [7.782e-03,7.953e-04,9.054e-05,1.023e-05,8.349e-07],
        "p65":     [2.199e-02,1.944e-03,2.414e-04,2.607e-05,2.512e-06],
        "p75":     [2.894e-02,4.480e-03,8.024e-03,2.220e-04,5.697e-05],
    },
  2: {
        "T_total": [4.715e+04,5.813e+05,7.113e+06,8.661e+07,7.001e+08],
        "median":  [5.856e-03,4.731e-04,5.369e-05,6.229e-06,8.568e-07],
        "p25":     [3.207e-03,2.620e-04,2.516e-05,2.621e-06,3.637e-07],
        "p35":     [4.135e-03,3.179e-04,3.426e-05,3.582e-06,5.419e-07],
        "p65":     [7.818e-03,7.247e-04,9.384e-05,8.951e-06,1.690e-06],
        "p75":     [1.126e-02,9.689e-04,1.166e-04,1.432e-05,9.637e-06],
    },
  2.4: {
        "T_total": [6.484e+04,5.325e+05,6.516e+06,7.934e+07,6.413e+08],
        "median":  [1.969e-03,2.533e-04,2.296e-05,1.717e-06,2.383e-07],
        "p25":     [1.203e-03,1.689e-04,1.464e-05,1.063e-06,1.473e-07],
        "p35":     [1.539e-03,2.059e-04,1.834e-05,1.286e-06,1.824e-07],
        "p65":     [2.544e-03,3.497e-04,2.989e-05,2.264e-06,3.301e-07],
        "p75":     [3.154e-03,4.239e-04,3.616e-05,2.726e-06,4.012e-07],
    },
  3: {
        "T_total": [5.759e+04,7.095e+05,5.787e+06,7.047e+07,8.545e+08],
        "median":  [1.394e-03,1.312e-04,1.674e-05,1.569e-06,1.247e-07],
        "p25":     [9.672e-04,8.931e-05,1.123e-05,1.069e-06,8.205e-08],
        "p35":     [1.163e-03,1.066e-04,1.340e-05,1.279e-06,1.017e-07],
        "p65":     [1.684e-03,1.729e-04,1.964e-05,1.861e-06,1.599e-07],
        "p75":     [2.030e-03,1.974e-04,2.241e-05,2.156e-06,1.850e-07],
    },
}

# ---------------- helpers ----------------
def y_and_errors(med, p35, p65, p25=None, p75=None):
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

# ---------------- PRL-ish style ----------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "mathtext.default": "it",
})

# PRL single-column width ~3.37 in
fig = plt.figure(figsize=(3.37, 2.60))
ax = plt.gca()

# ---------------- plot ----------------
for nu, d in sorted(data.items()):
    T_total = np.asarray(d["T_total"], float)
    med     = np.asarray(d["median"], float)
    p35     = np.asarray(d["p35"], float)
    p65     = np.asarray(d["p65"], float)
    p25     = np.asarray(d["p25"], float)
    p75     = np.asarray(d["p75"], float)

    x = np.log10(T_total)
    y, yerr_box, yerr_whisk = y_and_errors(med, p35, p65, p25, p75)

    line = ax.errorbar(
        x, y, yerr=yerr_box, fmt='o-', markersize=3, capsize=2.5,
        elinewidth=1.0, alpha=0.95, label=rf'$\nu={nu}$'
    )
    color = line[0].get_color()
    ax.errorbar(
        x, y, yerr=yerr_whisk, fmt='none',
        ecolor=color, elinewidth=0.8, alpha=0.6, capsize=4
    )

# dotted slope-1 guide (any intercept)
x1, x2 = ax.get_xlim()
b = -1.5
ax.plot([x1, x2], [x1 + b, x2 + b], ':', color='k', linewidth=1.0, label='slope $1$')

# math axis labels
ax.set_xlabel(r'$\log_{10} T_{\mathrm{total}}$')
ax.set_ylabel(r'$\log_{10}(1/\epsilon_{\ell_2})$')

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(ncol=2, frameon=False, handlelength=2.0, columnspacing=0.8)

fig.tight_layout()
plt.show()

# To save for submission:
# fig.savefig("prl_single_column_plot.pdf", bbox_inches="tight")
# fig.savefig("prl_single_column_plot.png", bbox_inches="tight")
