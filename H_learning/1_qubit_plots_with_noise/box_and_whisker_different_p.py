import numpy as np
import matplotlib.pyplot as plt

# ---------------- data (as given) ----------------
data = {
#     '5p': {
#         "T_total": [5.759e+04,7.095e+05,5.787e+06,7.047e+07,8.545e+08],
#         "median":  [1.394e-03,1.312e-04,1.674e-05,1.569e-06,1.247e-07],
#         "p25":     [9.672e-04,8.931e-05,1.123e-05,1.069e-06,8.205e-08],
#         "p35":     [1.163e-03,1.066e-04,1.340e-05,1.279e-06,1.017e-07],
#         "p65":     [1.684e-03,1.729e-04,1.964e-05,1.861e-06,1.599e-07],
#         "p75":     [2.030e-03,1.974e-04,2.241e-05,2.156e-06,1.850e-07],
#     },
#   '10p': {
#         "T_total": [5.759e+04,7.095e+05,5.787e+06,7.047e+07,8.545e+08], 
#         "median":  [1.614e-03,1.446e-04,1.918e-05,1.622e-06,1.465e-07], 
#         "p25":     [1.044e-03,9.599e-05,1.368e-05,1.169e-06,9.299e-08],
#         "p35":     [1.233e-03,1.113e-04,1.551e-05,1.353e-06,1.138e-07],
#         "p65":     [2.035e-03,1.725e-04,2.488e-05,2.073e-06,1.741e-07],
#         "p75":     [2.449e-03,1.983e-04,2.988e-05,2.569e-06,2.081e-07],
#     },
  0.15: {
        "T_total": [5.759e+04,7.095e+05,5.787e+06,7.047e+07,8.545e+08],
        "median":  [1.882e-03,1.691e-04,2.209e-05,1.918e-06,1.862e-07],
        "p25":     [1.198e-03,1.079e-04,1.324e-05,1.425e-06,1.182e-07],
        "p35":     [1.485e-03,1.347e-04,1.688e-05,1.584e-06,1.343e-07],
        "p65":     [2.257e-03,2.121e-04,2.801e-05,2.680e-06,2.401e-07],
        "p75":     [2.707e-03,2.660e-04,3.442e-05,3.977e-06,3.189e-07],
    },
  0.2: {
        "T_total": [5.759e+04,7.095e+05,5.787e+06,7.047e+07,8.545e+08],
        "median":  [2.205e-03,1.972e-04,3.650e-05,4.067e-06,3.238e-07],
        "p25":     [1.527e-03,1.341e-04,2.132e-05,1.864e-06,1.666e-07],
        "p35":     [1.776e-03,1.663e-04,2.676e-05,2.463e-06,2.182e-07],
        "p65":     [2.782e-03,3.188e-04,5.469e-05,5.317e-05,9.665e-06],
        "p75":     [3.836e-03,4.314e-04,3.581e-04,9.350e-04,1.038e-04],
    },
  0.25: {
        "T_total": [5.759e+04,7.095e+05,5.787e+06,7.047e+07,8.545e+08],
        "median":  [3.637e-03,4.816e-04,4.167e-04,2.685e-04,1.962e-04],
        "p25":     [1.942e-03,2.201e-04,3.908e-05,4.341e-06,3.861e-07],
        "p35":     [2.344e-03,3.078e-04,5.840e-05,2.486e-05,6.254e-06],
        "p65":     [5.863e-03,4.729e-03,3.075e-03,1.560e-03,1.771e-03],
        "p75":     [1.666e-02,1.298e-02,8.234e-03,7.089e-03,5.580e-03],
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
    "font.size": 12,          # base font
    "axes.labelsize": 13,     # x/y labels
    "xtick.labelsize": 12,    # tick labels
    "ytick.labelsize": 12,
    "legend.fontsize": 9.5,    # legend
    "lines.linewidth": 1.6,   # thicker lines
    "axes.linewidth": 1.0,
    "mathtext.default": "it",
})

# PRL single-column width ~3.37 in
fig = plt.figure(figsize=(3.37, 2.9))
ax = fig.gca()

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
        elinewidth=1.0, alpha=0.95, label=rf'$\eta={nu}$'
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
ax.legend(ncol=2, frameon=False, handlelength=1.0, columnspacing=0.5)

fig.tight_layout()
plt.show()

# To save for submission:
# fig.savefig("prl_single_column_plot.pdf", bbox_inches="tight")
# fig.savefig("prl_single_column_plot.png", bbox_inches="tight")
