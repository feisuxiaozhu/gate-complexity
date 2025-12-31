import numpy as np
import matplotlib.pyplot as plt

# ---------------- constants for normdiff ----------------
x0 = np.array([0.11, 0.21, 0.32, 0.51, 0.63, 0.31, 0.22, 0.11, 0.11, 0.22, 0.11, 0.11, 0.33, 0.22, 0.15])
lambda_true = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.3, 0.22, 0.15])
delta = lambda_true - x0

def normdiff_from_scaling(scaling):
    return np.linalg.norm(delta * float(scaling))

# ---------------- data (as given) ----------------
data = {
  5: {
        "T_total":  [1.411e+08, 1.411e+08, 1.411e+08, 1.411e+08, 1.411e+08, 1.411e+08, 1.411e+08, 1.411e+08],
        "scaling":  [1, 5, 10, 15, 20, 30, 40, 50],
        "median":   [1.011e-05, 1.054e-05, 1.054e-05, 1.036e-05, 1.025e-05, 1.036e-05, 1.014e-05, 1.019e-05],
        "p25":      [8.891e-06, 8.919e-06, 8.934e-06, 8.760e-06, 8.672e-06, 8.972e-06, 8.835e-06, 8.769e-06],
        "p35":      [9.550e-06, 9.384e-06, 9.559e-06, 9.316e-06, 9.344e-06, 9.640e-06, 9.554e-06, 9.291e-06],
        "p65":      [1.110e-05, 1.179e-05, 1.149e-05, 1.115e-05, 1.126e-05, 1.162e-05, 1.103e-05, 1.110e-05],
        "p75":      [1.196e-05, 1.305e-05, 1.277e-05, 1.217e-05, 1.232e-05, 1.265e-05, 1.167e-05, 1.179e-05],
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

# ---------------- single-column supplementary style ----------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,

    "font.size": 12,
    "axes.titlesize": 16,

    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,

    "lines.linewidth": 2.0,
    "axes.linewidth": 1.2,
    "mathtext.default": "it",
})

fig = plt.figure(figsize=(6.5, 3.2))
ax = plt.gca()

# ---------------- plot ----------------
all_x = []
for nu, d in sorted(data.items()):
    scaling = np.asarray(d["scaling"], float)
    x = np.array([normdiff_from_scaling(s) for s in scaling], dtype=float)
    all_x.extend(list(x))

    med = np.asarray(d["median"], float)
    p35 = np.asarray(d["p35"], float)
    p65 = np.asarray(d["p65"], float)
    p25 = np.asarray(d["p25"], float)
    p75 = np.asarray(d["p75"], float)

    y, yerr_box, yerr_whisk = y_and_errors(med, p35, p65, p25, p75)

    line = ax.errorbar(
        x, y, yerr=yerr_box, fmt="o-", markersize=6, capsize=4,
        elinewidth=1.6, alpha=0.95, label=rf"$\nu={nu}$"
    )
    color = line[0].get_color()
    ax.errorbar(
        x, y, yerr=yerr_whisk, fmt="none",
        ecolor=color, elinewidth=1.2, alpha=0.6, capsize=6
    )

# Keep numeric normdiff tick labels (do not force every point as a tick)
# Matplotlib will choose reasonable ticks automatically.

# Smaller x-axis label (your request)
ax.set_xlabel(r'$\|\Delta\|_2 $', fontsize=14, labelpad=6)

# Keep y label larger
ax.set_ylabel(r'$\log_{10}(1/\epsilon_{\ell_2})$', fontsize=18, labelpad=8)

ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
ax.legend(ncol=1, frameon=False, handlelength=2.2)

fig.tight_layout()
plt.show()

# To save:
# fig.savefig("supp_single_column_plot.pdf", bbox_inches="tight")
# fig.savefig("supp_single_column_plot.png", bbox_inches="tight")
