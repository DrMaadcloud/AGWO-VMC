import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5df", "axes.axisbelow": True, "legend.frameon": False,
                     "axes.edgecolor": "#5f5e58", "xtick.color": "#5f5e58", "ytick.color": "#5f5e58"})
T = 200; t = np.linspace(0, T, 801); x = t / T; lam, gam = 1.9, 0.2
lin = 2 * (1 - x); base = 2 * (1 - x) ** lam; impr = base * (1 - gam)
fig, ax = plt.subplots(figsize=(8.4, 3.6))
ax.fill_between(t, impr, base, color="#7fc4b4", alpha=0.3, lw=0, label="Range reachable by the feedback")
ax.plot(t, lin, "--", color="#1f3864", lw=1.6, dashes=(4,2.5), label="Standard GWO, linear decay (Eq. 10)")
ax.plot(t, base, color="#7b2d26", lw=1.6, label=r"AGWO-VMC base schedule, $\bar{g}=0$ (stalled)")
ax.plot(t, impr, color="#0b3d2e", lw=1.8, label=r"AGWO-VMC under sustained improvement, $\bar{g}=1$")
ax.axhline(1, color="#9a9a94", lw=0.8, ls=":"); ax.text(3, 1.03, "a(t) = 1", fontsize=7.5, color="#5f5e58")
for frac, lab, c in [(1 - 0.625 ** (1 / lam), "0.22", "#0b3d2e"), (1 - 0.5 ** (1 / lam), "0.31", "#7b2d26"), (0.5, "0.50", "#1f3864")]:
    ax.axvline(frac * T, color=c, lw=0.8, ls=":")
    ax.text(frac * T + (3.5 if lab == "0.31" else 1.5), 1.9, r"%s $t_{max}$" % lab, fontsize=8, color=c)
ax.set_xlim(0, T); ax.set_ylim(0, 2.05); ax.set_xlabel("Iteration t"); ax.set_ylabel("Control coefficient a(t)")
ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.86))
fig.savefig(__import__("sys").argv[1] + "/fig2.png", dpi=600, bbox_inches="tight", facecolor="white")
