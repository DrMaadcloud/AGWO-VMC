import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
INK, INK2 = "#1f1f1e", "#5f5e58"
fig, ax = plt.subplots(figsize=(7.2, 6.0)); ax.set_xlim(0, 100); ax.set_ylim(-2, 84); ax.axis("off")

def box(x, y, w, h, fc, ec, lines, sizes, lw=1.0, bold_first=True, gap=3.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    n = len(lines); cy = y + h / 2; top = cy + gap * (n - 1) / 2
    for i, (t, s) in enumerate(zip(lines, sizes)):
        ax.text(x + w / 2, top - i * gap, t, ha="center", va="center", fontsize=s, color=INK, fontweight="bold" if (i == 0 and bold_first) else "normal")
def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=INK2))
    if label: ax.text(lx if lx is not None else x1 + 1.5, ly if ly is not None else (y1 + y2) / 2, label, fontsize=7.6, color=INK2, va="center")


box(3, 70, 90, 10, "#f1efe8", "#8a897f", ["Data center (CloudSim Plus 8.5.7)", "100 heterogeneous hosts (50 HP ProLiant ML110 G4, 50 G5) · 300 VMs", "network topology not modeled"], [10, 7.8, 7.8], gap=2.9)
arrow(48, 70, 48, 64.5, "VM demand and current placement", lx=49.5)
box(3, 55, 90, 9.5, "#e3f3ec", "#5aa985", ["Monitoring", "demand snapshot of this cycle and incumbent placement", "(both fixed while the optimizer runs)"], [10, 7.8, 7.8], gap=2.8)
arrow(48, 55, 48, 51.3)

ax.add_patch(FancyBboxPatch((3, 17.5), 90, 33.8, boxstyle="round,pad=0,rounding_size=1.2", fc="#e8f1fb", ec="#5b8fd0", lw=1.0))
ax.text(48, 48.6, "AGWO-VMC optimizer", ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
ax.text(48, 45.7, "N = 50 wolves, warm-started from the incumbent · 12,000 fitness evaluations per cycle", ha="center", va="center", fontsize=7.6, color=INK)
bw, bh, by = 27.6, 14.5, 25.6
box(5.5, by, bw, bh, "#f0effb", "#6b63c4", ["Improvement feedback", r"$a(t)=2(1-t/t_{max})^{\lambda}(1-\gamma\bar{g}_t)$", "Eqs. (11)–(12)"], [8.2, 7.2, 7.2], gap=3.6)
box(34.2, by, bw, bh, "#f0effb", "#6b63c4", ["Probabilistic decoding", r"softmax over candidate set $S_j$", r"temperature $\tau$ · Eq. (13)", "capacity repair"], [8.2, 7.2, 7.2, 7.2], gap=3.0)
box(62.9, by, bw, bh, "#f0effb", "#6b63c4", [r"Local search on $\varphi_{best}$", "pairwise swaps", "+ host evacuation", r"every $T_{ls}=5$ iterations"], [8.2, 7.2, 7.2, 7.2], gap=3.0)
ax.add_patch(FancyBboxPatch((5.5, 19.3), 85, 4.6, boxstyle="round,pad=0,rounding_size=0.8", fc="#ffffff", ec="#9db8de", lw=0.8))
ax.text(48, 21.6, r"Fitness $F(\varphi)$, Eq. (8): energy, SLA overload risk, migration cost, load imbalance", ha="center", va="center", fontsize=7.6, color=INK)

arrow(26, 17.5, 26, 12.6, r"best placement $\varphi^{*}$", lx=27.5, ly=15.1)
box(3, 3, 45, 9.6, "#fbece6", "#d27b57", ["Migration controller", r"live migrations from the", r"incumbent to $\varphi^{*}$"], [9.5, 7.6, 7.6], gap=2.7)
arrow(48, 7.8, 54, 7.8)
box(54, 3, 39, 9.6, "#fbf2e1", "#c99a3c", ["Measurement", "energy (Eqs. (1)–(2)), SLATAH,", "PDM, migrations"], [9.5, 7.6, 7.6], gap=2.7)
# next-cycle loop (dashed)
ax.plot([25.5, 25.5, 97, 97], [3, 0.2, 0.2, 75], color=INK2, lw=0.9, ls=(0, (3, 2)))
ax.add_patch(FancyArrowPatch((97, 75), (93.3, 75), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=INK2))
ax.text(61, -1.4, "the placement held by the simulator is the incumbent of the next cycle", fontsize=7.4, color=INK2, ha="center", va="center")
fig.savefig(__import__("sys").argv[1] + "/fig1.png", dpi=600, bbox_inches="tight", facecolor="white")
