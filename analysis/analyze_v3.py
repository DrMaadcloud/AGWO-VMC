#!/usr/bin/env python3
"""Analysis of the v3 campaign (NSGA-II and LAA/GSO/OOA/DRA, results_v3), paired with the AGWO-VMC runs of results_v2.
Writes results_v3.json and Fig. S3 (fronts)."""
import json, os, sys, glob
import numpy as np, pandas as pd
from scipy import stats
from pymoo.indicators.hv import HV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2 = sys.argv[1]; V3 = sys.argv[2]; OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
d2 = pd.read_csv(os.path.join(V2, "runs.csv")); d3 = pd.read_csv(os.path.join(V3, "runs.csv"))
d2 = d2[d2.experiment == "main"].copy(); d3 = d3[d3.experiment == "main3"].copy()
d = pd.concat([d2, d3])
d["slatah_pct"] = 100 * d.slatah; d["pdm_pct"] = 100 * d.pdm
SC = ["static", "dynamic", "flash", "hetero"]
NEW = ["NSGAKNEE_LS", "NSGAAHP_LS", "NSGAKNEE", "NSGAAHP", "LAA_LS", "GSO_LS", "OOA_LS", "DRA_LS", "LAA", "GSO", "OOA", "DRA"]
ML = {"AGWO": "AGWO-VMC", "BFD": "PABFD", "NSGAKNEE_LS": "NSGA-II+LS (knee)", "NSGAAHP_LS": "NSGA-II+LS (AHP-best)", "NSGAKNEE": "NSGA-II (knee)",
      "NSGAAHP": "NSGA-II (AHP-best)", "LAA": "LAA", "GSO": "GSO", "OOA": "OOA", "DRA": "DRA", "LAA_LS": "LAA+LS", "GSO_LS": "GSO+LS",
      "OOA_LS": "OOA+LS", "DRA_LS": "DRA+LS", "IGWO_LS": "I-GWO+LS", "GWO_LS": "GWO+LS"}
R = {"n_runs_v3": int(len(d3)), "n_new_methods": len(NEW)}

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()
def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))
def wil(x, y):
    dd = np.asarray(x) - np.asarray(y)
    if np.allclose(dd, 0): return 1.0, 0.0
    r = stats.wilcoxon(x, y, zero_method="wilcox", method="auto")
    return float(r.pvalue), float(r.statistic)
def dz(x, y):
    dd = np.asarray(x) - np.asarray(y); s = dd.std(ddof=1)
    return float(dd.mean() / s) if s > 0 else 0.0

cols = ["energy_kwh", "active_hosts_mean", "migrations", "failed_migrations", "slatah_pct", "pdm_pct", "slav_pct", "opt_time_s_median",
        "opt_time_s_max", "decision_change_frac", "final_fitness", "iters_per_cycle"]
main = {}
for sc in SC:
    main[sc] = {}
    for me in ["AGWO", "BFD", "IGWO_LS", "GWO_LS"] + NEW:
        s = d[(d.scenario == sc) & (d.method == me)]
        if len(s): main[sc][me] = {"n": int(len(s)), **{c: [float(s[c].mean()), float(s[c].std(ddof=1))] for c in cols}}
R["main"] = main

tests = []
for sc in SC:
    a = d[(d.scenario == sc) & (d.method == "AGWO")].set_index("seed")
    for me in NEW:
        b = d[(d.scenario == sc) & (d.method == me)].set_index("seed")
        j = a.index.intersection(b.index)
        if len(j) < 5: continue
        for met in ["energy_kwh", "migrations", "slav_pct"]:
            x, y = a.loc[j, met].values, b.loc[j, met].values
            p, W = wil(x, y)
            tests.append({"scenario": sc, "method": me, "metric": met, "n": int(len(j)), "agwo": float(x.mean()), "other": float(y.mean()),
                          "rel_pct": float(100 * (x.mean() - y.mean()) / y.mean()) if abs(y.mean()) > 1e-9 else None,
                          "p": p, "W": W, "dz": dz(x, y), "a12": a12(x, y),
                          "wins": int((x < y).sum()), "ties": int((x == y).sum()), "losses": int((x > y).sum())})
for t, pa in zip(tests, holm([t["p"] for t in tests])): t["p_holm"] = pa
R["tests"] = tests; R["n_comparisons"] = len(tests)

# ---- fronts: hypervolume, IGD, convexity of the energy-SLA and energy-migration projections
REF = np.full(4, 1.1); hvf = HV(ref_point=REF)
def nd(P):
    P = np.asarray(P); keep = []
    for i in range(len(P)):
        dom = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if not dom.any(): keep.append(i)
    return P[keep]
def igd(A, Z):
    A = np.asarray(A); return float(np.mean([np.min(np.linalg.norm(A - z, axis=1)) for z in Z]))
def supported_share(P2):
    """Share of the 2-D non-dominated points that lie on the lower-left convex hull (are reachable by a weighted sum)."""
    Q = np.unique(nd(P2), axis=0)
    if len(Q) <= 2: return 1.0, len(Q)
    Q = Q[np.argsort(Q[:, 0])]
    hull = [Q[0]]
    for q in Q[1:]:
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            if (a[0] - o[0]) * (q[1] - o[1]) - (a[1] - o[1]) * (q[0] - o[0]) <= 1e-15: hull.pop()
            else: break
        hull.append(q)
    H = {tuple(h) for h in hull}
    return float(np.mean([tuple(q) in H for q in Q])), len(Q)

fr = []
for f in sorted(glob.glob(os.path.join(V3, "fronts", "*.csv"))):
    name = os.path.basename(f)[:-4]; parts = name.split("__")
    if parts[0] != "main3": continue
    sc, me, seed = parts[1], parts[2], int(parts[5][1:])
    t = pd.read_csv(f)
    for cyc, g in t.groupby("cycle"):
        N = g[g.source == "NSGA"][["e", "sla", "mig", "cv"]].values
        A = g[g.source.str.startswith("AGWO")][["e", "sla", "mig", "cv"]].values
        if len(N) == 0 or len(A) == 0: continue
        U = nd(np.vstack([N, A]))
        nsa = sum(any(np.all(np.isclose(u, a)) for a in A) for u in U)
        s_es, n_es = supported_share(np.vstack([N, A])[:, [0, 1]])
        s_em, n_em = supported_share(np.vstack([N, A])[:, [0, 2]])
        fr.append({"scenario": sc, "method": me, "seed": seed, "cycle": int(cyc), "hv_nsga": float(hvf(N)), "hv_agwo": float(hvf(A)),
                   "hv_union": float(hvf(U)), "igd_nsga": igd(N, U), "igd_agwo": igd(A, U), "n_front": int(len(N)), "n_union": int(len(U)),
                   "share_union_agwo": float(nsa / len(U)), "conv_es": s_es, "n_es": n_es, "conv_em": s_em, "n_em": n_em,
                   "agwo_dominated": int(sum(any(np.all(n <= a) and np.any(n < a) for n in N) for a in A)),
                   "emin_nsga": float(N[:, 0].min()), "emin_agwo": float(A[:, 0].min()),
                   "agwo_opt_s": float(g[g.source.str.startswith("AGWO")].opt_s.mean())})
F = pd.DataFrame(fr)
F.to_csv(os.path.join(OUT, "fronts_summary.csv"), index=False)
fronts = {}
for me in ["NSGAKNEE_LS", "NSGAKNEE"]:
    fronts[me] = {}
    for sc in SC + ["all"]:
        s = F[(F.method == me) & ((F.scenario == sc) if sc != "all" else True)]
        if not len(s): continue
        e = {k: [float(s[k].mean()), float(s[k].std(ddof=1))] for k in ["hv_nsga", "hv_agwo", "hv_union", "igd_nsga", "igd_agwo", "n_front", "n_union",
                                                                         "share_union_agwo", "conv_es", "n_es", "conv_em", "n_em", "agwo_dominated", "emin_nsga", "emin_agwo", "agwo_opt_s"]}
        # paired over snapshots (seed x cycle)
        pv, W = wil(s.hv_agwo.values, s.hv_nsga.values); pi, Wi = wil(s.igd_agwo.values, s.igd_nsga.values)
        e.update({"n_snap": int(len(s)), "p_hv": pv, "p_igd": pi, "hv_agwo_higher": int((s.hv_agwo > s.hv_nsga).sum()),
                  "igd_agwo_lower": int((s.igd_agwo < s.igd_nsga).sum()), "frac_es_convex": float((s.conv_es >= 0.999).mean()),
                  "frac_em_convex": float((s.conv_em >= 0.999).mean())})
        fronts[me][sc] = e
R["fronts"] = fronts
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o): return None
    return o
json.dump(clean(R), open(os.path.join(OUT, "results_v3.json"), "w"), indent=1)

# ---- Fig. S3: one snapshot per scenario, energy vs migration projection
C1, C2, INK2, GRID = "#2a78d6", "#eb6834", "#5f5e58", "#e6e5df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300})
SCL = {"static": "Static", "dynamic": "Dynamic Steady", "flash": "Flash Crowd", "hetero": "Heterogeneous Mix"}
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
for ax, sc in zip(axs, SC):
    f = os.path.join(V3, "fronts", f"main3__{sc}__NSGAKNEE_LS__default__100__s01.csv")
    if os.path.exists(f):
        t = pd.read_csv(f); g = t[t.cycle == 72] if (t.cycle == 72).any() else t[t.cycle == t.cycle.min()]
        N = g[g.source == "NSGA"]; A = g[g.source.str.startswith("AGWO")]
        ax.scatter(N.mig, N.e, s=7, color=C1, label="NSGA-II+LS front", alpha=.8)
        ax.scatter(A.mig, A.e, s=16, marker="D", color=C2, label="AGWO-VMC (7 weightings)")
    ax.set_title(SCL[sc], fontsize=8); ax.set_xlabel("Migration term")
axs[0].set_ylabel("Energy term"); axs[0].legend(fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figS3_fronts.png")); plt.close(fig)
print("ok", len(d3), "runs;", len(F), "snapshots")
