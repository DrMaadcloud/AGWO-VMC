#!/usr/bin/env python3
"""Analysis of the 30-seed AGWO-VMC campaign (results_v2): writes results_v2.json and the figures."""
import json, os, sys, glob, re
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

RES = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis_v2"
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(os.path.join(RES, "runs.csv"))
d["slatah_pct"] = 100 * d.slatah
d["pdm_pct"] = 100 * d.pdm
SC = ["static", "dynamic", "flash", "hetero"]
SCL = {"static": "Static", "dynamic": "Dynamic Steady", "flash": "Flash Crowd", "hetero": "Heterogeneous Mix"}
METH = ["AGWO", "BFD", "GWO_LS", "GA_LS", "PSO_LS", "HHO_LS", "IGWO_LS", "GWO", "GA", "PSO", "HHO", "IGWO", "MGWO", "CGWO"]
ML = {"AGWO": "AGWO-VMC", "BFD": "PABFD", "GWO_LS": "GWO+LS", "GA_LS": "GA+LS", "PSO_LS": "PSO+LS", "HHO_LS": "HHO+LS",
      "IGWO_LS": "I-GWO+LS", "GWO": "GWO", "GA": "GA", "PSO": "PSO", "HHO": "HHO", "IGWO": "I-GWO", "MGWO": "mGWO", "CGWO": "Chaotic GWO"}
R = {"n_runs": int(len(d)), "experiments": {k: int(v) for k, v in d.experiment.value_counts().items()}}

C1, C2, C3, C4, CG = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9a9a94"
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": int(__import__("os").environ.get("FIGDPI", 300))})

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()

def a12(x, y):  # P(x < y) + 0.5 P(x = y): probability that AGWO-VMC (x) is lower
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))

def wil(x, y):
    dd = np.asarray(x) - np.asarray(y)
    if np.allclose(dd, 0): return 1.0, 0.0
    r = stats.wilcoxon(x, y, zero_method="wilcox", method="auto")
    return float(r.pvalue), float(r.statistic)

def dz(x, y):
    dd = np.asarray(x) - np.asarray(y); s = dd.std(ddof=1)
    return float(dd.mean() / s) if s > 0 else (0.0 if dd.mean() == 0 else float(np.sign(dd.mean()) * np.inf))

M = d[d.experiment == "main"].copy()
cols = ["energy_kwh", "active_hosts_mean", "migrations", "failed_migrations", "mig_energy_kwh", "slatah_pct", "pdm_pct", "slav_pct",
        "opt_time_s_median", "opt_time_s_max", "decision_change_frac", "final_fitness", "conv_iter95", "iters_per_cycle"]
main = {}
for sc in SC:
    main[sc] = {}
    for me in METH:
        s = M[(M.scenario == sc) & (M.method == me)]
        if len(s): main[sc][me] = {"n": int(len(s)), **{c: [float(s[c].mean()), float(s[c].std(ddof=1))] for c in cols},
                                   "opt_time_q": [float(s.opt_time_s_median.quantile(q)) for q in (0.25, 0.5, 0.75)]}
R["main"] = main

# ---- paired tests: AGWO-VMC vs every baseline, per scenario (30 seeds) and per metric; Holm over the whole family
tests = []
for sc in SC:
    a = M[(M.scenario == sc) & (M.method == "AGWO")].set_index("seed")
    for me in METH[1:]:
        b = M[(M.scenario == sc) & (M.method == me)].set_index("seed")
        j = a.index.intersection(b.index)
        for met in ["energy_kwh", "migrations", "slav_pct"]:
            x, y = a.loc[j, met].values, b.loc[j, met].values
            p, W = wil(x, y)
            tests.append({"scenario": sc, "method": me, "metric": met, "n": int(len(j)), "agwo": float(x.mean()), "other": float(y.mean()),
                          "rel_pct": float(100 * (x.mean() - y.mean()) / y.mean()) if abs(y.mean()) > 1e-9 else None,
                          "p": p, "W": W, "dz": dz(x, y), "a12": a12(x, y),
                          "wins": int((x < y).sum()), "ties": int((x == y).sum()), "losses": int((x > y).sum())})
for t, pa in zip(tests, holm([t["p"] for t in tests])): t["p_holm"] = pa
R["tests"] = tests
R["n_comparisons"] = len(tests)

# ---- Friedman / Kendall W / Nemenyi CD on energy (blocks = scenario x seed)
piv = M.pivot_table(index=["scenario", "seed"], columns="method", values="energy_kwh")[METH].dropna()
fr = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
N, k = piv.shape
ranks = piv.rank(axis=1).mean()
qa = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
R["friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue), "df": int(k - 1), "k": int(k), "N": int(N),
                 "W": float(fr.statistic / (N * (k - 1))), "mean_ranks": {m: float(v) for m, v in ranks.items()},
                 "cd": float(qa * np.sqrt(k * (k + 1) / (6 * N)))}

# ---- ablation (D1P1L1 = AGWO, D0P1L1 = GWO_LS from the main comparison)
A = d[d.experiment == "ablation"][["scenario", "seed", "config", "energy_kwh", "migrations", "slav_pct", "active_hosts_mean", "final_fitness"]].copy()
extra = M[M.method.isin(["AGWO", "GWO_LS"])][["scenario", "seed", "method", "energy_kwh", "migrations", "slav_pct", "active_hosts_mean", "final_fitness"]].copy()
extra["config"] = extra.method.map({"AGWO": "D1P1L1", "GWO_LS": "D0P1L1"}); extra = extra.drop(columns="method")
A = pd.concat([A, extra], ignore_index=True)
abl = {}
for sc in SC + ["all"]:
    s = A if sc == "all" else A[A.scenario == sc]
    full = s[s.config == "D1P1L1"].set_index(["scenario", "seed"])
    abl[sc] = {}
    for cfg, t in s.groupby("config"):
        t2 = t.set_index(["scenario", "seed"]); j = t2.index.intersection(full.index)
        row = {k2: [float(t[k2].mean()), float(t[k2].std(ddof=1))] for k2 in ["energy_kwh", "migrations", "slav_pct", "active_hosts_mean", "final_fitness"]}
        row["n"] = int(len(t))
        if cfg != "D1P1L1":
            row["d_energy_pct"] = float(100 * (t2.loc[j].energy_kwh.mean() - full.loc[j].energy_kwh.mean()) / full.loc[j].energy_kwh.mean())
            row["p"] = wil(t2.loc[j].energy_kwh.values, full.loc[j].energy_kwh.values)[0]
        abl[sc][cfg] = row
    ks = [c for c in abl[sc] if "p" in abl[sc][c]]
    for c, pa in zip(ks, holm([abl[sc][c]["p"] for c in ks])): abl[sc][c]["p_holm"] = pa
R["ablation"] = abl
# Shapley attribution over D, P, L of the energy reduction from plain GWO (D0P0L0) to AGWO-VMC (D1P1L1)
import itertools, math
shap = {}
for sc in SC + ["all"]:
    E = {c: abl[sc][c]["energy_kwh"][0] for c in abl[sc]}
    v = lambda S: E["D0P0L0"] - E["D%dP%dL%d" % ("D" in S, "P" in S, "L" in S)]
    pl = ["D", "P", "L"]; phi = {}
    for p in pl:
        others = [q for q in pl if q != p]; tot = 0.0
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                wgt = math.factorial(len(S)) * math.factorial(len(pl) - len(S) - 1) / math.factorial(len(pl))
                tot += wgt * (v(set(S) | {p}) - v(set(S)))
        phi[p] = tot
    total = v({"D", "P", "L"})
    shap[sc] = {"phi_kwh": phi, "total_kwh": total, "share_pct": {p: 100 * phi[p] / total if total else 0 for p in pl}}
R["shapley"] = shap

# ---- sensitivity
S = d[d.experiment == "sens"].copy()
S["param"] = S.config.str.split("-").str[0]; S["value"] = S.config.str.split("-", n=1).str[1]
sens = {}
for p, s in S.groupby("param"):
    rows = []
    for v_, t in s.groupby("value"):
        rows.append({"value": v_, "n": int(len(t)), "energy_kwh": float(t.energy_kwh.mean()), "energy_sd": float(t.energy_kwh.std(ddof=1)),
                     "slav_pct": float(t.slav_pct.mean()), "migrations": float(t.migrations.mean()), "active": float(t.active_hosts_mean.mean()),
                     "fitness": float(t.final_fitness.mean()), "opt_time": float(t.opt_time_s_median.mean())})
    def key(e):
        try: return (0, float(e["value"]))
        except ValueError: return (1, 0.0)
    sens[p] = sorted(rows, key=key)
R["sens"] = sens
LG = d[d.experiment == "lamgam"].copy()
LG["lam"] = LG.config.str.extract(r"lambda-([\d.]+)")[0].astype(float); LG["gam"] = LG.config.str.extract(r"gamma-([\d.]+)")[0].astype(float)
hm = LG.groupby(["lam", "gam"]).agg(E=("energy_kwh", "mean"), F=("final_fitness", "mean")).reset_index()
R["lamgam"] = {"rows": hm.to_dict("records"), "spread_pct": float(100 * (hm.E.max() - hm.E.min()) / hm.E.min()),
               "best": hm.loc[hm.E.idxmin()].to_dict(), "worst": hm.loc[hm.E.idxmax()].to_dict()}
# Friedman across the 63 grid cells (blocks = seeds)
pv = LG.pivot_table(index="seed", columns="config", values="energy_kwh").dropna(axis=1)
fr2 = stats.friedmanchisquare(*[pv[c].values for c in pv.columns])
R["lamgam"]["friedman_p"] = float(fr2.pvalue)

# ---- weights
W = d[d.experiment == "weights_named"].copy()
wn = {}
for (cfg, me), s in W.groupby(["config", "method"]):
    wn.setdefault(cfg.replace("w-", ""), {})[me] = {"n": int(len(s)), "energy_kwh": float(s.energy_kwh.mean()), "migrations": float(s.migrations.mean()),
                                                    "slav_pct": float(s.slav_pct.mean()), "slatah_pct": float(s.slatah_pct.mean()), "active": float(s.active_hosts_mean.mean())}
for me in ["AGWO", "GWO_LS"]:
    s = M[(M.scenario == "dynamic") & (M.method == me)]
    wn.setdefault("default", {})[me] = {"n": int(len(s)), "energy_kwh": float(s.energy_kwh.mean()), "migrations": float(s.migrations.mean()),
                                        "slav_pct": float(s.slav_pct.mean()), "slatah_pct": float(s.slatah_pct.mean()), "active": float(s.active_hosts_mean.mean())}
for prof in wn:
    a = W[(W.config == "w-" + prof) & (W.method == "AGWO")].set_index("seed") if prof != "default" else M[(M.scenario == "dynamic") & (M.method == "AGWO")].set_index("seed")
    b = W[(W.config == "w-" + prof) & (W.method == "GWO_LS")].set_index("seed") if prof != "default" else M[(M.scenario == "dynamic") & (M.method == "GWO_LS")].set_index("seed")
    j = a.index.intersection(b.index)
    wn[prof]["p_energy"] = wil(a.loc[j].energy_kwh.values, b.loc[j].energy_kwh.values)[0]
R["weights_named"] = wn
RW = d[d.experiment == "weights_random"].copy()
RW["vec"] = RW.config.str.slice(0, 4)
wv = RW.pivot_table(index="vec", columns="method", values=["energy_kwh", "final_fitness", "active_hosts_mean"])
fA, fG = wv["final_fitness"]["AGWO"], wv["final_fitness"]["GWO_LS"]
eA, eG = wv["energy_kwh"]["AGWO"], wv["energy_kwh"]["GWO_LS"]
R["weights_random"] = {"n_vectors": int(len(wv)), "agwo_lower_fitness": int((fA < fG).sum()), "agwo_lower_energy": int((eA < eG).sum()),
                       "p_fitness": wil(fA.values, fG.values)[0], "p_energy": wil(eA.values, eG.values)[0],
                       "median_rel_energy_pct": float(np.median(100 * (eA - eG) / eG)),
                       "consolidating_vectors": int((wv["active_hosts_mean"]["AGWO"] < 70).sum())}
w_cfg = RW.drop_duplicates("vec").set_index("vec").config.str.split("-", expand=True).iloc[:, 1:5].astype(float)
w_cfg.columns = ["wE", "wS", "wM", "wB"]
cons = (wv["active_hosts_mean"]["AGWO"] < 70)
R["weights_random"]["cons_mean_weights"] = w_cfg[cons].mean().to_dict()
R["weights_random"]["noncons_mean_weights"] = w_cfg[~cons].mean().to_dict() if (~cons).any() else {}

# ---- scalability
SCA = d[d.experiment == "scal"]
scal = {}
for me, s in SCA.groupby("method"):
    g = s.groupby("fleet_pms").agg(t=("opt_time_s_median", "median"), tq1=("opt_time_s_median", lambda x: x.quantile(.25)),
                                    tq3=("opt_time_s_median", lambda x: x.quantile(.75)), tmax=("opt_time_s_max", "max"),
                                    e=("energy_kwh", "mean"), act=("active_hosts_mean", "mean"), n=("run_id", "count"))
    sl = stats.linregress(np.log(g.index.values.astype(float)), np.log(g.t.values))
    scal[me] = {"rows": [{"hosts": int(i), **{k2: float(v2) for k2, v2 in r.items()}} for i, r in g.iterrows()], "slope": float(sl.slope), "r2": float(sl.rvalue ** 2)}
R["scal"] = scal

# ---- convergence and diversity from iteration logs
logs = []
for f in glob.glob(os.path.join(RES, "iters", "main__*.csv")):
    pr = os.path.basename(f)[:-4].split("__")
    t = pd.read_csv(f); t["scenario"] = pr[1]; t["method"] = pr[2]; t["seed"] = int(pr[5][1:]); logs.append(t)
L = pd.concat(logs, ignore_index=True)
L["run"] = L.scenario + L.method + L.seed.astype(str) + "_" + L.cycle.astype(str)
L["fnorm"] = L.groupby("run").best_fitness.transform(lambda x: x / x.iloc[0])
L["xpl"] = 100 * L.groupby("run").div_cont.transform(lambda x: x / x.max() if x.max() > 0 else x * 0)
conv = {}
for (me, sc), s in L.groupby(["method", "scenario"]):
    g = s.groupby("iter").fnorm
    conv.setdefault(me, {})[sc] = {"iter": g.median().index.tolist(), "med": g.median().tolist(), "q1": g.quantile(.25).tolist(), "q3": g.quantile(.75).tolist()}
R["convergence_final_med"] = {me: {sc: v["med"][-1] for sc, v in d2.items()} for me, d2 in conv.items()}
div = {}
for me, s in L.groupby("method"):
    gc = s.groupby("iter")
    div[me] = {"iter": gc.div_cont.median().index.tolist(), "cont": gc.div_cont.median().tolist(), "place": gc.div_place.median().tolist(),
               "cont_q1": gc.div_cont.quantile(.25).tolist(), "cont_q3": gc.div_cont.quantile(.75).tolist(),
               "place_q1": gc.div_place.quantile(.25).tolist(), "place_q3": gc.div_place.quantile(.75).tolist(),
               "xpl": gc.xpl.median().tolist(), "xpl_mean": float(s.xpl.mean()),
               "place_start": float(s[s.iter == 1].div_place.median()), "place_end": float(s[s.iter == s.iter.max()].div_place.median())}
    if me == "AGWO":
        st = s[s.gbar == 0]; mv = s[s.gbar > 0]
        div[me]["xpl_stalled"] = float(st.xpl.mean()) if len(st) else None
        div[me]["xpl_moving"] = float(mv.xpl.mean()) if len(mv) else None
        div[me]["stalled_frac"] = float((s.gbar == 0).mean())
R["diversity"] = {me: {k2: v2 for k2, v2 in x.items() if k2 not in ("iter", "cont", "place", "xpl", "cont_q1", "cont_q3", "place_q1", "place_q3")} for me, x in div.items()}

# ================================================================ figures
figs = {}
def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); figs[name] = p
col = lambda me: C1 if me == "AGWO" else C2 if me == "BFD" else (C3 if me.endswith("_LS") else CG)

# Fig 4: energy per scenario
fig, axs = plt.subplots(1, 4, figsize=(7.4, 3.6), sharey=True)
for ax, sc in zip(axs, SC):
    ms = METH[::-1]
    v = [main[sc][m]["energy_kwh"][0] for m in ms]; e = [main[sc][m]["energy_kwh"][1] for m in ms]
    ax.barh(range(len(ms)), v, xerr=e, color=[col(m) for m in ms], height=0.64, error_kw={"elinewidth": 0.7, "ecolor": INK2})
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([ML[m] for m in ms]); ax.tick_params(axis="y", length=0)
    ax.set_title(SCL[sc], fontsize=9); ax.grid(axis="y", visible=False); ax.set_xlabel("Energy (kWh, 24 h)"); ax.set_xlim(0, 280)
fig.tight_layout(w_pad=0.5); save(fig, "fig4_energy.png")

# Fig 5: energy vs migrations
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=False)
for ax, sc in zip(axs, SC):
    for me, c, mk, lab in [("BFD", C2, "^", "PABFD"), ("GWO_LS", C3, "s", "GWO+LS"), ("HHO_LS", C4, "v", "HHO+LS"), ("AGWO", C1, "D", "AGWO-VMC")]:
        s = M[(M.scenario == sc) & (M.method == me)]
        ax.scatter(s.migrations / 288, s.energy_kwh, s=12, c=c, marker=mk, label=lab, edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_title(SCL[sc], fontsize=9); ax.set_xlabel("Migrations / cycle")
axs[0].set_ylabel("Energy (kWh, 24 h)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "fig5_tradeoff.png")

# Fig 6: convergence, Dynamic Steady (median and IQR of best fitness / initial)
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for me, c, ls_ in [("AGWO", C1, "-"), ("GWO_LS", C3, "-"), ("GWO", C2, "--"), ("PSO", C4, ":"), ("GA", CG, "-.")]:
    v = conv[me]["dynamic"]; ax.plot(v["iter"], v["med"], color=c, lw=2, ls=ls_, label=ML[me])
    if me in ("AGWO", "GWO_LS"): ax.fill_between(v["iter"], v["q1"], v["q3"], color=c, alpha=0.10, lw=0)
ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness / incumbent fitness"); ax.set_ylim(0.58, 1.02)
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "fig6_convergence.png")

# Fig 7: diversity
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
for me, c in [("AGWO", C1), ("GWO_LS", C3), ("GWO", C2)]:
    axs[0].fill_between(div[me]["iter"], div[me]["cont_q1"], div[me]["cont_q3"], color=c, alpha=0.15, lw=0)
    axs[1].fill_between(div[me]["iter"], div[me]["place_q1"], div[me]["place_q3"], color=c, alpha=0.15, lw=0)
    axs[0].plot(div[me]["iter"], div[me]["cont"], color=c, lw=2, label=ML[me])
    axs[1].plot(div[me]["iter"], div[me]["place"], color=c, lw=2, label=ML[me])
axs[0].set_ylabel("Continuous diversity (median)"); axs[1].set_ylabel("Placement diversity (median)")
for ax in axs: ax.set_xlabel("Iteration")
axs[1].legend(fontsize=8); fig.tight_layout(); save(fig, "fig7_diversity.png")

# Fig 8: sensitivity (a-f) + heat map (g)
fig = plt.figure(figsize=(7.4, 5.0))
gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.45)
names = [("lambda", "(a) λ"), ("gamma", "(b) γ"), ("k", "(c) $k$"), ("beta", "(d) τ"), ("t_ls", "(e) $T_{ls}$"), ("l_max", "(f) $L_{max}$")]
dfl = {"lambda": "1.9", "gamma": "0.2", "k": "5", "beta": "1.5", "t_ls": "5", "l_max": "50"}
for i, (p, t) in enumerate(names):
    ax = fig.add_subplot(gs[i // 3, i % 3]) if i < 3 else fig.add_subplot(gs[1, i - 3])
    rows = [e for e in sens[p] if e["value"] != "round"]
    x = [float(e["value"]) for e in rows]; y = [e["energy_kwh"] for e in rows]; sd = [e["energy_sd"] / np.sqrt(e["n"]) for e in rows]
    ax.errorbar(x, y, yerr=sd, color=C1, lw=1.6, marker="o", ms=3.5, capsize=0, elinewidth=0.8)
    if p == "beta":
        ax.set_xscale("log"); r = next(e for e in sens[p] if e["value"] == "round"); ax.axhline(r["energy_kwh"], color=C2, lw=1, ls="--")
    ax.axvline(float(dfl[p]), color=INK2, lw=0.6, ls=":")
    ax.set_title(t, fontsize=9); ax.tick_params(labelsize=7)
    if i in (0, 3): ax.set_ylabel("Energy (kWh)")
ax = fig.add_subplot(gs[:, 3])
H = hm.pivot(index="lam", columns="gam", values="E")
im = ax.imshow(H.values, cmap="Blues_r", aspect="auto", origin="lower")
ax.set_yticks(range(len(H.index))); ax.set_yticklabels([f"{v:g}" for v in H.index], fontsize=7)
ax.set_xticks(range(len(H.columns))); ax.set_xticklabels([f"{v:g}" for v in H.columns], fontsize=7, rotation=90)
ax.set_xlabel("γ"); ax.set_ylabel("λ"); ax.set_title("(g) λ × γ energy", fontsize=9); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04); cb.ax.tick_params(labelsize=7); cb.set_label("Energy (kWh)", fontsize=7)
save(fig, "fig8_sensitivity.png")

# Fig 9: (a) leave-one-out (Dynamic) (b) Shapley per scenario
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1.15]})
loo = [("D0P1L1", "No feedback decay"), ("D1P0L1", "Hard rounding"), ("D1P1L0", "No local search"), ("D1P1L1E0", "Swaps only (no evacuation)")]
v = [abl["dynamic"][c]["d_energy_pct"] for c, _ in loo]
yy = list(range(len(loo)))[::-1]; base = 0.003
axs[0].barh(yy, [max(val, base) - base for val in v], left=base, color=[CG, CG, C2, C2], height=0.6)
axs[0].plot([max(val, base) for val in v], yy, "o", ms=5, mfc="white", mec=INK2, mew=1.1, zorder=3)
axs[0].set_xscale("log"); axs[0].set_xlim(base, 600)
axs[0].xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100])); axs[0].xaxis.set_minor_locator(NullLocator())
axs[0].xaxis.set_major_formatter(FuncFormatter(lambda q, _: f"{q:g}"))
axs[0].set_yticks(yy); axs[0].set_yticklabels([n for _, n in loo]); axs[0].set_xlabel("Energy increase vs full method (%, log scale)")
axs[0].grid(axis="y", visible=False); axs[0].set_title("(a) Leave-one-out, Dynamic Steady", fontsize=9)
for yv, val in zip(yy, v): axs[0].text(max(val, base) * 1.6, yv, f"{val:+.2f}%", va="center", fontsize=7.5, color=INK2)
x = np.arange(4); wdt = 0.26
for i, (p, c, lab) in enumerate([("D", C1, "Feedback decay"), ("P", C3, "Softmax decoding"), ("L", C2, "Local search")]):
    vals = [shap[sc]["phi_kwh"][p] for sc in SC]; xs_ = x + (i - 1) * wdt
    axs[1].bar(xs_, vals, wdt, color=c, label=lab, edgecolor=c, lw=0.8)
    if p == "D":
        axs[1].plot(xs_, vals, "o", ms=4.5, mfc=c, mec="white", mew=0.6, zorder=3)
    for xv, val in zip(xs_, vals):
        axs[1].text(xv, val + (3 if val >= 0 else -3), f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=5.8, color=INK2, rotation=90)
axs[1].set_xticks(x); axs[1].set_xticklabels(["Static", "Dynamic", "Flash", "Hetero."], fontsize=7.5); axs[1].set_ylabel("Shapley value (kWh saved)")
axs[1].set_ylim(-42, 135); axs[1].legend(fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.13), handlelength=1.2, columnspacing=0.9); axs[1].set_title("(b) Shapley attribution vs standard GWO", fontsize=9); axs[1].axhline(0, color=INK2, lw=0.6)
save(fig, "fig9_ablation.png")

# Fig 10: scalability
fig, ax = plt.subplots(figsize=(6.0, 3.0))
STY = {"AGWO": dict(ls="-", lw=1.6, marker="o", ms=5, zorder=4),
       "GWO_LS": dict(ls=(0, (5, 3)), lw=1.6, marker="s", ms=8, mfc="none", mew=1.4, zorder=3),
       "GWO": dict(ls="-", lw=1.6, marker="^", ms=5, zorder=2)}
for me, c in [("GWO", C2), ("GWO_LS", C3), ("AGWO", C1)]:
    r_ = scal[me]["rows"]; xs = [e["hosts"] for e in r_]
    ax.fill_between(xs, [e["tq1"] for e in r_], [e["tq3"] for e in r_], color=c, alpha=0.15, lw=0)
    ax.plot(xs, [e["t"] for e in r_], color=c, label=f"{ML[me]} (slope {scal[me]['slope']:.2f})", **STY[me])
h, l = ax.get_legend_handles_labels(); ax.legend(h[::-1], l[::-1], fontsize=7.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.xaxis.set_major_locator(FixedLocator([50, 100, 200, 400, 800, 1000])); ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 8])); ax.yaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.tick_params(axis="x", labelsize=7.5); ax.set_xlabel("Physical hosts"); ax.set_ylabel("Optimizer time per cycle (s)")
save(fig, "fig10_scalability.png")

# Fig S2: critical-difference style mean-rank chart
fig, ax = plt.subplots(figsize=(6.4, 3.2))
rk = sorted(R["friedman"]["mean_ranks"].items(), key=lambda kv: kv[1])
ax.barh(range(len(rk))[::-1], [v for _, v in rk], color=[col(m) for m, _ in rk], height=0.6)
ax.set_yticks(range(len(rk))[::-1]); ax.set_yticklabels([ML[m] for m, _ in rk]); ax.set_xlabel("Mean Friedman rank on energy (lower is better)")
best = rk[0][1]; ax.axvline(best + R["friedman"]["cd"], color=C2, lw=1, ls="--"); ax.grid(axis="y", visible=False)
ax.text(best + R["friedman"]["cd"], -0.9, f" CD = {R['friedman']['cd']:.2f}", color=C2, fontsize=7.5, va="center")
save(fig, "figS2_ranks.png")

R["figures"] = figs
json.dump(R, open(os.path.join(OUT, "results_v2.json"), "w"), indent=1, default=float)
print("ok", len(figs), "figures")
