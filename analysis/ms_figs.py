import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys, os; OUT=sys.argv[1].rstrip('/')+'/'; R=json.load(open(os.path.join(OUT,'results_v2.json')))
INK2, GRID="#5f5e58", "#e6e5df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,"axes.axisbelow":True,"legend.frameon":False,"figure.dpi":300,"savefig.dpi":int(__import__("os").environ.get("FIGDPI",300)),"axes.edgecolor":INK2,"xtick.color":INK2,"ytick.color":INK2})
SC=["static","dynamic","flash","hetero"]; SCL=["Static","Dynamic Steady","Flash Crowd","Heterogeneous Mix"]
def grouped(metric, ms, labels, cols, ylab, name, scale=1.0):
    fig,ax=plt.subplots(figsize=(7.0,2.9)); x=np.arange(4); wd=0.8/len(ms)
    for i,(m,l,c) in enumerate(zip(ms,labels,cols)):
        v=[R['main'][s][m][metric][0]*scale for s in SC]; e=[R['main'][s][m][metric][1]*scale for s in SC]
        ax.bar(x+(i-(len(ms)-1)/2)*wd, v, wd*0.92, yerr=[np.minimum(e,v), e], color=c, label=l, error_kw={"elinewidth":0.6,"ecolor":INK2})
    ax.set_xticks(x); ax.set_xticklabels(SCL); ax.set_ylabel(ylab); ax.grid(axis='x',visible=False)
    ax.legend(ncol=len(ms), fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5,-0.12))
    fig.savefig(OUT+name, bbox_inches='tight', facecolor='white'); plt.close(fig)
grouped('slatah_pct',["AGWO","BFD","GWO_LS","HHO_LS"],["AGWO-VMC","PABFD","GWO+LS","HHO+LS"],["#2a78d6","#eb6834","#1baf7a","#4a3aa7"],"SLATAH (% of active time)","ms_fig4_sla.png")
grouped('migrations',["AGWO","BFD","GWO_LS","HHO_LS","HHO"],["AGWO-VMC","PABFD","GWO+LS","HHO+LS","HHO"],["#2a78d6","#eb6834","#1baf7a","#4a3aa7","#9a9a94"],"Live migrations per 24 h","ms_fig5_mig.png")
print('ok')
