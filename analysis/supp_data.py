import json, glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; RES=sys.argv[1]; OUT=sys.argv[2]
R=json.load(open(OUT+'/results_v2.json'))
d=pd.read_csv(RES+'/runs.csv'); M=d[d.experiment=='main'].copy(); M['slatah_pct']=100*M.slatah
S={}
st=[]
for (sc,me),s in M.groupby(['scenario','method']):
    e=s.energy_kwh; a=s.slatah_pct
    st.append({'scenario':sc,'method':me,'cv':float(100*e.std()/e.mean()),'iqr':float(e.quantile(.75)-e.quantile(.25)),'p95':float(e.quantile(.95)),'sl_p95':float(a.quantile(.95)),'chg':float(100*s.decision_change_frac.mean())})
S['stab']=st
# incremental addition per scenario
ab=R['ablation']; inc={}
for sc in ['static','dynamic','flash','hetero','all']:
    E=lambda c: ab[sc][c]['energy_kwh'][0]
    inc[sc]=[('Standard GWO, hard rounding (D0P0L0)',E('D0P0L0')),('+ feedback decay (D1P0L0)',E('D1P0L0')),('+ softmax encoding (D1P1L0)',E('D1P1L0')),('+ local search = AGWO-VMC (D1P1L1)',E('D1P1L1'))]
S['inc']=inc
json.dump(S,open(OUT+'/supp.json','w'),indent=1)
# Fig S1: convergence in all scenarios
logs=[]
for f in glob.glob(RES+'/iters/main__*.csv'):
    pr=os.path.basename(f)[:-4].split('__'); t=pd.read_csv(f); t['scenario']=pr[1]; t['method']=pr[2]; logs.append(t)
L=pd.concat(logs); L['run']=L.scenario+L.method+L.cycle.astype(str)+L.index.astype(str)
L['key']=L.scenario+L.method
L['fn']=L.groupby([L.scenario,L.method,L.cycle,(L.iter==1).cumsum()]).best_fitness.transform(lambda x:x/x.iloc[0])
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#e6e5df","legend.frameon":False,"figure.dpi":300,"savefig.dpi":300})
fig,axs=plt.subplots(1,4,figsize=(7.4,2.4),sharey=True)
cols={'AGWO':'#2a78d6','GWO_LS':'#1baf7a','GWO':'#eb6834','PSO':'#4a3aa7','GA':'#9a9a94'}
lab={'AGWO':'AGWO-VMC','GWO_LS':'GWO+LS','GWO':'GWO','PSO':'PSO','GA':'GA'}
for ax,sc,t in zip(axs,['static','dynamic','flash','hetero'],['Static','Dynamic Steady','Flash Crowd','Heterog. Mix']):
    for me,c in cols.items():
        s=L[(L.scenario==sc)&(L.method==me)].groupby('iter').fn.median(); ax.plot(s.index,s.values,color=c,lw=1.6,label=lab[me])
    ax.set_title(t,fontsize=8.5); ax.set_xlabel('Iteration')
axs[0].set_ylabel('Median best / incumbent fitness'); axs[-1].legend(fontsize=6.5,loc='lower right')
fig.tight_layout(); fig.savefig(OUT+'/figS1_convergence.png',bbox_inches='tight',facecolor='white')
print('ok')
