import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=json.load(open('runs100q/analysis/b3_components.json'))
order=['Full','No lightcone','No wire','No global','No attention (GCN)']
m=np.array([d[k]['seed_mean'] for k in order]); e=np.array([d[k]['seed_std'] for k in order])
full=m[0]
fig,ax=plt.subplots(figsize=(7.4,4.0))
cols=['#55A868']+['#4C72B0']*3+['#C44E52']
b=ax.bar(range(len(order)),m,yerr=e,capsize=4,color=cols,edgecolor='none',
         error_kw=dict(ecolor='0.3',lw=1.1))
for i,(v,s) in enumerate(zip(m,e)):
    ax.text(i,v+s+0.00012,f'{v:.4f}',ha='center',fontsize=9)
ax.axhline(full,ls='--',lw=1,color='0.45')
ax.text(len(order)-0.45,full+0.00004,'full model',fontsize=8,color='0.45',ha='right')
ax.set_xticks(range(len(order)))
ax.set_xticklabels(['Full','No\nlightcone','No\nwire','No\nglobal','No attention\n(GCN)'],fontsize=9)
ax.set_ylabel('MAE vs ZNE  (mean over 5 seeds)')
ax.set_title('Component ablation, 100-qubit TFIM (error bars: std over seeds)',fontsize=10.5)
ax.set_ylim(0.0120,0.0143); ax.grid(axis='y',alpha=0.25,lw=0.6)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('/root/papers/68a48de7934fcfad1330c50a/Figures/component-ablation.png',dpi=200)
print(f"{'variant':22s} {'mean':>8s} {'std':>8s}  delta vs full (in std of full)")
for k in order:
    v=d[k]; delta=v['seed_mean']-full
    print(f"{k:22s} {v['seed_mean']:8.4f} {v['seed_std']:8.4f}  {delta:+.4f}  ({delta/max(e[0],1e-9):+.1f} sd)")
