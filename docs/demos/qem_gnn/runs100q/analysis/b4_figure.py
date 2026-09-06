import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=json.load(open('runs100q/analysis/b4_factorial.json'))
fig,ax=plt.subplots(1,2,figsize=(13,4.2),gridspec_kw={'width_ratios':[1,1.45]})

# (a) main effects
facs=[('local','Local context $p_m$'),('glob','Global context $g$'),
      ('lightcone','Lightcone support'),('attention','Attention backbone')]
eff=[];se=[]
for f,_ in facs:
    on=np.array([v['mean'] for v in d.values() if v[f]==1]); off=np.array([v['mean'] for v in d.values() if v[f]==0])
    eff.append(off.mean()-on.mean()); se.append(np.sqrt(on.var(ddof=1)/len(on)+off.var(ddof=1)/len(off)))
y=np.arange(len(facs))[::-1]
cols=['#55A868' if e-s>0 else '#B0B0B0' for e,s in zip(eff,se)]
ax[0].barh(y,eff,xerr=se,color=cols,capsize=4,error_kw=dict(ecolor='0.3',lw=1.1))
ax[0].axvline(0,color='0.3',lw=1)
ax[0].set_yticks(y); ax[0].set_yticklabels([l for _,l in facs],fontsize=9.5)
ax[0].set_xlabel('effect on MAE of removing the factor'); ax[0].set_title('(a) Main effects',fontsize=11)
for yy,e,s in zip(y,eff,se):
    ax[0].text(e+s+0.00008,yy,f'{e:+.4f}',va='center',fontsize=8.5)
ax[0].set_xlim(-0.0006,0.0031); ax[0].spines[['top','right']].set_visible(False); ax[0].grid(axis='x',alpha=.25,lw=.6)

# (b) all 16 configs, grouped by which contexts are present
groups=[((0,0),'neither context'),((1,0),'local only'),((0,1),'global only'),((1,1),'both')]
gc={'neither context':'#C44E52','local only':'#4C72B0','global only':'#55A868','both':'#8172B2'}
xs=[];hs=[];es=[];cs=[];ticks=[];labels=[];pos=0
for (L,G),gname in groups:
    cells=sorted([v for v in d.values() if v['local']==L and v['glob']==G],key=lambda v:v['mean'])
    for v in cells:
        xs.append(pos); hs.append(v['mean']); es.append(v['std']); cs.append(gc[gname])
        labels.append(('C' if v['lightcone'] else '–')+('A' if v['attention'] else 'G')); pos+=1
    ticks.append((xs[-4]+xs[-1])/2 if len(cells)==4 else xs[-1]); pos+=0.9
b=ax[1].bar(xs,hs,yerr=es,color=cs,capsize=3,error_kw=dict(ecolor='0.35',lw=.9))
for x,h,l in zip(xs,hs,labels): ax[1].text(x,h+0.00025,l,ha='center',fontsize=7,color='0.3')
full=d['L1G1C1A1']['mean']
ax[1].axhline(full,ls='--',lw=1,color='0.45')
ax[1].text(xs[0]-0.55,full+0.00012,'full model',fontsize=8,color='0.45',ha='left')
ax[1].set_xticks(ticks); ax[1].set_xticklabels([g for _,g in groups],fontsize=9.5)
ax[1].set_ylabel('MAE vs ZNE'); ax[1].set_ylim(0,0.0125)
ax[1].set_title('(b) All 16 configurations, grouped by context inputs',fontsize=11)
ax[1].spines[['top','right']].set_visible(False); ax[1].grid(axis='y',alpha=.25,lw=.6)
from matplotlib.lines import Line2D
ax[1].legend(handles=[Line2D([0],[0],marker='s',ls='',color=gc[g],label=g) for _,g in groups],
             frameon=False,fontsize=8,ncol=2,loc='upper left')
ax[1].text(0.5,-0.155,'bar labels:  C = lightcone support on,  \u2013 = off;  A = attention backbone,  G = GCN',
           transform=ax[1].transAxes,ha='center',fontsize=7.5,color='0.35')
plt.tight_layout(rect=[0,0.03,1,1])
plt.savefig('/root/papers/68a48de7934fcfad1330c50a/Figures/factorial-ablation.png',dpi=200)
print('saved')
for (L,G),g in groups:
    m=[v['mean'] for v in d.values() if v['local']==L and v['glob']==G]
    print(f"  {g:16s} best {min(m):.4f}  worst {max(m):.4f}")
