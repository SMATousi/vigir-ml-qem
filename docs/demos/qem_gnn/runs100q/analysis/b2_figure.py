import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=json.load(open('runs100q/analysis/b2_heldout.json'))
m=json.load(open('runs100q/results_100q.json'))
C={'noisy':'#4C72B0','rf':'#DD8452','qagt':'#55A868'}
fig,ax=plt.subplots(1,2,figsize=(10,3.9))

# (a) interpolation vs extrapolation
lbl=['Random split\n(interpolation)','Held-out steps 8-10\n(extrapolation)']
x=np.arange(2); w=0.26
vals={'Unmitigated':[m['noisy']['mae'],d['Unmitigated']['mae']],
      'Random Forest':[m['RF']['mae'],d['Random Forest']['mae']],
      'QAGT-MLP':[m['B_conditioned_5seed']['mae'],d['QAGT-MLP']['mae']]}
for i,(k,c) in enumerate(zip(vals,[C['noisy'],C['rf'],C['qagt']])):
    b=ax[0].bar(x+(i-1)*w,vals[k],w,label=k,color=c)
    ax[0].bar_label(b,fmt='%.4f',fontsize=7.5,padding=2)
ax[0].set_xticks(x); ax[0].set_xticklabels(lbl,fontsize=9)
ax[0].set_ylabel('MAE vs ZNE'); ax[0].set_title('(a) Interpolation vs extrapolation')
ax[0].legend(frameon=False,fontsize=8.5,ncol=1); ax[0].set_ylim(0,max(vals['Unmitigated'])*1.35)

# (b) per held-out step
st=sorted(d['per_step'],key=int); xs=np.arange(len(st))
for i,(k,c,lab) in enumerate([('noisy',C['noisy'],'Unmitigated'),('rf',C['rf'],'Random Forest'),
                              ('qagt',C['qagt'],'QAGT-MLP')]):
    b=ax[1].bar(xs+(i-1)*w,[d['per_step'][s][k] for s in st],w,color=c,label=lab)
    ax[1].bar_label(b,fmt='%.4f',fontsize=7.5,padding=2)
ax[1].set_xticks(xs); ax[1].set_xticklabels([f'step {s}' for s in st])
ax[1].set_ylabel('MAE vs ZNE'); ax[1].set_title('(b) Error on each held-out step')
ax[1].legend(frameon=False,fontsize=8.5)
ax[1].set_ylim(0,max(d['per_step'][s]['noisy'] for s in st)*1.35)
for a in ax: a.grid(axis='y',alpha=0.25,lw=0.6); a.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('/root/papers/68a48de7934fcfad1330c50a/Figures/heldout-step-extrapolation.png',dpi=200)
print('saved')
