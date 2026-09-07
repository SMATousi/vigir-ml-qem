import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=json.load(open('runs_ablation/conditioning.json'))
col=json.load(open('runs_ablation/collapse.json'))
loc={'100q_heldout':0.017,'ising_dataset':0.270,'ising_init_from_qasm_coherent':0.340,
     'haoran_mbd_random_cliffords':0.596,'mbd_theta_0p1pi_coherent':0.720}
lbl={'100q_heldout':'100q\nheld-out','ising_dataset':'Ising','ising_init_from_qasm_coherent':'Ising QASM\ncoherent',
     'haoran_mbd_random_cliffords':'Rand.\nCliffords','mbd_theta_0p1pi_coherent':r'MBD 0.1$\pi$'+'\ncoherent'}
V=['emb + wire','emb only','wire only','neither','shared query']
C=['#55A868','#6FB98F','#8CCBB0','#B7DDC8','#C44E52']
order=sorted(d,key=lambda k:loc[k])
fig,ax=plt.subplots(1,2,figsize=(13.5,4.3),gridspec_kw={'width_ratios':[1.75,1]})

x=np.arange(len(order)); w=0.16
for i,(v,c) in enumerate(zip(V,C)):
    base=np.array([min(d[k][u]['seed_mean'] for u in V[:4]) for k in order])
    vals=np.array([d[k][v]['seed_mean'] for k in order])/base
    errs=np.array([d[k][v]['seed_std'] for k in order])/base
    ax[0].bar(x+(i-2)*w, vals, w, yerr=errs, label=v, color=c, capsize=2,
              error_kw=dict(ecolor='0.35',lw=.8))
ax[0].axhline(1,color='0.35',lw=1,ls='--')
ax[0].set_xticks(x); ax[0].set_xticklabels([lbl[k] for k in order],fontsize=8.5)
ax[0].set_ylabel('MAE relative to best conditioned variant')
ax[0].set_title('(a) Per-qubit conditioning: which mechanism, and whether any',fontsize=10.5)
ax[0].legend(frameon=False,fontsize=8,ncol=5,loc='upper center')
ax[0].set_ylim(0,3.0); ax[0].grid(axis='y',alpha=.25,lw=.6); ax[0].spines[['top','right']].set_visible(False)
for xi,k in zip(x,order):
    ax[0].text(xi,0.06,f'cone {loc[k]:.2f}',ha='center',fontsize=6.8,color='0.35')

names=['shared query','qubit-conditioned']
met=[('cosine','cosine between\nthe M pooled vectors'),('std_across_circuits','pooled std\nacross circuits')]
xx=np.arange(len(met)); w2=0.3
for i,(nm,c) in enumerate(zip(names,['#C44E52','#55A868'])):
    vals=[col[nm][m] for m,_ in met]
    b=ax[1].bar(xx+(i-0.5)*w2, vals, w2, label=nm, color=c)
    ax[1].bar_label(b,fmt='%.2f',fontsize=8,padding=2)
ax[1].set_xticks(xx); ax[1].set_xticklabels([t for _,t in met],fontsize=8.5)
ax[1].set_title('(b) Representation, Ising QASM coherent',fontsize=10.5)
ax[1].legend(frameon=False,fontsize=8.5); ax[1].set_ylim(0,0.72)
ax[1].grid(axis='y',alpha=.25,lw=.6); ax[1].spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('/root/papers/68a48de7934fcfad1330c50a/Figures/conditioning-ablation.png',dpi=200)
print('saved')
