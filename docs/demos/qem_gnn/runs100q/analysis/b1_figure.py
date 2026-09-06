import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.load(open('runs100q/analysis/b1_per_step.json'))
s=np.array([x['step'] for x in r]); am=np.array([x['ref_absmean'] for x in r])
sd=np.array([x['ref_std'] for x in r]); mn=np.array([x['mae_noisy'] for x in r])
mq=np.array([x['mae_qagt'] for x in r]); nn=np.array([x['norm_noisy'] for x in r])
nq=np.array([x['norm_qagt'] for x in r])
C={'noisy':'#4C72B0','qagt':'#55A868','ref':'#C44E52','sd':'#8172B2'}
fig,ax=plt.subplots(1,3,figsize=(13.5,3.9))

ax[0].plot(s,am,'o-',color=C['ref'],label=r'mean $|\hat{y}^{\mathrm{ZNE}}|$')
ax[0].plot(s,sd,'s--',color=C['sd'],label=r'std of $\hat{y}^{\mathrm{ZNE}}$')
ax[0].set_title('(a) The reference contracts with depth'); ax[0].set_xlabel('Trotter step')
ax[0].set_ylabel('reference scale'); ax[0].legend(frameon=False,fontsize=9); ax[0].set_ylim(bottom=0)

ax[1].plot(s,mn,'o-',color=C['noisy'],label='Unmitigated')
ax[1].plot(s,mq,'s-',color=C['qagt'],label='QAGT-MLP')
ax[1].set_title('(b) Absolute error falls with depth'); ax[1].set_xlabel('Trotter step')
ax[1].set_ylabel('MAE vs ZNE'); ax[1].legend(frameon=False,fontsize=9); ax[1].set_ylim(bottom=0)

ax[2].plot(s,nn,'o-',color=C['noisy'],label='Unmitigated')
ax[2].plot(s,nq,'s-',color=C['qagt'],label='QAGT-MLP')
ax[2].set_title('(c) Normalised error is flat'); ax[2].set_xlabel('Trotter step')
ax[2].set_ylabel(r'MAE / std$(\hat{y}^{\mathrm{ZNE}})$')
ax[2].set_yscale('log'); ax[2].legend(frameon=False,fontsize=9)
ax[2].annotate('step 1: reference spread\nnear-degenerate (std 0.02)', xy=(1,nn[0]),
               xytext=(2.4,1.05), fontsize=7.5, color='0.35',
               arrowprops=dict(arrowstyle='->',color='0.55',lw=0.8))
for a in ax:
    a.set_xticks(s); a.grid(alpha=0.25,lw=0.6); a.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('/root/papers/68a48de7934fcfad1330c50a/Figures/per-step-normalised-error.png',dpi=200)
print('saved | mean normalised error, steps 2-10: noisy %.3f  QAGT %.3f'%(nn[1:].mean(),nq[1:].mean()))
print('       spread of normalised error over steps 2-10: noisy %.3f  QAGT %.3f'%(nn[1:].std(),nq[1:].std()))
