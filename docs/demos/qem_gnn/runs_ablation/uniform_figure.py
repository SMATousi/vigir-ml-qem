"""Figure: does attention pooling beat averaging the same causal support?

(a) Per setting, the three pooling schemes normalized to the best conditioned
    attention variant: attention with a conditioned query, attention with a
    single shared query, and the uniform mean over the same lightcone.
(b) Paired difference (uniform mean - conditioned attention) in absolute MAE,
    with the seed spread of both variants combined, so the effect size can be
    read against the noise.
"""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PAPER = '/root/papers/68a48de7934fcfad1330c50a/Figures'
_FIG = os.environ.get('QAGT_FIG_DIR') or (_PAPER if os.path.isdir(_PAPER) else
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures'))
os.makedirs(_FIG, exist_ok=True)

r = json.load(open('runs_ablation/conditioning.json'))
loc = {'100q_heldout': 0.017, 'ising_dataset': 0.270,
       'ising_init_from_qasm_coherent': 0.340,
       'haoran_mbd_random_cliffords': 0.596, 'mbd_theta_0p1pi_coherent': 0.720}
nice = {'100q_heldout': '100q\nheld-out', 'ising_dataset': 'Ising',
        'ising_init_from_qasm_coherent': 'Ising QASM\ncoherent',
        'haoran_mbd_random_cliffords': 'Random\nCliffords',
        'mbd_theta_0p1pi_coherent': r'MBD $0.1\pi$'}
COND = ['emb + wire', 'emb only', 'wire only', 'neither']
order = sorted(r, key=lambda k: loc[k])
missing = [k for k in order if 'uniform mean' not in r[k]]
if missing:
    raise SystemExit(f'uniform mean not yet run for: {missing}')

best = [min(r[k][v]['seed_mean'] for v in COND) for k in order]
shared = [r[k]['shared query']['seed_mean'] for k in order]
mean_ = [r[k]['uniform mean']['seed_mean'] for k in order]
sd = lambda k, v: r[k][v]['seed_std']

fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
x = np.arange(len(order)); w = 0.26
series = [('attention, conditioned query', best, '#2c6fbb'),
          ('attention, shared query', shared, '#b0b0b0'),
          ('uniform mean over the cone', mean_, '#e08a1e')]
for i, (lab, vals, c) in enumerate(series):
    ax[0].bar(x + (i - 1) * w, np.array(vals) / np.array(best), w, label=lab,
              color=c, edgecolor='white', linewidth=0.6)
ax[0].axhline(1.0, color='0.3', lw=0.8, ls='--', zorder=0)
ax[0].set_xticks(x); ax[0].set_xticklabels([nice[k] for k in order], fontsize=8.5)
ax[0].set_ylabel('MAE / best conditioned variant')
ax[0].set_title('(a) Three ways to reduce the causal support', fontsize=10.5)
ax[0].legend(frameon=False, fontsize=8.2, ncol=1, loc='upper left')
for xi, k, b in zip(x, order, best):
    ax[0].text(xi, 0.04, f'cone {loc[k]:.2f}', ha='center', fontsize=6.8,
               color='0.35', transform=ax[0].get_xaxis_transform())
    ax[0].text(xi - w, 1.02, f'{b:.4f}', ha='center', va='bottom', fontsize=6.6,
               color='#2c6fbb', rotation=90)

diff = np.array(mean_) - np.array(best)
err = [np.hypot(sd(k, 'uniform mean'),
                min(((r[k][v]['seed_mean'], r[k][v]['seed_std']) for v in COND))[1])
       for k in order]
cols = ['#c2453a' if d > 0 else '#2e8b57' for d in diff]
ax[1].barh(x, diff, xerr=err, color=cols, height=0.55, capsize=3,
           error_kw=dict(lw=1.0, ecolor='0.35'))
ax[1].axvline(0, color='0.25', lw=1.0)
ax[1].set_yticks(x); ax[1].set_yticklabels([nice[k].replace('\n', ' ') for k in order], fontsize=8.5)
ax[1].set_xlabel('MAE(uniform mean) $-$ MAE(conditioned attention)')
ax[1].set_title('(b) Cost of replacing attention with an average', fontsize=10.5)
ax[1].text(0.985, 0.06, 'attention better $\\rightarrow$', ha='right', fontsize=7.5,
           color='0.35', transform=ax[1].transAxes)
for xi, d, k in zip(x, diff, order):
    if abs(d) < 5e-5:
        ax[1].text(2e-4, xi, 'no difference (both at the floor)', va='center',
                   fontsize=7.2, color='0.45')
for sp in ('top', 'right'):
    ax[0].spines[sp].set_visible(False); ax[1].spines[sp].set_visible(False)
plt.tight_layout()
out = os.path.join(_FIG, 'uniform-pooling-ablation.png')
plt.savefig(out, dpi=200)
print('saved', out)

print(f"\n{'setting':32s} {'cone':>5s} {'cond':>8s} {'shared':>8s} {'mean':>8s} {'mean-cond':>10s}")
for k, b, s_, m in zip(order, best, shared, mean_):
    print(f"{k:32s} {loc[k]:5.3f} {b:8.4f} {s_:8.4f} {m:8.4f} {m-b:+10.4f}")
