"""Assemble the benchmark comparison and draw it.

Everything here is scored by `random_circuits.mitigation.score_mitigation` from
the base paper's own repository, on version 0 of the full test split, so the
rows are directly comparable. Their checkpoint numbers are read from the
results they shipped; ours are produced by `train_bench.py`.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THEIRS = '/root/stousi_missouri.edu_01M26D5DP6FR5334PATFM1NQMK/outputs/results'
B = 'bench_quantinuum'
_PAPER = '/root/papers/68a48de7934fcfad1330c50a/Figures'
FIG = os.environ.get('QAGT_FIG_DIR') or (_PAPER if os.path.isdir(_PAPER) else f'{B}/figures')
os.makedirs(FIG, exist_ok=True)

NICE = {
    'Encoder-Model-Training-pauli-real-algiers': 'Encoder',
    'RNN-Model-Training-pauli-real-algiers': 'RNN',
    'PercLoss-kl-pauli-real-algiers': 'Perceiver',
    'PercLoss-kl-pauli-real-algiers-FTfrom-algiers-simulated': 'Perceiver FT',
    'PercLoss-kl-pauli-real-hanoi-FTfrom-algiers-real': 'Perceiver FT (from real)',
    'PercLoss-kl-pauli-real-hanoi-FTfrom-algiers-simulated': 'Perceiver FT (from sim)',
}


def rows(device):
    out = []
    theirs = json.load(open(f'{THEIRS}/{device}_pretrained.json'))['mitigation']
    for k, v in theirs.items():
        label = 'unmitigated' if k == 'unmitigated' else NICE.get(k, k)
        out.append(dict(label=label, kl=v['kl'], tv=v['tv'], hell=v['hellinger'],
                        rel=v.get('kl_rel_change', 0.0), sd=0.0,
                        kind='unmit' if k == 'unmitigated' else 'theirs'))
    for tag, label, kind in [('qagt', 'QAGT-MLP (ours)', 'ours'),
                             ('qagt_nograph', 'QAGT-MLP, graph off', 'ours_ab')]:
        p = f'{B}/results_{device}_{tag}.json'
        if os.path.exists(p):
            m = json.load(open(p))[tag]
            out.append(dict(label=label, kl=m['mean']['kl'], tv=m['mean']['tv'],
                            hell=m['mean']['hellinger'], rel=m['mean']['kl_rel_change'],
                            sd=m['std']['kl'], kind=kind))
    p = f'{B}/results_{device}_mlp.json'
    if os.path.exists(p):
        m = json.load(open(p))['mlp_noisy_only']
        out.append(dict(label='MLP, circuit-blind', kl=m['mean']['kl'], tv=m['mean']['tv'],
                        hell=m['mean']['hellinger'], rel=m['mean']['kl_rel_change'],
                        sd=m['std']['kl'], kind='blind'))
    return sorted(out, key=lambda r: -r['kl'])


COL = {'unmit': '#9e9e9e', 'theirs': '#6f8fb5', 'ours': '#c2453a',
       'ours_ab': '#e3a9a3', 'blind': '#2e8b57'}

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))
table = {}
for ax, device in zip(axes, ('algiers', 'hanoi')):
    R = rows(device)
    table[device] = R
    y = np.arange(len(R))
    ax.barh(y, [r['kl'] for r in R], xerr=[r['sd'] for r in R],
            color=[COL[r['kind']] for r in R], height=0.62,
            error_kw=dict(lw=1.0, ecolor='0.3'))
    ax.set_yticks(y); ax.set_yticklabels([r['label'] for r in R], fontsize=8.8)
    ax.invert_yaxis()
    for i, r in enumerate(R):
        ax.text(r['kl'] + 0.06, i, f"{r['kl']:.2f}", va='center', fontsize=7.8, color='0.25')
    ax.set_xlabel('KL to ideal (lower is better)')
    ax.set_title(f'{device}  ({"27,000" if device=="algiers" else "28,125"} test circuits)',
                 fontsize=10.5)
    ax.set_xlim(0, max(r['kl'] for r in R) * 1.14)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ('unmit', 'theirs', 'ours', 'ours_ab', 'blind')]
fig.legend(h, ['unmitigated', 'Placidi et al. checkpoints', 'QAGT-MLP (ours)',
               'ours, graph pathway off', 'MLP on measured distribution only'],
           loc='lower center', ncol=5, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, -0.03))
plt.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join(FIG, 'quantinuum-benchmark.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)

for device, R in table.items():
    print(f'\n### {device}\n')
    print('| method | KL | TV | Hellinger | rel. change |')
    print('|---|---|---|---|---|')
    for r in R:
        sd = f" ± {r['sd']:.3f}" if r['sd'] else ''
        print(f"| {r['label']} | {r['kl']:.3f}{sd} | {r['tv']:.3f} | {r['hell']:.3f} | {r['rel']:+.1%} |")
json.dump(table, open(f'{B}/comparison.json', 'w'), indent=1)
