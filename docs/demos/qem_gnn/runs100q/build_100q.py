"""Rebuild the 100q Brisbane graphs + labels with the CORRECTED lightcone.

Reproduces the pipeline of 01_build_graphs.ipynb / 02_GNN_Transformer.ipynb:
  circuits : docs/tutorials/data/ising_zne_hardware/100q_brisbane/  (500 files, steps 1-10)
  labels   : docs/tutorials/zne_mitigated/twirl_100q_brisbane/stepNN.json
             ZNE = nf1 - (nf3 - nf1)/2 on twirl-averaged values
  split    : per step, first 10 circuits train / next 40 test  -> 100 train / 400 test
"""
import sys, os, json, pickle, importlib, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from qiskit.dagcircuit import DAGOpNode

def assign_moments_topo(dag):
    m = {}
    for n in dag.topological_op_nodes():
        pr = [p for p in dag.predecessors(n) if isinstance(p, DAGOpNode)]
        m[n] = 0 if not pr else 1 + max(m[x] for x in pr)
    return m

import lightcone; lightcone.assign_moments = assign_moments_topo
import circuits_to_graph; importlib.reload(circuits_to_graph)

CIRC = '../../tutorials/data/ising_zne_hardware/100q_brisbane'
ZNE  = '../../tutorials/zne_mitigated/twirl_100q_brisbane'
OUT  = 'runs100q'
M, NSTEP, PER_STEP, K = 5, 10, 50, 10
MEASURED = [0, 1, 2, 3, 4]

# ---- labels, exactly as cell 17 of 02_GNN_Transformer.ipynb ----
nf1, nf3 = [], []
for s in range(1, NSTEP + 1):
    d = json.load(open(f'{ZNE}/step%02d.json' % s))
    nf1.append(np.array(d['noise_factor_1'])); nf3.append(np.array(d['noise_factor_3']))
nf1 = np.concatenate(nf1); nf3 = np.concatenate(nf3)
nf1 = nf1.reshape(nf1.shape[0], M, 5).mean(-1)
nf3 = nf3.reshape(nf3.shape[0], M, 5).mean(-1)
zne = nf1 - (nf3 - nf1) / 2
noisy = nf1
print(f'labels: {zne.shape}', flush=True)

# ---- circuits, in the same (step, J) order the labels assume ----
paths = [f'{CIRC}/step_%02d_J%02d.pk' % (s, j) for s in range(1, NSTEP + 1) for j in range(PER_STEP)]
assert len(paths) == len(zne), (len(paths), len(zne))

rows, graphs = [], []
t0 = time.time()
for i, p in enumerate(paths):
    qc = pickle.load(open(p, 'rb'))[0]['circuit']
    g = circuits_to_graph.circuit_to_gategraph_data(qc, MEASURED, max_params=2)
    key = f'{p}::#0'
    g.circuit_path = key
    graphs.append(g)
    rows.append({'circuit_path': key,
                 'noisy_z_json': json.dumps(list(map(float, noisy[i]))),
                 'target_y_json': json.dumps(list(map(float, zne[i])))})
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{len(paths)}  {time.time()-t0:.0f}s  nodes={g.num_nodes}', flush=True)

os.makedirs(OUT, exist_ok=True)
torch.save(graphs, f'{OUT}/graphs_100q.pt')
pd.DataFrame(rows).to_csv(f'{OUT}/labels_100q.csv', index=False)

# ---- split: per step, first K train, rest test ----
tr = [paths[s * PER_STEP + j] + '::#0' for s in range(NSTEP) for j in range(K)]
te = [paths[s * PER_STEP + j] + '::#0' for s in range(NSTEP) for j in range(K, PER_STEP)]
json.dump({'train': tr, 'test': te}, open(f'{OUT}/split_100q.json', 'w'))

n = np.array([g.num_nodes for g in graphs])
lc = np.array([g.lightcone_masks.numpy().mean() for g in graphs])
wm = np.array([g.wire_masks.numpy().mean() for g in graphs])
print(f'\ngraphs {len(graphs)} | nodes mean {n.mean():.0f} min {n.min()} max {n.max()}')
print(f'true-lightcone coverage mean {lc.mean():.4f} | wire-mask coverage mean {wm.mean():.4f}')
print(f'train {len(tr)} test {len(te)} | {time.time()-t0:.0f}s')
