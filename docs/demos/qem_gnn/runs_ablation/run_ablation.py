"""Graph-vs-descriptors ablation across all 11 simulator datasets.

Three variants, identical in every other respect:
  descriptors only : graph path zeroed (verified: encoder gradient is exactly 0)
  graph only       : the model as reported in the main results
  both             : graph + the Random Forest's circuit-level descriptors

Reuses the processed graphs already in runs/<dataset>/, so no rebuild.
Writes incrementally so partial results are usable.
"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch_geometric.loader import DataLoader
from data_utils import load_graphs, build_loaders
from sim_data_utils import attach_labels_and_noisy_exact, load_circuit_label_lists
from qem_levels import LevelSet
from qem_descriptors import attach_descriptors_from_paths, DESCRIPTOR_DIM
from qem_train import TrainConfig, train_qem, collect, metrics

SEEDS = (0, 1, 2)
OUT = 'runs_ablation/ablation.json'
DIRS = {  # dataset -> (raw train dir, raw val dir or None)
 'haoran_mbd_coherent_random_cliffords': ('../../tutorials/data/haoran_mbd_coherent/random_cliffords', None),
 'haoran_mbd_random_brickwork': ('../../tutorials/data/haoran_mbd/random_brickwork', None),
 'haoran_mbd_random_cliffords': ('../../tutorials/data/haoran_mbd/random_cliffords', '../../tutorials/data/haoran_mbd/random_cliffords/val'),
 'ising_dataset': ('../../tutorials/data/ising_dataset/train', '../../tutorials/data/ising_dataset/val'),
 'ising_dataset_random_init': ('../../tutorials/data/ising_dataset_random_init/train', '../../tutorials/data/ising_dataset_random_init/val'),
 'ising_init_0110': ('../../tutorials/data/ising_init_0110/train', '../../tutorials/data/ising_init_0110/val'),
 'ising_init_from_qasm': ('../../tutorials/data/ising_init_from_qasm/train', '../../tutorials/data/ising_init_from_qasm/val'),
 'ising_init_from_qasm_coherent': ('../../tutorials/data/ising_init_from_qasm_coherent/train', '../../tutorials/data/ising_init_from_qasm_coherent/val'),
 'ising_init_from_qasm_no_readout': ('../../tutorials/data/ising_init_from_qasm_no_readout/train', '../../tutorials/data/ising_init_from_qasm_no_readout/val'),
 'mbd_theta_0p05pi': ('runs/mbd_theta_0p05pi/pk_cache/train', 'runs/mbd_theta_0p05pi/pk_cache/val'),
 'mbd_theta_0p1pi_coherent': ('runs/mbd_theta_0p1pi_coherent/pk_cache/train', 'runs/mbd_theta_0p1pi_coherent/pk_cache/val'),
}
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for name, (trdir, vadir) in DIRS.items():
    if name in res and len(res[name]) == 3:
        print(f'skip {name} (done)', flush=True); continue
    R = f'runs/{name}'
    if not os.path.exists(f'{R}/processed/graphs_train.pt'):
        print(f'skip {name}: no processed graphs', flush=True); continue
    t0 = time.time()
    tr, M = attach_labels_and_noisy_exact(load_graphs(f'{R}/processed/graphs_train.pt'), f'{R}/labels_train.csv')
    if os.path.exists(f'{R}/processed/graphs_val.pt'):
        va, _ = attach_labels_and_noisy_exact(load_graphs(f'{R}/processed/graphs_val.pt'), f'{R}/labels_val.csv')
        trl = DataLoader(tr, batch_size=16, shuffle=True); val = DataLoader(va, batch_size=16, shuffle=False)
    else:
        trl, val = build_loaders(tr, batch_size=16, val_frac=0.2, seed=42)

    # descriptors, computed from each graph's own source circuit
    attach_descriptors_from_paths(list(trl.dataset) + list(val.dataset), 'cx')

    ytr = torch.cat([g.y for g in trl.dataset]); levels = LevelSet.detect(ytr)
    d_model = 64 if name == 'haoran_mbd_coherent_random_cliffords' else 128
    base = dict(d_model=d_model, layers=3, heads=4, dropout=0.05, use_noisy=True, pool='conditioned')
    if levels is not None:
        base.update(head='level', levels=levels)
        tc = dict(loss='ce', select_on='acc')
    else:
        base.update(head='residual')
        tc = dict(loss='smooth_l1', huber_beta=0.05, select_on='mae')

    res.setdefault(name, {})
    for vname, kw in [('descriptors_only', dict(desc_dim=DESCRIPTOR_DIM, graph_off=True)),
                      ('graph_only', dict()),
                      ('both', dict(desc_dim=DESCRIPTOR_DIM))]:
        if vname in res[name]: continue
        preds = []
        for s in SEEDS:
            cfg = TrainConfig(epochs=80, lr=1e-3, wd=1e-4, patience=25, scheduler='cosine',
                              warmup_epochs=5, seed=s, verbose=False, **tc)
            net, _ = train_qem(dict(base, **kw), (trl, val), cfg,
                               best_ckpt_path=f'runs_ablation/_tmp_{name}_{vname}_{s}.pt')
            raw, y, _ = collect(net, val, 'cuda')
            preds.append(torch.softmax(raw.float(), -1).numpy() if net.is_classifier else raw.numpy())
            Y = y.numpy()
        P = np.mean(preds, 0)
        pred = levels.argmax(torch.log(torch.tensor(P).clamp_min(1e-12))).numpy() if levels is not None else P
        res[name][vname] = metrics(pred, Y, levels)
        res[name][vname]['params'] = sum(p.numel() for p in net.parameters())
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f'  {name:34s} {vname:17s} MAE {res[name][vname]["mae"]:.4f}', flush=True)
    for f in glob.glob('runs_ablation/_tmp_*.pt'): os.remove(f)
    print(f'{name} done in {time.time()-t0:.0f}s', flush=True)
print('ABLATION COMPLETE')
