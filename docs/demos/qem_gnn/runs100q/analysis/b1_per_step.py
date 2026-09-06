"""B1: per-step spread of the ZNE reference, per-step MAE, and depth-normalised error.

Reviewer 1 observed that error-vs-ZNE is largest for the shallowest circuits.
This quantifies the alternative explanation: <Z> contracts toward zero with
Trotter depth, so the absolute error shrinks for reasons unrelated to
mitigation quality. No retraining -- the model predictions are reloaded.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch_geometric.loader import DataLoader
from sim_data_utils import attach_labels_and_noisy_exact
from qem_train import collect
from qem_ext import QEMGraphTransformerX

OUT='runs100q'; M=5; NSTEP=10; PER_STEP=50
graphs=torch.load(f'{OUT}/graphs_100q.pt', weights_only=False)
graphs,_=attach_labels_and_noisy_exact(graphs, f'{OUT}/labels_100q.csv')
sp=json.load(open(f'{OUT}/split_100q.json')); te=set(sp['test'])
test=[g for g in graphs if g.circuit_path in te]
# step index is encoded in the filename: step_%02d_J%02d.pk
steps=np.array([int(g.circuit_path.split('step_')[1][:2]) for g in test])
tel=DataLoader(test,batch_size=4,shuffle=False)

def preds(net_kw, ckpts):
    P=[]
    for c in ckpts:
        net=QEMGraphTransformerX(test[0].x.shape[1],**net_kw)
        net.load_state_dict(torch.load(c,map_location='cpu')); net.to('cuda')
        raw,y,n=collect(net,tel,'cuda'); P.append(raw.numpy())
    return np.mean(P,0), y.numpy(), n.numpy()

B=dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,head='residual',pool='conditioned')
p,y,z = preds(B,[f'{OUT}/best_B_s{s}.pt' for s in range(5)])
p,y,z = p.reshape(-1,M), y.reshape(-1,M), z.reshape(-1,M)

rows=[]
for s in range(1,NSTEP+1):
    k = steps==s
    ys, ps, zs = y[k], p[k], z[k]
    std = float(ys.std())                       # spread of the ZNE reference at this step
    mae = float(np.abs(ps-ys).mean())
    mae_z = float(np.abs(zs-ys).mean())
    rows.append(dict(step=s, n=int(k.sum()), ref_std=std, ref_absmean=float(np.abs(ys).mean()),
                     mae_qagt=mae, mae_noisy=mae_z,
                     norm_qagt=mae/std, norm_noisy=mae_z/std))
json.dump(rows, open(f'{OUT}/analysis/b1_per_step.json','w'), indent=1)
print(f"{'step':>4} {'n':>4} {'|ref|':>7} {'ref std':>8} {'MAE noisy':>10} {'MAE QAGT':>9} {'norm noisy':>11} {'norm QAGT':>10}")
for r in rows:
    print(f"{r['step']:4d} {r['n']:4d} {r['ref_absmean']:7.4f} {r['ref_std']:8.4f} "
          f"{r['mae_noisy']:10.4f} {r['mae_qagt']:9.4f} {r['norm_noisy']:11.4f} {r['norm_qagt']:10.4f}")
