"""Reproduce the ORIGINAL (pre-fix) 100q pipeline on the same split.

The broken predecessor map made lightcone == wire mask, and made every moment
index 0, so the 8-dim moment encoding was the constant sinusoidal_encoding(0).
Both are reproduced here from the already-built graphs.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch_geometric.loader import DataLoader
from sim_data_utils import attach_labels_and_noisy_exact
from qem_train import TrainConfig, train_qem, collect
from featurizers import sinusoidal_encoding

OUT='runs100q'; M=5
graphs=torch.load(f'{OUT}/graphs_100q.pt', weights_only=False)
graphs,_=attach_labels_and_noisy_exact(graphs, f'{OUT}/labels_100q.csv')
menc=torch.tensor(sinusoidal_encoding(0,8),dtype=torch.float32)
for g in graphs:
    g.lightcone_masks = g.wire_masks.clone()      # what the bug actually produced
    g.x[:, -8:] = menc                            # constant moment encoding
print("reverted to pre-fix graphs | lightcone coverage now",
      float(np.mean([g.lightcone_masks.numpy().mean() for g in graphs])), flush=True)

sp=json.load(open(f'{OUT}/split_100q.json')); trs,tes=set(sp['train']),set(sp['test'])
tr=[g for g in graphs if g.circuit_path in trs]; te=[g for g in graphs if g.circuit_path in tes]
trl=DataLoader(tr,batch_size=4,shuffle=True); tel=DataLoader(te,batch_size=4,shuffle=False)

net,_=train_qem(dict(d_model=128,layers=3,heads=4,dropout=0.1,use_noisy=True,head='scalar',pool='shared'),
                (trl,tel), TrainConfig(epochs=100,lr=1e-3,wd=1e-4,patience=30,seed=0,verbose=False),
                best_ckpt_path=f'{OUT}/best_prefix.pt')
raw,y,n=collect(net,tel,'cuda'); p=raw.numpy(); y=y.numpy()
r=dict(mae=float(np.abs(p-y).mean()), rmse=float(np.sqrt(((p-y)**2).mean())),
       per_qubit_mae=np.abs(p.reshape(-1,M)-y.reshape(-1,M)).mean(0).tolist(),
       params=sum(q.numel() for q in net.parameters()))
print(f"ORIGINAL (pre-fix) paper pipeline: MAE {r['mae']:.4f} RMSE {r['rmse']:.4f} params {r['params']}")
print("per-qubit:", [f"{x:.4f}" for x in r['per_qubit_mae']])
res=json.load(open(f'{OUT}/results_100q.json')); res['ORIGINAL_prefix_paper']=r
json.dump(res, open(f'{OUT}/results_100q.json','w'), indent=2)
