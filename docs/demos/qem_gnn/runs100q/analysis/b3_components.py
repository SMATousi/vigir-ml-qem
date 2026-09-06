"""B3: component ablation on the 100q Brisbane data.

Four factors removed one at a time from the full conditioned-pooling model:
  no lightcone  attention normalised over all nodes instead of over L_m
  no wire       drops the alpha * 1[n in W_m] bias from the attention logit
  no global     drops the mean-pooled g from the head input
  no attention  GCNConv backbone instead of TransformerConv
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch_geometric.loader import DataLoader
from sim_data_utils import attach_labels_and_noisy_exact
from qem_train import TrainConfig, train_qem, collect

OUT='runs100q'; M=5; SEEDS=(0,1,2,3,4)
graphs=torch.load(f'{OUT}/graphs_100q.pt', weights_only=False)
graphs,_=attach_labels_and_noisy_exact(graphs, f'{OUT}/labels_100q.csv')
sp=json.load(open(f'{OUT}/split_100q.json')); trs,tes=set(sp['train']),set(sp['test'])
tr=[g for g in graphs if g.circuit_path in trs]; te=[g for g in graphs if g.circuit_path in tes]
trl=DataLoader(tr,batch_size=4,shuffle=True); tel=DataLoader(te,batch_size=4,shuffle=False)
print(f'train {len(tr)} test {len(te)}', flush=True)

base=dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,head='residual',pool='conditioned')
VARIANTS=[('Full',{}), ('No lightcone',dict(use_lightcone=False)), ('No wire',dict(use_wire=False)),
          ('No global',dict(use_global=False)), ('No attention (GCN)',dict(backbone='gcn'))]
out=json.load(open(f'{OUT}/analysis/b3_components.json')) if os.path.exists(f'{OUT}/analysis/b3_components.json') else {}
for name,kw in VARIANTS:
    if name in out: print('skip',name,flush=True); continue
    P=[]; per_seed=[]
    for s in SEEDS:
        tc=TrainConfig(epochs=100,lr=1e-3,wd=1e-4,patience=30,loss='smooth_l1',huber_beta=0.05,
                       scheduler='cosine',warmup_epochs=5,select_on='mae',seed=s,verbose=False)
        net,_=train_qem(dict(base,**kw),(trl,tel),tc,best_ckpt_path=f'{OUT}/analysis/_b3_{s}.pt')
        raw,y,_=collect(net,tel,'cuda'); P.append(raw.numpy())
        per_seed.append(float(np.abs(raw.numpy()-y.numpy()).mean()))
    p=np.mean(P,0); yv=y.numpy()
    out[name]=dict(mae=float(np.abs(p-yv).mean()), rmse=float(np.sqrt(((p-yv)**2).mean())),
                   per_qubit=np.abs(p.reshape(-1,M)-yv.reshape(-1,M)).mean(0).tolist(),
                   params=sum(q.numel() for q in net.parameters()),
                   per_seed_mae=per_seed, seed_mean=float(np.mean(per_seed)),
                   seed_std=float(np.std(per_seed)))
    json.dump(out, open(f'{OUT}/analysis/b3_components.json','w'), indent=1)
    print(f"  {name:20s} ens MAE {out[name]['mae']:.4f}  per-seed {out[name]['seed_mean']:.4f} "
          f"+/- {out[name]['seed_std']:.4f}", flush=True)
for f in glob.glob(f'{OUT}/analysis/_b3_*.pt'): os.remove(f)
print('B3 COMPLETE')
