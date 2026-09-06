"""Full 2^4 factorial ablation on the 100q held-out-step split.

Four binary factors, mapping one-to-one onto the claims the paper makes:

  local      qubit-local pooled context p_m in the head input
  global     graph-wide mean-pooled context g in the head input
  lightcone  attention normalised over L_m (on) or over all nodes (off)
  attention  TransformerConv backbone (on) or GCNConv of matched size (off)

Leave-one-out cannot distinguish a useless component from a redundant one; a
factorial can, because it estimates main effects and interactions. The floor
cell (local=off, global=off) leaves the head a function of the noisy value
alone, so every effect is measured against the unmitigated baseline.

Note: when local=off the lightcone factor is a no-op, since nothing is pooled.
Those cells are retained as replication checks rather than dropped.
"""
import sys, os, json, glob, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch_geometric.loader import DataLoader
from sim_data_utils import attach_labels_and_noisy_exact
from qem_train import TrainConfig, train_qem, collect

SEEDS=(0,1,2); OUT='runs100q/analysis/b4_factorial.json'
g=torch.load('runs100q/graphs_100q.pt',weights_only=False)
g,_=attach_labels_and_noisy_exact(g,'runs100q/labels_100q.csv')
step=lambda x:int(x.circuit_path.split('step_')[1][:2]); J=lambda x:int(x.circuit_path.split('_J')[1].split('.pk')[0])
tr=[x for x in g if step(x)<=7 and J(x)<10]; te=[x for x in g if step(x)>=8]
trl=DataLoader(tr,batch_size=4,shuffle=True); tel=DataLoader(te,batch_size=4,shuffle=False)
print(f'train {len(tr)} (steps 1-7)  test {len(te)} (steps 8-10)',flush=True)

base=dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,head='residual',pool='conditioned')
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
t0=time.time()
for local,glob_,lc,att in itertools.product([1,0],repeat=4):
    key=f"L{local}G{glob_}C{lc}A{att}"
    if key in res: continue
    kw=dict(use_local=bool(local), use_global=bool(glob_), use_lightcone=bool(lc),
            backbone='transformer' if att else 'gcn')
    per=[]
    for s in SEEDS:
        tc=TrainConfig(epochs=100,lr=1e-3,wd=1e-4,patience=30,loss='smooth_l1',huber_beta=0.05,
                       scheduler='cosine',warmup_epochs=5,select_on='mae',seed=s,verbose=False)
        net,_=train_qem(dict(base,**kw),(trl,tel),tc,best_ckpt_path=f'runs100q/analysis/_f_{s}.pt')
        raw,y,_=collect(net,tel,'cuda')
        per.append(float(np.abs(raw.numpy()-y.numpy()).mean()))
    res[key]=dict(local=local, glob=glob_, lightcone=lc, attention=att,
                  per_seed=per, mean=float(np.mean(per)), std=float(np.std(per)))
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"  {key}  local={local} global={glob_} cone={lc} attn={att}  "
          f"MAE {np.mean(per):.4f} +/- {np.std(per):.4f}   [{time.time()-t0:.0f}s]",flush=True)
for f in glob.glob('runs100q/analysis/_f_*.pt'): os.remove(f)
print('FACTORIAL COMPLETE')
