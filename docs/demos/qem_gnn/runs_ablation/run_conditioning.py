"""Leave-one-in ablation of the per-qubit conditioning mechanisms.

QubitConditionedPoolingV2 conditions the pooling on the target qubit through
two independent paths: the learned embedding Emb(m) inside the query, and the
wire-membership bias alpha*1[n in W_m] in the attention logit. A leave-one-out
design cannot separate "useless" from "redundant" here, because removing either
leaves the other in place. This runs all four combinations, plus the shared
query of model.py as a no-conditioning reference.

Datasets are chosen to span the two causal-locality regimes revealed once
barriers are excluded from the lightcone:

  broad cones : mbd_theta_0p1pi_coherent (cov 0.72, Jaccard 0.66)
                haoran_mbd_random_cliffords (0.60, 0.51)
  local cones : ising_init_from_qasm_coherent (0.34, 0.15)
                ising_dataset (0.27, 0.06)
                100q held-out steps (0.017, 0.089)

The hypothesis under test is that the wire mask carries the per-qubit signal
where the cone is too broad to discriminate, and is redundant where it is not.
"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch_geometric.loader import DataLoader
from data_utils import load_graphs
from sim_data_utils import attach_labels_and_noisy_exact
from qem_levels import LevelSet
from qem_train import TrainConfig, train_qem, collect, metrics

SEEDS=(0,1,2); OUT='runs_ablation/conditioning.json'
VARIANTS=[('emb + wire',   dict(pool='conditioned')),
          ('emb only',     dict(pool='conditioned', use_wire=False)),
          ('wire only',    dict(pool='conditioned', use_qubit_emb=False)),
          ('neither',      dict(pool='conditioned', use_wire=False, use_qubit_emb=False)),
          ('shared query', dict(pool='shared')),
          # The control the four variants above cannot provide: they all pool by
          # a learned softmax and differ only in what enters the query, so none
          # of them tests attention pooling itself. This one keeps the same
          # causal support and averages it, with no pooling parameters at all.
          ('uniform mean',  dict(pool='mean'))]
res=json.load(open(OUT)) if os.path.exists(OUT) else {}

def run(tag, trl, val, base, tc, levels=None):
    res.setdefault(tag,{})
    for name,kw in VARIANTS:
        if name in res[tag]: continue
        P=[]; per=[]
        for s in SEEDS:
            cfg=TrainConfig(epochs=100 if tag.startswith('100q') else 80,
                            lr=1e-3,wd=1e-4,patience=30 if tag.startswith('100q') else 25,
                            scheduler='cosine',warmup_epochs=5,seed=s,verbose=False,**tc)
            net,_=train_qem(dict(base,**kw),(trl,val),cfg,best_ckpt_path=f'runs_ablation/_c_{s}.pt')
            raw,y,_=collect(net,val,'cuda')
            P.append(torch.softmax(raw.float(),-1).numpy() if net.is_classifier else raw.numpy())
            d=net.decode(raw,'argmax' if net.is_classifier else 'raw').numpy()
            per.append(float(np.abs(d-y.numpy()).mean()))
        p=np.mean(P,0); Y=y.numpy()
        pred=levels.argmax(torch.log(torch.tensor(p).clamp_min(1e-12))).numpy() if levels is not None else p
        m=metrics(pred,Y,levels)
        m.update(per_seed=per, seed_mean=float(np.mean(per)), seed_std=float(np.std(per)))
        res[tag][name]=m
        json.dump(res,open(OUT,'w'),indent=1)
        print(f"  [{tag}] {name:14s} MAE {m['seed_mean']:.4f} +/- {m['seed_std']:.4f}",flush=True)
    for f in glob.glob('runs_ablation/_c_*.pt'): os.remove(f)

# ---- 100q held-out steps (most local cones) ----
print('=== 100q held-out steps ===',flush=True)
g=torch.load('runs100q/graphs_100q.pt',weights_only=False)
g,_=attach_labels_and_noisy_exact(g,'runs100q/labels_100q.csv')
step=lambda x:int(x.circuit_path.split('step_')[1][:2]); J=lambda x:int(x.circuit_path.split('_J')[1].split('.pk')[0])
tr=[x for x in g if step(x)<=7 and J(x)<10]; te=[x for x in g if step(x)>=8]
run('100q_heldout', DataLoader(tr,batch_size=4,shuffle=True), DataLoader(te,batch_size=4,shuffle=False),
    dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,head='residual'),
    dict(loss='smooth_l1',huber_beta=0.05,select_on='mae'))
del g,tr,te

# ---- benchmark datasets, both regimes ----
for name in ['mbd_theta_0p1pi_coherent','haoran_mbd_random_cliffords',
             'ising_init_from_qasm_coherent','ising_dataset']:
    print(f'=== {name} ===',flush=True)
    R=f'runs/{name}'
    tr,M=attach_labels_and_noisy_exact(load_graphs(f'{R}/processed/graphs_train.pt'),f'{R}/labels_train.csv')
    va,_=attach_labels_and_noisy_exact(load_graphs(f'{R}/processed/graphs_val.pt'),f'{R}/labels_val.csv')
    lv=LevelSet.detect(torch.cat([x.y for x in tr]))
    base=dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,
              **(dict(head='level',levels=lv) if lv else dict(head='residual')))
    tc=dict(loss='ce',select_on='acc') if lv else dict(loss='smooth_l1',huber_beta=0.05,select_on='mae')
    run(name, DataLoader(tr,batch_size=16,shuffle=True), DataLoader(va,batch_size=16,shuffle=False), base, tc, lv)
    del tr,va
print('CONDITIONING ABLATION COMPLETE')
