"""Component ablation repeated in the settings where structure should matter most.

B3 on the 100q random split found no component carrying the result, and a GCN
backbone beating the graph transformer. That split is interpolation on a
dataset whose observables contract with depth. This repeats the ablation where
the margin over circuit-level features is largest:

  setting A  100q, held-out Trotter steps (train 1-7, test 8-10)   -- B2 showed 2.8x
  setting B  mbd_theta_0p1pi_coherent, ising_init_from_qasm_coherent
             -- the two 5-qubit datasets where QAGT-MLP leads RF by the most
"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch_geometric.loader import DataLoader
from data_utils import load_graphs
from sim_data_utils import attach_labels_and_noisy_exact
from qem_levels import LevelSet
from qem_train import TrainConfig, train_qem, collect, metrics

SEEDS=(0,1,2,3,4)
OUT='runs100q/analysis/b3b_hard.json'
VARIANTS=[('Full',{}), ('No lightcone',dict(use_lightcone=False)), ('No wire',dict(use_wire=False)),
          ('No global',dict(use_global=False)), ('No attention (GCN)',dict(backbone='gcn'))]
res=json.load(open(OUT)) if os.path.exists(OUT) else {}

def run(setting, trl, val, base, tc, levels=None):
    res.setdefault(setting,{})
    for name,kw in VARIANTS:
        if name in res[setting]: continue
        P=[]; per=[]
        for s in SEEDS:
            cfg=TrainConfig(epochs=100,lr=1e-3,wd=1e-4,patience=30,scheduler='cosine',
                            warmup_epochs=5,seed=s,verbose=False,**tc)
            net,_=train_qem(dict(base,**kw),(trl,val),cfg,best_ckpt_path=f'runs100q/analysis/_hb_{s}.pt')
            raw,y,_=collect(net,val,'cuda')
            P.append(torch.softmax(raw.float(),-1).numpy() if net.is_classifier else raw.numpy())
            d=net.decode(raw,'argmax' if net.is_classifier else 'raw').numpy()
            per.append(float(np.abs(d-y.numpy()).mean()))
        p=np.mean(P,0); Y=y.numpy()
        pred=levels.argmax(torch.log(torch.tensor(p).clamp_min(1e-12))).numpy() if levels is not None else p
        m=metrics(pred,Y,levels)
        m.update(per_seed_mae=per, seed_mean=float(np.mean(per)), seed_std=float(np.std(per)))
        res[setting][name]=m
        json.dump(res,open(OUT,'w'),indent=1)
        print(f"  [{setting}] {name:20s} ens {m['mae']:.4f} | per-seed {m['seed_mean']:.4f} +/- {m['seed_std']:.4f}",flush=True)
    for f in glob.glob('runs100q/analysis/_hb_*.pt'): os.remove(f)

# ---------- setting A: 100q held-out steps ----------
print('=== A: 100q held-out Trotter steps (train 1-7, test 8-10) ===',flush=True)
g=torch.load('runs100q/graphs_100q.pt',weights_only=False)
g,_=attach_labels_and_noisy_exact(g,'runs100q/labels_100q.csv')
step=lambda x:int(x.circuit_path.split('step_')[1][:2]); J=lambda x:int(x.circuit_path.split('_J')[1].split('.pk')[0])
tr=[x for x in g if step(x)<=7 and J(x)<10]; te=[x for x in g if step(x)>=8]
print(f'   train {len(tr)} test {len(te)}',flush=True)
run('100q_heldout_step',
    DataLoader(tr,batch_size=4,shuffle=True), DataLoader(te,batch_size=4,shuffle=False),
    dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,head='residual',pool='conditioned'),
    dict(loss='smooth_l1',huber_beta=0.05,select_on='mae'))
del g,tr,te

# ---------- setting B: hardest 5-qubit datasets ----------
for name in ['mbd_theta_0p1pi_coherent','ising_init_from_qasm_coherent']:
    print(f'=== B: {name} ===',flush=True)
    R=f'runs/{name}'
    tr,M=attach_labels_and_noisy_exact(load_graphs(f'{R}/processed/graphs_train.pt'),f'{R}/labels_train.csv')
    va,_=attach_labels_and_noisy_exact(load_graphs(f'{R}/processed/graphs_val.pt'),f'{R}/labels_val.csv')
    lv=LevelSet.detect(torch.cat([x.y for x in tr]))
    base=dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,pool='conditioned',
              **(dict(head='level',levels=lv) if lv else dict(head='residual')))
    tc=dict(loss='ce',select_on='acc') if lv else dict(loss='smooth_l1',huber_beta=0.05,select_on='mae')
    run(name, DataLoader(tr,batch_size=16,shuffle=True), DataLoader(va,batch_size=16,shuffle=False),
        base, tc, lv)
    del tr,va
print('HARD ABLATION COMPLETE')
