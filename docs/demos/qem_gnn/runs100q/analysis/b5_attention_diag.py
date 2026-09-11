"""Why is the attention-backbone main effect zero?

Hypothesis: it is not a statement about attention, but about the graph. Edges
are built forward-only along each qubit wire, so 85.6% of nodes have exactly
one incoming edge and 90.4% have at most one. A softmax over a single element
is identically 1.0, so on those nodes TransformerConv cannot express an
attention pattern at all -- it reduces to W_v h_pred + W_skip h_self, which is
the same span GCNConv covers with fixed weights.

Test: rerun transformer-vs-GCN unchanged, then rerun with the same graphs made
undirected, which raises the mean neighbourhood size. Masks are per-node and
unaffected. If the backbone effect appears only in the second condition, the
null is a property of the graph construction.

Same split, epochs, seeds and hyperparameters as b4_factorial.py.
"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from sim_data_utils import attach_labels_and_noisy_exact
from qem_train import TrainConfig, train_qem, collect

SEEDS=(0,1,2); OUT='runs100q/analysis/b5_attention.json'
g=torch.load('runs100q/graphs_100q.pt',weights_only=False)
g,_=attach_labels_and_noisy_exact(g,'runs100q/labels_100q.csv')
step=lambda x:int(x.circuit_path.split('step_')[1][:2]); J=lambda x:int(x.circuit_path.split('_J')[1].split('.pk')[0])

def indeg_stats(graphs):
    c=np.concatenate([torch.bincount(d.edge_index[1],minlength=d.num_nodes).numpy() for d in graphs[:20]])
    return dict(mean=float(c.mean()), frac_le1=float((c<=1).mean()), frac_ge3=float((c>=3).mean()))

base=dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,head='residual',pool='conditioned')
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
t0=time.time()

for cond in ('directed','undirected'):
    if cond=='undirected':
        for d in g:
            d.edge_index = to_undirected(d.edge_index, num_nodes=d.num_nodes)
    stats=indeg_stats(g)
    print(f"\n[{cond}] mean in-degree {stats['mean']:.2f}  "
          f"<=1: {100*stats['frac_le1']:.1f}%  >=3: {100*stats['frac_ge3']:.1f}%", flush=True)
    tr=[x for x in g if step(x)<=7 and J(x)<10]; te=[x for x in g if step(x)>=8]
    trl=DataLoader(tr,batch_size=4,shuffle=True); tel=DataLoader(te,batch_size=4,shuffle=False)
    for bb in ('transformer','gcn'):
        key=f'{cond}/{bb}'
        if key in res: print('  skip',key,flush=True); continue
        per=[]
        for s in SEEDS:
            tc=TrainConfig(epochs=100,lr=1e-3,wd=1e-4,patience=30,loss='smooth_l1',huber_beta=0.05,
                           scheduler='cosine',warmup_epochs=5,select_on='mae',seed=s,verbose=False)
            net,_=train_qem(dict(base,backbone=bb),(trl,tel),tc,
                            best_ckpt_path=f'runs100q/analysis/_b5_{s}.pt')
            raw,y,_=collect(net,tel,'cuda')
            per.append(float(np.abs(raw.numpy()-y.numpy()).mean()))
        res[key]=dict(cond=cond, backbone=bb, indeg=stats, per_seed=per,
                      mean=float(np.mean(per)), std=float(np.std(per)))
        json.dump(res,open(OUT,'w'),indent=1)
        print(f"  {key:24s} MAE {np.mean(per):.4f} +/- {np.std(per):.4f}   [{time.time()-t0:.0f}s]",flush=True)

for c in ('directed','undirected'):
    a=res.get(f'{c}/transformer'); b=res.get(f'{c}/gcn')
    if a and b:
        print(f"\n{c:11s} attention effect (gcn - transformer) = {b['mean']-a['mean']:+.5f}")
for f in glob.glob('runs100q/analysis/_b5_*.pt'): os.remove(f)
print('B5 COMPLETE')
