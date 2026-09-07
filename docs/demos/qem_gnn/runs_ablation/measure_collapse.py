"""Re-measure the pooled-vector collapse on barrier-free lightcones.

Table 4's mechanism numbers (cosine 1.0000, encoder constant) were computed
with contaminated masks. This retrains shared vs conditioned pooling on the
dataset where the conditioning gap is largest and re-measures.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, itertools
from torch_geometric.loader import DataLoader
from data_utils import load_graphs
from sim_data_utils import attach_labels_and_noisy_exact
from qem_train import TrainConfig, train_qem, collect

R='runs/ising_init_from_qasm_coherent'
tr,M=attach_labels_and_noisy_exact(load_graphs(f'{R}/processed/graphs_train.pt'),f'{R}/labels_train.csv')
va,_=attach_labels_and_noisy_exact(load_graphs(f'{R}/processed/graphs_val.pt'),f'{R}/labels_val.csv')
trl=DataLoader(tr,batch_size=16,shuffle=True); val=DataLoader(va,batch_size=16,shuffle=False)
base=dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,head='residual')
out={}
for name,kw in [('shared query',dict(pool='shared')),('qubit-conditioned',dict(pool='conditioned'))]:
    tc=TrainConfig(epochs=80,lr=1e-3,wd=1e-4,patience=25,loss='smooth_l1',huber_beta=0.05,
                   scheduler='cosine',warmup_epochs=5,select_on='mae',seed=0,verbose=False)
    net,_=train_qem(dict(base,**kw),(trl,val),tc,best_ckpt_path=f'runs_ablation/_mc.pt')
    raw,y,_=collect(net,val,'cuda'); mae=float(np.abs(raw.numpy()-y.numpy()).mean())
    b=next(iter(DataLoader(va,batch_size=128,shuffle=False))).to('cuda'); net.eval()
    with torch.no_grad():
        H=net.encoder(b.x,b.edge_index)
        P,_=(net.pool(H,b.lightcone_masks,b.ptr,measured_qubits=b.measured_qubits,
                      noisy_z=b.noisy_z,wire_masks=b.wire_masks) if kw['pool']=='conditioned'
             else net.pool(H,b.lightcone_masks,b.ptr))
    P=P.view(-1,M,P.shape[-1]).cpu().numpy()
    Pn=P/(np.linalg.norm(P,axis=-1,keepdims=True)+1e-9)
    cos=float(np.mean([(Pn[:,i]*Pn[:,j]).sum(-1).mean() for i,j in itertools.combinations(range(M),2)]))
    out[name]=dict(mae=mae, cosine=cos,
                   std_across_qubits=float(P.std(1).mean()),
                   std_across_circuits=float(P.reshape(-1,P.shape[-1]).std(0).mean()))
    print(f"  {name:20s} MAE {mae:.4f} | cosine {cos:.4f} | std across qubits {out[name]['std_across_qubits']:.4f} "
          f"| std across circuits {out[name]['std_across_circuits']:.4f}", flush=True)
os.path.exists('runs_ablation/_mc.pt') and os.remove('runs_ablation/_mc.pt')
json.dump(out, open('runs_ablation/collapse.json','w'), indent=1)
print('COLLAPSE MEASURED')
