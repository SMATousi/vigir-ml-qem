"""100q Brisbane: paper-config repro under the corrected lightcone, vs the corrected model, vs RF."""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from torch_geometric.loader import DataLoader
from sim_data_utils import attach_labels_and_noisy_exact
from qem_train import TrainConfig, train_qem, collect, metrics
from mlp import encode_data_v2_ecr
from sklearn.ensemble import RandomForestRegressor

OUT='runs100q'; M=5
graphs=torch.load(f'{OUT}/graphs_100q.pt', weights_only=False)
graphs,_=attach_labels_and_noisy_exact(graphs, f'{OUT}/labels_100q.csv')
sp=json.load(open(f'{OUT}/split_100q.json')); tr_set,te_set=set(sp['train']),set(sp['test'])
tr=[g for g in graphs if g.circuit_path in tr_set]; te=[g for g in graphs if g.circuit_path in te_set]
print(f'train {len(tr)} test {len(te)} | in_dim {tr[0].x.shape[1]}', flush=True)
trl=DataLoader(tr,batch_size=4,shuffle=True); tel=DataLoader(te,batch_size=4,shuffle=False)

res={}
def evaluate(net, tag):
    raw,y,n=collect(net,tel,'cuda')
    p=net.decode(raw,'raw').numpy(); y=y.numpy(); n=n.numpy()
    res[tag]=dict(mae=float(np.abs(p-y).mean()), rmse=float(np.sqrt(((p-y)**2).mean())),
                  per_qubit_mae=np.abs(p.reshape(-1,M)-y.reshape(-1,M)).mean(0).tolist(),
                  params=sum(q.numel() for q in net.parameters()))
    print(f'{tag}: MAE {res[tag]["mae"]:.4f} RMSE {res[tag]["rmse"]:.4f} params {res[tag]["params"]}', flush=True)

# --- A: the paper's exact configuration, only the lightcone corrected ---
print('\n=== A: paper config (shared pooling, scalar head, SmoothL1 beta=1, constant lr) ===', flush=True)
netA,_=train_qem(dict(d_model=128,layers=3,heads=4,dropout=0.1,use_noisy=True,head='scalar',pool='shared'),
                 (trl,tel), TrainConfig(epochs=100,lr=1e-3,wd=1e-4,patience=30,seed=0,verbose=False),
                 best_ckpt_path=f'{OUT}/best_A.pt')
evaluate(netA,'A_paper_config_fixed_lightcone')

# --- B: corrected model ---
print('\n=== B: conditioned pooling + residual head + Tier1 training ===', flush=True)
cfgB=dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,head='residual',pool='conditioned')
tcB=TrainConfig(epochs=100,lr=1e-3,wd=1e-4,patience=30,loss='smooth_l1',huber_beta=0.05,
                scheduler='cosine',warmup_epochs=5,select_on='mae',verbose=False)
preds=[]
for s in range(5):
    c=TrainConfig(**{**tcB.__dict__,'seed':s})
    net,_=train_qem(cfgB,(trl,tel),c,best_ckpt_path=f'{OUT}/best_B_s{s}.pt')
    raw,y,n=collect(net,tel,'cuda'); preds.append(net.decode(raw,'raw').numpy())
    if s==0: evaluate(net,'B_conditioned_seed0'); Y=y.numpy(); N=n.numpy()
    print(f'  seed {s} done', flush=True)
P=np.mean(preds,0)
res['B_conditioned_5seed']=dict(mae=float(np.abs(P-Y).mean()), rmse=float(np.sqrt(((P-Y)**2).mean())),
    per_qubit_mae=np.abs(P.reshape(-1,M)-Y.reshape(-1,M)).mean(0).tolist())
print(f'B 5-seed: MAE {res["B_conditioned_5seed"]["mae"]:.4f} RMSE {res["B_conditioned_5seed"]["rmse"]:.4f}', flush=True)

# --- noisy + RF baselines on the same split ---
res['noisy']=dict(mae=float(np.abs(N-Y).mean()), rmse=float(np.sqrt(((N-Y)**2).mean())),
                  per_qubit_mae=np.abs(N.reshape(-1,M)-Y.reshape(-1,M)).mean(0).tolist())
print(f'noisy: MAE {res["noisy"]["mae"]:.4f}', flush=True)

def circs(paths):
    out=[]
    for p in paths: out.append(pickle.load(open(p.split('::#')[0],'rb'))[0]['circuit'])
    return out
lab=pd.read_csv(f'{OUT}/labels_100q.csv').set_index('circuit_path')
def yn(keys): return ([json.loads(lab.loc[k,'target_y_json']) for k in keys],
                      [json.loads(lab.loc[k,'noisy_z_json']) for k in keys])
trk=[g.circuit_path for g in tr]; tek=[g.circuit_path for g in te]
tri,trn=yn(trk); tei,ten=yn(tek)
Xtr,ytr=encode_data_v2_ecr(circs(trk),tri,trn,obs_size=M)
Xte,yte=encode_data_v2_ecr(circs(tek),tei,ten,obs_size=M)
rf=np.stack([RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=0)
             .fit(Xtr.numpy(),ytr.numpy()[:,q]).predict(Xte.numpy()) for q in range(M)],1)
yv=yte.numpy()
res['RF']=dict(mae=float(np.abs(rf-yv).mean()), rmse=float(np.sqrt(((rf-yv)**2).mean())),
               per_qubit_mae=np.abs(rf-yv).mean(0).tolist())
print(f'RF: MAE {res["RF"]["mae"]:.4f} RMSE {res["RF"]["rmse"]:.4f}', flush=True)

json.dump(res, open(f'{OUT}/results_100q.json','w'), indent=2)
print('\n=== SUMMARY ===')
for k,v in res.items(): print(f'  {k:34s} MAE {v["mae"]:.4f}  RMSE {v["rmse"]:.4f}')
