"""B2: held-out-step extrapolation. Train on Trotter steps 1-7, test on 8-10.

The main split is random over circuits pooled across steps, which leaks
step-level structure and measures interpolation. This measures extrapolation.
Train size is held at 10 circuits/step (70 total) to stay comparable with the
main protocol's 10/step; the test set is every circuit from steps 8-10.
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, torch
from torch_geometric.loader import DataLoader
from sim_data_utils import attach_labels_and_noisy_exact
from qem_train import TrainConfig, train_qem, collect
from mlp import encode_data_v2_ecr
from sklearn.ensemble import RandomForestRegressor

OUT='runs100q'; M=5; TRAIN_STEPS=range(1,8); TEST_STEPS=range(8,11); K=10
graphs=torch.load(f'{OUT}/graphs_100q.pt', weights_only=False)
graphs,_=attach_labels_and_noisy_exact(graphs, f'{OUT}/labels_100q.csv')
step=lambda g:int(g.circuit_path.split('step_')[1][:2])
J   =lambda g:int(g.circuit_path.split('_J')[1].split('.pk')[0])
tr=[g for g in graphs if step(g) in TRAIN_STEPS and J(g)<K]
te=[g for g in graphs if step(g) in TEST_STEPS]
print(f'train {len(tr)} (steps 1-7)  test {len(te)} (steps 8-10)', flush=True)
trl=DataLoader(tr,batch_size=4,shuffle=True); tel=DataLoader(te,batch_size=4,shuffle=False)

res={}
cfg=dict(d_model=128,layers=3,heads=4,dropout=0.05,use_noisy=True,head='residual',pool='conditioned')
P=[]
for s in range(5):
    tc=TrainConfig(epochs=100,lr=1e-3,wd=1e-4,patience=30,loss='smooth_l1',huber_beta=0.05,
                   scheduler='cosine',warmup_epochs=5,select_on='mae',seed=s,verbose=False)
    net,_=train_qem(cfg,(trl,tel),tc,best_ckpt_path=f'{OUT}/analysis/_b2_s{s}.pt')
    raw,y,n=collect(net,tel,'cuda'); P.append(raw.numpy()); print(f'  seed {s} done',flush=True)
p=np.mean(P,0); y=y.numpy(); z=n.numpy()
f=lambda a: dict(mae=float(np.abs(a-y).mean()), rmse=float(np.sqrt(((a-y)**2).mean())),
                 per_qubit=np.abs(a.reshape(-1,M)-y.reshape(-1,M)).mean(0).tolist())
res['QAGT-MLP']=f(p); res['Unmitigated']=f(z)

lab=pd.read_csv(f'{OUT}/labels_100q.csv').set_index('circuit_path')
def enc(gs):
    cs=[pickle.load(open(g.circuit_path.split('::#')[0],'rb'))[0]['circuit'] for g in gs]
    yy=[json.loads(lab.loc[g.circuit_path,'target_y_json']) for g in gs]
    nn=[json.loads(lab.loc[g.circuit_path,'noisy_z_json']) for g in gs]
    return encode_data_v2_ecr(cs,yy,nn,obs_size=M)
Xtr,ytr=enc(tr); Xte,yte=enc(te)
rf=np.stack([RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=0)
             .fit(Xtr.numpy(),ytr.numpy()[:,q]).predict(Xte.numpy()) for q in range(M)],1)
yv=yte.numpy()
res['Random Forest']=dict(mae=float(np.abs(rf-yv).mean()), rmse=float(np.sqrt(((rf-yv)**2).mean())),
                          per_qubit=np.abs(rf-yv).mean(0).tolist())
# per-step breakdown on the held-out steps
st=np.array([step(g) for g in te])
res['per_step']={}
for s in TEST_STEPS:
    k=st==s
    res['per_step'][int(s)]=dict(
        noisy=float(np.abs(z.reshape(-1,M)[k]-y.reshape(-1,M)[k]).mean()),
        rf=float(np.abs(rf[k]-yv[k]).mean()),
        qagt=float(np.abs(p.reshape(-1,M)[k]-y.reshape(-1,M)[k]).mean()))
json.dump(res, open(f'{OUT}/analysis/b2_heldout.json','w'), indent=1)
for k in ['Unmitigated','Random Forest','QAGT-MLP']:
    print(f"{k:16s} MAE {res[k]['mae']:.4f}  RMSE {res[k]['rmse']:.4f}")
print('per-step:', json.dumps(res['per_step']))
