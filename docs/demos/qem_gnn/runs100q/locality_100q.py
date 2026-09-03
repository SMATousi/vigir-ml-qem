"""Paper Table 1, recomputed on the real 100q graphs under both mask definitions."""
import sys, os, json, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

CIRC='../../tutorials/data/ising_zne_hardware/100q_brisbane'
MEAS=[0,1,2,3,4]
sp=json.load(open('runs100q/split_100q.json'))
paths=[p.split('::#')[0] for p in sp['train']]          # the training graphs, as the paper states

def masks_and_edges(qc, mode):
    dag=circuit_to_dag(qc); nodes=list(dag.op_nodes()); idx={n:i for i,n in enumerate(nodes)}; N=len(nodes)
    nbq={}
    for n in nodes:
        for q in [x.index for x in n.qargs]: nbq.setdefault(q,[]).append(n)
    E=[]
    for q,lst in nbq.items():
        for a,b in zip(lst[:-1],lst[1:]): E.append((idx[a],idx[b]))
    pf=(lambda n: []) if mode=='wire' else (lambda n: [p for p in dag.predecessors(n) if isinstance(p,DAGOpNode)])
    Ms=[]
    for q in MEAS:
        st=list(nbq.get(q,[])); seen=set()
        while st:
            n=st.pop()
            if n in seen: continue
            seen.add(n); st.extend(pf(n))
        m=np.zeros(N,bool); m[[idx[n] for n in seen]]=True; Ms.append(m)
    return np.array(Ms).T, E, N

for mode in ('wire','lightcone'):
    per=[[] for _ in MEAS]; per_i=[[] for _ in MEAS]; per_b=[[] for _ in MEAS]; jac=[]; Ns=[]
    for p in paths:
        qc=pickle.load(open(p,'rb'))[0]['circuit']
        Mk,E,N=masks_and_edges(qc,mode); Ns.append(N)
        for k in range(len(MEAS)):
            L=Mk[:,k]
            touch=[(u,v) for u,v in E if L[u] or L[v]]
            ins=sum(1 for u,v in touch if L[u] and L[v])
            per[k].append(L.mean()); per_i[k].append(ins/max(1,len(touch))); per_b[k].append(1-ins/max(1,len(touch)))
        jac += [float((Mk[:,a]&Mk[:,b]).sum()/max(1,(Mk[:,a]|Mk[:,b]).sum())) for a,b in itertools.combinations(range(len(MEAS)),2)]
    print(f"\n=== mode: {mode}  (mean N = {np.mean(Ns):.0f}, {len(paths)} training circuits) ===")
    print(f"{'qubit':6s} {'coverage':>9s} {'internal':>9s} {'boundary':>9s}")
    for k in range(len(MEAS)):
        print(f"q{k:<5d} {np.mean(per[k]):9.4f} {np.mean(per_i[k]):9.4f} {np.mean(per_b[k]):9.4f}")
    print(f"mean   {np.mean([np.mean(x) for x in per]):9.4f} {np.mean([np.mean(x) for x in per_i]):9.4f} "
          f"{np.mean([np.mean(x) for x in per_b]):9.4f}   pairwise Jaccard {np.mean(jac):.4f}")
print("\npaper Table 1:      0.0136    0.0452    0.9548   Jaccard 0.0840")
