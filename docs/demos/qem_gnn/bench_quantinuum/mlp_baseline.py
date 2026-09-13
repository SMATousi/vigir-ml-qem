"""Circuit-blind control: predict the mitigated distribution from the measured one.

This model never sees the circuit. It takes only the 32 measured probabilities
and predicts a correction to them, trained and selected exactly like the graph
model and scored with the same `score_mitigation`. It exists because a learned
mitigator on this benchmark should have to beat the part of the task that is
solvable without any circuit information at all, and that floor is not reported
in the base paper.

    python bench_quantinuum/mlp_baseline.py --device algiers --seeds 3
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/root/stousi_missouri.edu_01M26D5DP6FR5334PATFM1NQMK/data_preparation')
from random_circuits.mitigation import score_mitigation

ROOT = ('/root/stousi_missouri.edu_01M26D5DP6FR5334PATFM1NQMK/DATA/pauli/'
        '%s_pauli_real_%s/t_3_4_5_6_9/mixed_tf')
EPS = 1e-12


def load(device, split, version=0):
    r = ROOT % (device, split)
    return (np.load(f'{r}/data_ideal_outputs_version{version}.npy'),
            np.load(f'{r}/data_noisy_outputs_version{version}.npy'))


def fwd_kl(logp, q):
    return (q * (torch.log(q.clamp_min(EPS)) - logp)).sum(-1).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='algiers')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--hidden', type=int, default=512)
    a = p.parse_args()
    dev = 'cuda'
    T = lambda x: torch.tensor(x, dtype=torch.float, device=dev)
    (ti, tn), (vi, vn), (si, sn) = (load(a.device, s) for s in ('train', 'val', 'test'))
    X, Y, XV, YV = T(tn), T(ti), T(vn), T(vi)
    XT = T(sn)

    per, preds = [], []
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(32, a.hidden), nn.GELU(),
                            nn.Linear(a.hidden, a.hidden), nn.GELU(),
                            nn.Linear(a.hidden, 32)).to(dev)
        nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, bad = float('inf'), None, 0
        for ep in range(a.epochs):
            net.train()
            perm = torch.randperm(len(X), device=dev)
            for i in range(0, len(X), 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                lg = torch.log_softmax(torch.log(X[b].clamp_min(EPS)) + net(X[b]), -1)
                fwd_kl(lg, Y[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(fwd_kl(torch.log_softmax(torch.log(XV.clamp_min(EPS)) + net(XV), -1), YV))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            P = torch.softmax(torch.log(XT.clamp_min(EPS)) + net(XT), -1).cpu().numpy()
        sc = score_mitigation(P.astype(np.float64), sn, si)
        per.append(sc); preds.append(P)
        print(f'  seed {seed}: KL {sc["kl"]:.4f}  tv {sc["tv"]:.4f}  hell {sc["hellinger"]:.4f}  '
              f'rel {sc["kl_rel_change"]:+.4f}  (val fwd KL {best:.4f})', flush=True)

    ens = score_mitigation(np.mean(preds, 0).astype(np.float64), sn, si)
    out = f'bench_quantinuum/results_{a.device}_mlp.json'
    json.dump({'mlp_noisy_only': {
        'per_seed': [{k: v for k, v in s.items() if isinstance(v, float)} for s in per],
        'mean': {k: float(np.mean([s[k] for s in per]))
                 for k in ('kl', 'tv', 'hellinger', 'rmse', 'kl_rel_change', 'fraction_improved')},
        'std': {k: float(np.std([s[k] for s in per])) for k in ('kl', 'tv', 'hellinger')},
        'ensemble': {k: v for k, v in ens.items() if isinstance(v, float)}}},
        open(out, 'w'), indent=1)
    m = {k: float(np.mean([s[k] for s in per])) for k in ('kl', 'tv', 'hellinger', 'kl_rel_change')}
    print(f"{a.device} MLP(noisy only): KL {m['kl']:.4f}  TV {m['tv']:.4f}  "
          f"Hell {m['hellinger']:.4f}  rel {m['kl_rel_change']:+.4f}  ens KL {ens['kl']:.4f}")
    print('wrote', out)


if __name__ == '__main__':
    main()
