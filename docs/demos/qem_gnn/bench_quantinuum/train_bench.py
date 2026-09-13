"""Train QAGT-MLP on the Placidi et al. Pauli benchmark and score it their way.

Protocol matched to `run_pretrained_baseline.py`, which produced the numbers in
`outputs/results/{device}_pretrained.json`: version 0, the full test split, and
the same `score_mitigation` code, imported from their repository rather than
reimplemented, so the comparison cannot drift through a redefined metric.

Training uses their objective and their direction. Note the two are not the
same direction: `Pipeline/models/lightning_perceiver_threshold_losses.py`
computes `loss_type="kl"` as `_kl_vec(ideal, pred)`, i.e. forward
KL(ideal || pred), which is mass-covering, while `mitigation.score_mitigation`
reports `kl_divergence(mitigated, ideal)`, the reverse direction. Their code
offers "reverse_kl" as a separate option, so the asymmetry is deliberate.

Training on the reported direction does not work: reverse KL is zero-forcing and
is minimised by collapsing onto a single high-probability bitstring. We measured
that -- a model trained that way reached mean predictive entropy 0.16 against
the ideal 1.09, put 0.96 of its mass on one outcome, and scored a competitive
mean KL of 4.09 while making total variation (0.76 vs 0.53 unmitigated) and the
per-circuit relative change worse than doing nothing. We therefore train on
forward KL, as they do, and score with their metric unchanged.

Early stopping is on validation forward KL, matching their selection criterion.

    python bench_quantinuum/train_bench.py --device algiers --seeds 3
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/root/stousi_missouri.edu_01M26D5DP6FR5334PATFM1NQMK/data_preparation')
from random_circuits.mitigation import score_mitigation      # their scorer, verbatim
from bench_quantinuum.model_dist import QAGTDistribution

EPS = 1e-12


def kl_loss(logits, ideal):
    """Mean forward KL(ideal || pred), their training objective."""
    logp = torch.log_softmax(logits, -1)
    q = ideal.clamp_min(EPS)
    return (q * (torch.log(q) - logp)).sum(-1).mean()


@torch.no_grad()
def predict(net, loader, dev):
    net.eval()
    P, N, I = [], [], []
    for b in loader:
        b = b.to(dev)
        p = torch.softmax(net(b), -1)
        P.append(p.cpu().numpy())
        N.append(b.noisy_dist.view(p.shape[0], -1).cpu().numpy())
        I.append(b.ideal_dist.view(p.shape[0], -1).cpu().numpy())
    return np.concatenate(P), np.concatenate(N), np.concatenate(I)


def run_seed(tr, va, seed, args, dev='cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    net = QAGTDistribution(pool=args.pool, backbone=args.backbone,
                           dropout=args.dropout, d_model=args.d_model,
                           desc_dim=args.desc_dim,
                           use_local=not args.no_local,
                           use_global=not args.no_global).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_state, bad = float('inf'), None, 0
    for ep in range(args.epochs):
        net.train()
        for b in tr:
            b = b.to(dev)
            opt.zero_grad()
            loss = kl_loss(net(b), b.ideal_dist.view(b.num_graphs, -1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        P, _, I = predict(net, va, dev)
        # forward KL(ideal || pred), the quantity trained on
        vkl = float(np.mean(np.sum((I + EPS) * np.log((I + EPS) / (P + EPS)), -1)))
        if vkl < best - 1e-6:
            best, bad = vkl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f'    seed {seed} ep {ep:3d}  val KL {vkl:.4f}  best {best:.4f}', flush=True)
        if bad >= args.patience:
            break
    net.load_state_dict(best_state)
    return net, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='algiers')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--pool', default='conditioned')
    p.add_argument('--backbone', default='transformer')
    p.add_argument('--desc-dim', type=int, default=101,
                   help='backend calibration block; 0 disables it')
    p.add_argument('--no-local', action='store_true',
                   help='drop the per-qubit pooled context')
    p.add_argument('--no-global', action='store_true',
                   help='drop the graph-wide pooled context')
    p.add_argument('--tag', default='qagt')
    a = p.parse_args()

    B = 'bench_quantinuum'
    load = lambda s: torch.load(f'{B}/graphs_{a.device}_{s}_v0.pt', weights_only=False)
    g_tr, g_va, g_te = load('train'), load('val'), load('test')
    print(f'{a.device}: train {len(g_tr)}  val {len(g_va)}  test {len(g_te)}', flush=True)
    tr = DataLoader(g_tr, batch_size=a.batch_size, shuffle=True)
    va = DataLoader(g_va, batch_size=512)
    te = DataLoader(g_te, batch_size=512)

    out_path = f'{B}/results_{a.device}_{a.tag}.json'
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    per_seed, preds = [], []
    t0 = time.time()
    for s in range(a.seeds):
        net, vkl = run_seed(tr, va, s, a)
        P, N, I = predict(net, te, 'cuda')
        sc = score_mitigation(P, N, I)
        per_seed.append(sc); preds.append(P)
        print(f'  seed {s}: test KL {sc["kl"]:.4f}  tv {sc["tv"]:.4f}  '
              f'hell {sc["hellinger"]:.4f}  rel {sc["kl_rel_change"]:+.4f}  '
              f'[{time.time()-t0:.0f}s]', flush=True)
        torch.save(net.state_dict(), f'{B}/{a.device}_{a.tag}_seed{s}.pt')

    ens = score_mitigation(np.mean(preds, 0), N, I)
    res[a.tag] = {
        'config': vars(a),
        'per_seed': [{k: v for k, v in s.items() if isinstance(v, float)} for s in per_seed],
        'mean': {k: float(np.mean([s[k] for s in per_seed]))
                 for k in ('kl', 'tv', 'hellinger', 'rmse', 'kl_rel_change',
                           'tv_rel_change', 'hellinger_rel_change', 'fraction_improved')},
        'std': {k: float(np.std([s[k] for s in per_seed]))
                for k in ('kl', 'tv', 'hellinger', 'kl_rel_change')},
        'ensemble': {k: v for k, v in ens.items() if isinstance(v, float)},
        'unmitigated': {k: float(v) for k, v in
                        score_mitigation(N, N, I).items() if isinstance(v, float)},
    }
    json.dump(res, open(out_path, 'w'), indent=1)
    m, sd = res[a.tag]['mean'], res[a.tag]['std']
    print(f"\n{a.device} {a.tag}: KL {m['kl']:.4f}+/-{sd['kl']:.4f}  "
          f"TV {m['tv']:.4f}  Hell {m['hellinger']:.4f}  "
          f"rel {m['kl_rel_change']:+.4f}  ens KL {ens['kl']:.4f}")
    print('wrote', out_path)


if __name__ == '__main__':
    main()
