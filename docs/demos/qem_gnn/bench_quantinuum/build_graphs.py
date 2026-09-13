"""Turn the Placidi et al. Pauli benchmark into QAGT-MLP circuit graphs.

Their datasets ship circuits as a padded array of shape ``[T, 5, 5]`` per
circuit: T timesteps, 5 qubits, 5 channels. We verified the semantics against
the released flat encoding ``data_inputs_version0.npy``, which is exactly
``concat(circuit.reshape(T, 25), backend_info)`` and matches index for index:

  channels 0, 1, 2, 4  single-qubit; never more than one qubit set jointly with
                       channel 3 on the same cell (checked: 0 co-occurrences).
                       Channel 4 is the measurement and closes each wire.
  channel 3            two-qubit gates. Nonzero entries always come in pairs on
                       *adjacent* qubits with opposite signs, i.e. a 1D chain.

That is everything the graph needs. We do not have to name the gates: the node
carries the raw channel values, and the topology comes from the pairing.

Nodes are operations, as in `circuits_to_graph.py`: a two-qubit gate is one node
on both wires, a single-qubit gate is one node on its own. Edges run forward
between consecutive operations on a wire, so the causal lightcone and wire masks
are the same objects the model uses everywhere else.

    python bench_quantinuum/build_graphs.py --device algiers --split test
"""
import argparse, os, sys, time
import numpy as np
import torch
from torch_geometric.data import Data

DATA_ROOT = os.environ.get(
    'QEM_BENCH_ROOT',
    '/root/stousi_missouri.edu_01M26D5DP6FR5334PATFM1NQMK/DATA/pauli')
NQ, NCH, DEPTH_DIR = 5, 5, 't_3_4_5_6_9'
FEAT_DIM = 20


def _moment_bits(m: int, width: int = 8) -> list:
    return [float((m >> b) & 1) for b in range(width)]


def circuit_to_graph(circ: np.ndarray) -> Data:
    """[T, 5, 5] -> gate graph with lightcone and wire masks."""
    T = circ.shape[0]
    active = (circ != 0).any(-1)                       # [T, 5]
    ops = []                                           # (moment, qubits, feats)
    for t in range(T):
        if not active[t].any():
            continue
        two = np.nonzero(circ[t, :, 3])[0]
        used = set()
        for a, b in zip(two[::2], two[1::2]):          # verified: adjacent pairs
            ops.append((t, (int(a), int(b)), circ[t, [a, b]].max(0)))
            used |= {int(a), int(b)}
        for q in np.nonzero(active[t])[0]:
            if int(q) in used:
                continue
            ops.append((t, (int(q),), circ[t, int(q)].copy()))

    n = len(ops)
    if n == 0:
        raise ValueError('empty circuit')

    # node features: 5 channels, arity, qubit one-hot, moment bits, moment frac
    x = np.zeros((n, FEAT_DIM), dtype=np.float32)
    max_moment = max(o[0] for o in ops) or 1
    for i, (t, qs, ch) in enumerate(ops):
        x[i, :NCH] = ch
        x[i, NCH] = len(qs)
        for q in qs:
            x[i, NCH + 1 + q] = 1.0
        x[i, NCH + 1 + NQ:NCH + 1 + NQ + 8] = _moment_bits(t)
        x[i, -1] = t / max_moment

    # wire adjacency, forward in time
    by_wire = {q: [] for q in range(NQ)}
    for i, (_, qs, _) in enumerate(ops):
        for q in qs:
            by_wire[q].append(i)
    src, dst = [], []
    for q in range(NQ):
        seq = by_wire[q]
        src += seq[:-1]
        dst += seq[1:]
    edge_index = (np.array([src, dst], dtype=np.int64) if src
                  else np.zeros((2, 0), dtype=np.int64))

    preds = {i: [] for i in range(n)}
    for s, d in zip(src, dst):
        preds[d].append(s)

    wire = np.zeros((n, NQ), dtype=np.float32)
    cone = np.zeros((n, NQ), dtype=np.float32)
    for q in range(NQ):
        for i in by_wire[q]:
            wire[i, q] = 1.0
        if not by_wire[q]:
            continue
        stack, seen = [by_wire[q][-1]], set()
        while stack:                                   # backward closure
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(preds[i])
        for i in seen:
            cone[i, q] = 1.0

    d = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index))
    d.num_nodes = n
    d.lightcone_masks = torch.from_numpy(cone)
    d.wire_masks = torch.from_numpy(wire)
    d.measured_qubits = torch.arange(NQ, dtype=torch.long)
    d.num_measured = NQ
    return d


def marginals(dist: np.ndarray) -> np.ndarray:
    """P(qubit m = 1) for each of the 5 qubits, from a [., 32] distribution."""
    idx = np.arange(dist.shape[-1])
    bits = np.stack([((idx >> m) & 1).astype(np.float64) for m in range(NQ)], 1)
    return dist @ bits


def build(device: str, split: str, version: int, out: str, limit=None):
    root = f'{DATA_ROOT}/{device}_pauli_real_{split}/{DEPTH_DIR}'
    ideal = np.load(f'{root}/mixed_tf/data_ideal_outputs_version{version}.npy')
    noisy = np.load(f'{root}/mixed_tf/data_noisy_outputs_version{version}.npy')
    n = len(ideal) if limit is None else min(limit, len(ideal))
    marg = marginals(noisy[:n])
    graphs, t0 = [], time.time()
    for i in range(n):
        circ = np.load(f'{root}/mixed/{i}/input_circuit_info_array_0.npy')
        g = circuit_to_graph(circ)
        g.noisy_z = torch.tensor(marg[i], dtype=torch.float)
        g.noisy_dist = torch.tensor(noisy[i], dtype=torch.float).view(1, -1)
        g.ideal_dist = torch.tensor(ideal[i], dtype=torch.float).view(1, -1)
        g.index = i
        graphs.append(g)
        if (i + 1) % 5000 == 0:
            print(f'  {i+1}/{n}  {time.time()-t0:.0f}s', flush=True)
    torch.save(graphs, out)
    nn = np.array([g.num_nodes for g in graphs])
    print(f'saved {out}: {len(graphs)} graphs, nodes mean {nn.mean():.1f} '
          f'min {nn.min()} max {nn.max()}, {time.time()-t0:.0f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='algiers')
    p.add_argument('--split', default='test')
    p.add_argument('--version', type=int, default=0)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    out = a.out or f'bench_quantinuum/graphs_{a.device}_{a.split}_v{a.version}.pt'
    build(a.device, a.split, a.version, out, a.limit)
