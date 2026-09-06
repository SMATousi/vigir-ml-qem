"""Generate circuit families with a tunable degree of causal locality.

Motivation: on the real transpiled 100q data the backward lightcone covers
>95% of the graph even at one Trotter step, and the M cones overlap with
Jaccard ~1.0, so the lightcone mask has nothing to discriminate. To test
whether that is a property of the mask or of the workload, we generate
circuits whose causal locality we control directly.

Construction: partition n qubits into disjoint blocks of size B. Two-qubit
gates act only within a block, so the backward lightcone of an observable is
exactly its own block. One observable is placed per block (cycling if there
are fewer blocks than observables). Sweeping B from 2 to n moves the cone
coverage from ~B/n to 1 and the pairwise Jaccard from 0 to 1.

This is not an artificial construction: running several small independent
circuits side by side on one large device is standard practice for amortising
queue time, and the resulting workload has exactly this block structure.

No transpilation is applied -- routing would reintroduce coupling between
blocks, which is precisely what destroys locality in the real data.
"""
import os, sys, json, pickle, argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

N_QUBITS = 12
M_OBS = 4
DEPTH = 6
SHOTS = 2000


def noise_model(p1=0.001, p2=0.02, p_ro=0.02):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['rz', 'sx', 'x'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def make_circuit(rng, block_size):
    """Random circuit whose 2q gates never cross a block boundary."""
    n = N_QUBITS
    blocks = [list(range(s, min(s + block_size, n))) for s in range(0, n, block_size)]
    qc = QuantumCircuit(n, M_OBS)
    for q in range(n):                      # random product initial state
        qc.ry(float(rng.uniform(0, np.pi)), q)
    for _ in range(DEPTH):
        for b in blocks:
            if len(b) < 2:
                continue
            order = list(b); rng.shuffle(order)
            for a, c in zip(order[::2], order[1::2]):
                qc.cx(a, c)
        for q in range(n):
            qc.rz(float(rng.uniform(-np.pi, np.pi)), q)
            qc.sx(q)
    # M distinct observables, spread over blocks as evenly as possible: one per
    # block round-robin, taking a different qubit within a block on each pass so
    # the targets are never the same qubit twice
    obs, used = [], {}
    for i in range(M_OBS):
        b = blocks[i % len(blocks)]
        k = used.get(i % len(blocks), 0)
        obs.append(b[(k * max(1, len(b) // M_OBS)) % len(b)])
        used[i % len(blocks)] = k + 1
    assert len(set(obs)) == M_OBS, f'observables not distinct: {obs} (block_size={block_size})'
    qc.barrier()
    for i, q in enumerate(obs):
        qc.measure(q, i)
    return qc, obs


def ideal_z(qc, obs):
    """Exact <Z_o> per observable, via single-qubit marginals of the statevector."""
    q = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(q)
    out = []
    for o in obs:
        p0, p1 = sv.probabilities([o])   # marginal over qubit o, vectorised
        out.append(float(p0 - p1))
    return out


def noisy_z(sim, qc, nm):
    r = sim.run(qc, noise_model=nm, shots=SHOTS).result().get_counts()
    tot = sum(r.values()); out = []
    for i in range(M_OBS):
        e = sum(c * (1 - 2 * int(b.replace(' ', '')[::-1][i])) for b, c in r.items()) / tot
        out.append(float(e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block-sizes', type=int, nargs='+', default=[2, 3, 4, 6, 12])
    ap.add_argument('--n-circuits', type=int, default=1500)
    ap.add_argument('--out', default='../../tutorials/data/locality_sweep')
    a = ap.parse_args()
    sim = AerSimulator(); nm = noise_model()
    for B in a.block_sizes:
        d = os.path.join(a.out, f'block{B:02d}')
        os.makedirs(d, exist_ok=True)
        rng = np.random.default_rng(1000 + B)
        entries = []
        for i in range(a.n_circuits):
            qc, obs = make_circuit(rng, B)
            entries.append(dict(circuit=qc, ideal_exp_value=ideal_z(qc, obs),
                                noisy_exp_values=[noisy_z(sim, qc, nm)],
                                observable=obs, block_size=B))
            if (i + 1) % 250 == 0:
                print(f'  B={B}  {i+1}/{a.n_circuits}', flush=True)
        # split into shards so the loader's os.walk sees several files
        per = len(entries) // 6
        for s in range(6):
            with open(os.path.join(d, f'shard_{s}.pk'), 'wb') as f:
                pickle.dump(entries[s * per:(s + 1) * per], f)
        iv = np.array([e['ideal_exp_value'] for e in entries])
        nv = np.array([e['noisy_exp_values'][0] for e in entries])
        print(f'B={B}: {len(entries)} circuits | unmitigated MAE {np.abs(nv-iv).mean():.4f} '
              f'| |ideal| {np.abs(iv).mean():.3f}', flush=True)
    print('GENERATION COMPLETE')


if __name__ == '__main__':
    main()
