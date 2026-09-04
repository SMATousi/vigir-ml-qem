# qem_descriptors.py
"""Circuit-level descriptor features, for the graph-vs-descriptors ablation.

These are exactly the circuit-level half of `mlp.encode_data_v2_ecr` -- the
feature vector the Random Forest baseline of liao2024machine is built on:

    5   gate counts over {two_q_gate, sx, x, id, rz},  scaled by 0.01
    160 rotation-angle histogram bins of width 0.025*pi over [-2pi, 2pi], x0.01

`encode_data_v2_ecr` appends the noisy expectation values after these; the
graph model already receives those as `noisy_z`, so they are omitted here and
the descriptor vector is 165-dimensional.

The point of the ablation is that QAGT-MLP and the Random Forest currently see
disjoint information -- graph structure versus circuit-level summary
statistics. Attaching these lets us measure whether the graph contributes
anything the summary statistics do not, and whether the two compose.
"""
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

BIN_SIZE = 0.025 * np.pi
NUM_ANGLE_BINS = int(np.ceil(4 * np.pi / BIN_SIZE))
DESCRIPTOR_DIM = 5 + NUM_ANGLE_BINS  # 165


def _angle_histogram(circuit) -> np.ndarray:
    angles = [float(instr.params[0])
              for instr, qargs, _ in circuit.data
              if instr.name in ("rx", "ry", "rz") and len(qargs) == 1]
    edges = np.arange(-2 * np.pi, 2 * np.pi + BIN_SIZE, BIN_SIZE)
    counts, _ = np.histogram(angles, bins=edges)
    return counts.astype(np.float32)


def circuit_descriptors(circuit, two_q_gate: str = "cx") -> np.ndarray:
    """[165] descriptor vector for one circuit, matching encode_data_v2_ecr."""
    gate_set = [two_q_gate, "sx", "x", "id", "rz"]
    ops = circuit.count_ops()
    counts = np.array([ops.get(g, 0) for g in gate_set], dtype=np.float32)
    hist = _angle_histogram(circuit)
    # 0.01 scaling, as in encode_data_v2_ecr, to match the magnitude of the
    # expectation values the head also receives
    return np.concatenate([counts, hist]) * 0.01


def attach_descriptors_from_paths(graphs, two_q_gate: str = "cx") -> int:
    """Attach descriptors by loading each graph's own source circuit.

    Every graph records `circuit_path` as "<pickle file>::#<index>", so the
    circuit can be recovered without knowing which directory a dataset was
    staged from -- some are read from the raw tree, others from a flattened or
    QASM-converted cache.
    """
    import pickle
    cache = {}
    n = 0
    for g in graphs:
        key = g.circuit_path if isinstance(g.circuit_path, str) else g.circuit_path[0]
        fn, _, idx = key.partition("::#")
        if fn not in cache:
            cache[fn] = pickle.load(open(fn, "rb"))
        obj = cache[fn]
        entry = obj[int(idx)] if isinstance(obj, list) else obj
        qc = entry["circuit"] if isinstance(entry, dict) else entry
        g.descriptors = torch.from_numpy(circuit_descriptors(qc, two_q_gate)).float().view(1, -1)
        n += 1
    if n == 0:
        raise ValueError("no graphs to attach descriptors to")
    return n


def attach_descriptors(graphs, paths: Sequence[str], circuits: Iterable,
                       two_q_gate: str = "cx") -> int:
    """Attach `descriptors` [1, 165] to each graph, matched by circuit_path.

    Stored with a leading singleton dimension so PyG batching concatenates it
    to [B, 165] rather than flattening it.
    """
    by_path = {p: c for p, c in zip(paths, circuits)}
    n = 0
    for g in graphs:
        key = g.circuit_path if isinstance(g.circuit_path, str) else g.circuit_path[0]
        qc = by_path.get(key)
        if qc is None:
            continue
        d = circuit_descriptors(qc, two_q_gate)
        g.descriptors = torch.from_numpy(d).float().view(1, -1)
        n += 1
    if n == 0:
        raise ValueError("no graphs matched the supplied circuit paths")
    return n
