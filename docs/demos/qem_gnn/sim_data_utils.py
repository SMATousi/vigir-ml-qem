"""Adapters for the simulator-generated Ising datasets under docs/tutorials/data/
(ising_dataset, ising_init_0110, ising_init_from_qasm*, haoran_mbd*, mbd_datasets2).

Unlike the Brisbane hardware dataset, each circuit entry here already carries
its own ideal_exp_value / noisy_exp_values, so no ZNE-derivation step is
needed -- we just need a labels CSV that lines up with the circuit_path keys
dataset.py assigns when it builds the graphs.

NOTE: data_utils.attach_labels_and_noisy() (used by the Brisbane notebooks)
matches by trimming a fixed 4 characters off the graph's circuit_path
(stripping a "::#0"-style suffix). That only works when a pickle file holds a
single circuit. These datasets hold hundreds of circuits per file (indices
"::#0".."::#999"), so that fixed-width trim silently corrupts the join.
attach_labels_and_noisy_exact() below matches on the exact circuit_path
string instead.
"""
import json
import os
import pickle
from typing import List, Optional, Tuple

import pandas as pd
import torch
from qiskit import QuantumCircuit
from torch_geometric.data import Data


def stage_flat_dir(src_dir: str, dst_dir: str, exclude_subdirs=("val",)) -> str:
    """Symlink only the direct-child files of src_dir into dst_dir, skipping
    nested subdirectories (e.g. a nested val/ split that must not leak into
    the train set through a recursive os.walk).
    """
    os.makedirs(dst_dir, exist_ok=True)
    excluded = set(exclude_subdirs)
    for fn in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, fn)
        if os.path.isdir(src_path):
            continue
        if fn in excluded:
            continue
        dst_path = os.path.join(dst_dir, fn)
        if not os.path.exists(dst_path):
            os.symlink(os.path.abspath(src_path), dst_path)
    return dst_dir


def convert_qasm_json_dir_to_pk(json_dir: str, out_pk_dir: str) -> str:
    """Convert step_N.json files (circuit stored as a QASM string) into
    step_N.pk pickle files with real QuantumCircuit objects, so the rest of
    the pipeline can treat this dataset like the others.
    """
    os.makedirs(out_pk_dir, exist_ok=True)
    for fn in sorted(os.listdir(json_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(json_dir, fn)) as f:
            entries = json.load(f)
        for e in entries:
            e["circuit"] = QuantumCircuit.from_qasm_str(e["circuit"])
        out_fn = fn[: -len(".json")] + ".pk"
        with open(os.path.join(out_pk_dir, out_fn), "wb") as f:
            pickle.dump(entries, f)
    return out_pk_dir


def _flatten_noisy(noisy):
    # noisy_exp_values is stored as [[v0, v1, ...]] (one realization); unwrap it.
    if isinstance(noisy, list) and len(noisy) == 1 and isinstance(noisy[0], list):
        return noisy[0]
    return noisy


def build_labels_csv(
    input_dir: str,
    out_csv: str,
    file_extensions: Optional[List[str]] = None,
) -> str:
    """Walk input_dir the same way utils.load_circuits_from_dir does, and for
    each circuit entry write a row keyed by the exact circuit_path string
    dataset.py will store on the corresponding graph.
    """
    if file_extensions is None:
        file_extensions = [".pk", ".pickle"]
    rows = []
    for root, _, files in os.walk(input_dir):
        for fn in sorted(files):
            if not any(fn.endswith(ext) for ext in file_extensions):
                continue
            path = os.path.join(root, fn)
            obj = pickle.load(open(path, "rb"))
            if not isinstance(obj, list):
                continue
            for i, entry in enumerate(obj):
                if not (isinstance(entry, dict) and "circuit" in entry):
                    continue
                noisy = _flatten_noisy(entry["noisy_exp_values"])
                ideal = entry["ideal_exp_value"]
                rows.append(
                    {
                        "circuit_path": f"{path}::#{i}",
                        "noisy_z_json": json.dumps(list(noisy)),
                        "target_y_json": json.dumps(list(ideal)),
                    }
                )
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def build_measured_qubits_map(
    input_dir: str,
    out_json: str,
    file_extensions: Optional[List[str]] = None,
) -> str:
    """For each circuit, figure out which physical qubit each classical bit
    reads from its 'measure' instructions, and record the physical-qubit
    order sorted by classical-bit index.

    This matters because ideal_exp_value[i] / noisy_exp_values[i] are indexed
    by classical-bit order, and -- across these datasets -- neither the set
    nor the order of physically measured qubits is fixed at [0..M-1]: it
    shifts across Trotter steps and even varies circuit-to-circuit within a
    step (confirmed empirically; e.g. haoran_mbd/random_cliffords mixes dozens
    of distinct qubit orders within a single step file). Passing a flat
    [0, 1, ..., M-1] as `measured_qubits` to ConvertConfig silently
    mislabels the lightcone_masks columns, which for many circuits leaves a
    "measured" column with an empty lightcone -> softmax over an all
    -inf masked row -> NaN loss during GNN training.
    """
    if file_extensions is None:
        file_extensions = [".pk", ".pickle"]
    mapping = {}
    for root, _, files in os.walk(input_dir):
        for fn in sorted(files):
            if not any(fn.endswith(ext) for ext in file_extensions):
                continue
            path = os.path.join(root, fn)
            obj = pickle.load(open(path, "rb"))
            if not isinstance(obj, list):
                continue
            for i, entry in enumerate(obj):
                if not (isinstance(entry, dict) and "circuit" in entry):
                    continue
                qc = entry["circuit"]
                pairs = []
                for instr in qc.data:
                    if instr.operation.name == "measure":
                        qidx = qc.find_bit(instr.qubits[0]).index
                        cidx = qc.find_bit(instr.clbits[0]).index
                        pairs.append((cidx, qidx))
                pairs.sort(key=lambda p: p[0])
                mapping[f"{path}::#{i}"] = [q for _, q in pairs]
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(mapping, f)
    return out_json


def load_circuit_label_lists(input_dir: str, file_extensions: Optional[List[str]] = None):
    """Walk input_dir like build_labels_csv and return parallel lists of
    (circuit_path, circuit, ideal_exp_value, noisy_exp_value) for RF-style
    featurization. circuit_path lines up exactly with Data.circuit_path, so
    callers can align an RF split with a GNN split built from the same
    directory (e.g. by filtering on the val_loader's graph circuit_paths).
    """
    if file_extensions is None:
        file_extensions = [".pk", ".pickle"]
    paths, circuits, ideal, noisy = [], [], [], []
    for root, _, files in os.walk(input_dir):
        for fn in sorted(files):
            if not any(fn.endswith(ext) for ext in file_extensions):
                continue
            path = os.path.join(root, fn)
            obj = pickle.load(open(path, "rb"))
            if not isinstance(obj, list):
                continue
            for i, entry in enumerate(obj):
                if not (isinstance(entry, dict) and "circuit" in entry):
                    continue
                paths.append(f"{path}::#{i}")
                circuits.append(entry["circuit"])
                ideal.append(entry["ideal_exp_value"])
                noisy.append(_flatten_noisy(entry["noisy_exp_values"]))
    return paths, circuits, ideal, noisy


def attach_labels_and_noisy_exact(
    graphs: List[Data],
    labels_csv: str,
    noisy_col: str = "noisy_z_json",
    target_col: str = "target_y_json",
) -> Tuple[List[Data], int]:
    df = pd.read_csv(labels_csv)
    by_path = {row["circuit_path"]: row for _, row in df.iterrows()}
    M_ref = None
    matched = 0
    for g in graphs:
        path = g.circuit_path if isinstance(g.circuit_path, str) else g.circuit_path[0]
        row = by_path.get(path)
        if row is None:
            continue
        noisy = torch.tensor(json.loads(row[noisy_col]), dtype=torch.float32)
        y = torch.tensor(json.loads(row[target_col]), dtype=torch.float32)
        M = g.lightcone_masks.shape[1]
        assert noisy.numel() == M and y.numel() == M, (
            f"Mismatch for {path}: graph M={M}, noisy={noisy.numel()}, y={y.numel()}"
        )
        g.noisy_z = noisy
        g.y = y
        matched += 1
        if M_ref is None:
            M_ref = M
        else:
            assert M_ref == M, "All graphs must share M"
    assert matched > 0, "No graphs matched labels CSV by circuit_path"
    return graphs, M_ref
