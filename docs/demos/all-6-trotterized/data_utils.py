import json
import pandas as pd
import torch
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Iterable, Optional
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import os


def load_graphs(graphs_pt_path: str) -> List[Data]:
    graphs = torch.load(graphs_pt_path, weights_only=False)
    assert isinstance(graphs, list) and all(isinstance(g, Data) for g in graphs), "Expected list[Data]"
    return graphs

def attach_labels_and_noisy(graphs: List[Data], labels_csv: str, noisy_col="noisy_z_json", target_col="target_y_json") -> Tuple[List[Data], int]:
    df = pd.read_csv(labels_csv)
    by_path = {row["circuit_path"]: row for _, row in df.iterrows()}
    # print(by_path)
    M_ref = None
    matched = 0
    for g in graphs:
        path = g.circuit_path if isinstance(g.circuit_path, str) else g.circuit_path[0]
        path = path[:-4]
        # print("The circuit path = ", path)
        row = by_path.get(path)
        # print("The CSV circuit path = ", path)
        if row is None:
            continue
        noisy = torch.tensor(json.loads(row[noisy_col]), dtype=torch.float32)
        y = torch.tensor(json.loads(row[target_col]), dtype=torch.float32)
        M = g.lightcone_masks.shape[1]
        assert noisy.numel() == M and y.numel() == M, f"Mismatch for {path}: graph M={M}, noisy={noisy.numel()}, y={y.numel()}"
        g.noisy_z = noisy
        g.y = y
        matched += 1
        if M_ref is None: M_ref = M
        else: assert M_ref == M, "All graphs must share M"
    assert matched > 0, "No graphs matched labels CSV by circuit_path"
    return graphs, M_ref

def build_loaders(graphs: List[Data], batch_size: int = 8, val_frac: float = 0.2, seed: int = 42):
    import random
    rnd = random.Random(seed)
    idx = list(range(len(graphs)))
    rnd.shuffle(idx)
    n_val = max(1, int(len(graphs) * val_frac))
    val_idx = set(idx[:n_val])
    train_graphs = [graphs[i] for i in range(len(graphs)) if i not in val_idx]
    val_graphs = [graphs[i] for i in range(len(graphs)) if i in val_idx]
    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        DataLoader(val_graphs, batch_size=batch_size, shuffle=False),
    )

def _as_str_path(p):
    return p if isinstance(p, str) else (p[0] if isinstance(p, (list, tuple)) and len(p) else str(p))

def _index_graphs_by_path(graphs: List[Data]):
    by_path = {}
    for g in graphs:
        key = _as_str_path(g.circuit_path[:-4])
        by_path[key] = g
    return by_path

def build_loaders_by_train_val_paths(
    graphs: List[Data],
    train_paths: Iterable[str],
    val_paths: Iterable[str],
    *,
    batch_size: int = 8,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build PyG DataLoaders from explicit train/val circuit path lists.
    Paths must match the stored `Data.circuit_path` strings.
    """
    by_path = _index_graphs_by_path(graphs)
    # print(by_path)
    train_keys = set(map(str, train_paths))
    val_keys   = set(map(str, val_paths))
    # print(train_keys)
    # print(val_keys)

    # sanity: no overlap
    overlap = train_keys & val_keys
    if overlap:
        print(f"[build_loaders_by_train_val_paths] Warning: {len(overlap)} overlapping paths; removing from val.")
        val_keys -= overlap

    def pick(keys):
        picked, missing = [], []
        for k in keys:
            if k in by_path: picked.append(by_path[k])
            else: missing.append(k)
        if missing:
            print(f"[build_loaders_by_train_val_paths] Missing {len(missing)} paths (showing 3): {missing[:3]}")
        return picked

    train_graphs = pick(train_keys)
    # print(train_graphs)
    val_graphs   = pick(val_keys)

    # ensure constant M across splits
    def M_of(gs): return int(gs[0].lightcone_masks.shape[1]) if gs else 0
    Ms = {m for m in (M_of(train_graphs), M_of(val_graphs)) if m}
    assert len(Ms) <= 1, f"Inconsistent measured-qubit count across splits: {Ms}"

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    print(f"[build] train={len(train_graphs)} | val={len(val_graphs)} | M={next(iter(train_loader)).lightcone_masks.shape[1] if len(train_graphs)>0 else 'N/A'}")
    return train_loader, val_loader

