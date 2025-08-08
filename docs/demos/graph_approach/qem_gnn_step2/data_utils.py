import json
import pandas as pd
import torch
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_graphs(graphs_pt_path: str) -> List[Data]:
    graphs = torch.load(graphs_pt_path)
    assert isinstance(graphs, list) and all(isinstance(g, Data) for g in graphs), "Expected list[Data]"
    return graphs

def attach_labels_and_noisy(graphs: List[Data], labels_csv: str, noisy_col="noisy_z_json", target_col="target_y_json") -> Tuple[List[Data], int]:
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
