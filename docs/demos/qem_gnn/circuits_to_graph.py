import numpy as np
from typing import Dict, List
from qiskit.converters import circuit_to_dag
from torch_geometric.data import Data

from featurizers import build_node_feature, DEFAULT_GATE_DICT
from lightcone import compute_lightcone_nodes, assign_moments

def to_torch(arr, dtype: str = "float"):
    import torch
    if dtype == "float":
        return torch.from_numpy(np.ascontiguousarray(arr)).float()
    elif dtype == "long":
        return torch.from_numpy(np.ascontiguousarray(arr)).long()
    else:
        raise ValueError("dtype must be 'float' or 'long'")

def circuit_to_gategraph_data(
    qc,
    measured_qubits: List[int],
    max_params: int = 2,
    gate_dict: Dict[str, int] = DEFAULT_GATE_DICT,
) -> Data:
    dag = circuit_to_dag(qc)
    moments = assign_moments(dag)

    nodes = list(dag.op_nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    x = []
    for n in nodes:
        feat = build_node_feature(n, gate_dict, moments[n], max_params)
        x.append(feat)
    x = np.array(x, dtype=np.float32)

    edge_src, edge_dst = [], []
    qubit_to_nodes = {}
    for n in nodes:
        for q in [q.index for q in n.qargs]:
            qubit_to_nodes.setdefault(q, []).append(n)
    for q, nlist in qubit_to_nodes.items():
        nlist_sorted = sorted(nlist, key=lambda n: moments[n])
        for a, b in zip(nlist_sorted[:-1], nlist_sorted[1:]):
            edge_src.append(node_idx[a])
            edge_dst.append(node_idx[b])
    if len(edge_src) == 0:
        edge_index = np.zeros((2,0), dtype=np.int64)
    else:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)

    lc = compute_lightcone_nodes(dag, measured_qubits)
    masks = []
    for q in measured_qubits:
        m = np.zeros(len(nodes), dtype=np.float32)
        for n in lc[q]:
            m[node_idx[n]] = 1.0
        masks.append(m)
    lightcone_masks = (np.stack(masks, axis=1) if masks
                   else np.zeros((len(nodes), 0), dtype=np.float32))  # (num_nodes, num_measured)


    data = Data(
        x=to_torch(x, "float"),
        edge_index=to_torch(edge_index, "long")
    )
    data.num_nodes = x.shape[0]
    data.lightcone_masks = to_torch(lightcone_masks, "float")
    import numpy as _np
    data.measured_qubits = to_torch(_np.array(measured_qubits, dtype=_np.int64), "long")
    data.num_measured = len(measured_qubits)
    data.num_qubits = qc.num_qubits
    data.depth_layers = int(max(moments.values()) + 1) if len(moments) else 0
    return data
