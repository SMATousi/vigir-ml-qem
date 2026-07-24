import math
from typing import Dict, List
from qiskit.dagcircuit import DAGOpNode

DEFAULT_GATE_DICT = {
    "x": 0, "y": 1, "z": 2,
    "rx": 3, "ry": 4, "rz": 5,
    "h": 6, "s": 7, "sdg": 8, "t": 9, "tdg": 10,
    "cx": 11, "cz": 12, "swap": 13, "iswap": 14,
    "rxx": 15, "ryy": 16, "rzz": 17, "rzx": 18,
    "id": 19, "sx": 20, "sxdg": 21,
}

def get_gate_id(name: str, gate_dict: Dict[str, int]) -> int:
    return gate_dict.get(name.lower(), len(gate_dict))

def sinusoidal_encoding(val: int, dim: int = 8, base: float = 10000.0):
    out = [0.0] * dim
    for i in range(dim // 2):
        denom = base ** (2 * i / dim)
        out[2*i] = math.sin(val / denom)
        out[2*i + 1] = math.cos(val / denom)
    return out

def extract_param_vector(node: DAGOpNode, max_params: int):
    params = []
    for p in getattr(node.op, "params", [])[:max_params]:
        try:
            params.append(float(p))
        except Exception:
            params.append(0.0)
    while len(params) < max_params:
        params.append(0.0)
    return params

def build_node_feature(
    node: DAGOpNode,
    gate_dict: Dict[str, int],
    moment_idx: int,
    max_params: int,
    qubit_enc_dim: int = 8,
    moment_enc_dim: int = 8,
):
    gid = get_gate_id(node.op.name, gate_dict)
    arity = len(node.qargs)
    params = extract_param_vector(node, max_params)
    q_enc = [0.0] * qubit_enc_dim
    for q in [q.index for q in node.qargs]:
        enc = sinusoidal_encoding(q, qubit_enc_dim)
        q_enc = [a + b for a, b in zip(q_enc, enc)]
    m_enc = sinusoidal_encoding(moment_idx, moment_enc_dim)
    feat = [float(gid), float(arity)] + params + q_enc + m_enc
    return feat
