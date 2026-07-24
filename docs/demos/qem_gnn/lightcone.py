from typing import Dict, List, Set
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

def node_qubits(node: DAGOpNode):
    return [q.index for q in node.qargs]

def build_predecessor_map(dag: DAGCircuit):
    pred = {}
    for node in dag.op_nodes():
        pred[node] = [e.node for e in dag.predecessors(node) if hasattr(e, "node") and isinstance(e.node, DAGOpNode)]
    return pred

def compute_lightcone_nodes(dag: DAGCircuit, measured_qubits: List[int]):
    """For each measured qubit q, return the set of DAGOpNodes in its backward lightcone."""
    preds = build_predecessor_map(dag)
    nodes_by_qubit = {}
    for n in dag.op_nodes():
        for q in node_qubits(n):
            nodes_by_qubit.setdefault(q, []).append(n)

    out = {q: set() for q in measured_qubits}
    for q in measured_qubits:
        stack = list(nodes_by_qubit.get(q, []))
        visited = set()
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            out[q].add(n)
            for p in preds.get(n, []):
                stack.append(p)
    return out

def assign_moments(dag: DAGCircuit):
    """Assign a topological 'layer index' to each DAGOpNode."""
    layer_index = {}
    current = 0
    for layer in dag.layers():
        for node in layer["graph"].op_nodes():
            layer_index[node] = current
        current += 1
    return layer_index
