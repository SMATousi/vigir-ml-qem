import os
import pickle
from typing import Dict, List, Optional, Tuple
from qiskit import QuantumCircuit
from qiskit.qpy import load as qpy_load

def load_circuits_from_dir(
    input_dir: str,
    file_extensions: Optional[List[str]] = None,
) -> List[Tuple[str, QuantumCircuit]]:
    """Return list of (path, QuantumCircuit). Supports .qpy, .qasm, and pickle files."""
    if file_extensions is None:
        file_extensions = [".qpy", ".qasm", ".pk", ".pickle"]

    out: List[Tuple[str, QuantumCircuit]] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if not any(fn.endswith(ext) for ext in file_extensions):
                continue
            path = os.path.join(root, fn)
            if fn.endswith(".qpy"):
                with open(path, "rb") as f:
                    circuits = list(qpy_load(f))
                for i, c in enumerate(circuits):
                    out.append((f"{path}::#{i}", c))
            elif fn.endswith(".qasm"):
                qc = QuantumCircuit.from_qasm_file(path)
                out.append((path, qc))
            else:
                obj = pickle.load(open(path, "rb"))
                if isinstance(obj, list):
                    for i, entry in enumerate(obj):
                        if isinstance(entry, dict) and "circuit" in entry:
                            out.append((f"{path}::#{i}", entry["circuit"]))
                elif isinstance(obj, dict) and "circuit" in obj:
                    out.append((path, obj["circuit"]))
    return out

def measured_qubits_for_path(
    path: str,
    global_measured: Optional[List[int]],
    measured_map: Optional[Dict[str, List[int]]],
) -> List[int]:
    """Resolve measured qubits using (1) measured_map by exact key or basename, else (2) global."""
    if measured_map:
        if path in measured_map:
            return measured_map[path]
        base = os.path.basename(path.split('::')[0])
        if base in measured_map:
            return measured_map[base]
    if global_measured is None:
        raise ValueError("Measured qubits not provided (global or map).")
    return global_measured
