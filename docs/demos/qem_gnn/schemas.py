from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ConvertConfig:
    input_dir: str                     # folder containing circuits (.qpy/.qasm/.pk)
    split: str                         # "train" or "test"
    output_path: str                   # path to .pt file to save a list[Data]
    measured_qubits: Optional[List[int]] = None  # global list of measured qubits
    measured_map_json: Optional[str] = None      # optional: per-circuit mapping file
    max_params: int = 2                # pad/truncate gate parameter vector length
    max_qubits: Optional[int] = None   # if None, infer from circuit
    verbose: bool = True
    file_extensions: Optional[List[str]] = None  # defaults: [".qpy",".qasm",".pk",".pickle"]
