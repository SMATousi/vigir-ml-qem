import json, os, torch
from tqdm.auto import tqdm
from torch_geometric.data import InMemoryDataset
from schemas import ConvertConfig
from utils import load_circuits_from_dir, measured_qubits_for_path
from circuits_to_graph import circuit_to_gategraph_data

class QEMCircuitDataset(InMemoryDataset):
    def __init__(self, root: str, cfg: ConvertConfig, transform=None, pre_transform=None):
        self.cfg = cfg
        super().__init__(root, transform, pre_transform)
        # We no longer read from processed_paths; we just run process() which writes a list.
        # Keep this minimal to satisfy the base class:
        self.data, self.slices = None, None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # Unused, but must exist
        return ["unused.pt"]

    def download(self):
        pass

    def process(self):
        measured_map = None
        if self.cfg.measured_map_json and os.path.exists(self.cfg.measured_map_json):
            with open(self.cfg.measured_map_json, "r") as f:
                measured_map = json.load(f)

        pairs = load_circuits_from_dir(self.cfg.input_dir, self.cfg.file_extensions)

        graphs = []
        for path, qc in tqdm(pairs, desc=f"Converting {self.cfg.split} circuits", unit="circuit"):
            measured = measured_qubits_for_path(path, self.cfg.measured_qubits, measured_map)
            d = circuit_to_gategraph_data(qc, measured, max_params=self.cfg.max_params)
            d.circuit_path = path
            d.split = 0 if self.cfg.split == "train" else 1
            if self.pre_transform is not None:
                d = self.pre_transform(d)
            graphs.append(d)

        os.makedirs(os.path.dirname(self.cfg.output_path), exist_ok=True)
        torch.save(graphs, self.cfg.output_path)

def convert_folder_to_pt(cfg: ConvertConfig):
    _ = QEMCircuitDataset(root=".", cfg=cfg)  # just to run process()
    return cfg.output_path

