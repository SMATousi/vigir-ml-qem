import json, os, torch
from torch_geometric.data import InMemoryDataset
from typing import Optional, Dict, List
from schemas import ConvertConfig
from utils import load_circuits_from_dir, measured_qubits_for_path
from circuits_to_graph import circuit_to_gategraph_data

class QEMCircuitDataset(InMemoryDataset):
    def __init__(self, root: str, cfg: ConvertConfig, transform=None, pre_transform=None):
        self.cfg = cfg
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        base = os.path.splitext(os.path.basename(self.cfg.output_path))[0]
        return [f"{base}.pt"]

    def download(self):
        pass

    def process(self):
        measured_map = None
        if self.cfg.measured_map_json and os.path.exists(self.cfg.measured_map_json):
            with open(self.cfg.measured_map_json, "r") as f:
                measured_map = json.load(f)

        pairs = load_circuits_from_dir(self.cfg.input_dir, self.cfg.file_extensions)
        data_list = []
        for path, qc in pairs:
            measured = measured_qubits_for_path(path, self.cfg.measured_qubits, measured_map)
            d = circuit_to_gategraph_data(qc, measured, max_params=self.cfg.max_params)
            d.circuit_path = path
            d.split = 0 if self.cfg.split == "train" else 1
            data_list.append(d)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.cfg.output_path), exist_ok=True)
        torch.save((data, slices), self.cfg.output_path)

def convert_folder_to_pt(cfg: ConvertConfig):
    _ = QEMCircuitDataset(root=".", cfg=cfg)
    return cfg.output_path
