# QEM-GNN — Step 1: Graph Builder + Lightcone Tags

This starter kit covers:
1. Loading Qiskit circuits (from `.qpy`, `.qasm`, or pickled dicts).
2. Computing measured-qubit lightcones on the circuit DAG.
3. Converting circuits → gate-graph with node/edge features.
4. Packaging graphs as PyTorch Geometric `Data` objects and saving `.pt` files.

Keep this step focused on data prep. Modeling comes next.

## Files

- `circuits_to_graph.py` — circuit → graph conversion
- `lightcone.py` — lightcone computation on Qiskit DAG
- `featurizers.py` — encoders (gate types, sinusoidal, params)
- `dataset.py` — PyG dataset + directory-to-pt converter
- `utils.py` — loaders for QPY/QASM/pickle, helpers
- `schemas.py` — dataclasses for config I/O
- `example_convert_folder.py` — small CLI to convert a folder
- `config_example.json` — example config for conversion

## Quick start

1) Install deps:
    pip install qiskit torch torch-geometric networkx

(For PyG wheels, see the PyG docs for your CUDA/CPU version.)

2) Edit `config_example.json` as needed (paths, measured qubits).

3) Run conversion:
    python example_convert_folder.py --config config_example.json

This will create `graphs_train.pt` / `graphs_test.pt` with a list of PyG `Data` objects.
