# QEM-GNN — Step 2: Graph Transformer + Qubit-Conditioned Pooling

This step provides a small Graph Transformer with qubit-conditioned pooling and a training loop.
Assumes Step 1 exported a **list of `Data`** graphs and you have a CSV mapping `circuit_path` to `noisy_z` and target `y`.
