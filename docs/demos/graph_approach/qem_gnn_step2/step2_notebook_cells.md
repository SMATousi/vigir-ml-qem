# Step 2 — Notebook cells

## 1) Imports
```python
# !pip install torch torch-geometric pandas
import torch
from data_utils import load_graphs, attach_labels_and_noisy, build_loaders
from train_loop import train, evaluate
```

## 2) Load graphs + labels
```python
GRAPHS_PT = "graphs_train.pt"   # list[Data] from Step 1
LABELS_CSV = "labels_train.csv" # columns: circuit_path,noisy_z_json,target_y_json
graphs = load_graphs(GRAPHS_PT)
graphs, M = attach_labels_and_noisy(graphs, LABELS_CSV)
print(f"Loaded {len(graphs)} graphs with M={M}")
```

## 3) DataLoaders
```python
train_loader, val_loader = build_loaders(graphs, batch_size=4, val_frac=0.2, seed=42)
```

## 4) Train
```python
model_cfg = dict(d_model=128, layers=3, heads=4, dropout=0.2, use_noisy=True)
net = train(model_cfg, (train_loader, val_loader), epochs=100, lr=1e-3, wd=1e-4, patience=15)
torch.save(net.state_dict(), "qem_graph_transformer.pt")
print("Saved model to qem_graph_transformer.pt")
```

## 5) Eval
```python
import torch.nn as nn
crit = nn.SmoothL1Loss()
metrics = evaluate(net, val_loader, crit, device="cuda" if torch.cuda.is_available() else "cpu")
metrics
```
