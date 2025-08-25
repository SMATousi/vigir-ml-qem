# If not installed already (uncomment as needed):
# !pip install torch torch-geometric pandas tqdm

import os, json, torch, pandas as pd
from pathlib import Path
import importlib

import data_utils
importlib.reload(data_utils)

from data_utils import load_graphs, attach_labels_and_noisy, build_loaders
from train_loop import train, evaluate
import torch, numpy as np, random
from torch_geometric.loader import DataLoader

import os
import math
import json
import random
import numpy as np
import torch
import optuna

# Reproducibility helpers
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# EDIT THESE:
GRAPHS_PT = "/home/macula/SMATousi/Desktop/all_6q_new/trotter/all_6_trotter_graphs_1200.pt"      # <-- Step 1 output (a list[Data])
BATCH_SIZE  = 16
NUM_WORKERS = 0
PIN_MEMORY  = torch.cuda.is_available()

graphs = torch.load(GRAPHS_PT, weights_only=False)   # list[Data] or InMemoryDataset
N = len(graphs)
print(f"Loaded {N} graphs")

def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_loaders_random(graphs, train_ratio=0.8, seed=42, batch_size=BATCH_SIZE):
    _set_seed(seed)
    idx = np.arange(len(graphs))
    np.random.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    train_set = [graphs[i] for i in train_idx]
    val_set   = [graphs[i] for i in val_idx]
    print(f"Random split: train={len(train_set)}, val={len(val_set)}")
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        train_idx, val_idx
    )

def build_loaders_by_indices(graphs, train_idx, val_idx, batch_size=BATCH_SIZE):
    train_set = [graphs[i] for i in train_idx]
    val_set   = [graphs[i] for i in val_idx]
    print(f"By indices: train={len(train_set)}, val={len(val_set)}")
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    )

def build_loaders_stratified_by_step(graphs, train_ratio=0.8, seed=42, batch_size=BATCH_SIZE):
    """Stratify by data.step (trotter_step). If missing, falls back to random."""
    # collect indices per step
    buckets = {}
    missing = []
    for i, g in enumerate(graphs):
        if hasattr(g, "step") and g.step is not None:
            s = int(float(g.step.item())) if torch.is_tensor(g.step) else int(g.step)
            buckets.setdefault(s, []).append(i)
        else:
            missing.append(i)
    if not buckets:
        print("No `step` metadata; falling back to random.")
        return build_loaders_random(graphs, train_ratio, seed, batch_size)

    _set_seed(seed)
    train_idx, val_idx = [], []
    for s, idxs in buckets.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        cut = int(len(idxs) * train_ratio)
        train_idx.extend(idxs[:cut].tolist())
        val_idx.extend(idxs[cut:].tolist())
    # distribute any missing-step items proportionally
    if missing:
        miss = np.array(missing); np.random.shuffle(miss)
        cut = int(len(miss) * train_ratio)
        train_idx += miss[:cut].tolist()
        val_idx   += miss[cut:].tolist()

    train_set = [graphs[i] for i in train_idx]
    val_set   = [graphs[i] for i in val_idx]
    print(f"Stratified by step: train={len(train_set)}, val={len(val_set)}; steps={sorted(buckets.keys())}")
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        np.array(train_idx), np.array(val_idx)
    )

# ---- Choose ONE of the following depending on your need ----

# A) Simple random split
train_loader, val_loader, train_idx, val_idx = build_loaders_random(graphs, train_ratio=0.625, seed=123, batch_size=16)

from pathlib import Path

def run_train_once(model_cfg: dict,
                   train_loader,
                   val_loader,
                   epochs: int = 1,
                   lr: float = 1e-3,
                   wd: float = 1e-4,
                   patience: int = 50,
                   scheduler_patience: int = 10,
                   scheduler_factor: int = 0.5,
                   ckpt_path: str = None):
    """
    Wrap your train() and return (best_mae, best_epoch, net, hist).
    Assumes your train() already does early-stopping on val MAE and saves best checkpoint if ckpt_path is given.
    """
    net, hist = train(
        model_cfg,
        (train_loader, val_loader),
        epochs=epochs,
        lr=lr,
        wd=wd,
        patience=patience,
        best_ckpt_path=ckpt_path,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        device=device,
    )

    # hist is assumed to be a list of dicts with 'val_mae' (adapt if your key differs)
    # Fallbacks are added for safety.
    val_maes = []
    for e, rec in enumerate(hist):
        # if isinstance(rec, dict):
        #     if "val_mae" in rec:
        #         val_maes.append((e, float(rec["val_mae"])))
        #     elif "val_loss" in rec:
        #         val_maes.append((e, float(rec["val_loss"])))
        # elif hasattr(rec, "get"):
        #     v = rec.get("val_mae", rec.get("val_loss", None))
        #     if v is not None:
        #         val_maes.append((e, float(v)))
        val_maes.append((e, float(rec[-2])))

    if len(val_maes) == 0:
        raise RuntimeError("History does not contain 'val_mae' or 'val_loss'.")

    best_epoch, best_mae = min(val_maes, key=lambda t: t[1])
    return best_mae, best_epoch, net, hist

def suggest_heads_for_dmodel(trial, d_model):
    # choose heads that divide d_model
    possible_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    if not possible_heads:
        # fallback to 1 head if needed (no multihead, but keeps code safe)
        possible_heads = [1]
    return trial.suggest_categorical("heads", possible_heads)

def build_model_cfg_from_trial(trial):
    # Keep sizes modest to avoid overfitting on small data
    d_model = trial.suggest_categorical("d_model", [64, 96, 128, 160, 192])
    layers  = trial.suggest_int("layers", 2, 6)
    heads   = suggest_heads_for_dmodel(trial, d_model)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    use_noisy = trial.suggest_categorical("use_noisy", [True, False])  # let study decide

    cfg = dict(
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
        use_noisy=use_noisy,
    )
    return cfg

def objective(trial: optuna.Trial):
    set_seed(42 + trial.number)

    # Model + optimizer hparams
    model_cfg = build_model_cfg_from_trial(trial)
    lr        = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    wd        = trial.suggest_float("wd", 1e-10, 1e-3, log=True)
    patience  = trial.suggest_int("patience", 30, 100)
    epochs    = trial.suggest_int("epochs", 150, 400)
    scheduler_patience = trial.suggest_int("scheduler_patience", 5, 50)
    scheduler_factor = trial.suggest_int("scheduler_factor", 0.5, 0.9)

    ckpt_path = f"optuna_chkpts/optuna_ckpt_trial_{trial.number}.pt"

    best_mae, best_epoch, net, hist = run_train_once(
        model_cfg,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        wd=wd,
        patience=patience,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        ckpt_path=ckpt_path,
    )

    # Log some useful attrs
    trial.set_user_attr("best_epoch", int(best_epoch))
    trial.set_user_attr("best_mae", float(best_mae))
    trial.set_user_attr("ckpt_path", ckpt_path)
    trial.set_user_attr("model_cfg", json.dumps(model_cfg))

    # Report the final score to Optuna (direction='minimize')
    return float(best_mae)


# You can switch pruners/samplers if you like; MedianPruner is a safe default.
pruner  = optuna.pruners.MedianPruner(n_warmup_steps=5)
sampler = optuna.samplers.TPESampler(seed=123)

study = optuna.create_study(
    study_name="qem_graph_transformer_arch_search",
    direction="minimize",
    sampler=sampler,
    pruner=pruner,
)

N_TRIALS = 250  # adjust based on your compute budget
print("Starting study with", N_TRIALS, "trials...")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n=== Study done ===")
print("Best value (val MAE):", study.best_value)
print("Best trial number:", study.best_trial.number)
print("Best params:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")



