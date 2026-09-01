"""Generate one notebook per simulator-generated Ising dataset: build graphs,
train the QEMGraphTransformer, train an RF baseline, and compare both against
the ideal expectation values -- mirroring 02_GNN_Transformer.ipynb, adapted
for datasets that already carry ideal_exp_value/noisy_exp_values per circuit
(no ZNE-derivation step needed).

Run from docs/demos/qem_gnn/:
    /opt/miniconda3/envs/blackwater/bin/python generate_sim_notebooks.py

Then execute each with:
    /opt/miniconda3/envs/blackwater/bin/jupyter nbconvert --to notebook \
        --execute --inplace --ExecutePreprocessor.timeout=3600 <notebook>.ipynb
"""
import nbformat as nbf

DATA_ROOT = "../../tutorials/data"  # relative to docs/demos/qem_gnn/

DATASETS = [
    dict(
        name="ising_dataset",
        title="Ising dataset (baseline simulator)",
        train_dir=f"{DATA_ROOT}/ising_dataset/train",
        val_dir=f"{DATA_ROOT}/ising_dataset/val",
        qasm_json=False,
        flat_stage=False,
    ),
    dict(
        name="ising_dataset_random_init",
        title="Ising dataset, randomized initial states",
        train_dir=f"{DATA_ROOT}/ising_dataset_random_init/train",
        val_dir=f"{DATA_ROOT}/ising_dataset_random_init/val",
        qasm_json=False,
        flat_stage=False,
    ),
    dict(
        name="ising_init_0110",
        title="Ising, fixed |0110>-style initial state",
        train_dir=f"{DATA_ROOT}/ising_init_0110/train",
        val_dir=f"{DATA_ROOT}/ising_init_0110/val",
        qasm_json=False,
        flat_stage=False,
    ),
    dict(
        name="ising_init_from_qasm",
        title="Ising, initial state from QASM",
        train_dir=f"{DATA_ROOT}/ising_init_from_qasm/train",
        val_dir=f"{DATA_ROOT}/ising_init_from_qasm/val",
        qasm_json=False,
        flat_stage=False,
    ),
    dict(
        name="ising_init_from_qasm_coherent",
        title="Ising from QASM, coherent noise",
        train_dir=f"{DATA_ROOT}/ising_init_from_qasm_coherent/train",
        val_dir=f"{DATA_ROOT}/ising_init_from_qasm_coherent/val",
        qasm_json=False,
        flat_stage=False,
    ),
    dict(
        name="ising_init_from_qasm_no_readout",
        title="Ising from QASM, no readout error",
        train_dir=f"{DATA_ROOT}/ising_init_from_qasm_no_readout/train",
        val_dir=f"{DATA_ROOT}/ising_init_from_qasm_no_readout/val",
        qasm_json=False,
        flat_stage=False,
    ),
    dict(
        name="haoran_mbd_random_brickwork",
        title="haoran_mbd: random brickwork circuits",
        train_dir=f"{DATA_ROOT}/haoran_mbd/random_brickwork",
        # val/ here only has circuit_graph/ideal/noisy, no 'circuit' object -> unusable.
        val_dir=None,
        qasm_json=False,
        flat_stage=True,  # exclude nested val/ from the recursive directory walk
    ),
    dict(
        name="haoran_mbd_random_cliffords",
        title="haoran_mbd: random Clifford circuits",
        train_dir=f"{DATA_ROOT}/haoran_mbd/random_cliffords",
        val_dir=f"{DATA_ROOT}/haoran_mbd/random_cliffords/val",
        qasm_json=False,
        flat_stage=True,
    ),
    dict(
        name="haoran_mbd_coherent_random_cliffords",
        title="haoran_mbd_coherent: random Clifford circuits, coherent noise",
        train_dir=f"{DATA_ROOT}/haoran_mbd_coherent/random_cliffords",
        val_dir=None,  # no val split provided at all
        qasm_json=False,
        flat_stage=False,  # flat already, nothing nested to exclude
    ),
    dict(
        name="mbd_theta_0p05pi",
        title="mbd_datasets2: theta=0.05pi",
        train_dir=f"{DATA_ROOT}/mbd_datasets2/theta_0.05pi/train",
        val_dir=f"{DATA_ROOT}/mbd_datasets2/theta_0.05pi/val",
        qasm_json=True,  # circuits stored as QASM strings inside JSON
        flat_stage=False,
    ),
    dict(
        name="mbd_theta_0p1pi_coherent",
        title="mbd_datasets2: theta=0.1pi, coherent noise",
        train_dir=f"{DATA_ROOT}/mbd_datasets2/theta_0.1pi_coherent/train",
        val_dir=f"{DATA_ROOT}/mbd_datasets2/theta_0.1pi_coherent/val",
        qasm_json=False,
        flat_stage=False,
    ),
]


# --------------------------------------------------------------------------
# Tier 0 + Tier 1 section (qem_levels.py / qem_ext.py / qem_train.py).
# Kept in one place so both the generator and any hand-patched notebook stay
# in sync. model.py and train_loop.py are untouched by all of this.
# --------------------------------------------------------------------------

TIER01_INTRO = """## Tier 0 + Tier 1: discrete-level head and a configurable training loop

Opt-in additions living in `qem_levels.py` / `qem_ext.py` / `qem_train.py`. `model.py` and
`train_loop.py` are untouched, so the other applications that import them are unaffected --
`QEMGraphTransformerX(head="scalar")` has a bit-identical `state_dict` to
`QEMGraphTransformer`, and `TrainConfig()`'s defaults reproduce the original loop exactly.

**Tier 0 -- match the head to the target distribution.** Some of these datasets have
targets that are not continuous: random Clifford circuits give `<Z>` in {-1, 0, +1}
exactly. Regressing on such a target makes the model hedge -- it emits 0.87 where the
truth is 1.0 and pays MAE on a point whose class it already got right.
`LevelSet.detect` inspects the **training** labels only and returns a level set just when
they really do collapse onto a few atoms, otherwise `None`, in which case the cells below
fall back to regression. Across this repo it fires for `haoran_mbd/random_cliffords` and
`haoran_mbd_coherent/random_cliffords`, and returns `None` for the `ising_*`,
`mbd_theta_*` and `random_brickwork` datasets.

**Tier 1 -- training-loop fixes.** `nn.SmoothL1Loss()` defaults to `beta=1.0`, and every
residual here lives in [-2, 2], so the loss never leaves the quadratic regime: it is
exactly `0.5*MSE`, with none of the robustness the name suggests. The LR was held constant
for the whole run. And a single unseeded run is not a like-for-like comparison against a
100-tree forest. `TrainConfig` exposes all three.

Both new heads zero-initialise their last layer, so training *starts* at a known baseline
and can only improve on it: `residual` starts at `y_hat = noisy_z` (the unmitigated
value), and `level` carries a `-gamma*(noisy_z - l_k)^2` prior term so its argmax at step 0
is exactly "snap `noisy_z` to the nearest level"."""

TIER01_DETECT = """import qem_levels, qem_ext, qem_train
importlib.reload(qem_levels); importlib.reload(qem_ext); importlib.reload(qem_train)

from qem_levels import LevelSet, level_report
from qem_ext import QEMGraphTransformerX
from qem_train import TrainConfig, train_qem, train_ensemble, collect, ensemble_predict, metrics

# Detect the level set from the TRAINING labels only (never the val split).
# Returns None when the targets are genuinely continuous -- the next cell then
# falls back to the residual regression head.
y_train_all = torch.cat([d.y for d in train_loader.dataset])
levels = LevelSet.detect(y_train_all)
print(levels, "|", level_report(y_train_all, levels))"""

TIER01_TRAIN = """# Capacity/dropout follow Tier 1: the original run underfits, so keep d_model at
# least 64 and drop the dropout that was regularising overfitting that isn't
# there.
cfg_common = dict(d_model=128, layers=3, heads=4,
                  dropout=0.05, use_noisy=True)

if levels is not None:
    # Tier 0: K-way classification over the detected levels.
    model_cfg_x = dict(cfg_common, head="level", levels=levels)
    train_cfg = TrainConfig(epochs=80, lr=1e-3, wd=1e-4, patience=25,
                            loss="ce", scheduler="cosine", warmup_epochs=5,
                            select_on="acc", seed=0)
else:
    # Continuous targets: stay a regressor, but keep the Tier 1 fixes --
    # residual head, a beta that actually reaches the linear regime, cosine LR.
    model_cfg_x = dict(cfg_common, head="residual")
    train_cfg = TrainConfig(epochs=80, lr=1e-3, wd=1e-4, patience=25,
                            loss="smooth_l1", huber_beta=0.05, scheduler="cosine",
                            warmup_epochs=5, select_on="mae", seed=0)

net_x, hist_x = train_qem(model_cfg_x, (train_loader, val_loader), train_cfg,
                          best_ckpt_path=f"{RUN_DIR}/best_qem_x.pt")
print("Total params:", sum(p.numel() for p in net_x.parameters()))"""

TIER01_ENSEMBLE = """# 5-seed ensemble -- a single unseeded run is not comparable to a 100-tree forest.
nets_x, hists_x = train_ensemble(
    model_cfg_x, (train_loader, val_loader),
    TrainConfig(**{**train_cfg.__dict__, "verbose": False}),
    seeds=(0, 1, 2, 3, 4), ckpt_prefix=f"{RUN_DIR}/qem_x",
)
print(f"trained {len(nets_x)} members")"""

TIER01_COMPARE = """from sklearn.ensemble import RandomForestClassifier

rows = {}
rows["Noisy (unmitigated)"] = metrics(noisy_np, ideal_np, levels)
rows["RF regressor"] = metrics(rf_preds, y_val.numpy(), levels)

if levels is not None:
    lv = np.asarray(levels.levels)
    to_cls = lambda a: np.abs(np.asarray(a)[..., None] - lv).argmin(-1)
    snap_np = lambda a: lv[to_cls(a)]

    # Snapping costs nothing and applies to every baseline, so compare like with like.
    rows["Noisy, snapped"] = metrics(snap_np(noisy_np), ideal_np, levels)
    rows["RF regressor, snapped"] = metrics(snap_np(rf_preds), y_val.numpy(), levels)
    rf_clf_pred = np.stack([
        lv[RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
           .fit(X_train.numpy(), to_cls(y_train.numpy())[:, q]).predict(X_val.numpy())]
        for q in range(M)
    ], axis=1)
    rows["RF classifier"] = metrics(rf_clf_pred, y_val.numpy(), levels)

# argmax is the MAE-optimal decoding of the level head; sum_k p_k * l_k is the
# RMSE-optimal one (hedging *is* correct under squared error). Report both.
raw_x, y_x, _ = collect(net_x, val_loader, device)
modes = ("argmax", "expect") if net_x.is_classifier else ("raw",)
for mode in modes:
    rows[f"QAGT-MLP Tier0+1, 1 seed ({mode})"] = metrics(
        net_x.decode(raw_x, mode).numpy(), y_x.numpy(), levels)
for mode in modes:
    p_ens, y_ens, _ = ensemble_predict(nets_x, val_loader, device, decode=mode)
    rows[f"QAGT-MLP Tier0+1, 5-seed ens ({mode})"] = metrics(p_ens, y_ens, levels)

comparison = pd.DataFrame(rows).T
print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))
comparison.to_csv(f"{RUN_DIR}/comparison_tier01.csv")"""

TIER01_FIGURE = """# Per-qubit RMSE: unmitigated vs RF vs QAGT-MLP (Tier 0+1), same style as the
# figure above. A level head is decoded with "expect" (sum_k p_k * l_k), which
# is the RMSE-optimal decoding -- and RMSE is what this figure reports.
p_fig, y_fig, _ = ensemble_predict(nets_x, val_loader, device,
                                   decode="expect" if net_x.is_classifier else "raw")
p_fig = p_fig.reshape(-1, M)
y_fig = y_fig.reshape(-1, M)

rmse_un_x = rmse_per_qubit(noisy_np, ideal_np)
rmse_rf_x = rmse_per_qubit(rf_preds, y_val.numpy())
rmse_qagt_x = rmse_per_qubit(p_fig, y_fig)

x = np.arange(M)
width = 0.25
plt.figure(figsize=(8, 5))
bars = [
    plt.bar(x - width, rmse_un_x, width, label="Noisy (unmitigated)"),
    plt.bar(x, rmse_rf_x, width, label="RF"),
    plt.bar(x + width, rmse_qagt_x, width, label="QAGT-MLP (Tier 0+1)"),
]
for b in bars:
    plt.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
# headroom so the legend never sits on top of a bar label
plt.ylim(0, max(rmse_un_x.max(), rmse_rf_x.max(), rmse_qagt_x.max()) * 1.25)
plt.xticks(x, [f"q{q}" for q in range(M)])
plt.ylabel("RMSE vs ideal")
plt.title(f"{DATASET_NAME}: RMSE by qubit (lower is better)")
plt.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(f"{RUN_DIR}/rmse_comparison_tier01.png", dpi=150)
plt.show()

rmse_table = pd.DataFrame({
    "qubit": [f"q{q}" for q in range(M)] + ["overall"],
    "rmse_noisy": list(rmse_un_x) + [rmse_un_x.mean()],
    "rmse_rf": list(rmse_rf_x) + [rmse_rf_x.mean()],
    "rmse_qagt": list(rmse_qagt_x) + [rmse_qagt_x.mean()],
})
print(rmse_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
rmse_table.to_csv(f"{RUN_DIR}/rmse_comparison_tier01.csv", index=False)"""


def tier01_cells():
    """The Tier 0 + Tier 1 section, as nbformat cells."""
    return [
        nbf.v4.new_markdown_cell(TIER01_INTRO),
        nbf.v4.new_code_cell(TIER01_DETECT),
        nbf.v4.new_code_cell(TIER01_TRAIN),
        nbf.v4.new_code_cell(TIER01_ENSEMBLE),
        nbf.v4.new_code_cell(TIER01_COMPARE),
        nbf.v4.new_code_cell(TIER01_FIGURE),
    ]


def make_notebook(cfg: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        f"# {cfg['title']}\n\n"
        f"Build graphs from `{cfg['train_dir']}`, train a per-qubit Random Forest "
        "baseline, and train the graph-Transformer QEM model in the Tier 0+1 "
        "section at the end; both are compared against the ideal "
        "expectation values on the validation split.\n\n"
        "Unlike the Brisbane hardware notebooks, this dataset already carries "
        "`ideal_exp_value` / `noisy_exp_values` per circuit, so there is no "
        "ZNE-derivation step -- labels come straight from the data."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys, os, json, importlib, random, pickle\n"
        "sys.path.insert(0, os.path.abspath(\".\"))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import matplotlib.pyplot as plt\n"
        "from qiskit.dagcircuit import DAGOpNode\n"
        "\n"
        "# lightcone.assign_moments (dag.layers()-based) silently drops some\n"
        "# nodes (e.g. barriers) in some qiskit versions; use a topological-order\n"
        "# version instead, matching 01_build_graphs.ipynb's fix.\n"
        "def assign_moments_topo(dag):\n"
        "    moments = {}\n"
        "    for node in dag.topological_op_nodes():\n"
        "        preds = [e.node for e in dag.predecessors(node)\n"
        "                 if hasattr(e, \"node\") and isinstance(e.node, DAGOpNode)]\n"
        "        moments[node] = 0 if not preds else 1 + max(moments[p] for p in preds)\n"
        "    return moments\n"
        "\n"
        "import lightcone\n"
        "lightcone.assign_moments = assign_moments_topo\n"
        "import circuits_to_graph\n"
        "importlib.reload(circuits_to_graph)\n"
        "import dataset\n"
        "importlib.reload(dataset)\n"
        "\n"
        "from schemas import ConvertConfig\n"
        "from sim_data_utils import (\n"
        "    build_labels_csv, attach_labels_and_noisy_exact, build_measured_qubits_map,\n"
        "    stage_flat_dir, convert_qasm_json_dir_to_pk, load_circuit_label_lists,\n"
        ")\n"
        "from data_utils import load_graphs, build_loaders\n"
        "from torch_geometric.loader import DataLoader\n"
        "from train_loop import train, evaluate\n"
        "from model import QEMGraphTransformer\n"
        "from mlp import encode_data_v2_ecr\n"
        "from sklearn.ensemble import RandomForestRegressor"
    ))

    cells.append(nbf.v4.new_code_cell(
        f"DATASET_NAME = {cfg['name']!r}\n"
        f"RAW_TRAIN_DIR = {cfg['train_dir']!r}\n"
        f"RAW_VAL_DIR = {cfg['val_dir']!r}  # None if no usable val split exists\n"
        "RUN_DIR = f\"runs/{DATASET_NAME}\"\n"
        "os.makedirs(RUN_DIR, exist_ok=True)"
    ))

    if cfg["qasm_json"]:
        staging_src = (
            "# Circuits are stored as QASM strings inside JSON here -- convert once\n"
            "# to pickled QuantumCircuit objects so the rest of the pipeline (which\n"
            "# expects .pk files with a 'circuit' key) can treat this like the others.\n"
            "TRAIN_DIR = convert_qasm_json_dir_to_pk(RAW_TRAIN_DIR, f\"{RUN_DIR}/pk_cache/train\")\n"
            "VAL_DIR = convert_qasm_json_dir_to_pk(RAW_VAL_DIR, f\"{RUN_DIR}/pk_cache/val\") if RAW_VAL_DIR else None"
        )
    elif cfg["flat_stage"]:
        staging_src = (
            "# Step files sit directly in this directory alongside a nested val/\n"
            "# subfolder; os.walk is recursive, so stage a symlinked copy of only\n"
            "# the top-level step files to keep val circuits out of train.\n"
            "TRAIN_DIR = stage_flat_dir(RAW_TRAIN_DIR, f\"{RUN_DIR}/train_only\", exclude_subdirs=(\"val\",))\n"
            "VAL_DIR = RAW_VAL_DIR"
        )
    else:
        staging_src = "TRAIN_DIR = RAW_TRAIN_DIR\nVAL_DIR = RAW_VAL_DIR"
    cells.append(nbf.v4.new_code_cell(staging_src))

    cells.append(nbf.v4.new_markdown_cell(
        "**Per-circuit measured-qubit map.** `ideal_exp_value[i]` / "
        "`noisy_exp_values[i]` are indexed by *classical-bit* order, but "
        "neither the set nor the order of physically measured qubits is "
        "fixed across circuits here -- it shifts across Trotter steps and "
        "even varies circuit-to-circuit within a step. Passing a flat "
        "`[0, 1, 2, 3]` as the measured-qubit list would silently mislabel "
        "the graph's lightcone columns (and for many circuits leaves a "
        "'measured' column with an empty lightcone -> NaN loss). So instead "
        "we derive the correct per-circuit qubit order straight from each "
        "circuit's own `measure` instructions."
    ))

    cells.append(nbf.v4.new_code_cell(
        "def build_split(input_dir, split):\n"
        "    measured_map_json = build_measured_qubits_map(\n"
        "        input_dir, f\"{RUN_DIR}/measured_map_{split}.json\", file_extensions=[\".pk\"],\n"
        "    )\n"
        "    cfg = ConvertConfig(\n"
        "        input_dir=input_dir, split=split,\n"
        "        output_path=f\"{RUN_DIR}/processed/graphs_{split}.pt\",\n"
        "        measured_qubits=None, measured_map_json=measured_map_json,\n"
        "        max_params=2, verbose=True,\n"
        "        file_extensions=[\".pk\"],\n"
        "    )\n"
        "    dataset.convert_folder_to_pt(cfg)\n"
        "    build_labels_csv(input_dir, f\"{RUN_DIR}/labels_{split}.csv\", file_extensions=[\".pk\"])\n"
        "    graphs = load_graphs(cfg.output_path)\n"
        "    graphs, M = attach_labels_and_noisy_exact(graphs, f\"{RUN_DIR}/labels_{split}.csv\")\n"
        "    return graphs, M\n"
        "\n"
        "train_graphs, M = build_split(TRAIN_DIR, \"train\")\n"
        "print(f\"train graphs: {len(train_graphs)} | M={M}\")\n"
        "\n"
        "if VAL_DIR is not None:\n"
        "    val_graphs, M_val = build_split(VAL_DIR, \"val\")\n"
        "    assert M_val == M\n"
        "    print(f\"val graphs: {len(val_graphs)} | M={M_val}\")\n"
        "    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)\n"
        "    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)\n"
        "else:\n"
        "    print(\"No usable val split for this dataset; using an internal 80/20 split of train.\")\n"
        "    train_loader, val_loader = build_loaders(train_graphs, batch_size=16, val_frac=0.2, seed=42)"
    ))

    cells.append(nbf.v4.new_code_cell(
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "\n"
        "# The original QEMGraphTransformer training used to live here. It cost ~35 min\n"
        "# per notebook and its only surviving product was the ideal/noisy pair below,\n"
        "# which needs no model at all -- so it is gone. The Tier 0+1 section further\n"
        "# down trains the model this notebook actually reports on.\n"
        "@torch.no_grad()\n"
        "def collect_target_noisy(loader, M):\n"
        "    Y, N = [], []\n"
        "    for batch in loader:\n"
        "        Y.append(batch.y.detach().cpu())\n"
        "        N.append(batch.noisy_z.detach().cpu())\n"
        "    return torch.cat(Y).view(-1, M), torch.cat(N).view(-1, M)\n"
        "\n"
        "Y, N = collect_target_noisy(val_loader, M)\n"
        "print(f\"val points: {Y.shape[0]} circuits x {M} qubits\")"
    ))

    cells.append(nbf.v4.new_markdown_cell("## Random Forest baseline\n\nTrained/evaluated on the *same* train/val circuits as the GNN (matched by `circuit_path`)."))

    cells.append(nbf.v4.new_code_cell(
        "train_paths_set = {g.circuit_path for g in train_loader.dataset}\n"
        "val_paths_set = {g.circuit_path for g in val_loader.dataset}\n"
        "\n"
        "def gather_split(input_dir, wanted_paths):\n"
        "    paths, circuits, ideal, noisy = load_circuit_label_lists(input_dir, file_extensions=[\".pk\"])\n"
        "    keep = [i for i, p in enumerate(paths) if p in wanted_paths]\n"
        "    return [circuits[i] for i in keep], [ideal[i] for i in keep], [noisy[i] for i in keep]\n"
        "\n"
        "train_circuits, train_ideal, train_noisy = gather_split(TRAIN_DIR, train_paths_set)\n"
        "val_source_dir = VAL_DIR if VAL_DIR is not None else TRAIN_DIR\n"
        "val_circuits, val_ideal, val_noisy = gather_split(val_source_dir, val_paths_set)\n"
        "print(f\"RF train circuits: {len(train_circuits)} | RF val circuits: {len(val_circuits)}\")"
    ))

    cells.append(nbf.v4.new_code_cell(
        "X_train, y_train = encode_data_v2_ecr(train_circuits, train_ideal, train_noisy, obs_size=M, two_q_gate='cx')\n"
        "X_val, y_val = encode_data_v2_ecr(val_circuits, val_ideal, val_noisy, obs_size=M, two_q_gate='cx')\n"
        "\n"
        "rf_models = []\n"
        "for q in range(M):\n"
        "    rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)\n"
        "    rfr.fit(X_train.numpy(), y_train.numpy()[:, q])\n"
        "    rf_models.append(rfr)\n"
        "    print(f\"RF trained for qubit {q}\")\n"
        "\n"
        "rf_preds = np.stack([m.predict(X_val.numpy()) for m in rf_models], axis=1)\n"
        "rf_mae = np.abs(rf_preds - y_val.numpy()).mean(axis=0).tolist()\n"
        "print(\"Per-qubit MAE (RF):\", rf_mae)"
    ))

    cells.append(nbf.v4.new_markdown_cell("## Comparison: unmitigated noisy vs RF"))

    cells.append(nbf.v4.new_code_cell(
        "noisy_np = N.numpy()\n"
        "ideal_np = Y.numpy()\n"
        "\n"
        "def rmse_per_qubit(pred, ideal):\n"
        "    return np.sqrt(((pred - ideal) ** 2).mean(axis=0))\n"
        "\n"
        "rmse_noisy = rmse_per_qubit(noisy_np, ideal_np)\n"
        "rmse_rf = rmse_per_qubit(rf_preds, y_val.numpy())\n"
        "\n"
        "results = pd.DataFrame({\n"
        "    \"qubit\": [f\"q{q}\" for q in range(M)],\n"
        "    \"rmse_noisy\": rmse_noisy,\n"
        "    \"rmse_rf\": rmse_rf,\n"
        "})\n"
        "overall = pd.DataFrame([{\"qubit\": \"overall\", \"rmse_noisy\": rmse_noisy.mean(),\n"
        "                        \"rmse_rf\": rmse_rf.mean()}])\n"
        "results = pd.concat([results, overall], ignore_index=True)\n"
        "print(results)\n"
        "\n"
        "results.to_csv(f\"{RUN_DIR}/comparison_results.csv\", index=False)\n"
        "with open(f\"{RUN_DIR}/comparison_results.json\", \"w\") as f:\n"
        "    json.dump({\n"
        "        \"dataset\": DATASET_NAME,\n"
        "        \"rmse_noisy\": rmse_noisy.tolist(), \"rmse_rf\": rmse_rf.tolist(),\n"
        "        \"n_train\": len(train_loader.dataset), \"n_val\": len(val_loader.dataset),\n"
        "    }, f, indent=2)"
    ))

    cells.append(nbf.v4.new_code_cell(
        "x = np.arange(M)\n"
        "width = 0.35\n"
        "plt.figure(figsize=(8, 5))\n"
        "plt.bar(x - width / 2, rmse_noisy, width, label=\"Noisy (unmitigated)\")\n"
        "plt.bar(x + width / 2, rmse_rf, width, label=\"RF\")\n"
        "plt.xticks(x, [f\"q{q}\" for q in range(M)])\n"
        "plt.ylabel(\"RMSE vs ideal\")\n"
        "plt.title(f\"{DATASET_NAME}: RMSE by qubit (lower is better)\")\n"
        "plt.legend()\n"
        "plt.tight_layout()\n"
        "plt.savefig(f\"{RUN_DIR}/rmse_comparison.png\")\n"
        "plt.show()"
    ))

    cells.extend(tier01_cells())

    nb["cells"] = cells
    return nb


def main():
    for cfg in DATASETS:
        nb = make_notebook(cfg)
        out_path = f"sim_{cfg['name']}.ipynb"
        with open(out_path, "w") as f:
            nbf.write(nb, f)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
