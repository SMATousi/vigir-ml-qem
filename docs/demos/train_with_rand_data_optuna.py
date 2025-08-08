import json, os, pickle
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import qiskit.circuit.random
import torch, random
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import optuna
from optuna.trial import TrialState

import numpy as np
import json, os, pickle
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from qiskit import QuantumCircuit

import sys
sys.path.append('../tutorials/')
from mlp import encode_data, encode_data_v2_ecr

def check_f(f, f_ext, step_indices):
    return f.endswith(f_ext) and any([f"step_%02d"%step_index in f for step_index in step_indices])

def load_circuits(data_dir, step_indices, f_ext='.pk'):
    Js = []
    circuits = []
    data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if check_f(f, f_ext, step_indices)])
    for data_file in tqdm(data_files, leave=True):
        for entry in pickle.load(open(data_file, 'rb')):
            Js.append(entry['J'])
            circuits.append(entry['circuit'])
    return circuits, Js


# circuits, Js = load_circuits('../tutorials/data/ising_zne_hardware/100q_brisbane/', list(range(1, 11)))

for step_index in [1]:
    with open('../tutorials/zne_mitigated/twirl_100q_brisbane/step%02d.json'%step_index, 'r') as file:
        loaded = json.load(file)
    noise_factor_1 = np.array(loaded['noise_factor_1'])
    noise_factor_3 = np.array(loaded['noise_factor_3'])

for step_index in tqdm([2, 3, 4, 5, 6, 7, 8, 9, 10]):
    with open('../tutorials/zne_mitigated/twirl_100q_brisbane/step%02d.json'%step_index, 'r') as file:
        loaded = json.load(file)
    noise_factor_1 = np.concatenate([noise_factor_1, loaded['noise_factor_1']])
    noise_factor_3 = np.concatenate([noise_factor_3, loaded['noise_factor_3']])

noise_factor_1_tw_avg = noise_factor_1.reshape(noise_factor_1.shape[0], 5, 5).mean(axis=-1)
noise_factor_3_tw_avg = noise_factor_3.reshape(noise_factor_3.shape[0], 5, 5).mean(axis=-1)

slope = (noise_factor_3_tw_avg - noise_factor_1_tw_avg) / 2
zne_mitigated_vals = (noise_factor_1_tw_avg - slope).tolist()
noisy_vals = noise_factor_1_tw_avg.tolist()
len(zne_mitigated_vals)

def count_gates_by_rotation_angle(circuit, bin_size):
    angles = []
    for instr, qargs, cargs in circuit.data:
        if instr.name in ['rx', 'ry', 'rz'] and len(qargs) == 1:
            angles += [float(instr.params[0])]
    bin_edges = np.arange(-2 * np.pi, 2 * np.pi + bin_size, bin_size)
    counts, _ = np.histogram(angles, bins=bin_edges)
    bin_labels = [f"{left:.2f} to {right:.2f}" for left, right in zip(bin_edges[:-1], bin_edges[1:])]
    angle_bins = {label: count for label, count in zip(bin_labels, counts)}
    return list(angle_bins.values())


def recursive_dict_loop(my_dict, parent_key=None, out=None, target_key1=None, target_key2=None):
    if out is None: out = []

    for key, val in my_dict.items():
        if isinstance(val, dict):
            recursive_dict_loop(val, key, out, target_key1, target_key2)
        else:
            if parent_key and target_key1 in str(parent_key) and key == target_key2:
                out += [val]
    return out or 0.


def encode_data_v2_ecr_CLIP_encoder(circuits, 
                                    ideal_exp_vals, 
                                    noisy_exp_vals, 
                                    obs_size, 
                                    tokenizer,
                                    text_model,
                                    device,
                                    max_len = 77,
                                    meas_bases=None, 
                                    two_q_gate='ecr'):
    
    if isinstance(noisy_exp_vals[0], list) and len(noisy_exp_vals[0]) == 1:
        noisy_exp_vals = [x[0] for x in noisy_exp_vals]

    if meas_bases is None:
        meas_bases = [[]]

    gates_set = [two_q_gate] + ['sx', 'x', 'id', 'rz']

    vec = []

    bin_size = 0.025 * np.pi
    num_angle_bins = int(np.ceil(4 * np.pi / bin_size))

    X = torch.zeros([len(circuits), 512 + len(vec) + len(gates_set) + num_angle_bins + obs_size + len(meas_bases[0])]) #512 for CLIP Embedding

    embedding_slice = slice(0,512)
    vec_slice = slice(512, 512+len(vec))
    gate_counts_slice = slice(512+len(vec), 512+len(vec)+len(gates_set))
    angle_bins_slice = slice(512+len(vec)+len(gates_set), 512+len(vec)+len(gates_set)+num_angle_bins)
    exp_val_slice = slice(512+len(vec)+len(gates_set)+num_angle_bins, 512+len(vec)+len(gates_set)+num_angle_bins+obs_size)
    meas_basis_slice = slice(512+len(vec)+len(gates_set)+num_angle_bins+obs_size, len(X[0]))

    # X[:, vec_slice] = vec[None, :]

    for i, circ in enumerate(tqdm(circuits)):
        circuit_qasm = circ.qasm()

        tokens = tokenizer(circuit_qasm, return_tensors="pt", truncation=False, padding=False)
        input_ids = tokens["input_ids"][0]  # remove batch dimension
        
        chunks = [input_ids[i:i + max_len] for i in range(0, len(input_ids), max_len)]
        
        # Encode each chunk and collect pooled outputs
        embeddings = []
        
        for chunk in chunks:
            chunk = chunk.unsqueeze(0).to(device)
            with torch.no_grad():
                output = text_model(input_ids=chunk)
                pooled = output.pooler_output  # shape: (1, hidden_dim)
            embeddings.append(pooled.cpu())  # keep CPU to save GPU memory
        
        # Combine embeddings (mean pooling)
        final_embedding = torch.mean(torch.stack(embeddings), dim=0)

        
        X[i, embedding_slice] = torch.tensor(final_embedding)

    
    for i, circ in enumerate(circuits):
        gate_counts_all = circ.count_ops()
        X[i, gate_counts_slice] = torch.tensor(
            [gate_counts_all.get(key, 0) for key in gates_set]
        ) * 0.01  # put it in the same order of magnitude as the expectation values

    for i, circ in enumerate(circuits):
        gate_counts = count_gates_by_rotation_angle(circ, bin_size)
        X[i, angle_bins_slice] = torch.tensor(gate_counts) * 0.01  # put it in the same order of magnitude as the expectation values

        if obs_size > 1: assert len(noisy_exp_vals[i]) == obs_size
        elif obs_size == 1: assert isinstance(noisy_exp_vals[i], float)

        X[i, exp_val_slice] = torch.tensor(noisy_exp_vals[i])

    if meas_bases != [[]]:
        assert len(meas_bases) == len(circuits)
        for i, basis in enumerate(meas_bases):
            X[i, meas_basis_slice] = torch.tensor(basis)

    y = torch.tensor(ideal_exp_vals, dtype=torch.float32)

    return X, y


# num_circ_per_step = 50
# k = train_test_split = 10
# train_circuits = []
# train_zne_vals = []
# train_noisy_vals = []
# test_circuits = []
# test_zne_vals = []
# test_noisy_vals = []
# test_Js = []
# for start_each_step in list(range(len(circuits))[::num_circ_per_step]):
#     train_circuits += circuits[start_each_step:start_each_step+k]
#     train_zne_vals += zne_mitigated_vals[start_each_step:start_each_step+k]
#     train_noisy_vals += noisy_vals[start_each_step:start_each_step+k]
#     test_circuits += circuits[start_each_step+k:start_each_step+num_circ_per_step]
#     test_zne_vals += zne_mitigated_vals[start_each_step+k:start_each_step+num_circ_per_step]
#     test_noisy_vals += noisy_vals[start_each_step+k:start_each_step+num_circ_per_step]
#     test_Js += Js[start_each_step+k:start_each_step+num_circ_per_step]

# print(len(train_circuits), len(train_zne_vals), len(train_noisy_vals))
# print(len(test_circuits), len(test_zne_vals), len(test_noisy_vals))

# few_normal_X_train, few_normal_y_train = encode_data_v2_ecr(train_circuits, train_zne_vals, train_noisy_vals, obs_size=5)
# few_normal_X_test, few_normal_y_test = encode_data_v2_ecr(test_circuits, test_zne_vals, test_noisy_vals, obs_size=5)


# torch.save(X_train, 'CLIP_Average_X_train.pt')
# torch.save(y_train, 'CLIP_Average_y_train.pt')
# torch.save(X_test, 'CLIP_Average_X_test.pt')
# torch.save(y_test, 'CLIP_Average_y_test.pt')
normal_X_train = torch.load("Normal_X_train_random_added.pt")
normal_y_train = torch.load("Normal_y_train_random_added.pt")
normal_X_test = torch.load("Normal_X_test_random_added.pt")
normal_y_test = torch.load("Normal_y_test_random_added.pt")

X_train = torch.load("CLIP_Average_X_train_random_added.pt")
y_train = torch.load("CLIP_Average_y_train_random_added.pt")
X_test = torch.load("CLIP_Average_X_test_random_added.pt")
y_test = torch.load("CLIP_Average_y_test_random_added.pt")

BATCH_SIZE = 32
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*1000, shuffle=False)

normal_train_dataset = TensorDataset(torch.Tensor(normal_X_train), torch.Tensor(normal_y_train))
normal_train_loader = DataLoader(normal_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
normal_test_dataset = TensorDataset(torch.Tensor(normal_X_test), torch.Tensor(normal_y_test))
normal_test_loader = DataLoader(normal_test_dataset, batch_size=BATCH_SIZE*1000, shuffle=False)

# few_normal_train_dataset = TensorDataset(torch.Tensor(few_normal_X_train), torch.Tensor(few_normal_y_train))
# few_normal_train_loader = DataLoader(few_normal_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# few_normal_test_dataset = TensorDataset(torch.Tensor(few_normal_X_test), torch.Tensor(few_normal_y_test))
# few_normal_test_loader = DataLoader(few_normal_test_dataset, batch_size=BATCH_SIZE*1000, shuffle=False)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

normal_X_train = pd.DataFrame(normal_X_train)
normal_y_train = pd.DataFrame(normal_y_train)
normal_X_test = pd.DataFrame(normal_X_test)
normal_y_test = pd.DataFrame(normal_y_test)

# few_normal_X_train = pd.DataFrame(few_normal_X_train)
# few_normal_y_train = pd.DataFrame(few_normal_y_train)
# few_normal_X_test = pd.DataFrame(few_normal_X_test)
# few_normal_y_test = pd.DataFrame(few_normal_y_test)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# rfr_tree_list = []
# for q in range(5):
#     rfr = RandomForestRegressor(n_estimators=500, verbose=1, n_jobs=-1)
#     rfr.fit(X_train, y_train.iloc[:, q])
#     rfr_tree_list.append(rfr)
#     print(f"Done with the {q} model")

# print(f"Done with the CLIP models. Moving to the normal ones ...")

# normal_rfr_tree_list = []
# for q in range(5):
#     rfr = RandomForestRegressor(n_estimators=100, verbose=0, n_jobs=-1)
#     rfr.fit(normal_X_train, normal_y_train.iloc[:, q])
#     normal_rfr_tree_list.append(rfr)
#     print(f"Done with the {q} model")

# few_normal_rfr_tree_list = []
# for q in range(5):
#     rfr = RandomForestRegressor(n_estimators=100, verbose=0, n_jobs=-1)
#     rfr.fit(few_normal_X_train, few_normal_y_train.iloc[:, q])
#     few_normal_rfr_tree_list.append(rfr)
#     print(f"Done with the {q} model")

# distances = []

# num_spins = 5

# for batch_X, batch_y in test_loader:
#     out = []
#     for q, model in enumerate(rfr_tree_list):
#         out.append(model.predict(batch_X[:, :]))
#     out = np.array(out).transpose()

#     for ideal, noisy, ngm_mitigated in zip(
#         batch_y.tolist(),
#         batch_X[:, -5:].tolist(),
#         out.tolist()
#     ):
#         for q in range(5):
#             ideal_q = ideal[q]
#             noisy_q = noisy[q]
#             ngm_mitigated_q = ngm_mitigated[q]
#             distances.append({
#                 f"ideal_{q}": ideal_q,
#                 f"noisy_{q}": noisy_q,
#                 f"ngm_mitigated_{q}": ngm_mitigated_q,
#                 f"dist_noisy_{q}": np.abs(ideal_q - noisy_q),
#                 f"dist_mitigated_{q}": np.abs(ideal_q - ngm_mitigated_q),
#                 f"dist_sq_noisy_{q}": np.square(ideal_q - noisy_q),
#                 f"dist_sq_mitigated_{q}": np.square(ideal_q - ngm_mitigated_q),
#             })

# plt.style.use({'figure.facecolor':'white'})

# df = pd.DataFrame(distances)

# for q in range(5):
#     print(f'RMSE_noisy_{q}:', np.sqrt(df[f"dist_sq_noisy_{q}"].mean()))
#     print(f'RMSE_mitigated_{q}:', np.sqrt(df[f"dist_sq_mitigated_{q}"].mean()))

# print(f'RMSE_noisy:', np.sqrt(np.mean([df[f"dist_sq_noisy_{q}"].mean() for q in range(4)])))
# print(f'RMSE_mitigated:', np.sqrt(np.mean([df[f"dist_sq_mitigated_{q}"].mean() for q in range(4)])))

# sns.boxplot(data=df[["dist_noisy_0", "dist_mitigated_0", "dist_noisy_1", "dist_mitigated_1", "dist_noisy_2", "dist_mitigated_2", "dist_noisy_3", "dist_mitigated_3", "dist_noisy_4", "dist_mitigated_4"]], orient="h", showfliers = False)
# plt.title("Dist to ideal exp value")
# plt.show()

# sns.histplot([df['ideal_0'], df['noisy_0'], df["ngm_mitigated_0"]], kde=True, bins=40)
# plt.title("Exp values distribution")
# # plt.show()

def evaluate_loader(test_loader, model_list, label: str, n_qbits=5):
    results = []

    for batch_X, batch_y in test_loader:
        predictions = []
        for q, model in enumerate(model_list):
            predictions.append(model.predict(batch_X[:, :]))
        predictions = np.array(predictions).transpose()

        for ideal, noisy_or_normal, mitigated in zip(
            batch_y.tolist(),
            batch_X[:, -5:].tolist(),
            predictions.tolist()
        ):
            for q in range(n_qbits):
                ideal_q = ideal[q]
                noisy_q = noisy_or_normal[q]
                mitigated_q = mitigated[q]

                results.append({
                    "source": label,
                    f"ideal_{q}": ideal_q,
                    f"input_{q}": noisy_q,
                    f"mitigated_{q}": mitigated_q,
                    f"dist_{q}": np.abs(ideal_q - noisy_q),
                    f"dist_mitigated_{q}": np.abs(ideal_q - mitigated_q),
                    f"dist_sq_{q}": np.square(ideal_q - noisy_q),
                    f"dist_sq_mitigated_{q}": np.square(ideal_q - mitigated_q),
                })
    return results

# ngm_results = evaluate_loader(test_loader, rfr_tree_list, label="RF+CLIP+LargeData")
# normal_results = evaluate_loader(normal_test_loader, normal_rfr_tree_list, label="RF+LargeData")
# few_normal_results = evaluate_loader(few_normal_test_loader, few_normal_rfr_tree_list, label="RF+FewData")
# all_results = ngm_results + normal_results + few_normal_results

# df = pd.DataFrame(all_results)

# for q in range(5):
#     for label in ["RF+CLIP+LargeData", "RF+LargeData", "RF+FewData"]:
#         subset = df[df["source"] == label]
#         rmse = np.sqrt(subset[f"dist_sq_{q}"].mean())
#         rmse_mitigated = np.sqrt(subset[f"dist_sq_mitigated_{q}"].mean())
#         print(f"[{label}] RMSE_input_{q}: {rmse:.4f}, RMSE_mitigated_{q}: {rmse_mitigated:.4f}")

# print("------ Overall RMSEs ------")
# for label in ["RF+CLIP+LargeData", "RF+LargeData", "RF+FewData"]:
#     subset = df[df["source"] == label]
#     rmse = np.sqrt(np.mean([subset[f"dist_sq_{q}"].mean() for q in range(5)]))
#     rmse_mitigated = np.sqrt(np.mean([subset[f"dist_sq_mitigated_{q}"].mean() for q in range(5)]))
#     print(f"[{label}] RMSE_input: {rmse:.4f}, RMSE_mitigated: {rmse_mitigated:.4f}")


# melted = pd.DataFrame()

# for q in range(5):
#     for metric in ["dist", "dist_mitigated"]:
#         temp = df[[f"{metric}_{q}", "source"]].copy()
#         temp = temp.rename(columns={f"{metric}_{q}": "value"})
#         temp["qubit"] = f"q{q}"
#         temp["type"] = "input" if "dist_" == metric else "mitigated"
#         melted = pd.concat([melted, temp], ignore_index=True)

# plt.figure(figsize=(12, 6))
# sns.boxplot(data=melted, x="value", y="qubit", hue="source", palette="Set2", showfliers=False)
# plt.title("Distance to Ideal Value by Qubit (Input Only)")
# plt.xlabel("Absolute Distance")
# plt.ylabel("Qubit")
# plt.savefig("Normal-vs-CLIP-RandomForests.png")
# plt.show()

number_of_epochs = 1000
number_heads = 1
model_depth = 8


import numpy as np
from sklearn.model_selection import train_test_split

few_X_train = torch.load("CLIP_Average_X_train.pt")
few_y_train = torch.load("CLIP_Average_y_train.pt")
few_X_test = torch.load("CLIP_Average_X_test.pt")
few_y_test = torch.load("CLIP_Average_y_test.pt")

X_train = torch.load("CLIP_Average_X_train_random_added.pt")
y_train = torch.load("CLIP_Average_y_train_random_added.pt")
X_test = torch.load("CLIP_Average_X_test_random_added.pt")
y_test = torch.load("CLIP_Average_y_test_random_added.pt")

from importlib import reload
import simple_transformer
reload(simple_transformer)
from simple_transformer import SimpleTransformerEstimator

# Import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Training will proceed without logging.")
    WANDB_AVAILABLE = False

# Import for timestamps
from datetime import datetime

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

few_X_train = pd.DataFrame(few_X_train)
few_y_train = pd.DataFrame(few_y_train)
few_X_test = pd.DataFrame(few_X_test)
few_y_test = pd.DataFrame(few_y_test)

normal_X_train = pd.DataFrame(normal_X_train)
normal_y_train = pd.DataFrame(normal_y_train)
normal_X_test = pd.DataFrame(normal_X_test)
normal_y_test = pd.DataFrame(normal_y_test)

# few_normal_X_train = pd.DataFrame(few_normal_X_train)
# few_normal_y_train = pd.DataFrame(few_normal_y_train)
# few_normal_X_test = pd.DataFrame(few_normal_X_test)
# few_normal_y_test = pd.DataFrame(few_normal_y_test)

# model = SimpleTransformerEstimator(n_epochs=100, 
#                                    verbose=True,
#                                    device=device,
#                                    seq_len=682)
# model.fit(X_train, y_train)

# Random Forest Baseline Training
# print("Training Random Forest baseline models...")
# normal_rfr_tree_list = []
# for q in range(5):
#     print(f"Training Random Forest baseline model for qubit {q}...")
#     rfr = RandomForestRegressor(n_estimators=100, verbose=0, n_jobs=-1)
#     rfr.fit(few_X_train, few_y_train.iloc[:, q])
#     normal_rfr_tree_list.append(rfr)
#     print(f"Done with the {q} model")

# # Evaluate Random Forest baseline
# print("Evaluating Random Forest baseline...")
# rf_results = evaluate_loader(test_loader, normal_rfr_tree_list, label="RandomForest_Baseline")
# rf_df = pd.DataFrame(rf_results)

# # Calculate RF baseline metrics to match transformer evaluation metrics exactly
# rf_baseline_metrics = {}
# for q in range(5):
#     # Match the exact metric names from SimpleTransformerEstimator._evaluate_model
#     rf_baseline_metrics[f"eval_dist_q{q}"] = rf_df[f"dist_{q}"].mean()
#     rf_baseline_metrics[f"eval_dist_mitigated_q{q}"] = rf_df[f"dist_mitigated_{q}"].mean()
#     rf_baseline_metrics[f"eval_rmse_input_q{q}"] = np.sqrt(rf_df[f"dist_sq_{q}"].mean())
#     rf_baseline_metrics[f"eval_rmse_mitigated_q{q}"] = np.sqrt(rf_df[f"dist_sq_mitigated_{q}"].mean())

# # Overall RF baseline metrics
# rf_baseline_metrics["overall_rmse_input"] = np.sqrt(np.mean([rf_df[f"dist_sq_{q}"].mean() for q in range(5)]))
# rf_baseline_metrics["overall_rmse_mitigated"] = np.sqrt(np.mean([rf_df[f"dist_sq_mitigated_{q}"].mean() for q in range(5)]))

# print(f"Random Forest Baseline - Overall RMSE Input: {rf_baseline_metrics['overall_rmse_input']:.4f}, RMSE Mitigated: {rf_baseline_metrics['overall_rmse_mitigated']:.4f}")

# # Create separate wandb run for RF baseline that logs to same figures as transformer training
# if WANDB_AVAILABLE:
#     timestamp_rf = datetime.now().strftime("%Y%m%d_%H%M%S")
#     wandb.init(project="vigir-ml-qem", 
#                name=f"RandomForest_Baseline_{timestamp_rf}",
#                config={
#                    "n_estimators": 100,
#                    "model_type": "RandomForest_Baseline",
#                    "data_type": "few_shot",
#                    "n_jobs": -1,
#                    "timestamp": timestamp_rf,
#                    "n_qubits": 5,
#                    "n_epochs": number_of_epochs
#                })
    
#     # Log RF baseline metrics for each epoch to create flat baseline lines
#     print("Logging RF baseline metrics for all epochs...")
#     for epoch in range(number_of_epochs):
#         log_dict = {"epoch": epoch}
#         log_dict.update(rf_baseline_metrics)
#         wandb.log(log_dict)
    
#     wandb.finish()
#     print("RF baseline logging completed.")
    



# # Create timestamp for this training run
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# # Initialize wandb for CLIP+Transformer training
# if WANDB_AVAILABLE:
#     wandb.init(project="vigir-ml-qem", 
#                name=f"CLIP_Transformer_Training_{timestamp}_{number_heads}_{model_depth}",
#                config={
#                    "n_epochs": number_of_epochs,
#                    "n_heads": number_heads,
#                    "model_depth": model_depth,
#                    "lr": 0.001,
#                    "lr_scheduler": "step",
#                    "step_size": 300,
#                    "model_type": "CLIP+Transformer",
#                    "seq_len": 682,
#                    "timestamp": timestamp,
#                    "n_qubits": 5
#                })

# trained_transformer_list_with_CLIP = []
# for q in range(5):
#     print(f"Training CLIP+Transformer model for qubit {q}...")
    
#     model = SimpleTransformerEstimator(n_epochs=number_of_epochs, 
#                                        nhead=number_heads,
#                                        d_model=model_depth,
#                                        lr=0.001,
#                                        verbose=False,
#                                        device=device,
#                                        lr_scheduler='step',
#                                        step_size=300,
#                                     #    pretrained_path=f'pretrained_q{q}.pt',
#                                        seq_len=682,
#                                        wandb_logging=WANDB_AVAILABLE,
#                                        eval_data=(X_test, y_test.iloc[:, q]),
#                                        qubit_idx=q)
    
#     # model.fit(X_train, y_train.iloc[:, q])
#     model.fit(few_X_train, few_y_train.iloc[:, q])
    
#     # Save model locally and as wandb artifact
#     local_path = f"/home/macula/SMATousi/Desktop/clip_transformer_q{q}_{timestamp}_epoch{number_of_epochs}.pt"
#     artifact_name = f"clip_transformer_q{q}_{timestamp}"
#     model.save(local_path, wandb_artifact_name=artifact_name)
    
#     trained_transformer_list_with_CLIP.append(model)
#     print(f"Done with {q}")

#     # Remove break to train all qubits
#     # break

# if WANDB_AVAILABLE:
#     wandb.finish()


# # Initialize wandb for regular Transformer training  
# # Create new timestamp for second training run
# timestamp_regular = datetime.now().strftime("%Y%m%d_%H%M%S")

# if WANDB_AVAILABLE:
#     wandb.init(project="vigir-ml-qem", 
#                name=f"Regular_Transformer_Training_{timestamp_regular}_{number_heads}_{model_depth}",
#                config={
#                    "n_epochs": number_of_epochs,
#                    "n_heads": number_heads, 
#                    "model_depth": model_depth,
#                    "lr": 0.001,
#                    "lr_scheduler": "step",
#                    "step_size": 300,
#                    "model_type": "Regular_Transformer",
#                    "seq_len": 682,
#                    "timestamp": timestamp_regular,
#                    "n_qubits": 5
#                })

# trained_transformer_list = []
# for q in range(5):
#     print(f"Training regular Transformer model for qubit {q}...")
    
#     model = SimpleTransformerEstimator(n_epochs=number_of_epochs, 
#                                        nhead=number_heads,
#                                        d_model=model_depth,
#                                        lr=0.001,
#                                        verbose=False,
#                                        device=device,
#                                        lr_scheduler='step',
#                                        step_size=30,
#                                        batch_size=16,
#                                        seq_len=170,
#                                        wandb_logging=WANDB_AVAILABLE,
#                                        eval_data=(normal_X_test, normal_y_test.iloc[:, q]),
#                                        qubit_idx=q)
    
#     # model.fit(normal_X_train, normal_y_train.iloc[:, q])
#     model.fit(few_normal_X_train, few_normal_y_train.iloc[:, q])
    
#     # Save model locally and as wandb artifact
#     local_path = f"/home/macula/SMATousi/Desktop/regular_transformer_q{q}_{timestamp_regular}_epoch{number_of_epochs}.pt"
#     artifact_name = f"regular_transformer_q{q}_{timestamp_regular}"
#     model.save(local_path, wandb_artifact_name=artifact_name)
    
#     trained_transformer_list.append(model)
#     print(f"Done with {q}")

#     # Remove break to train all qubits
#     # break

# if WANDB_AVAILABLE:
#     wandb.finish()




# trans_clip_results = evaluate_loader(test_loader, trained_transformer_list_with_CLIP, label="CLIP+Transformer+LargeData", n_qbits=1)
# normal_results = evaluate_loader(few_normal_test_loader, few_normal_rfr_tree_list, label="RandomForests+FewData", n_qbits=1)
# trans_results = evaluate_loader(normal_test_loader, trained_transformer_list, label="Transformer+LargeData", n_qbits=1)
# all_results = trans_clip_results + normal_results + trans_results

# df = pd.DataFrame(all_results)

# n_qbits = 1

# for q in range(n_qbits):
#     for label in ["CLIP+Transformer+LargeData", "RandomForests+FewData", "Transformer+LargeData"]:
#         subset = df[df["source"] == label]
#         rmse = np.sqrt(subset[f"dist_sq_{q}"].mean())
#         rmse_mitigated = np.sqrt(subset[f"dist_sq_mitigated_{q}"].mean())
#         print(f"[{label}] RMSE_input_{q}: {rmse:.4f}, RMSE_mitigated_{q}: {rmse_mitigated:.4f}")

# print("------ Overall RMSEs ------")
# for label in ["CLIP+Transformer+LargeData", "RandomForests+FewData", "Transformer+LargeData"]:
#     subset = df[df["source"] == label]
#     rmse = np.sqrt(np.mean([subset[f"dist_sq_{q}"].mean() for q in range(n_qbits)]))
#     rmse_mitigated = np.sqrt(np.mean([subset[f"dist_sq_mitigated_{q}"].mean() for q in range(n_qbits)]))
#     print(f"[{label}] RMSE_input: {rmse:.4f}, RMSE_mitigated: {rmse_mitigated:.4f}")


# melted = pd.DataFrame()

# for q in range(n_qbits):
#     for metric in ["dist", "dist_mitigated"]:
#         temp = df[[f"{metric}_{q}", "source"]].copy()
#         temp = temp.rename(columns={f"{metric}_{q}": "value"})
#         temp["qubit"] = f"q{q}"
#         temp["type"] = "input" if "dist_" == metric else "mitigated"
#         melted = pd.concat([melted, temp], ignore_index=True)

# plt.figure(figsize=(12, 6))
# sns.boxplot(data=melted, x="value", y="qubit", hue="source", palette="Set2", showfliers=False)
# plt.title("Distance to Ideal Value by Qubit (Input Only)")
# plt.xlabel("Absolute Distance")
# plt.ylabel("Qubit")
# plt.savefig("pretraining-1.png")
# plt.show()


def save_study_to_json(study, filename):
    """
    Save Optuna study results to a JSON file.
    
    Parameters:
    -----------
    study : optuna.Study
        The completed Optuna study
    filename : str
        Name of the JSON file to save results to
    """
    import json
    import numpy as np
    from datetime import datetime
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Prepare study data for JSON serialization
    study_data = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "n_complete_trials": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        "n_pruned_trials": len([t for t in study.trials if t.state == TrialState.PRUNED]),
        "n_failed_trials": len([t for t in study.trials if t.state == TrialState.FAIL]),
        "datetime_start": study.trials[0].datetime_start.isoformat() if study.trials else None,
        "datetime_complete": study.trials[-1].datetime_complete.isoformat() if study.trials and study.trials[-1].datetime_complete else None,
        "best_trial": None,
        "trials": []
    }
    
    # Add best trial information if available
    if study.best_trial is not None:
        study_data["best_trial"] = {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
            "datetime_start": study.best_trial.datetime_start.isoformat(),
            "datetime_complete": study.best_trial.datetime_complete.isoformat() if study.best_trial.datetime_complete else None,
            "duration": str(study.best_trial.duration) if study.best_trial.duration else None,
            "state": study.best_trial.state.name
        }
    
    # Add all trials information
    for trial in study.trials:
        trial_data = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
            "state": trial.state.name,
            "datetime_start": trial.datetime_start.isoformat(),
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            "duration": str(trial.duration) if trial.duration else None
        }
        study_data["trials"].append(trial_data)
    
    # Convert numpy types and save to JSON file
    study_data = convert_numpy_types(study_data)
    with open(filename, 'w') as f:
        json.dump(study_data, f, indent=2)
    
    print(f"Study results saved to: {filename}")


def train_with_rand_data_optuna(n_trials=100, timeout=3600, study_name="clip_transformer_optimization"):
    """
    Use Optuna to find the best model architecture for the CLIP transformer.
    
    Parameters:
    -----------
    n_trials : int
        Number of optimization trials to run
    timeout : int
        Maximum time in seconds for the optimization
    study_name : str
        Name for the Optuna study
        
    Returns:
    --------
    study : optuna.Study
        The completed Optuna study with results
    """
    
    # Import the SimpleTransformerEstimator
    from simple_transformer import SimpleTransformerEstimator
    
    # Load the data (using the existing data from the file)
    X_train_tensor = torch.load("CLIP_Average_X_train.pt")
    y_train_tensor = torch.load("CLIP_Average_y_train.pt")
    X_test_tensor = torch.load("CLIP_Average_X_test.pt")
    y_test_tensor = torch.load("CLIP_Average_y_test.pt")
    
    # Convert to numpy for easier handling
    X_train = X_train_tensor.numpy()
    y_train = y_train_tensor.numpy()
    X_test = X_test_tensor.numpy()
    y_test = y_test_tensor.numpy()
    
    def objective(trial):
        """
        Objective function for Optuna optimization.
        """
        # Suggest hyperparameters
        d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])
        nhead = trial.suggest_categorical('nhead', [1, 2, 4, 8])
        
        # Ensure nhead divides d_model
        while d_model % nhead != 0:
            nhead = trial.suggest_categorical('nhead', [1, 2, 4, 8])
            
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [64, 128, 256, 512])
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        n_epochs = trial.suggest_int('n_epochs', 50, 300)
        
        # Scheduler parameters
        use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
        lr_scheduler = 'plateau' if use_scheduler else None
        
        # Track metrics for all qubits
        total_rmse = 0.0
        qubit_metrics = {}
        
        try:
            # Train a model for each qubit (5 qubits total)
            for qubit_idx in range(5):
                # Create the model
                model = SimpleTransformerEstimator(
                    task="regression",
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    lr=lr,
                    weight_decay=weight_decay,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    lr_scheduler=lr_scheduler,
                    verbose=False,  # Reduce verbosity during optimization
                    seq_len=X_train.shape[1],  # Use actual sequence length
                    qubit_idx=qubit_idx
                )
                
                # Fit the model for this qubit
                model.fit(X_train, y_train[:, qubit_idx])
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                y_true = y_test[:, qubit_idx]
                
                # Calculate RMSE for this qubit
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                dist = np.mean(np.abs(y_true - y_pred))
                total_rmse += dist
                
                # Store individual qubit metrics
                qubit_metrics[f'rmse_q{qubit_idx}'] = rmse
                
                # Report intermediate results for pruning
                trial.report(total_rmse / (qubit_idx + 1), qubit_idx)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('inf')
        
        # Calculate average RMSE across all qubits
        avg_rmse = total_rmse / 5
        
        # Log additional metrics
        for key, value in qubit_metrics.items():
            trial.set_user_attr(key, value)
            
        return avg_rmse
    
    # Create and run the study
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    print(f"Starting Optuna optimization with {n_trials} trials...")
    print(f"Timeout: {timeout} seconds")
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("OPTUNA OPTIMIZATION RESULTS")
    print("="*50)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    if study.best_trial is not None:
        print(f"\nBest trial value (avg RMSE): {study.best_trial.value:.6f}")
        print("\nBest parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
            
        print("\nBest trial metrics by qubit:")
        for key, value in study.best_trial.user_attrs.items():
            if key.startswith('rmse_q'):
                print(f"  {key}: {value:.6f}")
    else:
        print("No successful trials completed.")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        # Create separate figures since ax parameter might not be supported
        fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
        if fig1 is not None:
            fig1.savefig(f'{study_name}_optimization_history.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
        
        # Parameter importances
        try:
            fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
            if fig2 is not None:
                fig2.savefig(f'{study_name}_param_importances.png', dpi=300, bbox_inches='tight')
                plt.close(fig2)
        except Exception as e:
            print(f"Could not create parameter importance plot: {e}")
        
        print(f"Plots saved as {study_name}_optimization_history.png and {study_name}_param_importances.png")
        
    except ImportError:
        print("Matplotlib not available for plotting.")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Save study results to JSON file
    save_study_to_json(study, f"{study_name}_results.json")
    
    return study


def train_best_model_from_study(study, save_path="best_clip_transformer_models"):
    """
    Train the final models using the best hyperparameters found by Optuna.
    
    Parameters:
    -----------
    study : optuna.Study
        The completed Optuna study
    save_path : str
        Directory to save the trained models
        
    Returns:
    --------
    models : list
        List of trained models for each qubit
    metrics : dict
        Final evaluation metrics
    """
    from simple_transformer import SimpleTransformerEstimator
    import os
    
    if study.best_trial is None:
        raise ValueError("No successful trials in the study.")
    
    # Get best parameters
    best_params = study.best_trial.params
    print(f"Training final models with best parameters: {best_params}")
    
    # Load data
    X_train_tensor = torch.load("CLIP_Average_X_train.pt")
    y_train_tensor = torch.load("CLIP_Average_y_train.pt")
    X_test_tensor = torch.load("CLIP_Average_X_test.pt")
    y_test_tensor = torch.load("CLIP_Average_y_test.pt")
    
    X_train = X_train_tensor.numpy()
    y_train = y_train_tensor.numpy()
    X_test = X_test_tensor.numpy()
    y_test = y_test_tensor.numpy()
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    models = []
    metrics = {}
    
    for qubit_idx in range(5):
        print(f"\nTraining model for qubit {qubit_idx}...")
        
        # Create model with best parameters
        model = SimpleTransformerEstimator(
            task="regression",
            d_model=best_params['d_model'],
            nhead=best_params['nhead'],
            dim_feedforward=best_params['dim_feedforward'],
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay'],
            n_epochs=best_params['n_epochs'],
            batch_size=best_params['batch_size'],
            lr_scheduler='plateau' if best_params['use_scheduler'] else None,
            verbose=True,
            seq_len=X_train.shape[1],
            qubit_idx=qubit_idx
        )
        
        # Train the model
        model.fit(X_train, y_train[:, qubit_idx])
        
        # Save the model
        model_path = os.path.join(save_path, f"best_model_qubit_{qubit_idx}.pt")
        model.save(model_path)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_true = y_test[:, qubit_idx]
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        metrics[f'qubit_{qubit_idx}'] = {
            'rmse': rmse,
            'mae': mae
        }
        
        models.append(model)
        
        print(f"Qubit {qubit_idx} - RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    
    # Calculate overall metrics
    overall_rmse = np.mean([metrics[f'qubit_{i}']['rmse'] for i in range(5)])
    overall_mae = np.mean([metrics[f'qubit_{i}']['mae'] for i in range(5)])
    
    metrics['overall'] = {
        'rmse': overall_rmse,
        'mae': overall_mae
    }
    
    print(f"\nOverall Performance:")
    print(f"Average RMSE: {overall_rmse:.6f}")
    print(f"Average MAE: {overall_mae:.6f}")
    
    # Save metrics
    import json
    with open(os.path.join(save_path, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return models, metrics


if __name__ == "__main__":
    """
    Main execution block to run the Optuna optimization.
    """
    print("Starting CLIP Transformer Architecture Optimization with Optuna")
    print("=" * 60)
    
    # Check if data files exist
    required_files = [
        "CLIP_Average_X_train.pt",
        "CLIP_Average_y_train.pt", 
        "CLIP_Average_X_test.pt",
        "CLIP_Average_y_test.pt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required data files: {missing_files}")
        print("Please ensure the CLIP data files are available in the current directory.")
        exit(1)
    
    print("All required data files found. Starting optimization...")
    
    # Run Optuna optimization with reasonable defaults
    try:
        study = train_with_rand_data_optuna(
            n_trials=100,  # Start with fewer trials for testing
            timeout=18000,  # 30 minutes timeout
            study_name="clip_transformer_qem_optimization"
        )
        
        # Display optimization results
        if study.best_trial is not None:
            print("\nOptimization completed successfully!")
            print(f"Best average RMSE: {study.best_trial.value:.6f}")
            print(f"\nBest parameters:")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")
            
            print(f"\nResults saved to: {study.study_name}_results.json")
            print(f"Optimization plots saved to: {study.study_name}_optimization_history.png and {study.study_name}_param_importances.png")
            
            print("\nTo train final models with these parameters, use:")
            print("models, metrics = train_best_model_from_study(study)")
                
        else:
            print("\nOptimization failed - no successful trials completed.")
            print("Consider:")
            print("1. Increasing the timeout")
            print("2. Reducing the complexity of hyperparameter space")
            print("3. Checking system resources")
            
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nError during optimization: {e}")
        print("Please check the error message above and try again.")

