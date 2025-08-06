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


circuits, Js = load_circuits('../tutorials/data/ising_zne_hardware/100q_brisbane/', list(range(1, 11)))

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


num_circ_per_step = 50
k = train_test_split = 10
train_circuits = []
train_zne_vals = []
train_noisy_vals = []
test_circuits = []
test_zne_vals = []
test_noisy_vals = []
test_Js = []
for start_each_step in list(range(len(circuits))[::num_circ_per_step]):
    train_circuits += circuits[start_each_step:start_each_step+k]
    train_zne_vals += zne_mitigated_vals[start_each_step:start_each_step+k]
    train_noisy_vals += noisy_vals[start_each_step:start_each_step+k]
    test_circuits += circuits[start_each_step+k:start_each_step+num_circ_per_step]
    test_zne_vals += zne_mitigated_vals[start_each_step+k:start_each_step+num_circ_per_step]
    test_noisy_vals += noisy_vals[start_each_step+k:start_each_step+num_circ_per_step]
    test_Js += Js[start_each_step+k:start_each_step+num_circ_per_step]

print(len(train_circuits), len(train_zne_vals), len(train_noisy_vals))
print(len(test_circuits), len(test_zne_vals), len(test_noisy_vals))

few_normal_X_train, few_normal_y_train = encode_data_v2_ecr(train_circuits, train_zne_vals, train_noisy_vals, obs_size=5)
few_normal_X_test, few_normal_y_test = encode_data_v2_ecr(test_circuits, test_zne_vals, test_noisy_vals, obs_size=5)


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

few_normal_train_dataset = TensorDataset(torch.Tensor(few_normal_X_train), torch.Tensor(few_normal_y_train))
few_normal_train_loader = DataLoader(few_normal_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
few_normal_test_dataset = TensorDataset(torch.Tensor(few_normal_X_test), torch.Tensor(few_normal_y_test))
few_normal_test_loader = DataLoader(few_normal_test_dataset, batch_size=BATCH_SIZE*1000, shuffle=False)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

normal_X_train = pd.DataFrame(normal_X_train)
normal_y_train = pd.DataFrame(normal_y_train)
normal_X_test = pd.DataFrame(normal_X_test)
normal_y_test = pd.DataFrame(normal_y_test)

few_normal_X_train = pd.DataFrame(few_normal_X_train)
few_normal_y_train = pd.DataFrame(few_normal_y_train)
few_normal_X_test = pd.DataFrame(few_normal_X_test)
few_normal_y_test = pd.DataFrame(few_normal_y_test)


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

# Custom wrapper to include RF baseline metrics in wandb logging
class SimpleTransformerEstimatorWithBaseline(SimpleTransformerEstimator):
    def __init__(self, *args, rf_baseline_metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rf_baseline_metrics = rf_baseline_metrics or {}
    
    def fit(self, X, y):
        """Override fit to include RF baseline metrics in wandb logging."""
        # Store original wandb logging state
        original_wandb_logging = self.wandb_logging
        
        # If wandb logging is enabled and we have baseline metrics, customize logging
        if original_wandb_logging and self.rf_baseline_metrics:
            # Monkey patch the wandb logging in the parent fit method
            import wandb
            original_wandb_log = wandb.log
            
            def custom_wandb_log(log_dict, *args, **kwargs):
                # Add RF baseline metrics to every log
                enhanced_log_dict = {**log_dict, **self.rf_baseline_metrics}
                return original_wandb_log(enhanced_log_dict, *args, **kwargs)
            
            # Temporarily replace wandb.log
            wandb.log = custom_wandb_log
            
            try:
                # Call parent fit method
                result = super().fit(X, y)
            finally:
                # Restore original wandb.log
                wandb.log = original_wandb_log
                
            return result
        else:
            # No custom logging needed, use parent method
            return super().fit(X, y)

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

few_normal_X_train = pd.DataFrame(few_normal_X_train)
few_normal_y_train = pd.DataFrame(few_normal_y_train)
few_normal_X_test = pd.DataFrame(few_normal_X_test)
few_normal_y_test = pd.DataFrame(few_normal_y_test)

# model = SimpleTransformerEstimator(n_epochs=100, 
#                                    verbose=True,
#                                    device=device,
#                                    seq_len=682)
# model.fit(X_train, y_train)

# Random Forest Baseline Training (moved to beginning for baseline comparison)
print("Training Random Forest baseline models...")
normal_rfr_tree_list = []
for q in range(5):
    print(f"Training Random Forest baseline model for qubit {q}...")
    rfr = RandomForestRegressor(n_estimators=100, verbose=0, n_jobs=-1)
    rfr.fit(few_X_train, few_y_train.iloc[:, q])
    normal_rfr_tree_list.append(rfr)
    print(f"Done with the {q} model")

# Evaluate Random Forest baseline and store metrics for epoch-by-epoch logging
print("Evaluating Random Forest baseline...")
rf_results = evaluate_loader(test_loader, normal_rfr_tree_list, label="RandomForest_Baseline")
rf_df = pd.DataFrame(rf_results)

# Store RF baseline metrics to be logged during transformer training
rf_baseline_metrics = {}
for q in range(5):
    rmse_input = np.sqrt(rf_df[f"dist_sq_{q}"].mean())
    rmse_mitigated = np.sqrt(rf_df[f"dist_sq_mitigated_{q}"].mean())
    
    rf_baseline_metrics[f"rf_baseline_rmse_input_q{q}"] = rmse_input
    rf_baseline_metrics[f"rf_baseline_rmse_mitigated_q{q}"] = rmse_mitigated
    rf_baseline_metrics[f"rf_baseline_mean_dist_sq_q{q}"] = rf_df[f"dist_sq_{q}"].mean()
    rf_baseline_metrics[f"rf_baseline_mean_dist_sq_mitigated_q{q}"] = rf_df[f"dist_sq_mitigated_{q}"].mean()

# Overall RF baseline metrics
rf_baseline_metrics["rf_baseline_overall_rmse_input"] = np.sqrt(np.mean([rf_df[f"dist_sq_{q}"].mean() for q in range(5)]))
rf_baseline_metrics["rf_baseline_overall_rmse_mitigated"] = np.sqrt(np.mean([rf_df[f"dist_sq_mitigated_{q}"].mean() for q in range(5)]))

print(f"Random Forest Baseline - Overall RMSE Input: {rf_baseline_metrics['rf_baseline_overall_rmse_input']:.4f}, RMSE Mitigated: {rf_baseline_metrics['rf_baseline_overall_rmse_mitigated']:.4f}")
    

number_of_epochs = 1000
number_heads = 1
model_depth = 8

# Create timestamp for this training run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize wandb for CLIP+Transformer training
if WANDB_AVAILABLE:
    wandb.init(project="vigir-ml-qem", 
               name=f"CLIP_Transformer_Training_{timestamp}_{number_heads}_{model_depth}",
               config={
                   "n_epochs": number_of_epochs,
                   "n_heads": number_heads,
                   "model_depth": model_depth,
                   "lr": 0.001,
                   "lr_scheduler": "step",
                   "step_size": 300,
                   "model_type": "CLIP+Transformer",
                   "seq_len": 682,
                   "timestamp": timestamp,
                   "n_qubits": 5
               })

trained_transformer_list_with_CLIP = []
for q in range(5):
    print(f"Training CLIP+Transformer model for qubit {q}...")
    
    model = SimpleTransformerEstimatorWithBaseline(n_epochs=number_of_epochs, 
                                                   nhead=number_heads,
                                                   d_model=model_depth,
                                                   lr=0.001,
                                                   verbose=False,
                                                   device=device,
                                                   lr_scheduler='step',
                                                   step_size=300,
                                                   pretrained_path=f'pretrained_q{q}.pt',
                                                   seq_len=682,
                                                   wandb_logging=WANDB_AVAILABLE,
                                                   eval_data=(X_test, y_test.iloc[:, q]),
                                                   qubit_idx=q,
                                                   rf_baseline_metrics=rf_baseline_metrics)
    
    # model.fit(X_train, y_train.iloc[:, q])
    model.fit(few_X_train, few_y_train.iloc[:, q])
    
    # Save model locally and as wandb artifact
    local_path = f"/home/macula/SMATousi/Desktop/clip_transformer_q{q}_{timestamp}_epoch{number_of_epochs}.pt"
    artifact_name = f"clip_transformer_q{q}_{timestamp}"
    model.save(local_path, wandb_artifact_name=artifact_name)
    
    trained_transformer_list_with_CLIP.append(model)
    print(f"Done with {q}")

    # Remove break to train all qubits
    # break

if WANDB_AVAILABLE:
    wandb.finish()


# Initialize wandb for regular Transformer training  
# Create new timestamp for second training run
timestamp_regular = datetime.now().strftime("%Y%m%d_%H%M%S")

if WANDB_AVAILABLE:
    wandb.init(project="vigir-ml-qem", 
               name=f"Regular_Transformer_Training_{timestamp_regular}_{number_heads}_{model_depth}",
               config={
                   "n_epochs": number_of_epochs,
                   "n_heads": number_heads, 
                   "model_depth": model_depth,
                   "lr": 0.001,
                   "lr_scheduler": "step",
                   "step_size": 300,
                   "model_type": "Regular_Transformer",
                   "seq_len": 682,
                   "timestamp": timestamp_regular,
                   "n_qubits": 5
               })

trained_transformer_list = []
for q in range(5):
    print(f"Training regular Transformer model for qubit {q}...")
    
    model = SimpleTransformerEstimatorWithBaseline(n_epochs=number_of_epochs, 
                                                   nhead=number_heads,
                                                   d_model=model_depth,
                                                   lr=0.001,
                                                   verbose=False,
                                                   device=device,
                                                   lr_scheduler='step',
                                                   step_size=30,
                                                   batch_size=16,
                                                   seq_len=170,
                                                   wandb_logging=WANDB_AVAILABLE,
                                                   eval_data=(normal_X_test, normal_y_test.iloc[:, q]),
                                                   qubit_idx=q,
                                                   rf_baseline_metrics=rf_baseline_metrics)
    
    # model.fit(normal_X_train, normal_y_train.iloc[:, q])
    model.fit(few_normal_X_train, few_normal_y_train.iloc[:, q])
    
    # Save model locally and as wandb artifact
    local_path = f"/home/macula/SMATousi/Desktop/regular_transformer_q{q}_{timestamp_regular}_epoch{number_of_epochs}.pt"
    artifact_name = f"regular_transformer_q{q}_{timestamp_regular}"
    model.save(local_path, wandb_artifact_name=artifact_name)
    
    trained_transformer_list.append(model)
    print(f"Done with {q}")

    # Remove break to train all qubits
    # break

if WANDB_AVAILABLE:
    wandb.finish()




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


