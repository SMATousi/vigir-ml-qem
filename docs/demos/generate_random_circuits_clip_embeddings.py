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


print(len(circuits), len(zne_mitigated_vals), len(noisy_vals))


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


from importlib import reload
import RandomCircuitGenerator
reload(RandomCircuitGenerator)
from RandomCircuitGenerator import generate_random_circuits


random_circuits, random_expect_ideal_per_qubit, random_expect_noise_per_qubit = generate_random_circuits(
    num_qubits=5, 
    num_circuits=10000, 
    circuit_depth=10
)

train_circuits.extend(random_circuits)
train_zne_vals.extend(random_expect_ideal_per_qubit)
train_noisy_vals.extend(random_expect_noise_per_qubit)

print(len(train_circuits), len(train_zne_vals), len(train_noisy_vals))
print(len(test_circuits), len(test_zne_vals), len(test_noisy_vals))

normal_X_train, normal_y_train = encode_data_v2_ecr(train_circuits, train_zne_vals, train_noisy_vals, obs_size=5)
normal_X_test, normal_y_test = encode_data_v2_ecr(test_circuits, test_zne_vals, test_noisy_vals, obs_size=5)


torch.save(normal_X_train, 'Normal_X_train_random_added.pt')
torch.save(normal_y_train, 'Normal_y_train_random_added.pt')
torch.save(normal_X_test, 'Normal_X_test_random_added.pt')
torch.save(normal_y_test, 'Normal_y_test_random_added.pt')

print(normal_X_train.shape, normal_y_train.shape)
print(normal_X_test.shape, normal_y_test.shape)


# CLIP Encoder
from transformers import CLIPTokenizer, CLIPTextModel
import torch
from tqdm import tqdm
import time

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_model.eval()

X_train, y_train = encode_data_v2_ecr_CLIP_encoder(train_circuits, 
                                                   train_zne_vals, 
                                                   train_noisy_vals, 
                                                   tokenizer=tokenizer,
                                                   text_model=text_model,
                                                   device=device,
                                                   obs_size=5)
                                                   
torch.save(X_train, 'CLIP_Average_X_train_random_added.pt')
torch.save(y_train, 'CLIP_Average_y_train_random_added.pt')

X_test, y_test = encode_data_v2_ecr_CLIP_encoder(test_circuits, 
                                                 test_zne_vals, 
                                                 test_noisy_vals, 
                                                 tokenizer=tokenizer,
                                                 text_model=text_model,
                                                 device=device,
                                                 obs_size=5)

torch.save(X_test, 'CLIP_Average_X_test_random_added.pt')
torch.save(y_test, 'CLIP_Average_y_test_random_added.pt')


