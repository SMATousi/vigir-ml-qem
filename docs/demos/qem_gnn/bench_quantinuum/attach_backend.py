"""Attach the per-circuit backend calibration vector to the cached graphs.

Their models receive this block: the released flat encoding is
``concat(circuit.reshape(T, 25), backend_info)``, 25 + 101 = 126 features per
timestep. It varies from circuit to circuit (per-feature spread up to ~200), so
withholding it from our model would make the comparison unfair in our
disfavour. It enters through the existing `desc_dim` pathway, as a graph-level
vector, rather than being repeated at every timestep.

The raw entries span roughly 0 to 1130, which would swamp a concatenated head
input, so they are standardized with statistics computed on the *training*
split alone and reused for val and test.
"""
import argparse, json, os
import numpy as np
import torch

DATA_ROOT = os.environ.get(
    'QEM_BENCH_ROOT',
    '/root/stousi_missouri.edu_01M26D5DP6FR5334PATFM1NQMK/DATA/pauli')
DEPTH_DIR = 't_3_4_5_6_9'
CLIP = 10.0


def raw_backend(device, split, n, version=0):
    root = f'{DATA_ROOT}/{device}_pauli_real_{split}/{DEPTH_DIR}/mixed'
    return np.stack([np.load(f'{root}/{i}/input_backend_info_array_{version}.npy')
                     for i in range(n)]).astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='algiers')
    p.add_argument('--version', type=int, default=0)
    a = p.parse_args()
    B = 'bench_quantinuum'
    stats_path = f'{B}/backend_stats_{a.device}.json'

    cache = {}
    for split in ('train', 'val', 'test'):
        g = torch.load(f'{B}/graphs_{a.device}_{split}_v0.pt', weights_only=False)
        cache[split] = (g, raw_backend(a.device, split, len(g), a.version))
        print(f'  {split}: {len(g)} graphs, backend {cache[split][1].shape}')

    mu = cache['train'][1].mean(0)
    sd = cache['train'][1].std(0)
    sd[sd < 1e-12] = 1.0                       # constant features carry nothing
    json.dump({'mean': mu.tolist(), 'std': sd.tolist(),
               'n_constant': int((cache['train'][1].std(0) < 1e-12).sum())},
              open(stats_path, 'w'))
    print(f'  standardized on train; {int((cache["train"][1].std(0) < 1e-12).sum())}'
          f'/101 features are constant across the training split')

    for split, (g, raw) in cache.items():
        z = np.clip((raw - mu) / sd, -CLIP, CLIP)
        for i, d in enumerate(g):
            d.descriptors = torch.tensor(z[i], dtype=torch.float).view(1, -1)
        torch.save(g, f'{B}/graphs_{a.device}_{split}_v0.pt')
        print(f'  {split}: attached, |z| max {np.abs(z).max():.2f}')


if __name__ == '__main__':
    main()
