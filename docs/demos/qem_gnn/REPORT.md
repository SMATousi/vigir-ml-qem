# QAGT-MLP: what changed, why, and what the model is now

Working notes for the `docs/demos/qem_gnn` pipeline. Starting question: *the
QAGT-MLP is not outperforming the Random Forest baseline, and the margin is not
big.* Four separate causes turned out to be responsible. Three of them were
bugs or mis-framings rather than model capacity, and fixing them in the wrong
order made results temporarily worse — that sequence is documented below,
because the intermediate failure is the evidence for the final design.

Branch: `qem-gnn/tier01-discrete-head` (11 commits, `f1462271`..`c1316e26`).

---

## 1. Results

Validation MAE / RMSE, 5-seed ensemble, all 11 simulator datasets. `noisy` is
the unmitigated expectation value; `RF` is the per-qubit Random Forest on the
same split.

| dataset | noisy MAE | RF MAE | **QAGT MAE** | noisy RMSE | RF RMSE | **QAGT RMSE** | winner |
|---|---|---|---|---|---|---|---|
| haoran_mbd_coherent_random_cliffords | 0.1317 | 0.0199 | **0.0138** | 0.1812 | 0.1043 | 0.1073 | QAGT |
| haoran_mbd_random_brickwork | 0.0679 | 0.0221 | **0.0198** | 0.0915 | 0.0307 | **0.0274** | QAGT |
| haoran_mbd_random_cliffords | 0.0870 | 0.0025 | **0.0025** | 0.1177 | 0.0057 | **0.0055** | QAGT |
| ising_dataset | 0.0725 | **0.0088** | 0.0109 | 0.1023 | **0.0117** | 0.0141 | RF (1.23x) |
| ising_dataset_random_init | 0.0700 | 0.0273 | **0.0148** | 0.1008 | 0.0415 | **0.0193** | QAGT |
| ising_init_0110 | 0.0858 | **0.0114** | 0.0132 | 0.1233 | **0.0155** | 0.0172 | RF (1.15x) |
| ising_init_from_qasm | 0.0791 | **0.0158** | 0.0170 | 0.1166 | 0.0233 | **0.0231** | RF (1.08x) |
| ising_init_from_qasm_coherent | 0.1089 | 0.0230 | **0.0186** | 0.1638 | 0.0412 | **0.0272** | QAGT |
| ising_init_from_qasm_no_readout | 0.0593 | **0.0157** | 0.0165 | 0.0977 | 0.0243 | **0.0220** | RF (1.05x) |
| mbd_theta_0p05pi | 0.1515 | **0.0061** | 0.0063 | 0.1989 | 0.0114 | **0.0091** | RF (1.03x) |
| mbd_theta_0p1pi_coherent | 0.1605 | 0.0566 | **0.0302** | 0.2547 | 0.1180 | **0.0605** | QAGT |

The `winner` column is by MAE. QAGT figures use `argmax` for the level head and
the raw output for the residual head; on `haoran_mbd_coherent_random_cliffords`
RF has the better RMSE in this column (0.1043 vs 0.1073), but the same trained
head decoded with `expect` gives **0.0984**, which wins — see §2.4.

**QAGT beats RF on 6/11 by MAE and 8/11 by RMSE** (9/11 if the RMSE-optimal
decoding is used where available). At the start of this work it
was 4/11, and where RF led it led by up to 3x; it now leads by at most 1.23x.
Every model beats the unmitigated baseline on every dataset.

On the two Clifford datasets the level head reports classification accuracy:
**0.9885** and **1.0000**.

### Progression

The same metric across the three stages of this work (5-seed ensemble MAE):

| dataset | (a) before | (b) after lightcone fix | (c) **final** |
|---|---|---|---|
| haoran_mbd_coherent_random_cliffords | 0.0136 | 0.0133 | 0.0138 |
| haoran_mbd_random_brickwork | 0.0233 | 0.0349 | **0.0198** |
| haoran_mbd_random_cliffords | 0.0025 | 0.0025 | 0.0025 |
| ising_dataset | 0.0121 | 0.0185 | **0.0109** |
| ising_dataset_random_init | 0.0161 | 0.0292 | **0.0148** |
| ising_init_0110 | 0.0158 | 0.0363 | **0.0132** |
| ising_init_from_qasm | 0.0239 | 0.0350 | **0.0170** |
| ising_init_from_qasm_coherent | 0.0688 | 0.0582 | **0.0186** |
| ising_init_from_qasm_no_readout | 0.0226 | 0.0342 | **0.0165** |
| mbd_theta_0p05pi | 0.0065 | 0.0092 | **0.0063** |
| mbd_theta_0p1pi_coherent | 0.0436 | 0.0991 | **0.0302** |

(c) beats (b) on 9/11 (mean −42%) and beats (a) on 9/11 (mean −19%). Stage (b)
is a *correctness* fix that made results worse; §4.3 explains why, and that
explanation is what motivated the final change.

### Datasets

All 11 are 5-qubit with 4 measured qubits. Medians are per circuit over the
training split; `measure`/`barrier` are excluded from gate counts.

| dataset | train | val | gates | 2q | depth med [min–max] | gate mix | targets |
|---|---|---|---|---|---|---|---|
| haoran_mbd_coherent_random_cliffords | 5000 | – | 31 | 10 | 19 [2–56] | rz 42% cx 32% sx 21% | 3-level |
| haoran_mbd_random_brickwork | 5000 | – | 143 | 13 | 57 [2–111] | rz 51% sx 39% cx 9% | continuous |
| haoran_mbd_random_cliffords | 5000 | 2000 | 32 | 10 | 21 [2–57] | rz 43% cx 31% sx 21% | 3-level |
| ising_dataset | 5000 | 2000 | 66 | 12 | 25 [1–48] | rz 53% sx 29% cx 19% | continuous |
| ising_dataset_random_init | 5000 | 2000 | 73 | 13 | 31 [2–63] | rz 52% sx 29% cx 19% | continuous |
| ising_init_0110 | 5000 | 2000 | 132 | 27 | 52 [4–103] | rz 52% sx 28% cx 20% | continuous |
| ising_init_from_qasm | 4500 | 1500 | 230 | 50 | 90 [8–168] | rz 51% sx 27% cx 22% | continuous |
| ising_init_from_qasm_coherent | 4500 | 1500 | 230 | 50 | 90 [8–168] | rz 51% sx 27% cx 22% | continuous |
| ising_init_from_qasm_no_readout | 4500 | 1500 | 230 | 50 | 90 [8–168] | rz 51% sx 27% cx 22% | continuous |
| mbd_theta_0p05pi | 5000 | 2000 | 133 | 13 | 55 [2–109] | rz 48% sx 42% cx 10% | continuous |
| mbd_theta_0p1pi_coherent | 5000 | 2000 | 133 | 13 | 55 [2–109] | rz 48% sx 42% cx 10% | continuous |

The three `ising_init_from_qasm` variants are the *same circuits* under three
noise models, so they form a controlled triple.

Combined figure: `figures/rmse_comparison_all_datasets.png`.

---

## 2. The final model

Predict, for each circuit and each measured qubit `m`, the ideal expectation
value `y_m`, given the circuit and the noisy measured value `noisy_z_m`.

### 2.1 Circuit → graph

`circuits_to_graph.circuit_to_gategraph_data`. One node per DAG op node
(including `measure` and `barrier`); edges connect consecutive operations along
each qubit wire, directed forward in time.

**Node features, 20 dims** (`featurizers.build_node_feature`):

| dims | content |
|---|---|
| 1 | gate id, integer from `DEFAULT_GATE_DICT` (22 = unknown/measure/barrier) |
| 1 | arity (number of qubits the gate acts on) |
| 2 | first two gate parameters, raw floats, zero-padded |
| 8 | sinusoidal encoding of qubit index, **summed** over the gate's qargs |
| 8 | sinusoidal encoding of the gate's moment (topological layer) |

**Per-graph tensors:**

| attribute | shape | meaning |
|---|---|---|
| `lightcone_masks` | `[N, M]` | node `n` is in the backward lightcone of measured qubit `m` |
| `wire_masks` | `[N, M]` | node `n` acts on measured qubit `m`'s **own wire** (added here) |
| `measured_qubits` | `[M]` | physical qubit index per output column (not `[0..M-1]`; e.g. `[2,1,3,4]`) |
| `noisy_z`, `y` | `[M]` | unmitigated and ideal values, attached from the labels CSV |

Batching is graph-major: `measured_qubits`, `noisy_z` and `y` all concatenate to
`[B*M]` in the same order as the pooled output, and the mask tensors concatenate
to `[N_total, M]`.

### 2.2 Encoder

`model.GraphEncoder`, unchanged. Linear projection to `d_model`, then `layers`
blocks of `TransformerConv` → GELU → dropout → LayerNorm with a residual. For
the continuous datasets: `d_model=128, layers=3, heads=4, dropout=0.05`.

### 2.3 Pooling — `QubitConditionedPoolingV2` (`qem_ext.py`)

This is the substantive architectural change. For each graph `b` and target `m`:

```
q_m      = MLP([ embed(physical_qubit_m), noisy_z_m ])        # [d]
l_nm     = <Key(h_n), q_m> / sqrt(d)  +  alpha * wire_nm      # [N_b]
a_nm     = softmax_n( l_nm  masked to lightcone_masks[:, m] )
pooled_m = sum_n a_nm * h_n                                   # [d]
global_b = mean_n h_n                                         # [d]
```

Two roles are deliberately separated:

- **`lightcone_masks` is the hard support** — which gates *can physically*
  influence the observable. Nodes outside it get `-inf` and cannot contribute.
- **`wire_masks` is a soft learned preference** — which gates act *most
  directly* on the observable, via a single learned scalar `alpha`.

The query depends on the target qubit's identity and on how corrupted its value
already is, so the `M` outputs attend differently over the same node set. The
predecessor of this module scored nodes with one shared `query` parameter, which
did not depend on `m` at all (§4.3).

`alpha` converges to **+0.59 … +1.22** across the 11 datasets — always
positive, i.e. every model learns to up-weight own-wire gates.

### 2.4 Head

Chosen automatically from the training labels by `qem_levels.LevelSet.detect`.

**Continuous targets → `QEMResidualHead`:**

```
y_hat_m = noisy_z_m + MLP([ pooled_m, global_b, noisy_z_m ])
```

The final linear layer is **zero-initialised**, so training starts exactly at
`y_hat = noisy_z` — the unmitigated baseline — and can only improve on it.
Without this the head must learn the identity map out of a `2d+1` concat in
which `noisy_z` is a single dimension competing with `2d` random ones.

**Discrete targets → `QEMLevelHead`:** `K` logits over the detected level set,

```
logits_mk = MLP([ pooled_m, global_b, noisy_z_m ])_k  -  gamma * (noisy_z_m - l_k)^2
```

with the MLP's last layer zero-initialised and `gamma` learned (softplus,
init 10). At step 0 the argmax is therefore exactly *"snap `noisy_z` to the
nearest level"* — a strong zero-training baseline the model starts from.

**Decoding.** `argmax` is MAE-optimal; `sum_k p_k * l_k` ("expect") is
RMSE-optimal, because hedging genuinely is correct under squared error. Both
are reported from the same trained head.

### 2.5 Training (`qem_train.TrainConfig`)

| setting | value | why |
|---|---|---|
| loss | SmoothL1 `beta=0.05` / cross-entropy | the PyTorch default `beta=1.0` never leaves the quadratic regime for targets in [-2,2] — it is exactly `0.5*MSE` |
| lr / schedule | 1e-3, cosine, 5-epoch warmup | the original loop held lr constant and the curve was flat from ~epoch 45 |
| epochs / patience | 80 / 25 | — |
| optimiser | AdamW, wd 1e-4, grad-clip 1.0 | unchanged |
| batch size | 16 | unchanged |
| seeds | 0–4, ensembled | a single unseeded run is not comparable to a 100-tree forest |
| selection | val MAE, or accuracy for the level head | — |

Ensembling averages **probabilities** (not logits) for the level head, so one
overconfident member cannot dominate.

### 2.6 Sizes

| configuration | total | encoder | pooling | head |
|---|---|---|---|---|
| residual + conditioned, `d_model=128` | 292,610 | 201,600 | 57,857 | 33,153 |
| residual + shared pooling (previous) | 251,393 | 201,600 | 16,640 | 33,153 |
| level + conditioned, `d_model=64` | 76,805 | 51,648 | 16,641 | 8,516 |

Conditioned pooling costs ~41k parameters over the shared version.

---

## 3. Modularity

`model.py` and `train_loop.py` are **untouched**. Everything is opt-in, so other
applications importing them are unaffected:

- `QEMGraphTransformerX(head="scalar", pool="shared")` has a **bit-identical
  `state_dict`** to `QEMGraphTransformer` (verified).
- `TrainConfig()` with default arguments reproduces the original training loop
  exactly (SmoothL1 `beta=1.0`, constant lr, no seeding, selection on MAE).
- `LevelSet.detect` returns `None` on continuous targets, so the classification
  path is never taken where it doesn't apply — no per-notebook special-casing.

Two changes are *not* opt-in because they are correctness fixes, and both alter
graph construction for every consumer: the `dag.predecessors()` fix in
`lightcone.py` (§4.2) and the added `wire_masks` attribute.

### New files

| file | contents |
|---|---|
| `qem_levels.py` | `LevelSet` — detection from training labels, `snap` / `argmax` / `expect` decoders |
| `qem_ext.py` | `QEMGraphTransformerX`, `QEMResidualHead`, `QEMLevelHead`, `QubitConditionedPoolingV2` |
| `qem_train.py` | `TrainConfig`, `train_qem`, `train_ensemble`, `ensemble_predict`, `metrics` |

### Modified files

| file | change |
|---|---|
| `lightcone.py` | fixed `build_predecessor_map` (§4.2) |
| `circuits_to_graph.py` | added `wire_masks` |
| `featurizers.py` | unchanged |
| `generate_sim_notebooks.py` | emits the Tier section via `tier01_cells()`; original-training cell removed |
| `sim_*.ipynb` (11) | Tier section added; original training removed; `assign_moments_topo` fixed |
| `01_build_graphs.ipynb` | `assign_moments_topo` fixed |

---

## 4. The four findings, in order

### 4.1 The targets were discrete, so the metric was measuring shrinkage

On `haoran_mbd_coherent/random_cliffords` the ideal `<Z>` is exactly −1, 0 or
+1: 68.1% of labels are ±1, 31.9% are ~0 (the ±0.002 spread is shot noise on an
exact zero), and **0.0% fall between 0.1 and 0.9**. An MSE-equivalent regressor
trained on a 3-atom distribution hedges — it emits 0.87 where the truth is 1.0
and pays 0.13 MAE on a point whose class it already got right.

98.9% of validation points were already classified correctly, and those points
contributed 0.0527 of the total 0.0596 MAE: **~87% of the reported error was
shrinkage, not mistakes.** Snapping the existing checkpoint's predictions to the
nearest level, with no retraining, moved it from MAE 0.0596 to 0.0138.

The original model was already *ahead* of every RF variant on class accuracy
(0.9885 vs 0.9872/0.9878). It was being scored on the calibration of an answer
it had already got right.

→ discrete-level head, with detection so the continuous datasets are unaffected.

### 4.2 `dag.predecessors()` was silently emptying every lightcone

`lightcone.build_predecessor_map`, and the `assign_moments_topo` patch at the
top of every notebook, filtered predecessors with:

```python
[e.node for e in dag.predecessors(node)
 if hasattr(e, "node") and isinstance(e.node, DAGOpNode)]
```

`dag.predecessors()` yields `DAGOpNode` objects **directly** in qiskit 0.24.1,
and those have no `.node` attribute. `hasattr(p, "node")` is `False` for every
one, so the comprehension returned `[]` unconditionally. Three silent effects:

1. `compute_lightcone_nodes` never walked backwards. Seeded with "every node
   touching wire q" and expanded through an always-empty predecessor map, the
   "backward lightcone" was **just the gates on that one wire**, missing every
   ancestor arriving through another qubit.
2. Every node got moment 0, so the 8-dim sinusoidal moment encoding was the
   constant `[0,1,0,1,0,1,0,1]` on every node — **8 of 20 input features carried
   no information**, and the model could not tell where in the circuit a gate
   sat.
3. `depth_layers` was 1 for every graph (unused; cosmetic).

Measured on 25 `ising_init_from_qasm` circuits, matched by `circuit_path`
(node counts identical, only masks and features change):

| | before | after |
|---|---|---|
| lightcone size | 35.1 nodes (36% of circuit) | 94.1 nodes (97%) |
| `depth_layers` | 1.0 | 46.8 |

### 4.3 Fixing that made results worse — and why

The correct lightcone regressed 8/11 datasets, up to +130%. The cause,
measured rather than assumed:

A true backward lightcone in a 5-qubit circuit of depth 20–90 is almost the
entire circuit, so all `M` masks become the same set:

| dataset | coverage | pairwise mask overlap |
|---|---|---|
| haoran_mbd_coherent_random_cliffords | 35% → 87% | 0.14 → 0.90 |
| ising_init_0110 | 36% → 96% | 0.20 → 0.97 |
| mbd_theta_0p1pi_coherent | 29% → 93% | 0.07 → 0.94 |
| ising_init_from_qasm_coherent | 36% → 98% | 0.20 → 0.98 |

The old `QubitConditionedPooling` scored nodes with a single shared `query`, so
the logit did not depend on `m` — **the mask was the only thing separating
output column `m` from `m'`**. Inspecting the trained models, the `M` pooled
vectors were bit-identical (cosine **1.0000**) on two of three datasets checked,
and on `ising_init_0110` the pooled vector was constant across *circuits* too —
the encoder had collapsed to a constant and the model had degenerated into
`noisy_z` plus a global offset.

So the bug had been accidentally supplying the qubit conditioning the
architecture never had: "the gates on wire q" is a sharply qubit-specific mask.
Removing it exposed the real design flaw.

The outputs did *not* collapse, because `noisy_z_m` feeds the head directly —
what was lost was the graph's contribution to telling the columns apart.

### 4.4 Conditioning the pooling deliberately

§2.3. Rather than reverting to a physically wrong lightcone for its accidental
benefit, the two roles were separated: correct lightcone as hard support, wire
membership as an explicit learned preference, and a query that actually depends
on the target qubit.

Pre-flight on a 1200/300 subset of `ising_init_0110`, 12 epochs:

| pooling | val MAE | pooled cosine across qubits | std across circuits |
|---|---|---|---|
| `shared` | 0.0908 | 1.0000 | 0.0000 (encoder dead) |
| `conditioned` | **0.0578** | 0.2819 | 0.4019 (encoder alive) |

The full suite then confirmed it: better than the accidental-conditioning
baseline on 9/11.

`ising_init_from_qasm_coherent` — an outlier that was 3x worse than RF and the
*only* dataset helped by the lightcone fix alone — went **0.0688 → 0.0186
(−73%)** and now beats RF. Its anomaly really was the pooling collapse.

---

## 5. Running it

Notebooks execute top to bottom; there is no longer a separate baseline-training
stage. Roughly 12–37 min per notebook on one RTX 3090, ~5.4 h for all 11:

```
build graphs → RF baseline → noisy vs RF → Tier 0+1+2 section
```

The Tier section detects the level set, trains one seeded model, trains a
5-seed ensemble, prints the comparison table and writes
`runs/<dataset>/rmse_comparison_tier01.{png,csv}`.

Batch execution:

```bash
for nb in sim_*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace --allow-errors \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb"
done
```

`runs/` is gitignored (`.gitignore:202`); committed figures live in `figures/`.

---

## 6. Known limitations

Measured or verified, not speculative:

- **Gate identity is an ordinal.** `float(gid)` is fed as a raw number, so `cx`
  (11) is literally "eleven times" `h` (6) to the input projection. Should be an
  embedding. Probably the largest remaining feature defect.
- **Angles are raw.** `rz(theta)` enters as a float, though gate action is
  2π-periodic; `(cos, sin)` is the right encoding. There are 70k `rz` gates in
  the Clifford dataset alone.
- **Qubit encodings are summed over qargs**, so for `cx(q0,q1)` control and
  target are indistinguishable and ordering is lost.
- **Edges are forward-in-time only**, so with 3 layers a node sees 3 gates of
  history and nothing downstream. No edge features (which wire, control vs
  target).
- **The split is random over circuits pooled from ~10 Trotter-step files.** Step
  index is a strong covariate, so step-level structure leaks. A held-out-step
  split is the harder and more meaningful test, and the one where a structural
  inductive bias should pay off.
- **Gradient imbalance.** With the zero-init head, encoder gradients run ~10x
  weaker than the head's and pooling ~1000x weaker in early steps. The head can
  fit `noisy_z` alone, which may starve the graph path.
- **Reproducibility.** `TransformerConv`'s GPU scatter ops are not
  bit-reproducible, so a fixed seed still moves run to run. On the Clifford
  datasets per-seed accuracy spans 0.9882–0.9890; differences inside that band
  are noise.
- **Pooling is a Python loop over the batch.** Correct but slow; the
  conditioned version added ~50% to suite runtime. `torch_geometric.utils.softmax`
  would vectorise it.

## 7. Notes on the record

- Two notebooks had inconsistent epoch counts (`coherent_random_cliffords` 5,
  `brickwork` 10, against 80 elsewhere) and were normalised to 80 so the suite
  is comparable. Their pre-normalisation numbers are not comparable to the rest.
- Before the original baseline training was removed, four notebooks had it
  pinned to `device='cpu'`, which is why it dominated runtime.
- `sim_haoran_mbd_coherent_random_cliffords.ipynb` keeps a results-discussion
  cell quoting the original model's numbers; those rows are annotated as coming
  from a run predating the baseline removal, since the notebook no longer
  reproduces them.
