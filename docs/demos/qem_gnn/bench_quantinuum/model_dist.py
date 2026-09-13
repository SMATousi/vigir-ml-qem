"""QAGT-MLP with a distribution head, for the Placidi et al. Pauli benchmark.

Their task differs from ours in what is predicted: a full 32-dimensional
distribution over 5-qubit bitstrings, not a per-qubit expectation value. The
part of our model under test is unchanged -- the circuit-graph encoder and the
qubit-conditioned attention pooling of Section 3.4 -- and only the head is
swapped, in the same opt-in way as every other variant in `qem_ext.py`.

The head keeps the residual construction we use elsewhere. It predicts a
correction in log space to the measured distribution,

    logits = log(noisy + eps) + Delta,     Delta = MLP([p_1..p_M, g, noisy])

with the last layer zero-initialised, so at step 0 the model outputs exactly the
unmitigated distribution and training can only depart from it deliberately. The
per-qubit query is conditioned on the marginal P(qubit m = 1) implied by the
measured distribution, which is the natural analogue of the noisy expectation
value z_m the pooling uses on our own datasets.
"""
from typing import Optional

import torch
import torch.nn as nn

from model import GraphEncoder
from qem_ext import GCNGraphEncoder, QubitConditionedPoolingV2, UniformLightconePooling, _zero_init_last_linear


class QAGTDistribution(nn.Module):
    def __init__(self, in_dim: int = 20, d_model: int = 128, layers: int = 3,
                 heads: int = 4, dropout: float = 0.1, n_out: int = 32,
                 n_measured: int = 5, hidden: int = 256, desc_dim: int = 0,
                 pool: str = 'conditioned', backbone: str = 'transformer',
                 use_wire: bool = True, use_lightcone: bool = True,
                 use_qubit_emb: bool = True, use_local: bool = True,
                 use_global: bool = True, zero_init: bool = True):
        super().__init__()
        Enc = GraphEncoder if backbone == 'transformer' else GCNGraphEncoder
        self.encoder = Enc(in_dim, d_model, layers, heads, dropout)
        if pool == 'mean':
            self.pool = UniformLightconePooling(d_model, use_lightcone=use_lightcone)
        else:
            self.pool = QubitConditionedPoolingV2(
                d_model, use_noisy=True, use_wire=use_wire,
                use_lightcone=use_lightcone, use_qubit_emb=use_qubit_emb)
        self.M, self.n_out = n_measured, n_out
        self.use_local, self.use_global = use_local, use_global
        self.desc_dim = desc_dim
        # The measured distribution is the input that carries most of the
        # signal, but as 32 raw probabilities (~0.03 each) it is a thin, small
        # slice of a concatenation dominated by M*d unit-scale graph features.
        # Measured: left that way the head barely trains -- gradient norms stay
        # under the 1.0 clip threshold and validation KL moves 0.885 -> 0.87 in
        # four epochs, against 0.758 for a plain MLP on the same 32 numbers.
        # Projecting each source to a common width and normalising the
        # concatenation fixes it.
        self.noisy_proj = nn.Sequential(nn.Linear(n_out, d_model), nn.GELU())
        self.desc_proj = (nn.Sequential(nn.Linear(desc_dim, d_model), nn.GELU())
                          if desc_dim else None)
        in_head = n_measured * d_model + d_model + d_model + (d_model if desc_dim else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(in_head),
            nn.Linear(in_head, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )
        if zero_init:
            _zero_init_last_linear(self.head)

    def forward(self, data):
        H = self.encoder(data.x, data.edge_index)
        pooled, glob = self.pool(H, data.lightcone_masks, data.ptr,
                                 measured_qubits=data.measured_qubits,
                                 noisy_z=data.noisy_z,
                                 wire_masks=getattr(data, 'wire_masks', None))
        B = data.ptr.numel() - 1
        pooled = pooled.view(B, self.M * pooled.shape[-1])
        if not self.use_local:
            pooled = torch.zeros_like(pooled)
        if not self.use_global:
            glob = torch.zeros_like(glob)
        noisy = data.noisy_dist.view(B, self.n_out)
        # log-probabilities are on a comparable scale to the other blocks
        parts = [pooled, glob, self.noisy_proj(torch.log(noisy.clamp_min(1e-12)))]
        if self.desc_dim:
            parts.append(self.desc_proj(data.descriptors.view(B, self.desc_dim)))
        delta = self.head(torch.cat(parts, dim=1))
        # residual in log space: at initialisation this is exactly the
        # unmitigated distribution
        return torch.log(noisy.clamp_min(1e-12)) + delta

    @staticmethod
    def probs(logits):
        return torch.softmax(logits, dim=-1)
