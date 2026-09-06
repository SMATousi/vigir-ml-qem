# qem_ext.py
"""Opt-in head variants for the QAGT-MLP (Tier 0 + part of Tier 1).

`model.py` is left untouched -- this module reuses its `GraphEncoder` and
`QubitConditionedPooling` verbatim and only swaps the regression head, so the
other notebooks that import `QEMGraphTransformer` are unaffected.

`QEMGraphTransformerX(head="scalar")` is behaviourally identical to
`QEMGraphTransformer`; the two new modes are:

  head="residual"  y_hat = noisy_z + delta, with the last layer zero-initialised
                   so training *starts* at the unmitigated baseline. Otherwise
                   the head has to learn the identity map out of a 2d+1 concat
                   in which noisy_z is a single dimension competing with 2d
                   randomly-initialised ones.

  head="level"     K logits over a discrete LevelSet instead of a scalar. With
                   `noisy_prior=True` the logits carry a -gamma*(noisy - l_k)^2
                   term and the last layer is zero-initialised, so at step 0 the
                   argmax is exactly "snap noisy_z to the nearest level" -- the
                   zero-training baseline -- and training can only improve on it.
"""
from typing import Optional

import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv

from model import GraphEncoder, QubitConditionedPooling, QEMHead
from qem_levels import LevelSet


def _zero_init_last_linear(seq: nn.Sequential) -> None:
    for layer in reversed(seq):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


class QEMResidualHead(nn.Module):
    """Scalar head that predicts a *correction* to noisy_z rather than the value."""

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.2,
                 zero_init: bool = True, desc_dim: int = 0):
        super().__init__()
        self.desc_dim = desc_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model + 1 + desc_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if zero_init:
            _zero_init_last_linear(self.mlp)

    def forward(self, pooled_all, global_all, noisy_z, desc=None):
        assert noisy_z is not None, "residual head requires use_noisy=True"
        BM = pooled_all.size(0)
        M = BM // global_all.size(0)
        g = torch.repeat_interleave(global_all, repeats=M, dim=0)
        n = noisy_z.view(BM, 1)
        feats = [pooled_all, g, n]
        if self.desc_dim:
            feats.append(torch.repeat_interleave(desc, repeats=M, dim=0))
        return (n + self.mlp(torch.cat(feats, dim=1))).squeeze(-1)


class QEMLevelHead(nn.Module):
    """K-way classification head over a discrete LevelSet."""

    def __init__(self, d_model: int, levels: LevelSet, use_noisy: bool = True,
                 hidden: int = 128, dropout: float = 0.2,
                 noisy_prior: bool = True, zero_init: bool = True,
                 prior_gamma: float = 10.0, desc_dim: int = 0):
        super().__init__()
        self.use_noisy = use_noisy
        self.K = levels.K
        self.desc_dim = desc_dim
        self.register_buffer("levels", levels.tensor())
        self.noisy_prior = noisy_prior and use_noisy
        in_dim = 2 * d_model + (1 if use_noisy else 0) + desc_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.K),
        )
        if zero_init:
            _zero_init_last_linear(self.mlp)
        if self.noisy_prior:
            # softplus-parameterised so gamma stays positive while being learnable
            inv = torch.log(torch.expm1(torch.tensor(float(prior_gamma))))
            self.raw_gamma = nn.Parameter(inv)

    def forward(self, pooled_all, global_all, noisy_z=None, desc=None):
        BM = pooled_all.size(0)
        M = BM // global_all.size(0)
        g = torch.repeat_interleave(global_all, repeats=M, dim=0)
        parts = [pooled_all, g]
        if self.use_noisy:
            assert noisy_z is not None, "noisy_z required when use_noisy=True"
            parts.append(noisy_z.view(BM, 1))
        if self.desc_dim:
            parts.append(torch.repeat_interleave(desc, repeats=M, dim=0))
        feats = torch.cat(parts, dim=1)

        logits = self.mlp(feats)  # [B*M, K]
        if self.noisy_prior:
            gamma = nn.functional.softplus(self.raw_gamma)
            d = noisy_z.view(BM, 1) - self.levels.view(1, self.K)
            logits = logits - gamma * d.pow(2)
        return logits


class GCNGraphEncoder(nn.Module):
    """GraphEncoder with GCNConv in place of TransformerConv.

    Used only for the "no attention" ablation: identical depth, width, residual
    and normalisation, so the comparison isolates the attention operator.
    """

    def __init__(self, in_dim: int, d_model: int = 128, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([GCNConv(d_model, d_model) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.proj(x)
        for conv, ln in zip(self.layers, self.norms):
            h_res = h
            h = self.dropout(nn.functional.gelu(conv(h, edge_index)))
            h = ln(h + h_res)
        return h


class QubitConditionedPoolingV2(nn.Module):
    """Attention pooling whose query actually depends on the target qubit.

    model.py's QubitConditionedPooling scores each node with a single shared
    `query` parameter, so the logit of node n does not depend on the target m
    at all -- the lightcone mask is the only thing separating output column m
    from m'. Once the lightcone is computed correctly it covers 87-98% of the
    nodes and overlaps 0.90-0.98 across targets, so the M pooled vectors
    collapse onto each other (measured: cosine 1.0000 on two of three
    datasets). This module restores the distinction on purpose:

      query   q_m       = MLP([embed(physical_qubit_m), noisy_z_m])
      logit   l_nm      = <K h_n, q_m> / sqrt(d)  +  alpha * wire_nm
      pooled  p_m       = softmax_n(l_nm masked to the lightcone) @ H

    The lightcone stays the hard support -- what can physically influence the
    observable -- while wire membership enters as a learned soft preference
    for gates sitting directly on the observable's own wire.
    """

    def __init__(self, d_model: int, max_qubits: int = 64, use_noisy: bool = True,
                 use_wire: bool = True, prior_alpha: float = 1.0,
                 use_lightcone: bool = True):
        super().__init__()
        self.use_noisy = use_noisy
        self.use_wire = use_wire
        self.use_lightcone = use_lightcone
        self.d_model = d_model
        self.key = nn.Linear(d_model, d_model)
        self.qubit_emb = nn.Embedding(max_qubits, d_model)
        in_q = d_model + (1 if use_noisy else 0)
        self.q_proj = nn.Sequential(
            nn.Linear(in_q, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        if use_wire:
            self.alpha = nn.Parameter(torch.tensor(float(prior_alpha)))

    def forward(self, H, masks, ptr, measured_qubits=None, noisy_z=None, wire_masks=None):
        M = masks.shape[1]
        B = ptr.numel() - 1
        scale = self.d_model ** 0.5
        K = self.key(H)

        pooled_chunks, global_chunks = [], []
        for b in range(B):
            s, e = int(ptr[b]), int(ptr[b + 1])
            Hb, Kb, mb = H[s:e], K[s:e], masks[s:e, :]

            qids = measured_qubits[b * M:(b + 1) * M]              # [M]
            qfeat = self.qubit_emb(qids)                           # [M, d]
            if self.use_noisy:
                nz = noisy_z[b * M:(b + 1) * M].view(M, 1)
                qfeat = torch.cat([qfeat, nz], dim=1)
            q = self.q_proj(qfeat)                                 # [M, d]

            logits = (Kb @ q.t()) / scale                          # [Nb, M]
            if self.use_wire and wire_masks is not None:
                logits = logits + self.alpha * wire_masks[s:e, :]

            if self.use_lightcone:
                logits = logits.masked_fill(mb <= 0, float("-inf"))
            attn = torch.softmax(logits, dim=0)
            pooled_chunks.append(torch.einsum("nm,nd->md", attn, Hb))
            global_chunks.append(Hb.mean(dim=0))

        return torch.cat(pooled_chunks, dim=0), torch.stack(global_chunks, dim=0)

class QEMGraphTransformerX(nn.Module):
    """QEMGraphTransformer with a configurable head.

    head="scalar" reproduces the original model exactly; "residual" and "level"
    are the Tier 0 / Tier 1 variants. `forward` returns [B*M] for the scalar and
    residual heads and [B*M, K] logits for the level head; `decode` turns either
    into the [B*M] continuous prediction the metrics are computed on.
    """

    def __init__(self, in_dim: int, d_model: int = 128, layers: int = 3, heads: int = 4,
                 dropout: float = 0.2, use_noisy: bool = True,
                 head: str = "scalar", levels: Optional[LevelSet] = None,
                 noisy_prior: bool = True, zero_init_head: bool = True,
                 pool: str = "shared", desc_dim: int = 0, graph_off: bool = False,
                 use_wire: bool = True, use_lightcone: bool = True,
                 use_global: bool = True, backbone: str = "transformer"):
        super().__init__()
        if head not in ("scalar", "residual", "level"):
            raise ValueError(f"unknown head {head!r}")
        if head == "level" and levels is None:
            raise ValueError("head='level' requires a LevelSet")
        if head in ("residual",) and not use_noisy:
            raise ValueError("head='residual' requires use_noisy=True")

        self.head_kind = head
        self.levels = levels
        self.desc_dim = desc_dim
        self.graph_off = graph_off
        self.use_global = use_global
        if backbone not in ("transformer", "gcn"):
            raise ValueError(f"unknown backbone {backbone!r}")
        if pool not in ("shared", "conditioned"):
            raise ValueError(f"unknown pool {pool!r}")
        self.pool_kind = pool
        Enc = GraphEncoder if backbone == "transformer" else GCNGraphEncoder
        self.encoder = Enc(in_dim, d_model, layers, heads, dropout)
        self.pool = (QubitConditionedPooling(d_model) if pool == "shared"
                     else QubitConditionedPoolingV2(d_model, use_noisy=use_noisy,
                                                    use_wire=use_wire,
                                                    use_lightcone=use_lightcone))
        if head == "scalar":
            if desc_dim:
                raise ValueError("desc_dim is only supported for the residual/level heads")
            self.head = QEMHead(d_model, use_noisy=use_noisy, hidden=d_model, dropout=dropout)
        elif head == "residual":
            self.head = QEMResidualHead(d_model, hidden=d_model, dropout=dropout,
                                        zero_init=zero_init_head, desc_dim=desc_dim)
        else:
            self.head = QEMLevelHead(d_model, levels, use_noisy=use_noisy, hidden=d_model,
                                     dropout=dropout, noisy_prior=noisy_prior,
                                     zero_init=zero_init_head, desc_dim=desc_dim)

    @property
    def is_classifier(self) -> bool:
        return self.head_kind == "level"

    def forward(self, data):
        H = self.encoder(data.x, data.edge_index)
        noisy = getattr(data, "noisy_z", None)
        if self.pool_kind == "conditioned":
            pooled_all, global_all = self.pool(
                H, data.lightcone_masks, data.ptr,
                measured_qubits=data.measured_qubits, noisy_z=noisy,
                wire_masks=getattr(data, "wire_masks", None),
            )
        else:
            pooled_all, global_all = self.pool(H, data.lightcone_masks, data.ptr)

        if not self.use_global:
            # "no global" ablation: the head sees only the qubit-local context
            global_all = torch.zeros_like(global_all)

        if self.graph_off:
            # descriptors-only ablation: keep the graph path in the parameter
            # count and the gradient graph, but deny it any information
            pooled_all = torch.zeros_like(pooled_all)
            global_all = torch.zeros_like(global_all)

        if self.desc_dim:
            desc = getattr(data, "descriptors", None)
            if desc is None:
                raise ValueError("desc_dim set but graphs carry no `descriptors`")
            return self.head(pooled_all, global_all, noisy, desc.view(global_all.size(0), -1))
        return self.head(pooled_all, global_all, noisy)

    def decode(self, out: torch.Tensor, mode: str = "argmax") -> torch.Tensor:
        """Model output -> [B*M] continuous predictions.

        For the level head: "argmax" (MAE-optimal) or "expect" (= sum_k p_k l_k,
        RMSE-optimal). For the scalar/residual heads: "raw", or "snap" to round
        onto the level set when one is attached.
        """
        if self.is_classifier:
            if mode in ("argmax", "raw", "snap"):
                return self.levels.argmax(out)
            if mode == "expect":
                return self.levels.expect(out)
            raise ValueError(f"unknown decode mode {mode!r} for a level head")
        if mode in ("raw", "expect", "argmax"):
            return out
        if mode == "snap":
            if self.levels is None:
                raise ValueError("decode('snap') needs a LevelSet on the model")
            return self.levels.snap(out)
        raise ValueError(f"unknown decode mode {mode!r}")
