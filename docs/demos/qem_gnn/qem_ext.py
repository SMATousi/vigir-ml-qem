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
                 zero_init: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if zero_init:
            _zero_init_last_linear(self.mlp)

    def forward(self, pooled_all, global_all, noisy_z):
        assert noisy_z is not None, "residual head requires use_noisy=True"
        BM = pooled_all.size(0)
        M = BM // global_all.size(0)
        g = torch.repeat_interleave(global_all, repeats=M, dim=0)
        n = noisy_z.view(BM, 1)
        return (n + self.mlp(torch.cat([pooled_all, g, n], dim=1))).squeeze(-1)


class QEMLevelHead(nn.Module):
    """K-way classification head over a discrete LevelSet."""

    def __init__(self, d_model: int, levels: LevelSet, use_noisy: bool = True,
                 hidden: int = 128, dropout: float = 0.2,
                 noisy_prior: bool = True, zero_init: bool = True,
                 prior_gamma: float = 10.0):
        super().__init__()
        self.use_noisy = use_noisy
        self.K = levels.K
        self.register_buffer("levels", levels.tensor())
        self.noisy_prior = noisy_prior and use_noisy
        in_dim = 2 * d_model + (1 if use_noisy else 0)
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

    def forward(self, pooled_all, global_all, noisy_z=None):
        BM = pooled_all.size(0)
        M = BM // global_all.size(0)
        g = torch.repeat_interleave(global_all, repeats=M, dim=0)
        if self.use_noisy:
            assert noisy_z is not None, "noisy_z required when use_noisy=True"
            feats = torch.cat([pooled_all, g, noisy_z.view(BM, 1)], dim=1)
        else:
            feats = torch.cat([pooled_all, g], dim=1)

        logits = self.mlp(feats)  # [B*M, K]
        if self.noisy_prior:
            gamma = nn.functional.softplus(self.raw_gamma)
            d = noisy_z.view(BM, 1) - self.levels.view(1, self.K)
            logits = logits - gamma * d.pow(2)
        return logits


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
                 noisy_prior: bool = True, zero_init_head: bool = True):
        super().__init__()
        if head not in ("scalar", "residual", "level"):
            raise ValueError(f"unknown head {head!r}")
        if head == "level" and levels is None:
            raise ValueError("head='level' requires a LevelSet")
        if head in ("residual",) and not use_noisy:
            raise ValueError("head='residual' requires use_noisy=True")

        self.head_kind = head
        self.levels = levels
        self.encoder = GraphEncoder(in_dim, d_model, layers, heads, dropout)
        self.pool = QubitConditionedPooling(d_model)
        if head == "scalar":
            self.head = QEMHead(d_model, use_noisy=use_noisy, hidden=d_model, dropout=dropout)
        elif head == "residual":
            self.head = QEMResidualHead(d_model, hidden=d_model, dropout=dropout,
                                        zero_init=zero_init_head)
        else:
            self.head = QEMLevelHead(d_model, levels, use_noisy=use_noisy, hidden=d_model,
                                     dropout=dropout, noisy_prior=noisy_prior,
                                     zero_init=zero_init_head)

    @property
    def is_classifier(self) -> bool:
        return self.head_kind == "level"

    def forward(self, data):
        H = self.encoder(data.x, data.edge_index)
        pooled_all, global_all = self.pool(H, data.lightcone_masks, data.ptr)
        noisy = getattr(data, "noisy_z", None)
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
