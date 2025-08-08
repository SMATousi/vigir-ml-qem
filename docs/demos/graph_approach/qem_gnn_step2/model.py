# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, num_layers: int = 3, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([
            TransformerConv(d_model, d_model // heads, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.proj(x)
        for conv, ln in zip(self.layers, self.norms):
            h_res = h
            h = conv(h, edge_index)
            h = self.dropout(F.gelu(h))
            h = ln(h + h_res)
        return h  # [N, d]

class QubitConditionedPooling(nn.Module):
    """
    Batch-aware masked attention pooling.

    Inputs:
      H: [N, d]
      masks: [N, M]  (M is constant across graphs in a batch)
      ptr: [B+1]     cumulative node offsets (from PyG DataBatch)

    Returns:
      pooled_all: [B*M, d]
      global_all: [B, d]
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, H, masks, ptr):
        N, d = H.shape
        M = masks.shape[1]
        B = ptr.numel() - 1

        Hw = self.lin(H)                          # [N, d]
        logits = (Hw * self.query).sum(dim=1)     # [N]

        pooled_chunks = []
        global_chunks = []
        for b in range(B):
            s, e = int(ptr[b]), int(ptr[b+1])
            Hb = H[s:e]                           # [Nb, d]
            mb = masks[s:e, :]                    # [Nb, M]
            glb = Hb.mean(dim=0)                  # [d]

            logb = logits[s:e].unsqueeze(1).expand(Hb.size(0), M)  # [Nb, M]
            masked_logits = logb.masked_fill(mb <= 0, float('-inf'))
            attn = torch.softmax(masked_logits, dim=0)             # [Nb, M]
            pooled_b = torch.einsum('nm,nd->md', attn, Hb)         # [M, d]

            pooled_chunks.append(pooled_b)
            global_chunks.append(glb)

        pooled_all = torch.cat(pooled_chunks, dim=0)               # [B*M, d]
        global_all = torch.stack(global_chunks, dim=0)             # [B, d]
        return pooled_all, global_all

class QEMHead(nn.Module):
    """
    Batch-aware regression head.

    Inputs:
      pooled_all: [B*M, d]
      global_all: [B, d]
      noisy_z:    [B*M] or None (if use_noisy=False)

    Output:
      y_hat: [B*M]
    """
    def __init__(self, d_model: int, use_noisy: bool = True, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.use_noisy = use_noisy
        in_dim = d_model + d_model + (1 if use_noisy else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled_all, global_all, noisy_z=None):
        BM, d = pooled_all.shape
        B = global_all.size(0)
        assert BM % B == 0, f"pooled size {BM} not divisible by batch size {B}"
        M = BM // B
        global_tiled = torch.repeat_interleave(global_all, repeats=M, dim=0)  # [B*M, d]

        if self.use_noisy:
            assert noisy_z is not None, "noisy_z required when use_noisy=True"
            feats = torch.cat([pooled_all, global_tiled, noisy_z.view(BM, 1)], dim=1)
        else:
            feats = torch.cat([pooled_all, global_tiled], dim=1)

        return self.mlp(feats).squeeze(-1)  # [B*M]

class QEMGraphTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, layers: int = 3, heads: int = 4, dropout: float = 0.2, use_noisy: bool = True):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, d_model, layers, heads, dropout)
        self.pool = QubitConditionedPooling(d_model)
        self.head = QEMHead(d_model, use_noisy=use_noisy, hidden=d_model, dropout=dropout)

    def forward(self, data):
        H = self.encoder(data.x, data.edge_index)                              # [N, d]
        pooled_all, global_all = self.pool(H, data.lightcone_masks, data.ptr)  # [B*M, d], [B, d]
        noisy = getattr(data, "noisy_z", None)                                 # [B*M] if present
        return self.head(pooled_all, global_all, noisy)                        # [B*M]

