import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, num_layers: int = 3, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([
            TransformerConv(d_model, d_model // heads, heads=heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.proj(x)
        for conv, ln in zip(self.layers, self.norms):
            h_res = h
            h = conv(h, edge_index)  # [N, d_model]
            h = self.dropout(F.gelu(h))
            h = ln(h + h_res)
        return h  # [N, d_model]

class QubitConditionedPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, H, masks):
        # H: [N, d], masks: [N, M]
        global_emb = H.mean(dim=0)  # [d]
        Hw = self.lin(H)            # [N, d]
        logits = (Hw * self.query).sum(dim=1)  # [N]
        N = H.size(0); M = masks.size(1)
        logits = logits.unsqueeze(1).expand(N, M)  # [N, M]
        masked_logits = logits.masked_fill(masks <= 0, float('-inf'))
        attn = torch.softmax(masked_logits, dim=0)  # over nodes
        pooled = torch.einsum('nm,nd->md', attn, H)  # [M, d]
        return pooled, global_emb

class QEMHead(nn.Module):
    def __init__(self, d_model: int, use_noisy: bool = True, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.use_noisy = use_noisy
        in_dim = d_model + d_model + (1 if use_noisy else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, pooled_m, global_emb, noisy_z=None):
        M, d = pooled_m.shape
        global_expanded = global_emb.unsqueeze(0).expand(M, -1)
        if self.use_noisy:
            assert noisy_z is not None, "noisy_z required"
            feats = torch.cat([pooled_m, global_expanded, noisy_z.view(M,1)], dim=1)
        else:
            feats = torch.cat([pooled_m, global_expanded], dim=1)
        return self.mlp(feats).squeeze(-1)  # [M]

class QEMGraphTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, layers: int = 3, heads: int = 4, dropout: float = 0.2, use_noisy: bool = True):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, d_model, layers, heads, dropout)
        self.pool = QubitConditionedPooling(d_model)
        self.head = QEMHead(d_model, use_noisy=use_noisy, hidden=d_model, dropout=dropout)

    def forward(self, data):
        H = self.encoder(data.x, data.edge_index)
        pooled_m, global_emb = self.pool(H, data.lightcone_masks)  # masks [N,M]
        noisy = getattr(data, "noisy_z", None)
        return self.head(pooled_m, global_emb, noisy)  # [M]
