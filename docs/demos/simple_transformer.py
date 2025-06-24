"""
shallow_transformer_estimator.py
A scikit‑learn–style PyTorch estimator that fits a very small Transformer
with an attention mechanism to 1×684‑dimensional samples.

Usage example
-------------
    import numpy as np
    from shallow_transformer_estimator import SimpleTransformerEstimator

    X = np.random.rand(100, 684).astype(np.float32)
    y = np.random.rand(100)

    model = SimpleTransformerEstimator(task='regression', n_epochs=50)
    model.fit(X, y)
    preds = model.predict(X)
    print(preds[:5])

Author: ChatGPT
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class _PositionalEncoding(nn.Module):
    """Standard sine–cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 684):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        return x + self.pe[:, : x.size(1)]


class _ShallowTransformer(nn.Module):
    """A single‑layer Transformer encoder with average pooling head."""

    def __init__(
        self,
        seq_len: int = 684,
        d_model: int = 32,
        nhead: int = 4,
        dim_feedforward: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = _PositionalEncoding(d_model, seq_len)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L)
        x = x.unsqueeze(-1)  # (B, L, 1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling over sequence length
        return self.head(x)


class SimpleTransformerEstimator:
    """A minimal Transformer estimator with scikit-learn‑like API (fit / predict).

    Parameters
    ----------
    task            : 'regression' or 'classification'
    lr_scheduler    : None | 'step' | 'plateau'
                      - 'step': StepLR with *step_size* / *gamma*
                      - 'plateau': ReduceLROnPlateau with *gamma*, patience=10
    step_size       : epochs between LR drops for StepLR
    gamma           : LR multiplier when scheduler steps
    """

    def __init__(
        self,
        task: str = "regression",
        d_model: int = 32,
        nhead: int = 4,
        dim_feedforward: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 200,
        batch_size: int = 16,
        lr_scheduler: str = None,
        step_size: int = 20,
        gamma: float = 0.5,
        verbose: bool = True,
        device: str = None,
        seq_len: int = 684,
    ):
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        if lr_scheduler not in {None, "step", "plateau"}:
            raise ValueError("lr_scheduler must be None, 'step', or 'plateau'")
        self.task = task
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.verbose = verbose
        self.seq_len = seq_len
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_: _ShallowTransformer | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_model(self, n_outputs: int):
        self.model_ = _ShallowTransformer(
            seq_len=self.seq_len,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            num_layers=1,
            dropout=0.1,
            n_outputs=n_outputs,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit the model."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] != self.seq_len:
            raise ValueError(f"X must have shape (n_samples, {self.seq_len})")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_outputs = y.shape[1]
        self._build_model(n_outputs)

        criterion = (
            nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # ---------------- LR scheduler ----------------
        if self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.gamma
            )
        elif self.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.gamma, patience=10
            )
        else:
            scheduler = None
        # ------------------------------------------------

        dataset = TensorDataset(
            torch.tensor(X), torch.tensor(y, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in tqdm(range(self.n_epochs)):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model_(xb)
                if self.task == "classification":
                    yb = yb.squeeze().long()
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            # ---- scheduler step ----
            if scheduler is not None:
                if self.lr_scheduler == "plateau":
                    scheduler.step(epoch_loss / len(dataset))
                else:  # step lr
                    scheduler.step()
            # ------------------------

            if self.verbose and (
                epoch % max(self.n_epochs // 10, 1) == 0 or epoch == self.n_epochs - 1
            ):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs} - "
                    f"loss: {epoch_loss / len(dataset):.4f} - lr: {current_lr:.2e}"
                )
        return self

    @torch.no_grad()
    def predict(self, X):
        """Generate predictions for X."""
        if self.model_ is None:
            raise RuntimeError("You must call fit before predict.")
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.tensor(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model_.eval()
        preds = []
        for (xb,) in loader:
            xb = xb.to(self.device)
            out = self.model_(xb).cpu().numpy()
            preds.append(out)
        preds = np.vstack(preds)
        if self.task == "regression":
            return preds.squeeze()
        else:
            return preds.argmax(axis=-1)


if __name__ == "__main__":
    # Quick self‑test
    X_demo = np.random.rand(100, 684).astype(np.float32)
    y_demo = np.random.rand(100)
    model = SimpleTransformerEstimator(n_epochs=5, verbose=False)
    model.fit(X_demo, y_demo)
    print("Demo preds:", model.predict(X_demo[:5]))
