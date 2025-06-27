"""
shallow_transformer_estimator.py (updated)
====================================================
A scikit‑learn–style PyTorch wrapper around a *very* shallow Transformer
encoder (one attention block) for 1 × 684 inputs.  Now supports:

* **Learning‑rate schedulers**: `None`, `'step'`, or `'plateau'`.
* **Pre‑trained weight loading** via the `pretrained_path` kwarg *or* the
  `.load(path)` helper, with an optional `freeze_pretrained` flag.

Typical use
-----------
```python
from shallow_transformer_estimator import SimpleTransformerEstimator

# Train from scratch
est = SimpleTransformerEstimator(task="regression", n_epochs=30)
est.fit(X_train, y_train)
est.save("model.pt")

# Fine‑tune from a checkpoint, freezing earlier layers
ft = SimpleTransformerEstimator(pretrained_path="model.pt", freeze_pretrained=True)
ft.fit(new_X, new_y)
```
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Positional encoding
# -----------------------------------------------------------------------------
class _PositionalEncoding(nn.Module):
    """Sine–cosine positional encoding (no learned parameters)."""

    def __init__(self, d_model: int, max_len: int = 684):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # → (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        return x + self.pe[:, : x.size(1)]


# -----------------------------------------------------------------------------
# Shallow Transformer body
# -----------------------------------------------------------------------------
class _ShallowTransformer(nn.Module):
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
        x = self.input_proj(x.unsqueeze(-1))  # (B, L, D)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)


# -----------------------------------------------------------------------------
# Estimator wrapper
# -----------------------------------------------------------------------------
class SimpleTransformerEstimator:
    """Minimal Transformer with `fit` / `predict` & checkpoint utilities.

    Parameters
    ----------
    task : {'regression', 'classification'}
    lr_scheduler : None | 'step' | 'plateau'
    step_size / gamma : scheduler hyper‑parameters
    pretrained_path : path to a `.pt` file created via `.save()` or any state‑dict
    freeze_pretrained : if *True*, loaded parameters are frozen before training
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        task: str = "regression",
        d_model: int = 32,
        nhead: int = 4,
        dim_feedforward: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 200,
        batch_size: int = 16,
        lr_scheduler: str | None = None,
        step_size: int = 20,
        gamma: float = 0.5,
        pretrained_path: str | None = None,
        freeze_pretrained: bool = False,
        verbose: bool = True,
        device: str | None = None,
        seq_len: int = 684,
    ):
        # --- arg checks --------------------------------------------------
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        if lr_scheduler not in {None, "step", "plateau"}:
            raise ValueError("lr_scheduler must be None, 'step', or 'plateau'")
        # --- store -------------------------------------------------------
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
        self.pretrained_path = pretrained_path
        self.freeze_pretrained = freeze_pretrained
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
        # ------- load checkpoint if requested ---------------------------
        if self.pretrained_path is not None:
            if not os.path.isfile(self.pretrained_path):
                raise FileNotFoundError(self.pretrained_path)
            state_dict = torch.load(self.pretrained_path, map_location=self.device)
            missing, unexpected = self.model_.load_state_dict(state_dict, strict=False)
            if self.verbose:
                print(
                    f"Loaded pretrained weights from '{self.pretrained_path}' "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
            if self.freeze_pretrained:
                for p in self.model_.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save model weights to *path* (state_dict)."""
        if self.model_ is None:
            raise RuntimeError("No model to save; call fit/load first.")
        torch.save(self.model_.state_dict(), path)
        if self.verbose:
            print(f"Saved weights to '{path}'.")

    def load(self, path: str, *, freeze: bool | None = None):
        """Load weights into the *current* estimator.

        If the model is not built yet (no call to `fit`) the checkpoint will be
        remembered and applied when `_build_model` is invoked.
        """
        if freeze is not None:
            self.freeze_pretrained = freeze
        self.pretrained_path = path
        if self.model_ is not None:
            state_dict = torch.load(path, map_location=self.device)
            self.model_.load_state_dict(state_dict, strict=False)
            if self.freeze_pretrained:
                for p in self.model_.parameters():
                    p.requires_grad = False
        return self

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit (or fine‑tune) the model."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] != self.seq_len:
            raise ValueError(f"X must have shape (n_samples, {self.seq_len})")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_outputs = y.shape[1]
        self._build_model(n_outputs)  # also loads / freezes if needed

        criterion = (
            nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        )
        # Do not include frozen params in optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model_.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        # ---------------- LR scheduler ----------------
        if self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        elif self.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.gamma, patience=10
            )
        else:
            scheduler = None
        # ------------------------------------------------

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for xb, yb in tqdm(loader, disable=not self.verbose):
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
                else:
                    scheduler.step()
            # ------------------------

            if self.verbose and (
                epoch % max(self.n_epochs // 10, 1) == 0 or epoch == self.n_epochs - 1
            ):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs} | "
                    f"loss={epoch_loss / len(dataset):.4f} | lr={current_lr:.2e}"
                )
        return self

    @torch.no_grad()
    def predict(self, X):
        """Generate predictions for *X*."""
        if self.model_ is None:
            raise RuntimeError("You must call fit or load before predict.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.seq_len:
            raise ValueError(f"X must have shape (n_samples, {self.seq_len})")
        dataset = TensorDataset(torch.tensor(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model_.eval()
        preds = []
        for (xb,) in loader:
            xb = xb.to(self.device)
            out = self.model_(xb).cpu().numpy()
            preds.append(out)
        preds = np.vstack(preds)
        return preds.squeeze() if self.task == "regression" else preds.argmax(axis=-1)


if __name__ == "__main__":
    # Smoke‑test with and without pretraining
    X_demo = np.random.rand(40, 684).astype(np.float32)
    y_demo = np.random.rand(40)

    # Train & save
    m1 = SimpleTransformerEstimator(n_epochs=3, verbose=False)
    m1.fit(X_demo, y_demo)
    m1.save("demo.pt")

    # Fine‑tune
    m2 = SimpleTransformerEstimator(
        pretrained_path="demo.pt", freeze_pretrained=True, n_epochs=2, verbose=False
    )
    m2.fit(X_demo, y_demo)
    print("OK – fine‑tune preds:", m2.predict(X_demo[:3]))
