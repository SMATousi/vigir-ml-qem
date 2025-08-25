# train_loop.py

import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from model import QEMGraphTransformer

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        y_hat = model(batch)
        y = batch.y
        loss = criterion(y_hat, y)
        total_loss += loss.item() * y.numel()
        total_mae  += (y_hat - y).abs().sum().item()
        n += y.numel()
    return {"loss": total_loss / n, "mae": total_mae / n}

def train(
    model_cfg: dict,
    loaders: Tuple,
    epochs: int = 50,
    lr: float = 1e-3,
    wd: float = 1e-4,
    device: str = None,
    patience: int = 10,
    best_ckpt_path: str = "best_qem_graph_transformer.pt",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = loaders
    sample = next(iter(train_loader))
    in_dim = sample.x.shape[1]

    net = QEMGraphTransformer(in_dim, **model_cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.SmoothL1Loss()

    best = {"mae": float("inf"), "epoch": -1}
    history: List[Tuple[int, float, float, float]] = []  # (epoch, train_loss, val_loss, val_mae)
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        net.train()
        run_loss = 0.0
        count = 0

        for batch in train_loader:
            batch = batch.to(device)
            y_hat = net(batch)
            loss = criterion(y_hat, batch.y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            run_loss += loss.item() * batch.y.numel()
            count += batch.y.numel()

        train_loss = run_loss / max(1, count)
        val_metrics = evaluate(net, val_loader, criterion, device)
        history.append((epoch, train_loss, val_metrics["loss"], val_metrics["mae"]))
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_metrics['loss']:.4f} | MAE {val_metrics['mae']:.4f}")

        # ---- Early stopping & best checkpoint based on *MAE* ----
        if val_metrics["mae"] + 1e-8 < best["mae"]:
            best["mae"] = val_metrics["mae"]
            best["epoch"] = epoch
            torch.save(net.state_dict(), best_ckpt_path)
            bad_epochs = 0
            # print(f"  ↳ Saved new best to {best_ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    # Restore best
    if best["epoch"] > 0:
        net.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"Loaded best model from epoch {best['epoch']} (val MAE={best['mae']:.4f}).")

    return net, history
