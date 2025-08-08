import torch
import torch.nn as nn
from typing import Dict
from model import QEMGraphTransformer

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        y_hat = model(batch)
        y = batch.y
        loss = criterion(y_hat, y)
        total_loss += loss.item() * y.numel()
        total_mae += (y_hat - y).abs().sum().item()
        n += y.numel()
    return {"loss": total_loss / n, "mae": total_mae / n}

def train(model_cfg: dict, loaders, epochs=50, lr=1e-3, wd=1e-4, device=None, patience=10):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = loaders
    # infer input dim
    sample = next(iter(train_loader))
    in_dim = sample.x.shape[1]
    net = QEMGraphTransformer(in_dim, **model_cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.SmoothL1Loss()

    best = {"val_loss": float("inf"), "state": None, "epoch": -1}
    for epoch in range(1, epochs+1):
        net.train()
        run_loss = 0.0; count = 0
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
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_metrics['loss']:.4f} | MAE {val_metrics['mae']:.4f}")
        if val_metrics["loss"] < best["val_loss"] - 1e-6:
            best.update(val_loss=val_metrics["loss"], state=net.state_dict(), epoch=epoch)
        elif epoch - best["epoch"] >= patience:
            print('Early stopping.'); break
    if best["state"] is not None:
        net.load_state_dict(best["state"])
    return net
