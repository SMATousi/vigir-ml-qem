# qem_train.py
"""Configurable training loop for the QAGT-MLP (Tier 1).

`train_loop.py` is left untouched. Everything here is opt-in through
`TrainConfig`, whose defaults reproduce the original loop exactly
(SmoothL1 with beta=1.0, constant LR, no seeding, selection on val MAE), so
switching a notebook over is a no-op until you change a field.

What the fields buy you:

  loss / huber_beta   nn.SmoothL1Loss() defaults to beta=1.0. Every residual in
                      this problem lives in [-2, 2], so the loss never leaves
                      the quadratic regime -- it is exactly 0.5*MSE, with none
                      of the robustness the name suggests. Set beta ~ 0.05, or
                      use "l1", to actually optimise the metric you report.
                      "ce" trains the discrete level head.

  scheduler / warmup  the original loop holds lr=1e-3 for 100 epochs and the
                      curve is flat from ~epoch 45.

  seed / train_ensemble
                      a single unseeded GNN run against a 100-tree forest is
                      not a like-for-like comparison.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn

from qem_ext import QEMGraphTransformerX
from qem_levels import LevelSet


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    wd: float = 1e-4
    patience: int = 30
    device: Optional[str] = None
    grad_clip: float = 1.0

    loss: str = "smooth_l1"          # "smooth_l1" | "l1" | "mse" | "ce"
    huber_beta: float = 1.0          # 1.0 == the original (i.e. 0.5*MSE)

    scheduler: Optional[str] = None  # None | "cosine" | "plateau"
    warmup_epochs: int = 0
    min_lr_frac: float = 0.01        # cosine floor, as a fraction of lr
    plateau_factor: float = 0.5
    plateau_patience: int = 8

    seed: Optional[int] = None
    select_on: str = "mae"           # "mae" | "rmse" | "acc"
    decode: str = "auto"             # "auto" | "raw" | "snap" | "argmax" | "expect"
    verbose: bool = True


def _default_decode(cfg: TrainConfig, net) -> str:
    if cfg.decode != "auto":
        return cfg.decode
    return "argmax" if net.is_classifier else "raw"


def _make_criterion(cfg: TrainConfig):
    if cfg.loss == "smooth_l1":
        return nn.SmoothL1Loss(beta=cfg.huber_beta)
    if cfg.loss == "l1":
        return nn.L1Loss()
    if cfg.loss == "mse":
        return nn.MSELoss()
    if cfg.loss == "ce":
        return nn.CrossEntropyLoss()
    raise ValueError(f"unknown loss {cfg.loss!r}")


def _make_scheduler(cfg: TrainConfig, opt):
    if cfg.scheduler is None:
        return None
    if cfg.scheduler == "plateau":
        mode = "max" if cfg.select_on == "acc" else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=mode, factor=cfg.plateau_factor, patience=cfg.plateau_patience
        )
    if cfg.scheduler == "cosine":
        w, T = cfg.warmup_epochs, max(1, cfg.epochs - cfg.warmup_epochs)

        def lam(e):  # e is 0-based epoch index
            if e < w:
                return (e + 1) / max(1, w)
            p = (e - w) / T
            return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))

        return torch.optim.lr_scheduler.LambdaLR(opt, lam)
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


def metrics(pred: np.ndarray, y: np.ndarray, levels: Optional[LevelSet] = None) -> Dict[str, float]:
    """MAE / RMSE, plus class accuracy when a level set is available."""
    pred, y = np.asarray(pred).ravel(), np.asarray(y).ravel()
    out = {
        "mae": float(np.abs(pred - y).mean()),
        "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
    }
    if levels is not None:
        lv = np.asarray(levels.levels)
        out["acc"] = float(
            (np.abs(pred[:, None] - lv).argmin(1) == np.abs(y[:, None] - lv).argmin(1)).mean()
        )
    return out


@torch.no_grad()
def collect(net, loader, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run `net` over `loader`, returning (raw_out, y, noisy_z) on the CPU.

    raw_out is [N*M] for scalar/residual heads and [N*M, K] logits for a level
    head, so callers can average logits across an ensemble before decoding.
    """
    net.eval()
    outs, ys, ns = [], [], []
    for batch in loader:
        batch = batch.to(device)
        outs.append(net(batch).detach().cpu())
        ys.append(batch.y.detach().cpu())
        ns.append(batch.noisy_z.detach().cpu())
    return torch.cat(outs), torch.cat(ys), torch.cat(ns)


def train_qem(
    model_cfg: dict,
    loaders: Tuple,
    cfg: TrainConfig = TrainConfig(),
    best_ckpt_path: str = "best_qem_graph_transformer_x.pt",
):
    """Train a QEMGraphTransformerX. `model_cfg` is passed straight to it."""
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.seed is not None:
        set_seed(cfg.seed)

    train_loader, val_loader = loaders
    in_dim = next(iter(train_loader)).x.shape[1]

    net = QEMGraphTransformerX(in_dim, **model_cfg).to(device)
    levels = net.levels
    if cfg.loss == "ce" and not net.is_classifier:
        raise ValueError("loss='ce' requires head='level'")
    if net.is_classifier and cfg.loss != "ce":
        raise ValueError("head='level' requires loss='ce'")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = _make_scheduler(cfg, opt)
    criterion = _make_criterion(cfg)
    decode = _default_decode(cfg, net)

    better = (lambda a, b: a > b + 1e-8) if cfg.select_on == "acc" else (lambda a, b: a < b - 1e-8)
    best = {"score": -math.inf if cfg.select_on == "acc" else math.inf, "epoch": -1}
    history: List[dict] = []
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        net.train()
        run_loss, count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            out = net(batch)
            target = levels.to_class(batch.y) if net.is_classifier else batch.y
            loss = criterion(out, target)

            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()

            run_loss += loss.item() * batch.y.numel()
            count += batch.y.numel()

        raw, y, _ = collect(net, val_loader, device)
        val = metrics(net.decode(raw, decode).numpy(), y.numpy(), levels)
        rec = {"epoch": epoch, "train_loss": run_loss / max(1, count),
               "lr": opt.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)

        if cfg.verbose:
            acc = f" | acc {val['acc']:.4f}" if "acc" in val else ""
            print(f"Epoch {epoch:03d} | train {rec['train_loss']:.4f} | "
                  f"MAE {val['mae']:.4f} | RMSE {val['rmse']:.4f}{acc} | lr {rec['lr']:.2e}")

        score = val[cfg.select_on]
        if better(score, best["score"]):
            best.update(score=score, epoch=epoch)
            torch.save(net.state_dict(), best_ckpt_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if cfg.verbose:
                    print("Early stopping.")
                break

        if sched is not None:
            sched.step(score) if cfg.scheduler == "plateau" else sched.step()

    if best["epoch"] > 0:
        net.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        if cfg.verbose:
            print(f"Loaded best model from epoch {best['epoch']} "
                  f"(val {cfg.select_on}={best['score']:.4f}).")
    return net, history


def train_ensemble(
    model_cfg: dict,
    loaders: Tuple,
    cfg: TrainConfig = TrainConfig(),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    ckpt_prefix: str = "qem_x",
):
    """Train one model per seed. Returns (nets, histories)."""
    nets, hists = [], []
    for s in seeds:
        if cfg.verbose:
            print(f"\n=== seed {s} ===")
        c = TrainConfig(**{**cfg.__dict__, "seed": s})
        net, h = train_qem(model_cfg, loaders, c, best_ckpt_path=f"{ckpt_prefix}_seed{s}.pt")
        nets.append(net)
        hists.append(h)
    return nets, hists


@torch.no_grad()
def ensemble_predict(nets, loader, device=None, decode: str = "auto"):
    """Average across ensemble members, then decode. Returns (pred, y, noisy).

    Classifier members are averaged in probability space, not logit space, so a
    single overconfident member cannot dominate the vote.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref = nets[0]
    mode = decode if decode != "auto" else ("argmax" if ref.is_classifier else "raw")

    acc_out, y_ref, n_ref = None, None, None
    for net in nets:
        net.to(device)
        raw, y, n = collect(net, loader, device)
        cur = torch.softmax(raw.float(), dim=-1) if ref.is_classifier else raw.float()
        acc_out = cur if acc_out is None else acc_out + cur
        y_ref, n_ref = y, n
    acc_out = acc_out / len(nets)

    if ref.is_classifier:
        pred = ref.levels.argmax(acc_out) if mode == "argmax" else ref.levels.expect(torch.log(acc_out.clamp_min(1e-12)))
    else:
        pred = ref.decode(acc_out, mode)
    return pred.numpy(), y_ref.numpy(), n_ref.numpy()
