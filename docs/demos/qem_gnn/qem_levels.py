# qem_levels.py
"""Discrete-level support for QEM targets (Tier 0).

Some of these datasets have targets that are not continuous. Random Clifford
circuits, for instance, give <Z> in {-1, 0, +1} exactly -- on
haoran_mbd_coherent/random_cliffords, 68.1% of labels are +-1, 31.9% are ~0
(the +-0.002 spread is shot noise on an exact zero) and *nothing* falls
between 0.1 and 0.9. Training a plain regression head on such a target makes
the model hedge: it emits 0.87 where the truth is 1.0 and pays 0.13 MAE on a
point whose class it already got right.

`LevelSet.detect` inspects the *training* labels and reports whether they
collapse onto a small set of atoms. It returns None when they do not, so the
non-Clifford notebooks (ising_*, mbd_theta_*) keep the plain regression path
without any special-casing at the call site.
"""
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, List]


def _as_np(y: ArrayLike) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return np.asarray(y, dtype=np.float64).ravel()


@dataclass
class LevelSet:
    """A small set of atoms the targets collapse onto, plus encode/decode helpers."""

    levels: List[float]

    @property
    def K(self) -> int:
        return len(self.levels)

    def tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(self.levels, dtype=dtype, device=device)

    # ---- detection -------------------------------------------------------
    @classmethod
    def detect(
        cls,
        y: ArrayLike,
        gap: float = 0.05,
        max_levels: int = 8,
        max_spread: float = 0.2,
        coverage: float = 0.999,
        round_to: Optional[float] = 1e-2,
    ) -> Optional["LevelSet"]:
        """Return a LevelSet if `y` clusters onto <= max_levels atoms, else None.

        Values are sorted and split wherever consecutive values differ by more
        than `gap`. A cluster is accepted only if its own spread is at most
        `max_spread` (so a broad continuum is never mistaken for an atom), and
        the whole detection is accepted only if at least `coverage` of the
        points sit within `max_spread` of their cluster centre.
        """
        v = np.sort(_as_np(y))
        if v.size == 0:
            return None

        splits = np.nonzero(np.diff(v) > gap)[0] + 1
        clusters = np.split(v, splits)
        if len(clusters) > max_levels:
            return None
        if any(float(c[-1] - c[0]) > max_spread for c in clusters):
            return None

        centers = np.array([float(c.mean()) for c in clusters])
        if round_to:
            snapped = np.round(centers / round_to) * round_to
            centers = np.where(np.abs(snapped - centers) <= round_to, snapped, centers)
        centers = np.sort(centers)

        covered = (np.abs(v[:, None] - centers[None, :]).min(axis=1) <= max_spread).mean()
        if covered < coverage:
            return None
        return cls(levels=[float(c) for c in centers])

    # ---- encode / decode -------------------------------------------------
    def to_class(self, y: torch.Tensor) -> torch.Tensor:
        """Continuous targets -> nearest-level class indices (same shape as y)."""
        lv = self.tensor(device=y.device, dtype=y.dtype)
        return (y.unsqueeze(-1) - lv).abs().argmin(dim=-1)

    def snap(self, y: torch.Tensor) -> torch.Tensor:
        """Continuous predictions -> nearest level."""
        lv = self.tensor(device=y.device, dtype=y.dtype)
        return lv[(y.unsqueeze(-1) - lv).abs().argmin(dim=-1)]

    def argmax(self, logits: torch.Tensor) -> torch.Tensor:
        """[..., K] logits -> the level with the highest probability (MAE-optimal)."""
        lv = self.tensor(device=logits.device, dtype=torch.float32)
        return lv[logits.argmax(dim=-1)]

    def expect(self, logits: torch.Tensor) -> torch.Tensor:
        """[..., K] logits -> sum_k p_k * level_k (the RMSE-optimal decoding)."""
        lv = self.tensor(device=logits.device, dtype=torch.float32)
        return (torch.softmax(logits.float(), dim=-1) * lv).sum(dim=-1)

    def __repr__(self) -> str:
        return f"LevelSet({[round(l, 4) for l in self.levels]})"


def level_report(y: ArrayLike, ls: Optional[LevelSet]) -> str:
    """One-line human summary, for printing in a notebook cell."""
    if ls is None:
        return "no discrete level set detected -> continuous regression"
    v = _as_np(y)
    lv = np.asarray(ls.levels)
    cls = np.abs(v[:, None] - lv[None, :]).argmin(axis=1)
    frac = [f"{ls.levels[k]:+.3g}: {(cls == k).mean():.1%}" for k in range(ls.K)]
    resid = np.abs(v - lv[cls]).mean()
    return f"{ls.K} levels [{', '.join(frac)}] | oracle-snap MAE {resid:.4f}"
