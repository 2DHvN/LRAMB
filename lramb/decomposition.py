"""Convolution-level EP decomposition by source distance and local field value.

rho = |x-y| is a periodic source displacement, never a distance to a droplet
centre. Conditional bins always use the receiving field phi_mid(x).
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .model import ActiveModelB, inverse


@dataclass
class DecompositionResult:
    metadata: dict
    phi_edges: np.ndarray
    distance_edges: np.ndarray
    time: np.ndarray
    rate_lr: np.ndarray       # (time, receiving-phi bin, source-distance shell)
    rate_local: np.ndarray    # (time, receiving-phi bin, local C / local active)
    occupancy: np.ndarray     # number of site-step observations per interval/phi
    mu_lr: np.ndarray         # conditional mean, (phi, source distance)
    mu_local: np.ndarray      # conditional mean, (phi, local C / local active)

    @property
    def phi(self):
        return (self.phi_edges[:-1]+self.phi_edges[1:])/2

    @property
    def distance(self):
        return (self.distance_edges[:-1]+self.distance_edges[1:])/2

    @property
    def probability(self):
        count = self.occupancy.sum(axis=0)
        return count/count.sum()

    def conditional_rates(self):
        """EP rate density per occupied site, conditioned on phi_mid(x)."""
        duration = self.metadata["dt"]*self.occupancy.sum(axis=0)
        volume = self.metadata["dV"]
        interval = self.metadata["interval"]
        lr = np.divide(self.rate_lr.sum(axis=0)*interval,
                       volume*duration[:, None], out=np.full(self.mu_lr.shape, np.nan),
                       where=duration[:, None] > 0)
        local = np.divide(self.rate_local.sum(axis=0)*interval,
                          volume*duration[:, None], out=np.full(self.mu_local.shape, np.nan),
                          where=duration[:, None] > 0)
        return local, lr

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, metadata=json.dumps(self.metadata),
            **{k: v for k, v in vars(self).items() if k != "metadata"})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(metadata=json.loads(str(data["metadata"])),
                       **{k: data[k].copy() for k in data.files if k != "metadata"})


class KernelDecomposition:
    """Online midpoint mu and entropy bins, with an additive reconstruction.

    mu_K,rho(x) = g sum_(r in shell rho) w_r phi_mid(x-r).
    ds_K,rho(x) = -mu_K,rho(x) [phi_next(x)-phi_old(x)]/D.
    Outer phi bins include out-of-range tails; their count is reported.
    """
    def __init__(self, model: ActiveModelB, phi_edges=None, distance_edges=None,
                 shells=12):
        self.model = model
        self.phi_edges = np.asarray(np.linspace(-2, 2, 33) if phi_edges is None else phi_edges,
                                    dtype=float)
        if distance_edges is None:
            # Isolate the on-site r=0 weight, then use equal-width nonlocal shells.
            distance_edges = np.r_[-model.p.dx/2,
                np.linspace(model.p.dx/2, model.radius.max()+model.p.dx/2, shells)]
        self.distance_edges = np.asarray(distance_edges, dtype=float)
        for name, edges in (("phi", self.phi_edges), ("distance", self.distance_edges)):
            if edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
                raise ValueError(f"{name} edges must be finite and strictly increasing")
        if self.distance_edges[0] > 0 or self.distance_edges[-1] <= model.radius.max():
            raise ValueError("Distance shells must cover every periodic source displacement")
        self.nphi = len(self.phi_edges)-1
        self.nshell = len(self.distance_edges)-1
        labels = np.digitize(model.radius, self.distance_edges)-1
        self.multipliers = np.array([np.fft.fft2(model.weights*(labels == i)).real
                                     for i in range(self.nshell)])
        self._offset = np.arange(self.nshell)[:, None]*self.nphi
        self.result = None

    def start(self, production_steps, sample_every, blocks):
        self.steps, self.sample, self.blocks = production_steps, sample_every, blocks
        intervals = production_steps//sample_every
        self.rate_lr = np.zeros((intervals, self.nphi, self.nshell))
        self.rate_local = np.zeros((intervals, self.nphi, 2))
        self.occupancy = np.zeros((intervals, self.nphi), dtype=np.int64)
        self.mu_sum_lr = np.zeros((self.nphi, self.nshell))
        self.mu_sum_local = np.zeros((self.nphi, 2))
        self.updates = 0
        self.tail_count = 0
        self.max_mu_residual = 0.0
        self.result = None

    def update(self, old, new, old_hat, new_hat, entropy_density, mu):
        m = self.model
        mid = (old+new)/2
        delta = new-old
        labels = np.clip(np.digitize(mid.ravel(), self.phi_edges)-1, 0, self.nphi-1)
        self.tail_count += np.count_nonzero((mid < self.phi_edges[0]) | (mid >= self.phi_edges[-1]))
        joint = (self._offset+labels).ravel()
        mu_shell = m.kernel.strength*inverse(self.multipliers*(old_hat+new_hat)[None, :, :]/2)
        self.max_mu_residual = max(self.max_mu_residual, float(np.max(abs(mu_shell.sum(axis=0)-mu[1]))))
        ds_shell = -mu_shell*delta/m.p.D
        def bin_shell(values):
            return np.bincount(joint, weights=values.ravel(),
                minlength=self.nphi*self.nshell).reshape(self.nshell, self.nphi).T
        index = self.updates//self.sample
        self.rate_lr[index] += m.dV*bin_shell(ds_shell)
        self.mu_sum_lr += bin_shell(mu_shell)
        self.occupancy[index] += np.bincount(labels, minlength=self.nphi)
        for j, channel in enumerate((0, 2)):
            self.rate_local[index, :, j] += m.dV*np.bincount(labels,
                weights=entropy_density[channel].ravel(), minlength=self.nphi)
            self.mu_sum_local[:, j] += np.bincount(labels, weights=mu[channel].ravel(),
                                                  minlength=self.nphi)
        self.updates += 1

    def finish(self):
        if self.updates != self.steps:
            raise RuntimeError("Incomplete decomposition observation window")
        m = self.model
        count = self.occupancy.sum(axis=0)
        interval = m.p.dt*self.sample
        self.result = DecompositionResult(
            metadata=dict(dt=m.p.dt, dV=m.dV, area=m.area, interval=interval,
                          blocks=self.blocks, steps=self.steps,
                          distance="periodic source displacement |x-y|",
                          phi="receiving midpoint field phi_mid(x)",
                          local_channels=["local conservative", "local active"],
                          max_mu_residual=self.max_mu_residual,
                          phi_tail_fraction=self.tail_count/(self.steps*m.p.n**2)),
            phi_edges=self.phi_edges, distance_edges=self.distance_edges,
            time=(np.arange(len(self.rate_lr))+.5)*interval,
            rate_lr=self.rate_lr/interval, rate_local=self.rate_local/interval,
            occupancy=self.occupancy,
            mu_lr=np.divide(self.mu_sum_lr, count[:, None],
                           out=np.full_like(self.mu_sum_lr, np.nan), where=count[:, None] > 0),
            mu_local=np.divide(self.mu_sum_local, count[:, None],
                              out=np.full_like(self.mu_sum_local, np.nan), where=count[:, None] > 0))
        return self.result
