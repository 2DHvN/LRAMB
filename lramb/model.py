"""A mass-conserving lattice SDE and trajectory-level entropy bookkeeping.

All FFTs are orthonormal. The lattice Laplacian, kernel multipliers, and
conserved noise use the same nearest-neighbour Laplacian symbol q**2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.fft import fft2, ifft2


@dataclass(frozen=True)
class Parameters:
    n: int = 64
    dx: float = 1.0
    dt: float = 0.01
    a: float = -1.0
    b: float = 1.0
    kappa: float = 1.0
    lam: float = 0.8
    mobility: float = 1.0
    D: float = 0.05
    mean: float = -0.2
    seed: int = 17

    def __post_init__(self):
        if not isinstance(self.n, int) or self.n < 4:
            raise ValueError("n must be an integer >= 4")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("All parameters must be finite")
        if min(self.dx, self.dt, self.b, self.mobility, self.D) <= 0 or self.kappa < 0:
            raise ValueError("dx, dt, b, mobility, D must be positive; kappa >= 0")


@dataclass(frozen=True)
class KernelSpec:
    """Multiplier convention for nonzero lattice wave numbers.

    Riesz: (ell*q)^(-alpha); Poisson: exp(-ell*q);
    Linnik: [1+(ell*q)^alpha]^(-1). The zero mode is removed.
    Positive strength penalises the corresponding density modes.
    """
    kind: str = "riesz"
    strength: float = 0.35
    ell: float = 2.0
    alpha: float = 1.0

    def multiplier(self, q: np.ndarray) -> np.ndarray:
        if self.kind not in {"riesz", "poisson", "linnik"}:
            raise ValueError(f"Unknown kernel {self.kind!r}")
        if not np.isfinite([self.strength, self.ell, self.alpha]).all() or self.ell <= 0:
            raise ValueError("Finite parameters and ell > 0 are required")
        if self.kind == "riesz" and not 0 < self.alpha < 2:
            raise ValueError("The 2D Riesz potential uses 0 < alpha < 2")
        if self.kind == "linnik" and not 0 < self.alpha <= 2:
            raise ValueError("Linnik uses 0 < alpha <= 2")
        result = np.zeros_like(q)
        nz = q > 0
        z = self.ell * q[nz]
        if self.kind == "riesz":
            result[nz] = z ** (-self.alpha)
        elif self.kind == "poisson":
            result[nz] = np.exp(-z)
        else:
            result[nz] = 1 / (1 + z**self.alpha)
        return result


def transform(x):
    return fft2(x, norm="ortho")


def inverse(x):
    return ifft2(x, norm="ortho").real


def laplacian(phi: np.ndarray, dx: float) -> np.ndarray:
    return sum(np.roll(phi, 1, axis) + np.roll(phi, -1, axis) - 2 * phi
               for axis in (0, 1)) / dx**2


def gradient_squared(phi: np.ndarray, dx: float) -> np.ndarray:
    """Site-centred average of squared forward/backward bond gradients."""
    return sum((np.roll(phi, 1, ax) - phi)**2 +
               (np.roll(phi, -1, ax) - phi)**2 for ax in (0, 1)) / (2 * dx**2)


class ActiveModelB:
    """dphi = M Laplacian(mu_C + lambda |grad phi|²) dt + conserved noise.

    A custom multiplier array or callable(qx, qy) can replace KernelSpec.
    It must be real and inversion symmetric to define a reciprocal energy.
    """
    def __init__(self, parameters: Parameters | None = None,
                 kernel: KernelSpec | None = None,
                 multiplier: np.ndarray | Callable | None = None):
        self.p = parameters or Parameters()
        self.kernel = kernel or KernelSpec()
        self.custom_multiplier = multiplier is not None
        p = self.p
        self.dV = p.dx**2
        self.area = (p.n * p.dx)**2
        self.k = 2 * np.pi * np.fft.fftfreq(p.n, d=p.dx)
        self.kx, self.ky = np.meshgrid(self.k, self.k)
        self.qx = 2 / p.dx * np.sin(self.kx * p.dx / 2)
        self.qy = 2 / p.dx * np.sin(self.ky * p.dx / 2)
        self.q2 = self.qx**2 + self.qy**2
        self.q = np.sqrt(self.q2)
        raw = self.kernel.multiplier(self.q) if multiplier is None else (
            multiplier(self.qx, self.qy) if callable(multiplier) else multiplier)
        raw = np.asarray(raw)
        if raw.shape != self.q.shape or not np.isfinite(raw).all() or np.iscomplexobj(raw):
            raise ValueError("Custom multiplier must be a finite real (n,n) array")
        partner = (-np.arange(p.n)) % p.n
        if not np.allclose(raw, raw[np.ix_(partner, partner)], rtol=1e-12, atol=1e-12):
            raise ValueError("A conservative kernel requires K(q) = K(-q)")
        self.K = raw.copy()
        self.K[0, 0] = 0.0
        # weights w_r: convolution is sum_r w_r phi_(x-r), including cell volume.
        self.weights = np.fft.ifft2(self.K).real
        self.linear_mu = p.a + p.kappa * self.q2 + self.kernel.strength * self.K
        self.rng = np.random.default_rng(p.seed)
        self.noise_scale = np.sqrt(2 * p.mobility * p.D * p.dt / self.dV) * self.q
        # Half-width shell edges include q=0 and every Brillouin-zone corner.
        dq = 2 * np.sin(np.pi / p.n) / p.dx
        self.q_edges = np.arange(-0.5 * dq, self.q.max() + 1.51 * dq, dq)
        self.q_labels = np.clip(np.digitize(self.q.ravel(), self.q_edges) - 1,
                                0, len(self.q_edges) - 2)
        self.q_count = np.bincount(self.q_labels, minlength=len(self.q_edges)-1)
        self.q_centres = (self.q_edges[:-1] + self.q_edges[1:]) / 2
        self.q_valid = self.q_count > 0
        # Physical displacement shells; the r=0 diagonal is retained explicitly.
        offsets = np.fft.fftfreq(p.n) * p.n * p.dx
        xx, yy = np.meshgrid(offsets, offsets)
        self.radius = np.hypot(xx, yy)
        self.r_labels = np.rint(self.radius / p.dx).astype(int)
        self.r_centres = np.arange(self.r_labels.max() + 1) * p.dx
        # Each row is the real reciprocal multiplier of one displacement shell.
        self.r_multipliers = np.array([
            np.fft.fft2(self.weights * (self.r_labels == j)).real
            for j in range(len(self.r_centres))])

    def initial_field(self, mode="random", amplitude=0.08):
        p = self.p
        if mode == "random":
            field = amplitude * self.rng.normal(size=(p.n, p.n))
        elif mode == "stripe":
            x = np.arange(p.n) * p.dx
            field = np.tile(np.tanh(np.cos(2*np.pi*x/(p.n*p.dx)) * p.n / 10), (p.n, 1))
            field += amplitude * self.rng.normal(size=field.shape)
        else:
            raise ValueError("initial mode must be 'random' or 'stripe'")
        return field - field.mean() + p.mean

    def nonlinear_mu(self, phi):
        return self.p.b * phi**3 + self.p.lam * gradient_squared(phi, self.p.dx)

    def step(self, phi, phi_hat=None):
        """One explicit Euler-Maruyama step, evaluating all drift at phi."""
        p = self.p
        ph = transform(phi) if phi_hat is None else phi_hat
        noise = self.noise_scale * transform(self.rng.normal(size=phi.shape))
        mu_hat = self.linear_mu * ph + transform(self.nonlinear_mu(phi))
        new_hat = ph - p.dt * p.mobility * self.q2 * mu_hat + noise
        new_hat[0, 0] = ph[0, 0]  # fixed mass, also across long runs
        return inverse(new_hat), new_hat

    def energy_channels(self, phi, phi_hat=None):
        ph = transform(phi) if phi_hat is None else phi_hat
        p = self.p
        power = np.abs(ph)**2
        bulk = self.dV * np.sum(p.a / 2 * phi**2 + p.b / 4 * phi**4)
        gradient = self.dV / 2 * p.kappa * np.sum(self.q2 * power)
        lr_modes = self.dV / 2 * self.kernel.strength * self.K * power
        by_q = np.bincount(self.q_labels, weights=lr_modes.ravel(),
                           minlength=len(self.q_centres))
        by_r = self.dV / 2 * self.kernel.strength * np.einsum(
            "rij,ij->r", self.r_multipliers, power, optimize=False)
        return np.array([bulk, gradient, lr_modes.sum()]), by_q, by_r

    def chemical_potentials(self, phi, phi_hat=None):
        """Return local conservative, long-range conservative, and local active mu."""
        p = self.p
        ph = transform(phi) if phi_hat is None else phi_hat
        mu_local = p.a*phi + p.b*phi**3 - p.kappa*laplacian(phi, p.dx)
        mu_lr = self.kernel.strength * inverse(self.K * ph)
        mu_active = p.lam * gradient_squared(phi, p.dx)
        return np.array([mu_local, mu_lr, mu_active])

    def entropy_maps(self, old, new, old_hat=None, new_hat=None, *, return_mu=False):
        """Integrated EP density over one solver step (not an EP rate).

        All chemical potentials are evaluated at (old + new)/2. For the
        quadratic kernel this equals -Delta F_K/D after spatial integration;
        the quartic bulk has a finite midpoint quadrature residual.
        """
        mid = (old + new) / 2
        delta = new - old
        midhat = transform(mid) if old_hat is None else (old_hat + new_hat) / 2
        mu = self.chemical_potentials(mid, midhat)
        ds = -mu * delta / self.p.D
        return (ds, mu) if return_mu else ds

    def structure_factor(self, phi_hat):
        power = self.dV * np.abs(phi_hat)**2
        power = power.copy()
        power[0, 0] = 0  # connected density fluctuations
        shell = np.bincount(self.q_labels, weights=power.ravel(),
                            minlength=len(self.q_centres))
        return np.divide(shell, self.q_count, out=np.zeros_like(shell), where=self.q_count > 0)


@dataclass
class Run:
    """Saved fields/observables plus production-only, full-step EP statistics."""
    metadata: dict
    time: np.ndarray
    fields: np.ndarray
    energy: np.ndarray
    energy_q: np.ndarray
    energy_r: np.ndarray
    structure: np.ndarray
    variance: np.ndarray
    mass: np.ndarray
    prod_time: np.ndarray
    ep_rate: np.ndarray
    ep_map: np.ndarray
    ep_map_blocks: np.ndarray
    prod_energy_q: np.ndarray
    prod_energy_r: np.ndarray
    mean_structure_2d: np.ndarray
    balance_error: float

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: val for key, val in vars(self).items() if key != "metadata"}
        np.savez_compressed(path, metadata=json.dumps(self.metadata), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key].copy() for key in data.files if key != "metadata"}
            payload["balance_error"] = float(payload["balance_error"])
            return cls(metadata=json.loads(str(data["metadata"])), **payload)


def simulate(model: ActiveModelB, *, burn_steps=40000, production_steps=64000,
             sample_every=40, frame_every=200, blocks=8, initial=None, progress=True,
             decomposition=None, backend="numpy", device="auto", dtype="float64", cpu_threads=1):
    """Run burn-in then fixed-duration production, without claiming stationarity.

    EP maps use midpoint quadrature at every integration step. Kernel entropy
    is an exact quadratic boundary difference; the quartic bulk has a finite
    quadrature residual, recorded as balance_error. Only production fields
    contribute to averaged S.
    An optional KernelDecomposition observer bins full-step EP by local phi
    and source distance |x-y|, without changing the dynamics.
    backend='torch' uses device-resident CPU/CUDA tensors and rFFT. 'auto'
    selects torch when CUDA is available, otherwise NumPy. The Run schema and
    decomposition observer are shared. PyTorch is an optional dependency.
    """
    from .backends import backend_options
    options=backend_options(backend,device,dtype)
    if options["backend"]=="torch":
        from .torch_backend import simulate_torch
        return simulate_torch(model,burn_steps=burn_steps,production_steps=production_steps,
            sample_every=sample_every,frame_every=frame_every,blocks=blocks,initial=initial,
            progress=progress,decomposition=decomposition,cpu_threads=cpu_threads,
            device=options["device"],dtype=options["dtype"])
    if (burn_steps < 0 or production_steps <= 0 or sample_every <= 0 or frame_every <= 0
            or blocks < 2 or production_steps % (sample_every * blocks)
            or burn_steps % frame_every or production_steps % frame_every):
        raise ValueError("Use nonnegative burn-in and positive intervals; production_steps "
                         "must divide into sample_every*blocks, both phases into frame_every")
    p = model.p
    if decomposition is not None:
        decomposition.start(production_steps, sample_every, blocks)
    phi = model.initial_field() if initial is None else np.array(initial, dtype=float, copy=True)
    if phi.shape != (p.n, p.n) or not np.isfinite(phi).all():
        raise ValueError("initial must be a finite (n,n) field")
    ph = transform(phi)
    target_mean = phi.mean()
    records = {key: [] for key in ("time", "fields", "energy", "energy_q", "energy_r",
                                   "structure", "variance", "mass")}
    prod_time, rates, prod_q, prod_r = [], [], [], []
    maps = np.zeros((3, p.n, p.n))
    block_maps = np.zeros((blocks, 3, p.n, p.n))
    structure_2d = np.zeros_like(phi)
    interval = np.zeros(3)
    previous_energy = None
    max_balance_error = 0.0
    def record(step):
        energy, eq, er = model.energy_channels(phi, ph)
        for key, val in dict(time=step*p.dt, fields=phi.astype(np.float32), energy=energy,
                             energy_q=eq, energy_r=er, structure=model.structure_factor(ph),
                             variance=phi.var(), mass=phi.mean()).items():
            records[key].append(val)
    record(0)
    iterator = range(1, burn_steps + production_steps + 1)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc=f"{model.kernel.kind} {p.n}x{p.n}", unit="step")
    for step in iterator:
        new, new_hat = model.step(phi, ph)
        if step % 100 == 0 and (not np.isfinite(new).all() or np.max(np.abs(new)) > 30):
            raise FloatingPointError(f"Unstable trajectory at step {step}; reduce dt")
        if step == burn_steps + 1:
            previous_energy, eq, er = model.energy_channels(phi, ph)
            prod_q.append(eq)
            prod_r.append(er)
        if step > burn_steps:
            if decomposition is None:
                ds = model.entropy_maps(phi, new, ph, new_hat)
            else:
                ds, mu = model.entropy_maps(phi, new, ph, new_hat, return_mu=True)
                decomposition.update(phi, new, ph, new_hat, ds, mu)
            maps += ds
            block = min((step-burn_steps-1) * blocks // production_steps, blocks-1)
            block_maps[block] += ds
            interval += model.dV * ds.sum(axis=(1, 2))
        phi, ph = new, new_hat
        if step % frame_every == 0:
            record(step)
        if step > burn_steps and (step - burn_steps) % sample_every == 0:
            energy, eq, er = model.energy_channels(phi, ph)
            exact = -(energy.sum() - previous_energy.sum()) / p.D
            max_balance_error = max(max_balance_error, abs(interval[:2].sum() - exact))
            prod_time.append((step-burn_steps-0.5*sample_every)*p.dt)
            rates.append(interval / (sample_every*p.dt))
            prod_q.append(eq)
            prod_r.append(er)
            power = model.dV * np.abs(ph)**2
            power[0, 0] = 0
            structure_2d += power
            interval[:] = 0
            previous_energy = energy
    if abs(phi.mean()-target_mean) > 1e-10:
        raise FloatingPointError("Mass conservation was lost")
    if decomposition is not None:
        decomposition.finish()
    duration = production_steps * p.dt
    return Run(
        metadata=dict(parameters=asdict(p), kernel=asdict(model.kernel),
                      burn_steps=burn_steps, production_steps=production_steps,
                      sample_every=sample_every, frame_every=frame_every, blocks=blocks,
                      initial_mean=float(target_mean), custom_multiplier=model.custom_multiplier,
                      backend="numpy", device="cpu", dtype="float64",
                      integrator="Euler-Maruyama", entropy_convention="midpoint for all terms"),
        **{key: np.asarray(value) for key, value in records.items()},
        prod_time=np.asarray(prod_time), ep_rate=np.asarray(rates),
        ep_map=maps/duration, ep_map_blocks=block_maps/(duration/blocks),
        prod_energy_q=np.asarray(prod_q), prod_energy_r=np.asarray(prod_r),
        mean_structure_2d=structure_2d/len(rates), balance_error=max_balance_error)
