"""Block diagnostics and spectra. All rates are per unit physical time."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import linregress, t as student_t

from .model import ActiveModelB, Run


def block_summary(values, blocks=8):
    values = np.asarray(values)
    length = len(values) // blocks
    if length < 2:
        raise ValueError("At least two observations per block are required")
    means = values[:length*blocks].reshape(blocks, length, *values.shape[1:]).mean(axis=1)
    se = means.std(axis=0, ddof=1) / np.sqrt(blocks)
    return values.mean(axis=0), se, means


def stationarity_report(model: ActiveModelB, run: Run):
    """Finite-window drift screens, not a proof of steady state.

    Block SE assumes decorrelation between blocks; inspect the block-size scan.
    """
    burn = run.metadata["burn_steps"] * model.p.dt
    keep = run.time >= burn
    blocks = run.metadata["blocks"]
    q = model.q_centres
    structure = run.structure[keep]
    weight = structure * model.q_count
    qmean = (weight*q).sum(axis=1) / np.maximum(weight.sum(axis=1), 1e-30)
    series = {
        "F_C / area": run.energy[keep].sum(axis=1)/model.area,
        "Var(phi)": run.variance[keep],
        "mean q (S-weighted)": qmean,
        "active EP / area": run.ep_rate[:, 2]/model.area,
    }
    rows = []
    for label, values in series.items():
        mean, se, bm = block_summary(values, blocks)
        fit = linregress(np.arange(blocks), bm)
        critical = student_t.ppf(0.975, blocks-2)
        drift = fit.slope * (blocks-1)
        drift_ci = critical * fit.stderr * (blocks-1)
        rows.append(dict(observable=label, mean=float(mean), block_se=float(se),
                         drift=float(drift), drift_ci95=float(drift_ci),
                         drift_detected=bool(abs(drift) > drift_ci)))
    return rows


def kernel_rates(model: ActiveModelB, run: Run):
    interval = run.metadata["sample_every"] * model.p.dt
    return (-np.diff(run.prod_energy_q, axis=0)/(model.p.D*interval),
            -np.diff(run.prod_energy_r, axis=0)/(model.p.D*interval))


def temporal_spectrum(values, interval, nperseg=None):
    """One-sided Welch PSD; shell cross correlations are not discarded in totals."""
    count = len(values)
    if count < 16:
        raise ValueError("At least 16 samples are required for a fluctuation spectrum")
    segment = min(256, count//4) if nperseg is None else min(nperseg, count)
    return welch(values, fs=1/interval, axis=0, nperseg=segment,
                 noverlap=segment//2, detrend="constant", scaling="density")


def uncertainty_scan(values):
    """SE against block duration; a plateau supports the quoted uncertainty."""
    sizes, errors = [], []
    for length in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        blocks = len(values)//length
        if blocks < 8:
            break
        means = np.asarray(values[:length*blocks]).reshape(blocks, length).mean(axis=1)
        sizes.append(length)
        errors.append(means.std(ddof=1)/np.sqrt(blocks))
    return np.asarray(sizes), np.asarray(errors)
