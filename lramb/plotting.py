"""Notebook figures and profile animation; no simulation is hidden in plotting."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np

from .analysis import (block_summary, kernel_rates, stationarity_report,
                       temporal_spectrum, uncertainty_scan)


def _finish(fig, output, name):
    fig.tight_layout()
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output)/f"{name}.png", dpi=160, bbox_inches="tight")
    return fig


def plot_kernel(model, output=None):
    p = model.p
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    kmax = np.pi / p.dx
    extent = [-kmax, kmax, -kmax, kmax]
    image = axes[0, 0].imshow(np.fft.fftshift(model.K), origin="lower", extent=extent)
    fig.colorbar(image, ax=axes[0, 0], label=r"$\widehat K(q)$")
    axes[0, 0].set(title=f"{model.kernel.kind}: momentum space (zero mode removed)",
                   xlabel=r"$k_x$", ylabel=r"$k_y$")
    valid = model.q > 0
    order = np.argsort(model.q[valid])
    axes[0, 1].loglog(model.q[valid][order], model.K[valid][order], ".", ms=2)
    axes[0, 1].set(xlabel="Lattice wave number q", ylabel=r"$\widehat K(q)$",
                   title="Multiplier used in the solver")
    real = np.fft.fftshift(model.weights) / model.dV
    half = p.n * p.dx / 2
    image = axes[1, 0].imshow(real, origin="lower", extent=[-half, half, -half, half],
                             cmap="RdBu_r", norm=SymLogNorm(max(abs(real).max()*1e-3, 1e-14)))
    fig.colorbar(image, ax=axes[1, 0], label=r"$K_{\rm eff}(r)=w_r/\Delta V$")
    axes[1, 0].set(title="Periodic real-space kernel (signed, mean zero)",
                   xlabel="displacement x", ylabel="displacement y")
    labels = model.r_labels.ravel()
    count = np.bincount(labels)
    radial = np.bincount(labels, weights=(model.weights/model.dV).ravel())/count
    axes[1, 1].plot(model.r_centres, radial, "o-", ms=3)
    axes[1, 1].axhline(0, color="gray", lw=.7)
    axes[1, 1].set_yscale("symlog", linthresh=max(abs(radial).max()*1e-3, 1e-14))
    axes[1, 1].set(xlabel="Periodic displacement radius r",
                   ylabel="Radial mean K_eff(r)", title="Negative background follows zero-mode removal")
    return _finish(fig, output, "kernel")


def plot_diagnostics(model, run, output=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    burn = run.metadata["burn_steps"] * model.p.dt
    prod = run.time >= burn
    axes[0, 0].plot(run.time, run.mass-run.mass[0])
    axes[0, 0].set(ylabel=r"$\bar\phi(t)-\bar\phi(0)$", title="Mass conservation")
    for j, label in enumerate(("bulk", "gradient", "long range")):
        axes[0, 1].plot(run.time, run.energy[:, j]/model.area, label=label)
    axes[0, 1].plot(run.time, run.energy.sum(axis=1)/model.area, "k", label="total")
    axes[0, 1].set(ylabel="Free energy / area", title="Conservative free-energy relaxation")
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].plot(run.time, run.variance)
    axes[0, 2].set(ylabel="Var(phi)", title="Density variance")
    weighted = run.structure * model.q_count
    qmean = np.sum(weighted*model.q_centres, axis=1)/np.maximum(weighted.sum(axis=1), 1e-30)
    axes[1, 0].plot(run.time, qmean)
    axes[1, 0].set(ylabel="S-weighted mean q", title="Domain scale / residual coarsening")
    t = run.prod_time + burn
    for values, label in ((run.ep_rate[:, 2], "active"),
                           (run.ep_rate[:, :2].sum(axis=1), "conservative"),
                           (run.ep_rate.sum(axis=1), "medium")):
        _, _, bm = block_summary(values/model.area, run.metadata["blocks"])
        timeblocks = np.array_split(t, len(bm))
        centres = [x.mean() for x in timeblocks]
        axes[1, 1].plot(centres, bm, "o-", label=label)
    axes[1, 1].set(ylabel="Block EP rate / area", title="Production-window block averages")
    axes[1, 1].legend(fontsize=8)
    sizes, errors = uncertainty_scan(run.ep_rate[:, 2]/model.area)
    axes[1, 2].plot(sizes*run.metadata["sample_every"]*model.p.dt, errors, "o-")
    axes[1, 2].set(xscale="log", xlabel="Block duration", ylabel="SE of mean active EP / area",
                   title="SE plateau checks temporal correlations")
    for ax in axes.flat:
        if ax is not axes[1, 2]:
            ax.set_xlabel("Time")
            ax.axvline(burn, color="gray", ls="--", lw=.8)
        ax.grid(alpha=.2)
    rows = stationarity_report(model, run)
    any_drift = any(row["drift_detected"] for row in rows)
    fig.suptitle("Finite-window drift detected: extend burn-in / production" if any_drift else
                 "No drift detected in these blocks; inspect correlation and structure diagnostics", fontsize=12)
    return _finish(fig, output, "steady_state_diagnostics")


def plot_ep_maps(model, run, output=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    maps = [*run.ep_map, run.ep_map.sum(axis=0)]
    labels = ["Local conservative", "Long-range conservative", "Local active", "Medium (sum)"]
    half = model.p.n*model.p.dx
    for ax, values, label in zip(axes.flat, maps, labels):
        limit = max(np.max(np.abs(values)), 1e-15)
        image = ax.imshow(values, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                          extent=[0, half, 0, half])
        fig.colorbar(image, ax=ax, label="Mean EP rate density")
        ax.set(title=f"{label}; integral = {model.dV*values.sum():.4g}", xlabel="x", ylabel="y")
    block_total = run.ep_map_blocks.sum(axis=1)
    se = block_total.std(axis=0, ddof=1)/np.sqrt(len(block_total))
    image = axes[1, 1].imshow(se, origin="lower", cmap="magma", extent=[0, half, 0, half])
    fig.colorbar(image, ax=axes[1, 1], label="Block SE of medium map")
    axes[1, 1].set(title="Uncertainty from production blocks", xlabel="x", ylabel="y")
    axes[1, 2].imshow(run.fields[-1], origin="lower", cmap="RdBu_r", extent=[0, half, 0, half])
    axes[1, 2].set(title="Final density (laboratory coordinates)", xlabel="x", ylabel="y")
    fig.suptitle("Time averaged EP: full integration-step quadrature, production only", fontsize=13)
    return _finish(fig, output, "mean_ep_maps")


def plot_structure(model, run, output=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    valid = model.q_valid & (model.q_centres > 0)
    q = model.q_centres[valid]
    indices = sorted(set(np.linspace(1, len(run.time)-1, 7, dtype=int)))
    for i in indices:
        axes[0, 0].loglog(q, np.maximum(run.structure[i, valid], 1e-15), label=f"t={run.time[i]:g}")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set(xlabel="q", ylabel="Radial mean S(q)", title="Connected structure factor over time")
    values = run.structure[:, valid].T
    image = axes[0, 1].pcolormesh(run.time, q, np.maximum(values, 1e-12), shading="auto",
                                  norm=LogNorm(), cmap="viridis")
    fig.colorbar(image, ax=axes[0, 1], label="S(q,t)")
    axes[0, 1].axvline(run.metadata["burn_steps"]*model.p.dt, color="white", ls="--")
    axes[0, 1].set(xlabel="Time", ylabel="q", title="Drift of the characteristic scale")
    image = axes[1, 0].imshow(np.fft.fftshift(np.maximum(run.mean_structure_2d, 1e-12)),
                               origin="lower", extent=[-np.pi/model.p.dx, np.pi/model.p.dx]*2,
                               norm=LogNorm(), cmap="magma")
    fig.colorbar(image, ax=axes[1, 0], label="Production mean S(kx,ky)")
    axes[1, 0].set(xlabel="kx", ylabel="ky", title="Anisotropy and domain orientation")
    keep = np.flatnonzero(run.time >= run.metadata["burn_steps"]*model.p.dt)
    for group, label in zip(np.array_split(keep, 4), ("production Q1", "Q2", "Q3", "Q4")):
        axes[1, 1].loglog(q, np.maximum(run.structure[group][:, valid].mean(axis=0), 1e-15), label=label)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set(xlabel="q", ylabel="Mean S(q)", title="Production quarters: stationarity screen")
    return _finish(fig, output, "structure_factor")


def plot_kernel_fluctuations(model, run, output=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    rate_q, rate_r = kernel_rates(model, run)
    valid = model.q_valid & (model.q_centres > 0)
    interval = run.metadata["sample_every"] * model.p.dt
    q, r = model.q_centres[valid], model.r_centres
    for rates, coords, ax, xlabel in ((rate_q[:, valid], q, axes[0, 0], "q shell"),
                                      (rate_r, r, axes[0, 1], "displacement shell r")):
        mean, se, _ = block_summary(rates/model.area, run.metadata["blocks"])
        ax.errorbar(coords, mean, yerr=2*se, fmt=".-", capsize=2, label="Mean ± 2 block SE")
        ax.plot(coords, np.std(rates/model.area, axis=0), label="RMS centred fluctuation")
        ax.axhline(0, color="gray", lw=.6)
        ax.set_yscale("symlog", linthresh=max(np.std(rates/model.area,axis=0).max()*1e-3,1e-14))
        ax.set(xlabel=xlabel, ylabel="Shell EP rate / area",
               title=f"Conservative kernel decomposition: {xlabel}")
        ax.legend(fontsize=8)
    values = np.column_stack([run.ep_rate[:, 2], run.ep_rate[:, :2].sum(axis=1),
                              run.ep_rate.sum(axis=1)]) / model.area
    f, power = temporal_spectrum(values, interval)
    for j, label in enumerate(("active", "conservative", "medium")):
        axes[0, 2].loglog(f[1:], np.maximum(power[1:, j], 1e-30), label=label)
    axes[0, 2].set(xlabel="Temporal frequency f", ylabel="One-sided PSD", title="Centred EP-rate fluctuations")
    axes[0, 2].legend(fontsize=8)
    for rates, coords, ax, label in ((rate_q[:, valid], q, axes[1, 0], "q"),
                                    (rate_r, r, axes[1, 1], "r")):
        freq, psd = temporal_spectrum(rates/model.area, interval)
        image = ax.pcolormesh(coords, freq[1:], np.maximum(psd[1:], 1e-30),
                               shading="auto", norm=LogNorm(), cmap="magma")
        fig.colorbar(image, ax=ax, label="Shell-rate PSD")
        ax.set(xlabel=f"Kernel shell {label}", ylabel="Temporal frequency f", yscale="log",
               title=f"Fluctuation spectrum P({label}, f)")
    # Finite-difference derivative identity; do not sum shell PSDs (cross terms matter).
    fq = run.prod_energy_q[:, valid]/(model.p.D*model.area)
    freq, energy_psd = temporal_spectrum(fq[:-1].sum(axis=1), interval)
    _, rate_psd = temporal_spectrum(rate_q.sum(axis=1)/model.area, interval)
    factor = (2*np.sin(np.pi*freq*interval)/interval)**2
    axes[1, 2].loglog(freq[1:], rate_psd[1:], label="Total kernel EP-rate PSD")
    axes[1, 2].loglog(freq[1:], (factor*energy_psd)[1:], "--", label="Finite-difference factor × energy PSD")
    axes[1, 2].set(xlabel="Temporal frequency f", ylabel="PSD", title="Boundary fluctuations: low-frequency suppression")
    axes[1, 2].legend(fontsize=8)
    fig.suptitle("Signed shell sums recover LR entropy; individual shell PSDs are correlated", fontsize=13)
    return _finish(fig, output, "kernel_entropy_fluctuations")


def profile_animation(model, run, path=None, max_frames=100, fps=15):
    """Density movie + central line + transverse mean profile; returns animation."""
    indices = np.unique(np.linspace(0, len(run.fields)-1, min(max_frames, len(run.fields)), dtype=int))
    selected = run.fields[indices]
    x = np.arange(model.p.n)*model.p.dx
    limit = max(abs(selected.min()), abs(selected.max()), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=80)
    im = axes[0].imshow(selected[0], origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                         extent=[0, model.p.n*model.p.dx]*2)
    axes[0].axhline(model.p.n*model.p.dx/2, color="black", ls="--", alpha=.5)
    fig.colorbar(im, ax=axes[0], label="phi")
    centre, = axes[1].plot(x, selected[0, model.p.n//2], label="central row")
    average, = axes[1].plot(x, selected[0].mean(axis=0), label="y-mean")
    axes[1].set(xlabel="x", ylabel="phi", ylim=(-limit, limit))
    axes[1].legend()
    axes[0].set(xlabel="x", ylabel="y")
    title = fig.suptitle("")
    burn = run.metadata["burn_steps"]*model.p.dt
    def update(i):
        field = selected[i]
        im.set_data(field)
        centre.set_ydata(field[model.p.n//2])
        average.set_ydata(field.mean(axis=0))
        t = run.time[indices[i]]
        title.set_text(f"{model.kernel.kind}: t={t:g} ({'burn-in' if t < burn else 'production'})")
        return im, centre, average, title
    animation = FuncAnimation(fig, update, frames=len(indices), interval=1000/fps, blit=False)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return animation


def plot_kernel_decomposition(result, output=None):
    """Receiving-phi / source-distance mu and midpoint entropy spectra."""
    from matplotlib.colors import TwoSlopeNorm, Normalize
    phi, rho = result.phi, result.distance
    probability = result.probability
    valid = probability > max(probability.max()*1e-4, 1e-12)
    interval, area = result.metadata["interval"], result.metadata["area"]
    occupied = np.flatnonzero(valid)
    lo, hi = occupied[0], occupied[-1]+1
    def heat(fig, ax, values, title, label, signed=True):
        data = np.ma.masked_invalid(values[lo:hi])
        limit = max(float(np.max(abs(data))), 1e-15)
        image = ax.pcolormesh(result.distance_edges,result.phi_edges[lo:hi+1],data,
            shading="flat",cmap="RdBu_r" if signed else "magma",
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit) if signed else Normalize(0,limit))
        ax.set(xlabel=r"Source distance $\rho=|x-y|$",ylabel=r"Receiving $\phi_{mid}(x)$",title=title)
        fig.colorbar(image,ax=ax,label=label)
    fig_mu, axes = plt.subplots(2,2,figsize=(13,9))
    mu_lr = np.nansum(result.mu_lr,axis=1)
    for values,label in ((result.mu_local[:,0],"local conservative"),
                         (mu_lr,"sum of LR shells"),
                         (result.mu_local[:,0]+mu_lr,"mu_C"),
                         (result.mu_local[:,1],"local active")):
        axes[0,0].plot(phi[valid],values[valid],label=label)
    axes[0,0].set(xlabel="phi_mid(x)",ylabel="Conditional mean chemical potential",
                  title="mu_C = local + sum_rho mu_K,rho")
    axes[0,0].legend(fontsize=8)
    heat(fig_mu,axes[0,1],result.mu_lr,"Convolution before EP weighting",
         "Mean mu_K,rho conditioned on receiving phi")
    axes[1,0].stairs(probability/np.diff(result.phi_edges),result.phi_edges)
    axes[1,0].set(xlabel="phi_mid(x)",ylabel="Probability density",title="Site-time occupancy of phi bins")
    local, lr = result.conditional_rates()
    for values,label in ((local[:,0],"local conservative"),(lr.sum(axis=1),"long range"),
                         (local[:,1],"local active"),(local.sum(axis=1)+lr.sum(axis=1),"medium")):
        axes[1,1].plot(phi[valid],values[valid],label=label)
    axes[1,1].set(xlabel="phi_mid(x)",ylabel="EP rate density per occupied site",
                  title="Conditional mean EP (use occupancy weights for totals)")
    axes[1,1].legend(fontsize=8)
    _finish(fig_mu,output,"kernel_phi_distance_mu")

    rates = result.rate_lr/area
    fig_ep, axes = plt.subplots(2,3,figsize=(16,9))
    heat(fig_ep,axes[0,0],rates.mean(axis=0),"Mean EP contribution per (phi, rho) bin",
         "Mean bin rate / area")
    heat(fig_ep,axes[0,1],rates.std(axis=0),"Centred fluctuation per (phi, rho) bin",
         "RMS bin-rate fluctuation / area",signed=False)
    for values,label in ((result.rate_local[:,:,0]/area,"local conservative"),
                         (rates.sum(axis=2),"long range"),
                         (result.rate_local[:,:,1]/area,"local active")):
        mean,se,_=block_summary(values,result.metadata["blocks"])
        axes[0,2].plot(phi[valid],mean[valid],label=label)
        axes[0,2].fill_between(phi[valid],(mean-2*se)[valid],(mean+2*se)[valid],alpha=.2)
    axes[0,2].set(xlabel="phi_mid(x)",ylabel="Bin EP rate / area",title="Additive phi bins; mean +/- 2 block SE")
    axes[0,2].legend(fontsize=8)
    for values,coords,ax,label in ((rates.sum(axis=1),rho,axes[1,0],"Source distance rho"),
                                  (rates.sum(axis=2)[:,lo:hi],phi[lo:hi],axes[1,1],"Receiving phi_mid(x)")):
        freq,psd=temporal_spectrum(values,interval)
        image=ax.pcolormesh(coords,freq[1:],np.maximum(psd[1:],1e-30),shading="auto",
                            norm=LogNorm(),cmap="magma")
        ax.set(xlabel=label,ylabel="Temporal frequency f",yscale="log",title="PSD of summed midpoint EP bins")
        fig_ep.colorbar(image,ax=ax,label="Rate PSD / area^2")
    global_local=result.rate_local.sum(axis=1)/area
    global_lr=rates.sum(axis=(1,2))
    channels=np.column_stack((global_local[:,0],global_lr,global_local[:,1],
                              global_local.sum(axis=1)+global_lr))
    freq,psd=temporal_spectrum(channels,interval)
    for j,label in enumerate(("local conservative","long range","local active","medium")):
        axes[1,2].loglog(freq[1:],np.maximum(psd[1:,j],1e-30),label=label)
    axes[1,2].set(xlabel="Temporal frequency f",ylabel="Rate PSD / area^2",
                  title="Full rates summed before PSD (cross terms retained)")
    axes[1,2].legend(fontsize=8)
    fig_ep.suptitle("mu_K,rho(x) = g sum_(x-y in shell rho) w_(x-y) phi(y); EP uses receiving phi(x)")
    _finish(fig_ep,output,"kernel_phi_distance_fluctuations")

    fig_psd, axes = plt.subplots(1,3,figsize=(15,4.5))
    freq,joint_psd=temporal_spectrum(rates,interval)
    for ax,target in zip(axes,(-.8,0,.6)):
        index=occupied[np.argmin(abs(phi[occupied]-target))]
        image=ax.pcolormesh(rho,freq[1:],np.maximum(joint_psd[1:,index,:],1e-30),
                            shading="auto",norm=LogNorm(),cmap="magma")
        ax.set(xlabel="Source distance rho",ylabel="Temporal frequency f",yscale="log",
               title=f"Joint PSD at receiving phi = {phi[index]:.3g}")
        fig_psd.colorbar(image,ax=ax,label="P(phi, rho, f) / area^2")
    _finish(fig_psd,output,"kernel_phi_resolved_psd")
    return fig_mu,fig_ep,fig_psd


def plot_circle_stability(reference,perturbed,diagnostics,output=None):
    """Survival, radial motion, and same-noise small-shape response."""
    p=reference.metadata["parameters"]
    length=p["n"]*p["dx"]
    t=reference.time
    a,b=diagnostics["reference"],diagnostics["perturbed"]
    fig,axes=plt.subplots(3,3,figsize=(15,12))
    limit=max(abs(reference.fields).max(),abs(perturbed.fields).max())
    for ax,field,title in ((axes[0,0],reference.fields[0],"Initial reference circle"),
                           (axes[0,1],reference.fields[-1],"Final reference"),
                           (axes[0,2],perturbed.fields[-1],"Final perturbed circle (same noise)")):
        image=ax.imshow(field,origin="lower",extent=[0,length]*2,cmap="RdBu_r",vmin=-limit,vmax=limit)
        if field.min()<diagnostics["threshold"]<field.max():
            ax.contour(np.arange(p["n"])*p["dx"],np.arange(p["n"])*p["dx"],field,
                       levels=[diagnostics["threshold"]],colors="black",linewidths=.7)
        fig.colorbar(image,ax=ax,label="phi")
        ax.set(title=title,xlabel="x",ylabel="y")
    for obs,label in ((a,"reference"),(b,"perturbed")):
        axes[1,0].plot(t,obs["radius"],label=label)
        axes[1,2].step(t,obs["components"],where="mid",label=label)
    axes[1,0].set(xlabel="Time",ylabel="sqrt(largest-component area / pi)",title="Radial growth / shrinkage")
    axes[1,0].legend(fontsize=8)
    axes[1,2].set(xlabel="Time",ylabel="Number of components above size cutoff",title="Breakup / disappearance")
    axes[1,2].legend(fontsize=8)
    spec=reference.metadata["circle"]
    amplitudes=diagnostics["relative_modes"]
    shape_modes=list(range(2,min(7,amplitudes.shape[1])))
    if spec["mode"] not in shape_modes:
        shape_modes.append(spec["mode"])
    for mode in shape_modes:
        axes[1,1].plot(t,amplitudes[:,mode],label=f"m={mode}")
    axes[1,1].set(xlabel="Time",ylabel="Paired mode response / instantaneous radii",title="Dimensionless shape response, centre motion removed")
    axes[1,1].legend(fontsize=8)
    alive=np.flatnonzero(a["area"] > 0)
    for index in np.unique(alive[np.linspace(0,len(alive)-1,min(3,len(alive)),dtype=int)]):
        axes[2,0].plot(a["radial_coordinate"],a["radial_profile"][index],label=f"reference t={t[index]:g}")
    axes[2,0].axhline(diagnostics["threshold"],color="gray",ls="--")
    for value in (spec["phi_inside"],spec["phi_outside"]):
        axes[2,0].axhline(value,color="gray",ls=":",lw=.7)
    axes[2,0].set(xlabel="Distance to tracked cluster centre",ylabel="Azimuthal mean phi",title="Cluster radial profiles")
    axes[2,0].legend(fontsize=8)
    axes[2,1].semilogy(t,np.maximum(diagnostics["field_norm"]/diagnostics["field_norm"][0],1e-15))
    axes[2,1].set(xlabel="Time",ylabel="||delta phi(t)|| / ||delta phi(0)||",title="Equal-mass same-noise field response")
    for obs,label in ((a,"reference"),(b,"perturbed")):
        shift=(obs["centre"]-obs["centre"][0]+length/2)%length-length/2
        axes[2,2].plot(t,np.linalg.norm(shift,axis=1),label=label)
    axes[2,2].set(xlabel="Time",ylabel="Periodic centre displacement",title="Translation is recorded separately")
    axes[2,2].legend(fontsize=8)
    fit=diagnostics["fit"]
    growth=(f"; relative shape rate = {fit['rate']:.4g} +/- {fit['ci95']:.2g}" if fit["rate"] is not None else "")
    fig.suptitle(f"Finite-window circle test: {fit['interpretation']}{growth}",fontsize=12)
    return _finish(fig,output,"circle_stability")
