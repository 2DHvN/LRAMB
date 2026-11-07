"""Fresh dynamical runs while varying the Poisson/Linnik kernel scale ell."""
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from .model import ActiveModelB,Parameters,KernelSpec,Run,simulate
from .backends import backend_options
from .ep_ranges import ShellEP,shell_summary,fluctuation_statistics,save_shell_summary
from .scaling import scaling_comparison
from .analysis import stationarity_report,temporal_spectrum


GLOBAL_CHANNELS=["medium","kernel","local conservative","local active"]


def _finite_json(value):
    if isinstance(value,dict):
        return {k:_finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [_finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.generic):
        return value.item()
    return value


def run_range_sweep(kind,*,ells=(1.,2.,4.,8.),parameters=None,kernel=None,output,
                    burn_steps=40000,production_steps=64000,sample_every=40,
                    frame_every=200,blocks=8,draws=500,initial_mode="stripe",
                    simulation=None,reuse_cache=True,progress=True,
                    density=False,increment=False,rho_min=2.,rho_max=16.):
    """Change ell in the dynamics, keeping g, alpha and numerical protocol fixed.

    Every member starts from the same initial field and uses a common noise seed
    within its backend. No earlier trajectory is filtered/reweighted to obtain a
    different ell. Full-step midpoint kernel EP follows shell energy differences;
    the expensive receiving-phi observer is unnecessary after summing all phi.
    """
    if kind not in {"poisson","linnik"}:
        raise ValueError("A physical ell range sweep is defined here for Poisson and Linnik; "
                         "Riesz ell only rescales the coupling")
    ells=np.asarray(ells,dtype=float)
    if (ells.ndim!=1 or len(ells)<2 or not np.isfinite(ells).all()
            or np.any(ells<=0) or len(np.unique(ells))!=len(ells)):
        raise ValueError("Use at least two unique, finite positive ell values")
    p=parameters or Parameters()
    base=kernel or KernelSpec(kind,alpha=1.5 if kind=="linnik" else 1.)
    if base.kind!=kind:
        raise ValueError("kernel.kind must match the requested sweep")
    options=backend_options(**(simulation or dict(backend="auto")))
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    source=sha256(b"".join((Path(__file__).parent/f).read_bytes()
                  for f in ("model.py","torch_backend.py","backends.py"))).hexdigest()
    results=[]
    for ell in sorted(ells):
        spec=replace(base,ell=float(ell))
        model=ActiveModelB(p,spec)
        # Also advances each NumPy generator identically before solver noise.
        initial=model.initial_field(initial_mode)
        protocol=dict(kind=kind,parameters=asdict(p),kernel=asdict(spec),
            burn_steps=burn_steps,production_steps=production_steps,sample_every=sample_every,
            frame_every=frame_every,blocks=blocks,initial_mode=initial_mode,
            initial_hash=sha256(initial.tobytes()).hexdigest(),simulation=options,
            source_hash=source,multiplier_hash=sha256(model.K.tobytes()).hexdigest())
        signature=sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        member=folder/f"ell_{float(ell):.12g}"
        path=member/"trajectory.npz"
        run=Run.load(path) if reuse_cache and path.exists() else None
        started=time.perf_counter()
        if run is None or run.metadata.get("range_sweep_signature")!=signature:
            if progress:
                print(f"{kind} ell={ell:g}: integrating {burn_steps+production_steps:,} steps "
                      f"({options['backend']}/{options['device']}, g={spec.strength:g})",flush=True)
            run=simulate(model,initial=initial,burn_steps=burn_steps,production_steps=production_steps,
                sample_every=sample_every,frame_every=frame_every,blocks=blocks,progress=False,**options)
            run.metadata.update(initial_mode=initial_mode,range_sweep_signature=signature,
                                range_sweep_protocol=protocol,common_noise_seed=p.seed)
            run.save(path)
            if progress:
                print(f"  saved ell={ell:g}: {time.perf_counter()-started:.1f}s",flush=True)
        elif progress:
            print(f"{kind} ell={ell:g}: loaded matching dynamical trajectory",flush=True)
        series=ShellEP.from_run(run)
        summary=shell_summary(series,blocks=blocks,draws=draws,density=density,increment=increment)
        # Sum the signed channels before computing global variance or PSD.
        global_values=np.column_stack([series.medium_rate.sum(1),series.lr_rate.sum(1),
                                      series.local_rate[:,0],series.local_rate[:,1]])*summary["scale"]
        global_stats=fluctuation_statistics(global_values,blocks=blocks,draws=draws)
        frequency,psd=temporal_spectrum(global_values,series.metadata["interval"])
        scaling=scaling_comparison(run,series,summary,rho_min=rho_min,rho_max=rho_max)
        item=dict(ell=float(ell),model=model,run=run,series=series,summary=summary,
                  global_values=global_values,global_stats=global_stats,frequency=frequency,
                  psd=psd,scaling=scaling,stationarity=stationarity_report(model,run),folder=member)
        series.save(member/"rho_ep_time_series.npz")
        save_shell_summary(series,summary,member/"rho_ep_fluctuations.npz")
        results.append(item)
    save_range_sweep(results,folder)
    return results


def save_range_sweep(results,output):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    rho=results[0]["series"].rho
    metadata=dict(kind=results[0]["model"].kernel.kind,range_parameter="ell in the dynamical kernel",
        coupling="fixed strength g; no additional ell-dependent normalization",
        global_channels=GLOBAL_CHANNELS,
        uncertainty="single common seed; 95% temporal block-bootstrap intervals, not seed-to-seed uncertainty",
        members=[dict(ell=r["ell"],metadata=r["run"].metadata,stationarity=r["stationarity"],
                      scaling=r["scaling"]["report"]) for r in results])
    (folder/"range_sweep.json").write_text(json.dumps(_finite_json(metadata),indent=2,allow_nan=False),encoding="utf-8")
    np.savez_compressed(folder/"range_sweep.npz",ell=[r["ell"] for r in results],rho=rho,
        variance=[r["summary"]["shell"]["variance"] for r in results],
        variance_ci95=[r["summary"]["shell"]["variance_ci"] for r in results],
        variance_over_rho2=[r["scaling"]["scaled_variance"] for r in results],
        density_two_point=[r["scaling"]["two_point"]["correlation"] for r in results],
        global_variance=[r["global_stats"]["variance"] for r in results],
        global_variance_ci95=[r["global_stats"]["variance_ci"] for r in results],
        global_mean=[r["global_stats"]["mean"] for r in results],frequency=results[0]["frequency"],
        global_psd=[r["psd"] for r in results],metadata=json.dumps(_finite_json(metadata),allow_nan=False))


def _save(fig,output,name):
    fig.tight_layout()
    if output is not None:
        path=Path(output);path.mkdir(parents=True,exist_ok=True)
        fig.savefig(path/name,dpi=160,bbox_inches="tight")


def plot_range_fluctuations(results,output=None):
    """All curves correspond to newly integrated dynamics at the labeled ell."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    colors=plt.cm.viridis(np.linspace(.08,.9,len(results)))
    for r,color in zip(results,colors):
        rho=r["series"].rho;keep=rho>0
        s=r["summary"]["shell"]
        label=f"ell={r['ell']:g}"
        for ax,x,divisor in ((axes[0,0],rho,1.),(axes[0,1],rho,rho**2),
                              (axes[0,2],rho/r["ell"],rho**2)):
            y=s["variance"][keep]/(divisor[keep] if np.ndim(divisor) else divisor)
            band=s["variance_ci"][:,keep]/(divisor[keep] if np.ndim(divisor) else divisor)
            ax.loglog(x[keep],y,"o-",ms=2.5,color=color,label=label)
            ax.fill_between(x[keep],*band,color=color,alpha=.14)
        axes[1,2].loglog(r["frequency"][1:],np.maximum(r["psd"][1:,1],1e-30),color=color,label=label)
    axes[0,0].set(xlabel=r"Source distance $\rho$",ylabel=r"$\mathrm{Var}(j_\rho)$",title="Medium EP at fixed source distance")
    axes[0,1].set(xlabel=r"Source distance $\rho$",ylabel=r"$\mathrm{Var}(j_\rho)/\rho^2$",title="Physical distance, changing kernel ell")
    axes[0,2].set(xlabel=r"Rescaled distance $\rho/\ell$",ylabel=r"$\mathrm{Var}(j_\rho)/\rho^2$",title="Rescaled distance; amplitude left unchanged")
    ell=np.array([r["ell"] for r in results])
    for j,label,color in ((0,"total medium EP","C0"),(1,"total kernel EP","C1"),(3,"local active EP","C2")):
        y=np.array([r["global_stats"]["variance"][j] for r in results])
        ci=np.array([r["global_stats"]["variance_ci"][:,j] for r in results])
        axes[1,0].plot(ell,y,"o-",label=label,color=color)
        axes[1,0].fill_between(ell,ci[:,0],ci[:,1],alpha=.18,color=color)
    axes[1,0].set(xlabel=r"Kernel range parameter $\ell$",ylabel="Variance of signed global sum",yscale="log",
                  title="Sum all shells first; covariance retained")
    rho=results[0]["series"].rho
    for target in (2.,4.,8.,16.):
        j=int(np.argmin(abs(rho-target)))
        y=np.array([r["scaling"]["scaled_variance"][j] for r in results])
        ci=np.array([r["scaling"]["scaled_ci95"][:,j] for r in results])
        axes[1,1].loglog(ell,y,"o-",label=f"rho={rho[j]:g}")
        axes[1,1].fill_between(ell,ci[:,0],ci[:,1],alpha=.13)
    axes[1,1].set(xlabel=r"Kernel range parameter $\ell$",ylabel=r"$\mathrm{Var}(j_\rho)/\rho^2$",
                  title="Range dependence at fixed physical rho")
    axes[1,2].set(xlabel="Temporal frequency",ylabel="One-sided PSD of total kernel EP",
                  title="Temporal spectra across actual kernel ranges")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    spec=results[0]["model"].kernel
    fig.suptitle(f"{spec.kind.capitalize()}: fresh dynamics at each ell; strength g={spec.strength:g} fixed\n"
                 "Euler-Maruyama / midpoint EP; shading: 95% time-block bootstrap intervals",fontsize=12)
    _save(fig,output,"range_ep_fluctuations.png")
    return fig


def plot_range_kernels(results,output=None):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for r in results:
        m=r["model"];rho=r["series"].rho;label=f"ell={r['ell']:g}"
        q=np.sort(np.unique(m.q[m.q>0]))
        axes[0,0].semilogy(q,m.kernel.strength*m.kernel.multiplier(q),label=label)
        count=np.bincount(m.r_labels.ravel(),minlength=len(rho))
        weights=np.bincount(m.r_labels.ravel(),weights=m.weights.ravel(),minlength=len(rho))
        radial=np.divide(weights,count,out=np.zeros_like(weights),where=count>0)*m.kernel.strength
        axes[0,1].plot(rho,radial,"o-",ms=2.5,label=label)
        c=r["scaling"]["two_point"]
        axes[1,0].plot(rho,c["correlation"]/c["zero_distance"],"o-",ms=2.5,label=label)
        structure=np.bincount(m.q_labels,weights=r["run"].mean_structure_2d.ravel(),minlength=len(m.q_count))
        structure=np.divide(structure,m.q_count,out=np.zeros_like(structure),where=m.q_count>0)
        keep=(m.q_centres>0)&m.q_valid
        axes[1,1].loglog(m.q_centres[keep],np.maximum(structure[keep],1e-30),label=label)
    axes[0,0].set(xlabel="Lattice wave number q",ylabel=r"$g\widehat K_\ell(q)$",title="Kernel multiplier changes in the drift")
    axes[0,1].set(xlabel=r"Source distance $\rho$",ylabel="Signed real-space weight (radial mean)",yscale="symlog",
                  title="Periodic zero-mean kernel; on-site weight retained")
    axes[0,1].set_yscale("symlog",linthresh=1e-5)
    axes[1,0].axhline(0,color="gray",lw=.8)
    axes[1,0].set(xlabel=r"Separation $\rho$",ylabel=r"$C_\ell(\rho)/C_\ell(0)$",title="Each trajectory's connected density correlation")
    axes[1,1].set(xlabel="Lattice wave number q",ylabel="Production mean S(q)",title="The density structure responds to ell")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.suptitle(f"{results[0]['model'].kernel.kind.capitalize()}: kernel range and the resulting density structure",fontsize=12)
    _save(fig,output,"range_kernel_and_correlations.png")
    return fig


def plot_range_scaling(results,output=None):
    import matplotlib.pyplot as plt
    rows=int(np.ceil(len(results)/2))
    fig,axes=plt.subplots(rows,2,figsize=(13,4.3*rows),squeeze=False)
    for ax,r in zip(axes.flat,results):
        data=r["scaling"];rho=data["rho"];keep=rho>0
        ep=data["scaled_variance"];c=data["two_point"]["correlation"]
        branch=data["report"]["common_sign_branch"]
        ref=int(np.argmin(abs(rho-(branch["rho"][0] if branch["rho"] else 2.))))
        ax.loglog(rho[keep],ep[keep]/ep[ref],"o-",ms=2.5,label="Var(EP sum) / rho^2")
        for sign,marker in ((1,"o"),(-1,"x")):
            use=keep&(np.sign(c)==sign)
            ax.loglog(rho[use],abs(c[use])/abs(c[ref]),marker,ms=3.5,color="C1",label=f"|C|, sign {sign:+d}")
        if len(branch["rho"])>=3:
            ax.axvspan(branch["rho"][0],branch["rho"][-1],color="gray",alpha=.1)
            title=f"ell={r['ell']:g}, common rho={branch['rho'][0]:g}..{branch['rho'][-1]:g}\n"
            title+=f"EP slope {branch['ep_variance_over_rho2_slope']:.2f}; C slope {branch['two_point_abs_slope']:.2f}"
        else:
            title=f"ell={r['ell']:g}: too few bins before the C sign change for a shared fit"
        ax.set(xlabel=r"$\rho$",ylabel="Normalized at first selected rho",title=title)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    fig.suptitle("EP fluctuation / rho squared versus density two-point scaling at each physical kernel range\n"
                 "Finite-window effective slopes; kernel zero crossings and coarsening can affect fits",fontsize=12)
    _save(fig,output,"range_scaling_comparison.png")
    return fig


def plot_range_states(results,output=None):
    import matplotlib.pyplot as plt
    count=len(results);rows=int(np.ceil((count+2)/3))
    fig,axes=plt.subplots(rows,3,figsize=(15,4.1*rows),squeeze=False)
    flat=list(axes.flat)
    bound=max(np.max(abs(r["run"].fields[-1])) for r in results)
    for ax,r in zip(flat,results):
        ax.imshow(r["run"].fields[-1],origin="lower",cmap="RdBu_r",vmin=-bound,vmax=bound)
        ax.set(title=f"ell={r['ell']:g}, final phi",xlabel="x grid site",ylabel="y grid site")
    for r in results:
        run=r["run"];m=r["model"];label=f"ell={r['ell']:g}"
        flat[count].plot(run.time,run.variance,label=label)
        weight=run.structure*m.q_count
        qmean=(weight*m.q_centres).sum(1)/np.maximum(weight.sum(1),1e-30)
        flat[count+1].plot(run.time,qmean,label=label)
    flat[count].set(xlabel="Physical time",ylabel="Var(phi)",title="Equal durations; inspect structural drift")
    flat[count+1].set(xlabel="Physical time",ylabel="S-weighted mean q",title="Coarsening / characteristic-scale drift")
    burn=results[0]["run"].metadata["burn_steps"]*results[0]["model"].p.dt
    for ax in flat[count:count+2]:
        ax.axvline(burn,color="gray",ls=":",label="production starts")
        ax.legend(fontsize=8)
    for ax in flat[count+2:]:
        ax.set_visible(False)
    fig.suptitle("Fresh 64x64 trajectories across kernel ell; common color scale for all final fields",fontsize=12)
    _save(fig,output,"range_states_and_drift.png")
    return fig
