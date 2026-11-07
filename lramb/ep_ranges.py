"""Space-summed medium EP fluctuations by source-distance shell and cutoff.

The order is: sum signed EP at fixed rho, centre the resulting time series,
then compute its variance. Local conservative/active terms live at rho=0.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .model import Parameters, Run
from .decomposition import DecompositionResult
from .analysis import temporal_spectrum


@dataclass
class ShellEP:
    metadata: dict
    distance_edges: np.ndarray
    time: np.ndarray
    lr_rate: np.ndarray       # time x rho; already summed over all space and phi
    local_rate: np.ndarray    # time x (local conservative, local active)

    @property
    def rho(self):
        return (self.distance_edges[:-1]+self.distance_edges[1:])/2

    @property
    def medium_rate(self):
        values=self.lr_rate.copy()
        values[:,0]+=self.local_rate.sum(axis=1)
        return values

    @property
    def cumulative_rate(self):
        return np.cumsum(self.medium_rate,axis=1)

    @classmethod
    def from_decomposition(cls, result: DecompositionResult):
        """Collapse receiving-phi bins *before* taking any moments or PSD."""
        return cls(metadata=dict(result.metadata,source="phi-bin sum of full-step midpoint EP"),
                   distance_edges=result.distance_edges.copy(),time=result.time.copy(),
                   lr_rate=result.rate_lr.sum(axis=1),local_rate=result.rate_local.sum(axis=1))

    @classmethod
    def from_run(cls, run: Run, *, production_steps=None):
        """Fine dx-width shells; the quadratic midpoint identity telescopes exactly.

        Uses saved full-precision shell energies, not differences of sparse
        float32 fields. This is the space sum of the per-step midpoint kernel EP.
        """
        p=Parameters(**run.metadata["parameters"])
        interval=p.dt*run.metadata["sample_every"]
        steps=run.metadata["production_steps"] if production_steps is None else production_steps
        if steps<=0 or steps>run.metadata["production_steps"] or steps % run.metadata["sample_every"]:
            raise ValueError("Requested production prefix must align with saved EP intervals")
        count=steps//run.metadata["sample_every"]
        rates=-np.diff(run.prod_energy_r[:count+1],axis=0)/(p.D*interval)
        edges=(np.arange(rates.shape[1]+1)-.5)*p.dx
        metadata=dict(run.metadata,interval=interval,area=(p.n*p.dx)**2,dV=p.dx**2,
                      dt=p.dt,used_production_steps=steps,source="space-summed midpoint kernel EP via exact quadratic energy difference")
        return cls(metadata=metadata,distance_edges=edges,time=run.prod_time[:count].copy(),
                   lr_rate=rates,local_rate=run.ep_rate[:count][:,[0,2]].copy())

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,metadata=json.dumps(self.metadata),distance_edges=self.distance_edges,
                            time=self.time,lr_rate=self.lr_rate,local_rate=self.local_rate)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            return cls(metadata=json.loads(str(d["metadata"])),
                       **{k:d[k].copy() for k in ("distance_edges","time","lr_rate","local_rate")})


def fluctuation_statistics(values, *, blocks=8, draws=500, seed=291):
    """Variance/std and block-bootstrap intervals, retaining shell covariance.

    Complete nonoverlapping time blocks are resampled jointly across every rho.
    Variance is taken about the global mean, not a separately removed block mean.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or not np.isfinite(x).all() or blocks<2 or draws<2:
        raise ValueError("Use a finite time-by-shell matrix, blocks>=2 and draws>=2")
    length=len(x)//blocks
    if length<2:
        raise ValueError("At least two EP intervals per bootstrap block are needed")
    x=x[:blocks*length]
    mean=x.mean(axis=0)
    centred=x-mean
    variance=np.mean(centred**2,axis=0)
    covariance=centred.T@centred/len(x)
    # Centre once for numerical stability; moments of resampled blocks restore
    # their own global mean through E[y^2]-E[y]^2.
    grouped=centred.reshape(blocks,length,x.shape[1])
    first=grouped.mean(axis=1)
    second=(grouped**2).mean(axis=1)
    index=np.random.default_rng(seed).integers(blocks,size=(draws,blocks))
    variance_draws=np.maximum(second[index].mean(axis=1)-first[index].mean(axis=1)**2,0)
    return dict(mean=mean,variance=variance,std=np.sqrt(variance),covariance=covariance,
                variance_ci=np.quantile(variance_draws,[.025,.975],axis=0),
                std_ci=np.sqrt(np.quantile(variance_draws,[.025,.975],axis=0)),
                variance_draws=variance_draws,used_intervals=len(x),block_intervals=length)


def shell_summary(series: ShellEP, *, blocks=8, draws=500, density=False, increment=False):
    scale=(series.metadata["interval"] if increment else 1.)/(series.metadata["area"] if density else 1.)
    shell=series.medium_rate*scale
    cumulative=np.cumsum(shell,axis=1)
    a=fluctuation_statistics(shell,blocks=blocks,draws=draws)
    b=fluctuation_statistics(cumulative,blocks=blocks,draws=draws)
    # Covariances between different distances are essential to cumulative EP.
    predicted=np.cumsum(np.cumsum(a["covariance"],axis=0),axis=1).diagonal()
    np.testing.assert_allclose(b["variance"],predicted,rtol=1e-9,atol=1e-18)
    return dict(shell=a,cumulative=b,values=shell,cumulative_values=cumulative,
                scale=scale,density=density,increment=increment,blocks=blocks,draws=draws)


def decay_diagnostic(series: ShellEP,summary, *, rho_min=2.,rho_max=None):
    """Descriptive decay in rho, with a time-block bootstrap of the log slope."""
    rho=series.rho
    if rho_max is None:
        rho_max=np.sqrt(series.metadata["area"])/4
    std=summary["shell"]["std"]
    keep=(rho>=rho_min)&(rho<=rho_max)&(std>0)
    indices=np.flatnonzero(keep)
    if len(indices)<3:
        return dict(interpretation="insufficient nonzero rho bins",rho_min=rho_min,rho_max=rho_max)
    log_r=np.log(rho[keep]); design=log_r-log_r.mean()
    slope=float(np.dot(design,np.log(std[keep]))/np.dot(design,design))
    replicas=np.sqrt(summary["shell"]["variance_draws"][:,keep])
    slopes=np.log(np.maximum(replicas,1e-300))@design/np.dot(design,design)
    low,high=np.quantile(slopes,[.025,.975])
    return dict(rho_min=float(rho[indices[0]]),rho_max=float(rho[indices[-1]]),
                log_std_slope=slope,log_std_slope_ci95=[float(low),float(high)],
                spearman_rho=float(spearmanr(rho[keep],std[keep]).statistic),
                decreasing_adjacent_fraction=float(np.mean(np.diff(std[keep])<0)),
                interpretation="decreasing over this rho window" if high<0 else
                               "increasing over this rho window" if low>0 else "unresolved rho trend")


def plot_shell_summary(series,summary,output=None, *, rho_min=2.,rho_max=None,prefix="rho_ep"):
    """The requested 1D space-summed EP variation, cumulative range and PSD."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    rho=series.rho
    keep=rho>0
    values=summary["values"]
    s,c=summary["shell"],summary["cumulative"]
    unit="Entropy increment" if summary["increment"] else "EP rate"
    unit+=" / area" if summary["density"] else ""
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,key,ci,label in ((axes[0,0],"std","std_ci","Std"),
                            (axes[0,1],"variance","variance_ci","Variance")):
        ax.plot(rho[keep],s[key][keep],"o-",ms=3)
        ax.fill_between(rho[keep],s[ci][0,keep],s[ci][1,keep],alpha=.25,label="95% time-block bootstrap interval")
        ax.set(xlabel=r"Source distance $\rho$",ylabel=f"{label} of space-summed {unit}",yscale="log",
               title=f"{label} after summing every site / phi at fixed rho")
        ax.axvline(np.sqrt(series.metadata["area"])/2,color="gray",ls=":",label="L/2; outer bins include torus corners")
        ax.text(.03,.97,f"rho=0: {label} = {s[key][0]:.4g}",transform=ax.transAxes,
                va="top",fontsize=8,bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
        ax.legend(fontsize=7)
    chosen=np.unique(np.linspace(1,len(rho)-1,min(4,len(rho)-1),dtype=int))
    for j in chosen:
        axes[0,2].plot(series.time,values[:,j]-s["mean"][j],lw=.7,label=f"rho={rho[j]:g}")
    axes[0,2].set(xlabel="Production time",ylabel=f"Centred {unit}",title="One signed sum time signal for each rho")
    axes[0,2].legend(fontsize=8)
    edge=series.distance_edges[1:]
    axes[1,0].plot(edge,c["std"],"o-",ms=3)
    axes[1,0].fill_between(edge,*c["std_ci"],alpha=.25)
    axes[1,0].set(xlabel="Included source radius R (shell outer edge)",ylabel=f"Std of cumulative {unit}",
                  title="Same trajectory: local + sum_(rho <= R) EP")
    covariance=s["covariance"]
    denom=np.sqrt(np.outer(s["variance"],s["variance"]))
    correlation=np.divide(covariance,denom,out=np.zeros_like(covariance),where=denom>0)
    image=axes[1,1].imshow(correlation,origin="lower",cmap="RdBu_r",vmin=-1,vmax=1,
        extent=[series.distance_edges[0],series.distance_edges[-1]]*2,aspect="auto")
    fig.colorbar(image,ax=axes[1,1],label="Correlation of signed shell sums")
    axes[1,1].set(xlabel="rho",ylabel="rho'",title="Cross-shell correlations enter cumulative variance")
    frequency,psd=temporal_spectrum(values,series.metadata["interval"])
    image=axes[1,2].pcolormesh(rho[keep],frequency[1:],np.maximum(psd[1:,keep],1e-30),
                              shading="auto",cmap="magma",norm=LogNorm())
    axes[1,2].set(xlabel="rho",ylabel="Temporal frequency",yscale="log",title="PSD of space-summed rho signals")
    fig.colorbar(image,ax=axes[1,2],label=f"PSD of {unit}")
    diagnostic=decay_diagnostic(series,summary,rho_min=rho_min,rho_max=rho_max)
    fig.suptitle(f"Local terms at rho=0; nonlocal shell sums shown separately. {diagnostic['interpretation']}",fontsize=12)
    fig.tight_layout()
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        fig.savefig(output/f"{prefix}.png",dpi=160,bbox_inches="tight")
        (output/f"{prefix}_decay.json").write_text(json.dumps(diagnostic,indent=2),encoding="utf-8")
    return fig,diagnostic


def plot_cumulative_ranges(series,summary,output=None, *, prefix="interaction_range_ep"):
    """Auxiliary source-distance cutoff R, keeping physical kernel ell fixed.

    Actual Poisson/Linnik range sweeps integrate new runs via range_sweep.py.
    The legacy output prefix is retained for existing notebooks.
    """
    import matplotlib.pyplot as plt
    area_scale=summary["scale"]
    cutoff=series.distance_edges[1:]
    values=summary["cumulative_values"]
    kernel=np.cumsum(series.lr_rate*area_scale,axis=1)
    kernel_stats=fluctuation_statistics(kernel,blocks=summary["blocks"],draws=summary["draws"])
    full=summary["cumulative"]
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    axes[0].plot(cutoff,full["std"],"o-",ms=3,label="medium: local at 0 + kernel up to R")
    axes[0].fill_between(cutoff,*full["std_ci"],alpha=.2)
    axes[0].plot(cutoff,kernel_stats["std"],"s-",ms=3,label="kernel contribution up to R")
    axes[0].fill_between(cutoff,*kernel_stats["std_ci"],alpha=.2)
    axes[0].set(xlabel="Source-distance cutoff R (outer shell edge)",ylabel="Std of space-summed EP",
                title="Auxiliary distance cutoff; cross-shell covariance retained")
    axes[0].legend(fontsize=7)
    targets=(cutoff[0],np.sqrt(series.metadata["area"])/8,
             np.sqrt(series.metadata["area"])/4,cutoff[-1])
    indices=np.unique([np.argmin(abs(cutoff-r)) for r in targets])
    freq,psd=temporal_spectrum(values[:,indices],series.metadata["interval"])
    for j,index in enumerate(indices):
        signal=values[:,index]-values[:,index].mean()
        label=f"R={cutoff[index]:g}"
        axes[1].plot(series.time,signal,lw=.6,label=label)
        axes[2].loglog(freq[1:],np.maximum(psd[1:,j],1e-30),label=label)
    axes[1].set(xlabel="Production time",ylabel="Centred cumulative EP",title="Signed space sum with selected ranges included")
    axes[2].set(xlabel="Temporal frequency",ylabel="One-sided PSD",title="Fluctuation spectra versus source cutoff R")
    axes[1].legend(fontsize=8);axes[2].legend(fontsize=8)
    fig.suptitle("Auxiliary source-distance cutoff on one trajectory; physical kernel range ell is fixed",fontsize=12)
    fig.tight_layout()
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        fig.savefig(output/f"{prefix}.png",dpi=160,bbox_inches="tight")
    return fig,kernel_stats


def save_shell_summary(series,summary,path):
    frequency,psd=temporal_spectrum(summary["values"],series.metadata["interval"])
    np.savez_compressed(path,rho=series.rho,distance_edges=series.distance_edges,time=series.time,
        medium_rate=series.medium_rate,kernel_rate=series.lr_rate,local_rate=series.local_rate,
        mean=summary["shell"]["mean"],variance=summary["shell"]["variance"],
        std=summary["shell"]["std"],std_ci95=summary["shell"]["std_ci"],
        variance_ci95=summary["shell"]["variance_ci"],covariance=summary["shell"]["covariance"],
        cumulative_std=summary["cumulative"]["std"],cumulative_std_ci95=summary["cumulative"]["std_ci"],
        frequency=frequency,psd=psd,metadata=json.dumps(dict(series.metadata,
            density=summary["density"],increment=summary["increment"])))
