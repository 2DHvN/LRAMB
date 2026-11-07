"""Compare a shell EP variance divided by rho**2 with a density two-point function."""
import json
from pathlib import Path

import numpy as np

from .model import Parameters


def density_two_point(run, distance_edges):
    """Connected, equal-time, pair-averaged C(rho) on the same production window.

    mean_structure_2d averages every production EP endpoint, not sparse movies.
    Its orthonormal power divided by dV has inverse DFT <delta phi(x) delta
    phi(x+r)>_x. The conserved zero mode was removed during simulation.
    """
    p=Parameters(**run.metadata["parameters"])
    correlation=np.fft.ifft2(run.mean_structure_2d/p.dx**2).real
    offsets=np.fft.fftfreq(p.n)*p.n*p.dx
    xx,yy=np.meshgrid(offsets,offsets)
    labels=np.digitize(np.hypot(xx,yy).ravel(),distance_edges)-1
    count=len(distance_edges)-1
    if np.any((labels<0)|(labels>=count)):
        raise ValueError("Distance edges must cover every periodic displacement")
    pairs=np.bincount(labels,minlength=count)
    total=np.bincount(labels,weights=correlation.ravel(),minlength=count)
    radial=np.divide(total,pairs,out=np.full(count,np.nan),where=pairs>0)
    return dict(correlation=radial,displacement_count=pairs,map=correlation,
                zero_distance=float(correlation[0,0]))


def _slope(rho, values, keep):
    if np.count_nonzero(keep)<3:
        return float("nan")
    x=np.log(rho[keep]); x-=x.mean()
    return float(np.log(np.abs(values[keep]))@x/(x@x))


def _local_slopes(rho,values,width=5,sign_sensitive=False):
    result=np.full(len(rho),np.nan)
    for j in range(len(rho)):
        indices=np.arange(max(0,j-width//2),min(len(rho),j+width//2+1))
        if len(indices)!=width or np.any(rho[indices]<=0) or np.any(values[indices]==0):
            continue
        if sign_sensitive and np.any(np.sign(values[indices])!=np.sign(values[j])):
            continue
        keep=np.zeros(len(rho),dtype=bool); keep[indices]=True
        result[j]=_slope(rho,values,keep)
    return result


def scaling_comparison(run,series,summary,*,rho_min=2.,rho_max=16.):
    """Descriptive finite-window slopes; compare both observables on one sign branch."""
    if (len(series.time)!=len(run.prod_time)
            or not np.allclose(series.time,run.prod_time)
            or summary["shell"]["used_intervals"]!=len(run.prod_time)):
        raise ValueError("Use the full production window and a block count dividing its EP intervals, "
                         "so C and EP refer to the same observation window")
    rho=series.rho
    rho2=np.where(rho>0,rho**2,np.nan)
    scaled=summary["shell"]["variance"]/rho2
    ci=summary["shell"]["variance_ci"]/rho2
    replicas=summary["shell"]["variance_draws"]/rho2
    two_point=density_two_point(run,series.distance_edges)
    correlation=two_point["correlation"]
    window=(rho>=rho_min)&(rho<=rho_max)&np.isfinite(scaled)&(scaled>0)
    # C can oscillate. Never fit |C| through its zero or across opposite signs.
    common=np.zeros(len(rho),dtype=bool)
    indices=np.flatnonzero(window)
    if len(indices):
        sign=np.sign(correlation[indices[0]])
        for j in indices:
            if (not np.isfinite(correlation[j]) or np.sign(correlation[j])!=sign
                    or abs(correlation[j])<1e-12*abs(two_point["zero_distance"])):
                break
            common[j]=True
    def fit(mask,fit_c=True):
        ep=_slope(rho,scaled,mask)
        c=_slope(rho,correlation,mask) if fit_c else float("nan")
        if np.count_nonzero(mask)>=3:
            x=np.log(rho[mask]);x-=x.mean()
            samples=np.log(np.maximum(replicas[:,mask],1e-300))@x/(x@x)
            interval=np.quantile(samples,[.025,.975]).tolist()
        else:
            interval=[float("nan")]*2
        return dict(rho=rho[mask].tolist(),ep_variance_over_rho2_slope=ep,
                    ep_slope_ci95=interval,two_point_abs_slope=c,
                    slope_difference=ep-c)
    return dict(rho=rho,scaled_variance=scaled,scaled_ci95=ci,two_point=two_point,
                ep_local_slope=_local_slopes(rho,scaled),
                correlation_local_slope=_local_slopes(rho,correlation,sign_sensitive=True),
                report=dict(requested_window=[rho_min,rho_max],
                    ep_full_window=fit(window,fit_c=False),common_sign_branch=fit(common),
                    ep_sampling_interval=series.metadata["interval"],
                    correlation_sampling="every saved production EP endpoint",
                    correlation_uncertainty="not estimated: blockwise structure powers were not saved",
                    interpretation="descriptive finite-window slopes; not established asymptotic exponents"))


def plot_scaling_comparison(run,series,summary,output=None,*,rho_min=2.,rho_max=16.):
    import matplotlib.pyplot as plt
    data=scaling_comparison(run,series,summary,rho_min=rho_min,rho_max=rho_max)
    rho=data["rho"]; keep=rho>0
    ep=data["scaled_variance"];ci=data["scaled_ci95"]
    c=data["two_point"]["correlation"]; c0=data["two_point"]["zero_distance"]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    axes[0,0].loglog(rho[keep],ep[keep],"o-",ms=3)
    axes[0,0].fill_between(rho[keep],ci[0,keep],ci[1,keep],alpha=.25)
    axes[0,0].set(xlabel=r"Source distance $\rho$",ylabel=r"$\mathrm{Var}_t[j_\rho]/\rho^2$",
                  title="Signed space sum first, then variance / rho squared")
    axes[0,1].plot(rho,c/c0,"o-",ms=3)
    axes[0,1].axhline(0,color="gray",lw=.8)
    axes[0,1].set(xlabel=r"Separation $\rho$",ylabel=r"$C(\rho)/C(0)$",
                  title="Connected density two-point function (signed pair mean)")
    branch=data["report"]["common_sign_branch"]
    if branch["rho"]:
        ref=np.argmin(abs(rho-branch["rho"][0]))
        if ep[ref]>0 and abs(c[ref])>0:
            axes[1,0].loglog(rho[keep],ep[keep]/ep[ref],"o-",ms=3,
                label=r"$[\mathrm{Var}(j_\rho)/\rho^2]$ / reference")
            axes[1,0].fill_between(rho[keep],ci[0,keep]/ep[ref],ci[1,keep]/ep[ref],alpha=.2)
            for sign,marker,label in ((1,"o","C > 0"),(-1,"x","C < 0")):
                use=keep&(np.sign(c)==sign)
                axes[1,0].loglog(rho[use],abs(c[use])/abs(c[ref]),marker,ms=4,
                    color="C1",label=f"|C| / reference ({label})")
            fit_r=np.asarray(branch["rho"])
            for key,color in (("ep_variance_over_rho2_slope","C0"),("two_point_abs_slope","C1")):
                slope=branch[key]
                if np.isfinite(slope):
                    axes[1,0].loglog(fit_r,(fit_r/fit_r[0])**slope,"--",color=color,
                                    label=f"common-window slope {slope:.2f}")
    axes[1,0].set(xlabel=r"$\rho$",ylabel="Shape normalized at first fitted rho",
                  title="Scaling comparison: same window before first C sign change")
    axes[1,0].legend(fontsize=8)
    axes[1,1].plot(rho[keep],data["ep_local_slope"][keep],"o-",ms=3,label="Var(EP) / rho^2")
    axes[1,1].plot(rho[keep],data["correlation_local_slope"][keep],"o-",ms=3,label="|C(rho)|")
    axes[1,1].set(xlabel=r"$\rho$",ylabel=r"$d\log|f|/d\log\rho$ (5-shell fit)",
                  title="Local slopes; C windows spanning a sign change are omitted")
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.axvspan(rho_min,rho_max,color="gray",alpha=.07)
        ax.axvline(np.sqrt(series.metadata["area"])/2,color="gray",ls=":",lw=.8)
    fig.suptitle(f"{run.metadata['kernel']['kind'].capitalize()}: EP fluctuations and density correlations on one trajectory\n"
                "EP is a dynamical, kernel-weighted observable; C is an equal-time pair average",fontsize=12)
    fig.tight_layout()
    if output is not None:
        folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
        fig.savefig(folder/"rho2_scaling_vs_two_point.png",dpi=160,bbox_inches="tight")
        (folder/"rho2_scaling.json").write_text(json.dumps(data["report"],indent=2),encoding="utf-8")
        np.savez_compressed(folder/"rho2_scaling.npz",rho=rho,scaled_variance=ep,scaled_ci95=ci,
            density_two_point=c,displacement_count=data["two_point"]["displacement_count"],
            ep_local_slope=data["ep_local_slope"],correlation_local_slope=data["correlation_local_slope"])
    return fig,data


def plot_scaling_collection(comparisons,output):
    """A compact comparison of all kernels, for export beside the notebooks."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(comparisons),figsize=(5*len(comparisons),4.8),squeeze=False)
    reports={}
    for ax,(kind,data) in zip(axes[0],comparisons.items()):
        rho=data["rho"];ep=data["scaled_variance"]
        c=data["two_point"]["correlation"]
        fit=data["report"]["common_sign_branch"];reports[kind]=data["report"]
        if not fit["rho"]:
            ax.set_title(f"{kind}: no common sign branch")
            continue
        ref=np.argmin(abs(rho-fit["rho"][0]))
        keep=(rho>0)&(rho<=data["report"]["requested_window"][1])
        ax.loglog(rho[keep],ep[keep]/ep[ref],"o-",ms=3,label="Var(EP sum) / rho^2")
        ax.fill_between(rho[keep],*(data["scaled_ci95"][:,keep]/ep[ref]),alpha=.2)
        for sign,marker,label in ((1,"o","C > 0"),(-1,"x","C < 0")):
            use=keep&(np.sign(c)==sign)
            ax.loglog(rho[use],abs(c[use])/abs(c[ref]),marker,ms=4,color="C1",label=f"|C| ({label})")
        ax.axvspan(fit["rho"][0],fit["rho"][-1],color="gray",alpha=.1)
        ax.set(xlabel=r"Distance $\rho$",ylabel="Normalized at first fitted rho",
            title=f"{kind.capitalize()}: common fit {fit['rho'][0]:g}..{fit['rho'][-1]:g}\n"
                  f"EP slope {fit['ep_variance_over_rho2_slope']:.2f}; C slope {fit['two_point_abs_slope']:.2f}")
        ax.legend(fontsize=8)
    fig.suptitle("Same production trajectories: EP fluctuation / rho squared versus density two-point correlations\n"
                 "Grey: common fit window before the first C sign change; effective slopes, not asymptotic exponents",fontsize=11)
    fig.tight_layout()
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=170,bbox_inches="tight")
    path.with_suffix(".json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    return fig
