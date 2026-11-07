"""Planar-interface stability on the existing periodic conserved lattice.

A central rising wall necessarily has a falling companion at the periodic seam.
Both are tracked by orientation, with temporal unwrapping and graph diagnostics.
"""
from dataclasses import asdict,dataclass,replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import linregress,t as student_t

from .model import ActiveModelB,Parameters,Run,simulate


@dataclass(frozen=True)
class WallSpec:
    width: float=2**.5
    phi_positive: float=1.
    phi_negative: float=-1.
    centre: float|None=None
    amplitude: float=1.
    mode: int=2
    perturbation: str="central"  # central only; bend: in phase; thickness: opposite phase
    graph_fraction: float=.95
    track_modes: int=8

    @classmethod
    def from_bulk(cls,p: Parameters,*,width=None,**kwargs):
        if p.a>=0:
            raise ValueError("A two-phase wall requires a < 0")
        if width is None:
            if p.kappa<=0:
                raise ValueError("Specify the interface width for kappa=0")
            width=np.sqrt(2*p.kappa/(-p.a))
        phi0=float(np.sqrt(-p.a/p.b))
        return cls(width=float(width),phi_positive=phi0,phi_negative=-phi0,**kwargs)


def wall_initial(p: Parameters,spec: WallSpec,*,perturbed=False,mean=None,return_info=False):
    """A +/- bulk slab, with a central wall and its periodic companion.

    A tanh of signed periodic distance gives the passive planar interface width.
    Matching mean changes a constant central-wall offset, never the phi plateaus.
    """
    length=p.n*p.dx
    centre=length/2 if spec.centre is None else float(spec.centre)%length
    if not (np.isfinite([spec.width,spec.phi_positive,spec.phi_negative,spec.amplitude,centre]).all()
            and spec.width>0 and spec.phi_positive>spec.phi_negative and spec.amplitude>0
            and isinstance(spec.mode,int) and 1<=spec.mode<p.n/4
            and spec.track_modes>=spec.mode and spec.track_modes<p.n/2
            and .5<spec.graph_fraction<=1 and spec.perturbation in {"central","bend","thickness"}
            and length/2-2*spec.amplitude>max(6*spec.width,4*p.dx)):
        raise ValueError("Use resolved separated walls, positive contrast/amplitude and resolved Fourier modes")
    x=np.arange(p.n)*p.dx
    wave=spec.amplitude*np.cos(2*np.pi*spec.mode*np.arange(p.n)/p.n) if perturbed else np.zeros(p.n)
    central=centre+wave
    companion=centre+length/2+(
        wave if spec.perturbation=="bend" else -wave if spec.perturbation=="thickness" else 0.)
    def profile(offset):
        c=central+offset
        separation=companion-c
        midpoint=(companion+c)/2
        distance=(x[None,:]-midpoint[:,None]+length/2)%length-length/2
        fraction=(1+np.tanh((separation[:,None]/2-abs(distance))/spec.width))/2
        return spec.phi_negative+(spec.phi_positive-spec.phi_negative)*fraction
    offset=0.
    if mean is not None:
        if not np.isfinite(mean) or not spec.phi_negative<mean<spec.phi_positive:
            raise ValueError("Mean must lie between the bulk plateaus")
        bound=min(length/8,(length/2-2*spec.amplitude-4*spec.width)/2)
        target=lambda delta:float(profile(delta).mean()-mean)
        if target(-bound)*target(bound)>0:
            raise ValueError("Requested mean cannot be matched by a small interface displacement")
        offset=float(brentq(target,-bound,bound,xtol=1e-12))
    phi=profile(offset)
    info=dict(central_wall=centre,companion_wall=(centre+length/2)%length,
        central_mean_correction=offset,phi_positive=spec.phi_positive,phi_negative=spec.phi_negative,
        actual_min=float(phi.min()),actual_max=float(phi.max()),mean=float(phi.mean()),
        periodic_interfaces=2,perturbation=spec.perturbation if perturbed else "flat",
        mass_matching="central-wall position adjustment; no uniform phi shift")
    return (phi,info) if return_info else phi


def run_wall_pair(p,kernel,spec=None,*,steps=16000,sample_every=40,frame_every=200,
                  blocks=8,multiplier=None,progress=True,backend="numpy",device="auto",
                  dtype="float64",cpu_threads=1):
    spec=spec or WallSpec.from_bulk(p)
    reference,ri=wall_initial(p,spec,return_info=True)
    perturbed,pi=wall_initial(p,spec,perturbed=True,mean=reference.mean(),return_info=True)
    cp=replace(p,mean=float(reference.mean()))
    kwargs=dict(burn_steps=0,production_steps=steps,sample_every=sample_every,frame_every=frame_every,
        blocks=blocks,progress=progress,backend=backend,device=device,dtype=dtype,cpu_threads=cpu_threads)
    a=simulate(ActiveModelB(cp,kernel,multiplier=multiplier),initial=reference,**kwargs)
    b=simulate(ActiveModelB(cp,kernel,multiplier=multiplier),initial=perturbed,**kwargs)
    for run,info in ((a,ri),(b,pi)):
        run.metadata.update(wall=asdict(spec),wall_initialization=info,paired_noise=True)
    return a,b


def wall_observables(fields,dx,spec: WallSpec):
    """Extract rising/central and falling/companion phi-midpoint contours.

    Multiple crossings are recorded rather than silently reinterpreted as a wall.
    Fourier coefficients are withheld when too many rows cease to be a graph.
    A least-squares Fourier fit uses valid rows if a small fraction is missing.
    """
    fields=np.asarray(fields)
    if fields.ndim!=3 or fields.shape[1]!=fields.shape[2] or not np.isfinite(fields).all():
        raise ValueError("Use a finite time-by-n-by-n field array")
    n=fields.shape[-1];length=n*dx
    threshold=(spec.phi_positive+spec.phi_negative)/2
    centre=length/2 if spec.centre is None else float(spec.centre)%length
    previous=np.repeat(np.array([centre,((centre+length/2)%length)])[:,None],n,axis=1)
    modes=min(spec.track_modes,n//2-1)
    theta=2*np.pi*np.arange(n)/n
    design=np.column_stack([np.ones(n)]+[f(j*theta) for j in range(1,modes+1) for f in (np.cos,np.sin)])
    out={key:[] for key in ("height","coefficients","mean_position","roughness","rms_slope",
                            "coverage","graph_fraction","crossing_counts","width_mean","width_min")}
    for phi in fields:
        forward=np.roll(phi,-1,axis=1)
        up=(phi<=threshold)&(forward>threshold)
        down=(phi>threshold)&(forward<=threshold)
        counts=np.column_stack((up.sum(1),down.sum(1)))
        height=np.full((2,n),np.nan)
        for wall,crosses in enumerate((up,down)):
            for row in range(n):
                indices=np.flatnonzero(crosses[row])
                if not len(indices):
                    continue
                fraction=(threshold-phi[row,indices])/(forward[row,indices]-phi[row,indices])
                crossing=((indices+fraction)*dx)%length
                displacement=(crossing-previous[wall,row]+length/2)%length-length/2
                nearest=int(np.argmin(abs(displacement)))
                if abs(displacement[nearest])<length/4:
                    height[wall,row]=previous[wall,row]+displacement[nearest]
                    previous[wall,row]=height[wall,row]
        coverage=np.isfinite(height).mean(1)
        separation=(height[1]-height[0])%length
        valid=np.isfinite(height).all(0)
        graph=(counts==1).all(1)&valid&(separation>4*dx)&(separation<length-4*dx)
        graph_fraction=float(graph.mean())
        coefficients=np.full((2,modes+1),complex(np.nan,np.nan))
        roughness=np.full(2,np.nan);rms_slope=np.full(2,np.nan)
        if graph_fraction>=spec.graph_fraction:
            for wall in (0,1):
                use=graph&np.isfinite(height[wall])
                if use.sum()<2*modes+1:
                    continue
                if use.all():
                    c=np.fft.rfft(height[wall])/n
                    coefficients[wall]=c[:modes+1]
                else:
                    fit=np.linalg.lstsq(design[use],height[wall,use],rcond=None)[0]
                    coefficients[wall,0]=fit[0]
                    coefficients[wall,1:]=(fit[1::2]-1j*fit[2::2])/2
                c=coefficients[wall]
                roughness[wall]=np.std(height[wall,use])
                # Resolved low-mode slope, excluding the translation mode.
                ky=2*np.pi*np.arange(1,modes+1)/length
                rms_slope[wall]=np.sqrt(2*np.sum(ky**2*abs(c[1:])**2))
        finite_width=separation[graph] if graph_fraction>=spec.graph_fraction else np.array([])
        out["height"].append(height)
        out["coefficients"].append(coefficients)
        out["mean_position"].append(coefficients[:,0].real)
        out["roughness"].append(roughness);out["rms_slope"].append(rms_slope)
        out["coverage"].append(coverage);out["graph_fraction"].append(graph_fraction)
        out["crossing_counts"].append(counts)
        out["width_mean"].append(float(np.mean(finite_width)) if len(finite_width) else np.nan)
        out["width_min"].append(float(np.min(finite_width)) if len(finite_width) else np.nan)
    return {key:np.asarray(value) for key,value in out.items()}


def wall_diagnostics(reference: Run,perturbed: Run,spec=None,*,fit_until=40.):
    if not np.array_equal(reference.time,perturbed.time):
        raise ValueError("Wall pairs require identical observation times")
    spec=spec or WallSpec(**reference.metadata["wall"])
    p=Parameters(**reference.metadata["parameters"])
    a=wall_observables(reference.fields,p.dx,spec)
    b=wall_observables(perturbed.fields,p.dx,spec)
    delta=b["coefficients"]-a["coefficients"]
    amplitude=2*abs(delta[:,:,spec.mode])
    ky=2*np.pi*spec.mode/(p.n*p.dx)
    valid=(np.minimum(a["graph_fraction"],b["graph_fraction"])>=spec.graph_fraction)
    valid &= np.isfinite(amplitude[:,0])&(amplitude[:,0]>0)&(amplitude[:,0]*ky<.5)
    valid &= (np.minimum(a["width_min"],b["width_min"])>max(6*spec.width,4*p.dx))
    valid &= reference.time<=fit_until
    bad=np.flatnonzero(~valid);end=int(bad[0]) if len(bad) else len(valid)
    fit=dict(rate=None,ci95=None,points=end,mode=spec.mode,wave_number=float(ky),
             observable="paired central-wall displacement Fourier amplitude; m=0 translation removed",
             interpretation="insufficient resolved small-perturbation graph window")
    if end>=6:
        regression=linregress(reference.time[:end],np.log(amplitude[:end,0]))
        ci=student_t.ppf(.975,end-2)*regression.stderr
        fit.update(rate=float(regression.slope),ci95=float(ci),
            interpretation="finite-time amplification" if regression.slope-ci>0 else
                           "finite-time decay" if regression.slope+ci<0 else "unresolved growth sign")
    survival=[]
    for run,obs in ((reference,a),(perturbed,b)):
        lost=np.flatnonzero(obs["graph_fraction"]<spec.graph_fraction)
        survival.append(dict(final_graph_fraction=float(obs["graph_fraction"][-1]),
            first_graph_loss_time=float(run.time[lost[0]]) if len(lost) else None,
            final_mean_slab_width=float(obs["width_mean"][-1]) if np.isfinite(obs["width_mean"][-1]) else None,
            final_central_roughness=float(obs["roughness"][-1,0]) if np.isfinite(obs["roughness"][-1,0]) else None,
            interpretation="two resolved interface graphs survive this window" if not len(lost) else
                           "additional crossings, missing interfaces or unresolved slab during this window"))
    return dict(reference=a,perturbed=b,delta_coefficients=delta,amplitude=amplitude,
        centre_response=abs(delta[:,0]+delta[:,1]),thickness_response=2*abs(delta[:,1]-delta[:,0]),
        gain=amplitude[:,0]/amplitude[0,0],fit=fit,survival=survival,
        field_norm=np.sqrt(np.mean((perturbed.fields.astype(float)-reference.fields)**2,axis=(1,2))),
        threshold=(spec.phi_positive+spec.phi_negative)/2)


def save_wall_report(report,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(fit=report["fit"],survival=report["survival"]),indent=2),encoding="utf-8")
    np.savez_compressed(path.with_suffix('.npz'),delta_coefficients=report["delta_coefficients"],
        amplitude=report["amplitude"],centre_response=report["centre_response"],
        thickness_response=report["thickness_response"],
        **{f"{name}_{key}":val for name in ("reference","perturbed") for key,val in report[name].items()})
