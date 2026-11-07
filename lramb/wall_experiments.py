"""Cached planar-wall experiments and visualizations, alongside circle studies."""
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .model import ActiveModelB,KernelSpec,Run
from .backends import backend_options
from .wall import WallSpec,run_wall_pair,wall_diagnostics,save_wall_report


def run_wall_suite(p,kernel,*,output,control_output,spec=None,steps=16000,
                   sample_every=40,frame_every=200,blocks=8,fit_until=40.,
                   multiplier=None,simulation=None,reuse_cache=True,include_controls=True):
    spec=spec or WallSpec.from_bulk(p)
    opts=backend_options(**(simulation or dict(backend="auto")))
    source_hash=sha256(b"".join((Path(__file__).parent/f).read_bytes()
        for f in ("wall.py","model.py","backends.py","torch_backend.py"))).hexdigest()
    cases=[("local passive",replace(p,lam=0.),KernelSpec("poisson",strength=0.),None,"passive"),
           ("local active",p,KernelSpec("poisson",strength=0.),None,"active"),
           (f"active + {kernel.kind}",p,kernel,multiplier,None)]
    if not include_controls:
        cases=[r for r in cases if r[-1] is None]
    suite=[]
    for name,parameters,ks,custom,control in cases:
        model=ActiveModelB(parameters,ks,multiplier=custom)
        signature=sha256(json.dumps(dict(parameters=asdict(parameters),kernel=asdict(ks),wall=asdict(spec),
            steps=steps,sample_every=sample_every,frame_every=frame_every,blocks=blocks,simulation=opts,
            source_hash=source_hash,multiplier=sha256(model.K.tobytes()).hexdigest()),sort_keys=True).encode()).hexdigest()
        # Include the whole protocol in the path so modes/amplitudes can coexist.
        base=Path(control_output)/control if control else Path(output)
        folder=base/f"mode_{spec.mode}_{signature[:12]}"
        folder.mkdir(parents=True,exist_ok=True)
        a_path,b_path=folder/"reference.npz",folder/"perturbed.npz"
        a=Run.load(a_path) if reuse_cache and a_path.exists() else None
        b=Run.load(b_path) if reuse_cache and b_path.exists() else None
        if a is None or b is None or any(r.metadata.get("wall_signature")!=signature for r in (a,b)):
            print(f"Wall {name}, mode={spec.mode}, A={spec.amplitude:g}: integrating flat / perturbed pair",flush=True)
            a,b=run_wall_pair(parameters,ks,spec,steps=steps,sample_every=sample_every,frame_every=frame_every,
                blocks=blocks,multiplier=custom,progress=False,**opts)
            for run,path in ((a,a_path),(b,b_path)):
                run.metadata.update(wall_signature=signature,wall_case=name)
                run.save(path)
        else:
            print(f"Wall {name}: matching cache",flush=True)
        report=wall_diagnostics(a,b,spec,fit_until=fit_until)
        save_wall_report(report,folder/"stability.json")
        suite.append(dict(name=name,control=control,spec=spec,reference=a,perturbed=b,report=report,folder=folder))
    Path(output).mkdir(parents=True,exist_ok=True)
    (Path(output)/"wall_cases.json").write_text(json.dumps([
        dict(case=r["name"],folder=str(r["folder"]),wall=asdict(spec),
             parameters=r["reference"].metadata["parameters"],kernel=r["reference"].metadata["kernel"],
             reference_initial=r["reference"].metadata["wall_initialization"],
             perturbed_initial=r["perturbed"].metadata["wall_initialization"],
             fit=r["report"]["fit"],survival=r["report"]["survival"]) for r in suite],indent=2),encoding="utf-8")
    return suite


def _save(fig,output,name):
    fig.tight_layout()
    if output is not None:
        folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
        fig.savefig(folder/name,dpi=160,bbox_inches="tight")


def plot_wall_initialization(suite,output=None):
    import matplotlib.pyplot as plt
    r=next(r for r in suite if r["control"] is None)
    a,b=r["reference"],r["perturbed"];p=a.metadata["parameters"];spec=r["spec"]
    length=p["n"]*p["dx"];x=np.arange(p["n"])*p["dx"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for ax,run,label in ((axes[0],a,"Flat reference"),(axes[1],b,"Central-wall perturbation")):
        image=ax.imshow(run.fields[0],origin="lower",extent=[0,length]*2,cmap="RdBu_r",
                        vmin=spec.phi_negative,vmax=spec.phi_positive)
        ax.contour(x,x,run.fields[0],levels=[(spec.phi_positive+spec.phi_negative)/2],colors="black",linewidths=.8)
        ax.axvline(a.metadata["wall_initialization"]["central_wall"],color="gray",ls=":")
        ax.set(xlabel="x",ylabel="y",title=label)
        fig.colorbar(image,ax=ax,label="phi")
    axes[2].plot(x,a.fields[0].mean(0),label="reference, y average")
    axes[2].plot(x,b.fields[0,0],label="perturbed, y=0")
    axes[2].plot(x,b.fields[0].mean(0),"--",label="perturbed, y average")
    for value in (spec.phi_positive,spec.phi_negative):
        axes[2].axhline(value,color="gray",ls=":",lw=.7)
    axes[2].set(xlabel="x",ylabel="Initial phi",title="Bulk plateaus and periodic seam")
    axes[2].legend(fontsize=8)
    fig.suptitle("Central wall at L/2 plus its periodic companion at x=0; fixed bulk plateaus, equal mass",fontsize=12)
    _save(fig,output,"wall_initialization.png")
    return fig


def plot_wall_comparison(suite,output=None):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    for r in suite:
        a=r["reference"];report=r["report"];obs=report["reference"];label=r["name"]
        axes[0].plot(a.time,report["amplitude"][:,0],label=label)
        axes[1].plot(a.time,obs["roughness"][:,0],label=label)
        axes[2].plot(a.time,obs["width_mean"],label=label)
    axes[0].set(xlabel="Time",ylabel="Paired central-wall mode amplitude",title="Response of injected mode; translation removed")
    axes[1].set(xlabel="Time",ylabel="Reference wall height RMS",title="Reference roughness (graph-valid frames)")
    axes[2].set(xlabel="Time",ylabel="Positive-phase slab width",title="Mean phase-width relaxation")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Flat-wall stability: local passive, local active and the requested active+kernel model",fontsize=12)
    _save(fig,output,"wall_comparison.png")
    return fig


def plot_wall_stability(item,output=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    a,b=item["reference"],item["perturbed"];report=item["report"];spec=item["spec"]
    p=a.metadata["parameters"];n=p["n"];length=n*p["dx"];grid=np.arange(n)*p["dx"];time=a.time
    ao,bo=report["reference"],report["perturbed"]
    fig,axes=plt.subplots(3,3,figsize=(15,12))
    limit=max(float(abs(a.fields).max()),float(abs(b.fields).max()))
    for ax,field,title in ((axes[0,0],a.fields[0],"Initial flat wall + periodic companion"),
                           (axes[0,1],a.fields[-1],"Final reference"),(axes[0,2],b.fields[-1],"Final perturbed wall, same noise")):
        image=ax.imshow(field,origin="lower",extent=[0,length]*2,cmap="RdBu_r",vmin=-limit,vmax=limit)
        if field.min()<report["threshold"]<field.max():
            ax.contour(grid,grid,field,levels=[report["threshold"]],colors="black",linewidths=.6)
        fig.colorbar(image,ax=ax,label="phi")
        ax.set(xlabel="x",ylabel="y",title=title)
    graph_valid=np.flatnonzero(bo["graph_fraction"]>=spec.graph_fraction)
    snapshots=(graph_valid[np.unique([0,len(graph_valid)//2,len(graph_valid)-1])]
               if len(graph_valid) else [])
    for index in snapshots:
        h=bo["height"][index,0];valid=np.isfinite(h)
        if valid.any() and bo["graph_fraction"][index]>=spec.graph_fraction:
            axes[1,0].plot(grid,h-np.mean(h[valid]),label=f"central, t={time[index]:g}")
    axes[1,0].set(xlabel="Along-wall coordinate y",ylabel="h(y) - mean(h)",title="Perturbed central-wall shape, drift removed")
    axes[1,0].legend(fontsize=8)
    delta=report["delta_coefficients"]
    modes=list(range(1,min(7,delta.shape[-1])))
    if spec.mode not in modes:
        modes.append(spec.mode)
    for mode in modes:
        axes[1,1].plot(time,2*abs(delta[:,0,mode]),lw=1.7 if mode==spec.mode else .8,label=f"central m={mode}")
    axes[1,1].plot(time,report["amplitude"][:,1],"k--",lw=1,label=f"companion m={spec.mode}")
    axes[1,1].set(xlabel="Time",ylabel="Paired interface Fourier amplitude",title="Shape modes of both interfaces")
    axes[1,1].legend(fontsize=7,ncol=2)
    power=abs(delta[:,0,1:])**2
    image=axes[1,2].pcolormesh(np.arange(1,power.shape[1]+1),time,np.maximum(power,1e-14),
                              shading="auto",norm=LogNorm(),cmap="magma")
    fig.colorbar(image,ax=axes[1,2],label="|delta h_m|^2")
    axes[1,2].set(xlabel="Along-wall Fourier mode m",ylabel="Time",title="Perturbation spectrum (m=0 excluded)")
    for wall,label in ((0,"central"),(1,"periodic companion")):
        axes[2,0].plot(time,ao["mean_position"][:,wall]-ao["mean_position"][0,wall],label=label)
    axes[2,0].set(xlabel="Time",ylabel="Unwrapped mean displacement",title="Uniform wall translation recorded separately")
    axes[2,0].legend(fontsize=8)
    for obs,label in ((ao,"reference"),(bo,"perturbed")):
        axes[2,1].plot(time,obs["width_mean"],label=label)
        axes[2,2].plot(time,obs["graph_fraction"],label=label)
    axes[2,1].set(xlabel="Time",ylabel="Mean positive-phase width",title="Coupled-wall thickness relaxation")
    axes[2,1].legend(fontsize=8)
    axes[2,2].axhline(spec.graph_fraction,color="gray",ls=":",label="fit validity threshold")
    axes[2,2].set(xlabel="Time",ylabel="Fraction of rows with exactly two crossings",ylim=(-.02,1.03),
                  title="Overhang / extra domains stop the linear fit")
    axes[2,2].legend(fontsize=8)
    fit=report["fit"]
    suffix=f"; rate={fit['rate']:.4g} +/- {fit['ci95']:.2g}" if fit["rate"] is not None else ""
    graph_note=(f"final graph fraction: reference={ao['graph_fraction'][-1]:.2f}, "
                f"perturbed={bo['graph_fraction'][-1]:.2f}")
    fig.suptitle(f"{item['name']}: early m={spec.mode} response: {fit['interpretation']}{suffix}\n"
                 f"{graph_note}; periodic {n}x{n}; Euler-Maruyama / midpoint EP",fontsize=12)
    _save(fig,output,"wall_stability.png")
    return fig
