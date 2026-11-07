"""Bulk-initialized circle pairs, two radii, and named local-model controls."""
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .model import KernelSpec,Run
from .backends import backend_options
from .circle import CircleSpec,run_circle_pair,circle_diagnostics,save_circle_report


def run_circle_suite(p,kernel,*,output,control_output,radii=(10.,16.),steps=16000,
                     frame_every=200,sample_every=40,blocks=8,fit_until=40.,
                     multiplier=None,simulation=None,reuse_cache=True,include_controls=True):
    """Keep bulk plateaus fixed in all cases; change radius to set each mean.

    Controls share cached trajectories across the three kernel notebooks. They
    are explicitly separate models; the requested active+kernel model is intact.
    """
    opts=backend_options(**(simulation or dict(backend="auto")))
    source_hash=sha256(b"".join((Path(__file__).parent/f).read_bytes()
        for f in ("circle.py","model.py","backends.py","torch_backend.py"))).hexdigest()
    cases=[("local passive",replace(p,lam=0.),KernelSpec("poisson",strength=0.),None,"passive"),
           ("local active",p,KernelSpec("poisson",strength=0.),None,"active"),
           (f"active + {kernel.kind}",p,kernel,multiplier,None)]
    if not include_controls:
        cases=[case for case in cases if case[-1] is None]
    suite=[]
    for radius in radii:
        spec=CircleSpec.from_bulk(p,radius=float(radius),perturbation=.05,mode=2)
        for name,parameters,spec_kernel,custom,control in cases:
            folder=(Path(control_output)/control if control else Path(output))/f"radius_{radius:g}"
            folder.mkdir(parents=True,exist_ok=True)
            from .model import ActiveModelB
            model=ActiveModelB(parameters,spec_kernel,multiplier=custom)
            signature=sha256(json.dumps(dict(parameters=asdict(parameters),kernel=asdict(spec_kernel),
                circle=asdict(spec),steps=steps,frame_every=frame_every,sample_every=sample_every,
                blocks=blocks,simulation=opts,source_hash=source_hash,
                multiplier_hash=sha256(model.K.tobytes()).hexdigest()),sort_keys=True).encode()).hexdigest()
            a_path,b_path=folder/"reference.npz",folder/"perturbed.npz"
            a=Run.load(a_path) if reuse_cache and a_path.exists() else None
            b=Run.load(b_path) if reuse_cache and b_path.exists() else None
            if a is None or b is None or any(r.metadata.get("circle_signature")!=signature for r in (a,b)):
                print(f"Circle R={radius:g}, {name}, initial plateaus "
                      f"{spec.phi_inside:g}/{spec.phi_outside:g}: integrating paired runs",flush=True)
                a,b=run_circle_pair(parameters,spec_kernel,spec,steps=steps,frame_every=frame_every,
                    sample_every=sample_every,blocks=blocks,multiplier=custom,progress=False,**opts)
                for run,path in ((a,a_path),(b,b_path)):
                    run.metadata["circle_signature"]=signature
                    run.metadata["circle_case"]=name
                    run.save(path)
            else:
                print(f"Circle R={radius:g}, {name}: matching cache",flush=True)
            report=circle_diagnostics(a,b,spec,fit_until=fit_until)
            save_circle_report(report,folder/"stability.json")
            suite.append(dict(name=name,control=control,radius=float(radius),spec=spec,
                              reference=a,perturbed=b,report=report,folder=folder))
    Path(output).mkdir(parents=True,exist_ok=True)
    (Path(output)/"initialization_and_stability.json").write_text(json.dumps([
        dict(case=r["name"],radius=r["radius"],parameters=r["reference"].metadata["parameters"],
             kernel=r["reference"].metadata["kernel"],circle=asdict(r["spec"]),
             reference_initial=r["reference"].metadata["circle_initialization"],
             perturbed_initial=r["perturbed"].metadata["circle_initialization"],
             relative_shape_fit=r["report"]["fit"],survival=r["report"]["survival"])
        for r in suite],indent=2),encoding="utf-8")
    return suite


def plot_circle_initialization(suite,p,output=None):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    phi0=np.sqrt(-p.a/p.b)
    phi=np.linspace(-1.5*phi0,1.5*phi0,400)
    axes[0].plot(phi,p.a*phi**2/2+p.b*phi**4/4)
    axes[0].plot([-phi0,phi0],[-p.a**2/(4*p.b)]*2,"o",label="local quartic minima")
    axes[0].set(xlabel="phi",ylabel="f_bulk(phi)",title=f"Local minima at +/-sqrt(-a/b) = +/-{phi0:g}")
    axes[0].legend(fontsize=8)
    for item in suite:
        if item["control"] is not None:
            continue
        obs=item["report"]["reference"]
        axes[1].plot(obs["radial_coordinate"],obs["radial_profile"][0],label=f"R0={item['radius']:g}")
    for value in (-phi0,phi0):
        axes[1].axhline(value,color="gray",ls=":",lw=.8)
    axes[1].set(xlabel="Distance from cluster centre",ylabel="Initial azimuthal mean phi",
                title="Fixed plateau densities; mass matched by radius adjustment")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    if output is not None:
        fig.savefig(Path(output)/"circle_initialization.png",dpi=160,bbox_inches="tight")
    return fig


def plot_circle_comparison(suite,output=None):
    import matplotlib.pyplot as plt
    radii=sorted(set(r["radius"] for r in suite))
    fig,axes=plt.subplots(len(radii),3,figsize=(15,4.2*len(radii)),squeeze=False)
    for row,radius in enumerate(radii):
        for r in suite:
            if r["radius"]!=radius:
                continue
            t=r["reference"].time;obs=r["report"]["reference"];label=r["name"]
            axes[row,0].plot(t,obs["radius"],label=label)
            axes[row,1].plot(t,obs["relative_roughness"],label=label)
            axes[row,2].semilogy(t,r["report"]["relative_amplitude"],label=label)
        axes[row,0].set(xlabel="Time",ylabel="Area radius",title=f"R0={radius:g}: radial relaxation / survival")
        axes[row,1].set(xlabel="Time",ylabel="RMS angular radius fluctuation / mean radius",
                        title=f"R0={radius:g}: reference circularity")
        axes[row,2].set(xlabel="Time",ylabel="Paired m=2 response / instantaneous radii",
                        title=f"R0={radius:g}: dimensionless shape perturbation")
        for ax in axes[row]:
            ax.legend(fontsize=8)
    fig.suptitle("Bulk minima initialize all cases; common noise, equal-mass reference/perturbed pairs\n"
                 "Radial shrinkage and angular deformation are diagnosed separately",fontsize=12)
    fig.tight_layout()
    if output is not None:
        fig.savefig(Path(output)/"circle_comparison.png",dpi=160,bbox_inches="tight")
    return fig
