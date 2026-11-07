"""Finite-time circular-cluster survival and small shape-perturbation experiment."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label, map_coordinates
from scipy.optimize import brentq
from scipy.stats import linregress, t as student_t

from .model import ActiveModelB, Parameters, Run, simulate


@dataclass(frozen=True)
class CircleSpec:
    radius: float = 10.0
    width: float = 2**.5
    phi_inside: float = 1.0
    phi_outside: float = -1.0
    perturbation: float = 0.05
    mode: int = 2
    angles: int = 128
    min_component_sites: int = 8

    @classmethod
    def from_bulk(cls,p: Parameters,*,width=None,**kwargs):
        """Local quartic minima +/-sqrt(-a/b); passive planar tanh width.

        These initialize the field; they are not assumed to be the binodals of
        a curved, noisy, active model with nonlocal interactions.
        """
        if p.a>=0:
            raise ValueError("Two local bulk minima require a < 0")
        if width is None:
            if p.kappa<=0:
                raise ValueError("Specify an interface width when kappa=0")
            width=np.sqrt(2*p.kappa/(-p.a))
        phi0=float(np.sqrt(-p.a/p.b))
        return cls(phi_inside=phi0,phi_outside=-phi0,width=float(width),**kwargs)


def periodic_offsets(n, dx, centre=None):
    length = n*dx
    cx, cy = (length/2, length/2) if centre is None else centre
    x = (np.arange(n)*dx-cx+length/2) % length-length/2
    y = (np.arange(n)*dx-cy+length/2) % length-length/2
    return np.meshgrid(x, y)


def circle_initial(p: Parameters, spec: CircleSpec, perturbed=False, mean=None, centre=None,
                   *, return_info=False):
    """Smooth droplet with optional cos(m theta) radius perturbation.

    Setting mean adjusts the base radius to conserve the reference mass.
    Plateau densities are never shifted to match a requested mean.
    """
    if not (0 < spec.radius < p.n*p.dx/3 and spec.width > 0 and
            spec.phi_inside > spec.phi_outside and 2 <= spec.mode < spec.angles/2 and
            0 < spec.perturbation < .2 and spec.angles >= 16):
        raise ValueError("Use a resolved droplet, positive width/contrast and a small m>=2 perturbation")
    x, y = periodic_offsets(p.n, p.dx, centre)
    distance=np.hypot(x,y)
    angular=1+spec.perturbation*np.cos(spec.mode*np.arctan2(y,x)) if perturbed else 1.
    def profile(base_radius):
        fraction=(1+np.tanh((base_radius*angular-distance)/spec.width))/2
        return spec.phi_outside+(spec.phi_inside-spec.phi_outside)*fraction
    radius=spec.radius
    if mean is not None:
        if not np.isfinite(mean) or not spec.phi_outside<mean<spec.phi_inside:
            raise ValueError("Mean must lie strictly between the two plateau densities")
        lo,hi=0.,np.nextafter(p.n*p.dx/3,0.)
        target=lambda r: float(profile(r).mean()-mean)
        if target(lo)*target(hi)>0:
            raise ValueError("Requested mean cannot be reached by a resolved droplet with fixed plateau densities")
        radius=float(brentq(target,lo,hi,xtol=1e-12))
    phi=profile(radius)
    if return_info:
        return phi,dict(base_radius=float(radius),nominal_radius=spec.radius,
            phi_inside=spec.phi_inside,phi_outside=spec.phi_outside,
            actual_min=float(phi.min()),actual_max=float(phi.max()),mean=float(phi.mean()),
            mass_matching="radius adjustment, no uniform field shift",perturbed=bool(perturbed))
    return phi


def periodic_components(mask):
    """Four-neighbour connected components with boundary labels merged on a torus."""
    labels, count = label(mask)
    if count == 0:
        return labels, np.array([mask.size])
    parent = np.arange(count+1)
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in np.concatenate((np.column_stack((labels[:,0], labels[:,-1])),
                               np.column_stack((labels[0], labels[-1])))):
        if a and b:
            parent[root(int(b))] = root(int(a))
    lookup = np.array([root(i) for i in range(count+1)])
    merged = lookup[labels]
    return merged, np.bincount(merged.ravel(), minlength=count+1)


def _centre(mask, dx):
    n = len(mask)
    phase = np.exp(2j*np.pi*np.arange(n)/n)
    zx = np.dot(mask.sum(axis=0), phase)
    zy = np.dot(mask.sum(axis=1), phase)
    scale = max(mask.sum(), 1)
    valid = min(abs(zx), abs(zy))/scale > .1
    return (np.angle(zx) % (2*np.pi)*n*dx/(2*np.pi),
            np.angle(zy) % (2*np.pi)*n*dx/(2*np.pi)), valid


def cluster_observables(fields, dx, threshold, *, angles=128, modes=6, min_sites=8):
    """Track largest component and outward threshold crossings about its centre.

    A centre outside the dense phase, missing crossings, or a torus-spanning
    component yields NaN shape coefficients; breakup/dissolution remain visible.
    """
    n = fields.shape[-1]
    theta = 2*np.pi*np.arange(angles)/angles
    radii = np.arange(0, n*dx/2-dx, dx/4)
    out = {k: [] for k in ("area", "radius", "components", "largest_fraction", "centre",
                           "shape", "interface_radius", "coverage", "radial_profile")}
    for phi in fields:
        labels, sizes = periodic_components(phi > threshold)
        positive = sizes.copy()
        positive[0] = 0
        largest = int(positive.argmax())
        area = positive[largest]*dx**2
        component = (labels == largest) if area else np.zeros_like(phi, dtype=bool)
        centre, centre_valid = _centre(component, dx)
        if not area:
            centre = (n*dx/2, n*dx/2)
        coords = [(centre[1]+np.sin(theta)[:,None]*radii)/dx,
                  (centre[0]+np.cos(theta)[:,None]*radii)/dx]
        ray = map_coordinates(phi, coords, order=1, mode="grid-wrap")
        inside = ray[:,0] > threshold
        below = ray <= threshold
        crosses = inside & below.any(axis=1)
        first = np.argmax(below, axis=1)
        boundary = np.full(angles, np.nan)
        good = np.flatnonzero(crosses & (first > 0))
        j = first[good]
        left, right = ray[good,j-1], ray[good,j]
        boundary[good] = radii[j-1]+(left-threshold)/(left-right)*(radii[j]-radii[j-1])
        coverage = len(good)/angles
        coefficient = np.full(modes+1, complex(np.nan,np.nan))
        if centre_valid and coverage == 1:
            coefficient = np.fft.rfft(boundary)/angles
            coefficient = coefficient[:modes+1]
        out["area"].append(area)
        out["radius"].append(np.sqrt(area/np.pi))
        out["components"].append(np.count_nonzero(positive >= min_sites))
        out["largest_fraction"].append(area/max(positive.sum()*dx**2, dx**2))
        out["centre"].append(centre if area else (np.nan,np.nan))
        out["shape"].append(coefficient)
        out["interface_radius"].append(boundary)
        out["coverage"].append(coverage)
        out["radial_profile"].append(ray.mean(axis=0))
    out = {k: np.asarray(v) for k,v in out.items()}
    mean_radius=np.real(out["shape"][:,0])
    out["mean_interface_radius"]=mean_radius
    out["relative_modes"]=np.divide(2*abs(out["shape"]),mean_radius[:,None],
        out=np.full(out["shape"].shape,np.nan),where=mean_radius[:,None]>0)
    out["relative_roughness"]=np.sqrt(np.mean(
        (out["interface_radius"]-mean_radius[:,None])**2,axis=1))/np.where(mean_radius>0,mean_radius,np.nan)
    out["radial_coordinate"] = radii
    return out


def run_circle_pair(p, kernel, spec=None, *, steps=16000, frame_every=200,
                    sample_every=40, blocks=8, multiplier=None, progress=True,
                    backend="numpy",device="auto",dtype="float64",cpu_threads=1):
    """Same-noise, equal-mass reference/perturbed droplets, using Euler-Maruyama."""
    spec = spec or CircleSpec.from_bulk(p)
    reference,reference_info = circle_initial(p, spec,return_info=True)
    perturbed,perturbed_info = circle_initial(p, spec, perturbed=True, mean=reference.mean(),return_info=True)
    cp = replace(p, mean=float(reference.mean()))
    kwargs = dict(burn_steps=0, production_steps=steps, frame_every=frame_every,
                  sample_every=sample_every, blocks=blocks, progress=progress,
                  backend=backend,device=device,dtype=dtype,cpu_threads=cpu_threads)
    a = simulate(ActiveModelB(cp, kernel, multiplier=multiplier), initial=reference, **kwargs)
    b = simulate(ActiveModelB(cp, kernel, multiplier=multiplier), initial=perturbed, **kwargs)
    for run,info in ((a,reference_info),(b,perturbed_info)):
        run.metadata["circle"] = asdict(spec)
        run.metadata["paired_noise"] = True
        run.metadata["circle_initialization"] = info
    return a,b


def circle_diagnostics(reference: Run, perturbed: Run, spec=None, fit_until=40.0):
    spec = spec or CircleSpec(**reference.metadata["circle"])
    dx = reference.metadata["parameters"]["dx"]
    if not np.array_equal(reference.time, perturbed.time):
        raise ValueError("Paired droplets must have identical observation times")
    threshold = (spec.phi_inside+spec.phi_outside)/2
    kw = dict(angles=spec.angles, modes=max(6,spec.mode), min_sites=spec.min_component_sites)
    a = cluster_observables(reference.fields, dx, threshold, **kw)
    b = cluster_observables(perturbed.fields, dx, threshold, **kw)
    delta = 2*(b["shape"]-a["shape"])
    amplitude = np.abs(delta[:,spec.mode])
    ar=a["mean_interface_radius"][:,None];br=b["mean_interface_radius"][:,None]
    relative_delta=2*(np.divide(b["shape"],br,out=np.full_like(b["shape"],complex(np.nan,np.nan)),where=br>0)
                     -np.divide(a["shape"],ar,out=np.full_like(a["shape"],complex(np.nan,np.nan)),where=ar>0))
    relative_amplitude=abs(relative_delta[:,spec.mode])
    # Compare each interface in its own centre frame, removing m=1 translation.
    gain = amplitude/amplitude[0] if amplitude[0] > 0 else np.full_like(amplitude,np.nan)
    field_norm = np.sqrt(np.mean((perturbed.fields.astype(float)-reference.fields)**2,axis=(1,2)))
    shape_valid = (a["components"] == 1) & (b["components"] == 1)
    # Stop the fit at the first lost interface, split, or large deformation.
    in_linear_range = shape_valid & np.isfinite(relative_amplitude) & (relative_amplitude>0) & (
        relative_amplitude<.15) & (reference.time<=fit_until) & (np.minimum(ar[:,0],br[:,0])>4*dx)
    bad = np.flatnonzero(~in_linear_range)
    end = int(bad[0]) if len(bad) else len(amplitude)
    def shape_fit(values,observable):
        fit=dict(rate=None,ci95=None,points=end,observable=observable,
                 interpretation="insufficient single-droplet small-perturbation window")
        if end>=6:
            regression=linregress(reference.time[:end],np.log(values[:end]))
            ci=student_t.ppf(.975,end-2)*regression.stderr
            fit.update(rate=float(regression.slope),ci95=float(ci),
                interpretation=("finite-time amplification" if regression.slope-ci>0 else
                                "finite-time decay" if regression.slope+ci<0 else "unresolved growth sign"))
        return fit
    fit=shape_fit(relative_amplitude,"paired interface mode normalized by each instantaneous mean radius")
    absolute_fit=shape_fit(amplitude,"paired absolute interface displacement")
    survival = []
    for run,obs in ((reference,a),(perturbed,b)):
        tail = slice(max(1,len(run.time)*3//4),None)
        ratio = float(np.median(obs["area"][tail])/max(obs["area"][0],dx**2))
        split = bool(np.any(obs["components"][tail] > 1))
        roughness=obs["relative_roughness"][tail]
        roughness=roughness[np.isfinite(roughness)]
        median_roughness=float(np.median(roughness)) if len(roughness) else None
        survival.append(dict(final_area_ratio=ratio,
            final_components=int(obs["components"][-1]), breakup_in_last_quarter=split,
            median_tail_relative_roughness=median_roughness,
            circularity=("no resolved interface" if median_roughness is None else
                         "approximately circular" if median_roughness<.1 and not split else "deformed or fragmented"),
            interpretation=("dissolved" if ratio < .1 else "fragmented" if split else
                            "radial growth" if ratio > 1.2 else "radial shrinkage" if ratio < .8
                            else "single cluster survives this window")))
    return dict(reference=a,perturbed=b,amplitude=amplitude,gain=gain,
                relative_amplitude=relative_amplitude,relative_modes=abs(relative_delta),
                relative_gain=relative_amplitude/relative_amplitude[0],
                field_norm=field_norm,fit=fit,absolute_fit=absolute_fit,survival=survival,threshold=threshold)


def save_circle_report(diagnostics, path):
    Path(path).write_text(json.dumps({"fit":diagnostics["fit"],"absolute_fit":diagnostics["absolute_fit"],
        "survival":diagnostics["survival"]},indent=2),encoding="utf-8")
