"""Device-resident Euler-Maruyama with rFFT and full-step midpoint entropy.

Only saved arrays are copied back to NumPy. PyTorch is an optional dependency;
this module is imported lazily by simulate(backend='torch').
"""
from contextlib import contextmanager
from dataclasses import asdict
import time

import numpy as np
import torch

from .model import Run
from .decomposition import DecompositionResult


@torch.jit.script
def _neighbours(phi: torch.Tensor,dx: float):
    left=torch.roll(phi,1,0);right=torch.roll(phi,-1,0)
    down=torch.roll(phi,1,1);up=torch.roll(phi,-1,1)
    lap=(left+right+down+up-4*phi)/(dx*dx)
    grad=((left-phi).square()+(right-phi).square()+
          (down-phi).square()+(up-phi).square())/(2*dx*dx)
    return lap,grad


@torch.jit.script
def _step(phi: torch.Tensor,ph: torch.Tensor,normal: torch.Tensor,
          linear: torch.Tensor,q2: torch.Tensor,noise_scale: torch.Tensor,
          dt_mobility: float,b: float,lam: float,dx: float):
    _,grad=_neighbours(phi,dx)
    nonlinear=b*phi.pow(3)+lam*grad
    drift=linear*ph+torch.fft.rfft2(nonlinear,norm="ortho")
    noise=noise_scale*torch.fft.rfft2(normal,norm="ortho")
    new_hat=ph-dt_mobility*q2*drift+noise
    new_hat[0,0]=ph[0,0]
    new=torch.fft.irfft2(new_hat,s=[phi.size(0),phi.size(1)],norm="ortho")
    return new,new_hat


@torch.jit.script
def _midpoint(old: torch.Tensor,new: torch.Tensor,ph: torch.Tensor,new_hat: torch.Tensor,
              kernel: torch.Tensor,a: float,b: float,kappa: float,lam: float,
              strength: float,dx: float,D: float):
    mid=(old+new)/2
    lap,grad=_neighbours(mid,dx)
    mu_local=a*mid+b*mid.pow(3)-kappa*lap
    mu_kernel=strength*torch.fft.irfft2(kernel*(ph+new_hat)/2,
                                     s=[old.size(0),old.size(1)],norm="ortho")
    mu=torch.stack([mu_local,mu_kernel,lam*grad])
    ds=-mu.to(torch.float64)*(new-old).to(torch.float64)/D
    return ds,mu


@contextmanager
def _cpu_threads(count,device):
    previous=torch.get_num_threads()
    change=device.type=="cpu" and count is not None
    if count is not None and (not isinstance(count,int) or count<1):
        raise ValueError("cpu_threads must be a positive integer or None")
    if change:
        torch.set_num_threads(count)
    try:
        yield
    finally:
        if change:
            torch.set_num_threads(previous)


def _numpy(value):
    return value.detach().cpu().numpy().copy()


def _full_power(half,n):
    """Restore real-field inversion symmetry along the missing Fourier columns."""
    partner=(-np.arange(n))%n
    stop=-1 if n%2==0 else None
    return np.concatenate([half,half[partner,1:stop][:,::-1]],axis=1)


class _DeviceDecomposition:
    def __init__(self,observer,device,dtype,steps,sample,blocks):
        self.observer=observer;self.steps=steps;self.sample=sample;self.blocks=blocks
        self.m=observer.model; self.p=self.m.p
        self.nphi=observer.nphi;self.nshell=observer.nshell
        nh=self.p.n//2+1
        self.multipliers=torch.as_tensor(observer.multipliers[:,:,:nh],dtype=dtype,device=device)
        self.phi_edges=torch.as_tensor(observer.phi_edges,dtype=dtype,device=device)
        self.offset=torch.arange(self.nshell,device=device)[:,None]*self.nphi
        kw=dict(dtype=torch.float64,device=device)
        self.lr=torch.zeros((steps//sample,self.nphi,self.nshell),**kw)
        self.local=torch.zeros((steps//sample,self.nphi,2),**kw)
        self.occupancy=torch.zeros((steps//sample,self.nphi),dtype=torch.int64,device=device)
        self.mu_lr=torch.zeros((self.nphi,self.nshell),**kw)
        self.mu_local=torch.zeros((self.nphi,2),**kw)
        self.tail=torch.zeros((),dtype=torch.int64,device=device)
        self.residual=torch.zeros((),**kw)
        observer.result=None

    def update(self,index,old,new,ph,new_hat,ds,mu):
        mid=(old+new)/2;delta=(new-old).to(torch.float64)
        labels=(torch.bucketize(mid.flatten().contiguous(),self.phi_edges,right=True)-1).clamp(0,self.nphi-1)
        self.tail+=((mid<self.phi_edges[0])|(mid>=self.phi_edges[-1])).sum()
        joint=(self.offset+labels).flatten()
        mu_r=self.m.kernel.strength*torch.fft.irfft2(self.multipliers*(ph+new_hat)[None,:,:]/2,
                                                    s=(self.p.n,self.p.n),norm="ortho")
        self.residual=torch.maximum(self.residual,(mu_r.sum(0)-mu[1]).abs().max().double())
        def bins(values):
            out=torch.zeros(self.nshell*self.nphi,dtype=torch.float64,device=values.device)
            out.scatter_add_(0,joint,values.double().flatten())
            return out.reshape(self.nshell,self.nphi).T
        slot=index//self.sample
        self.lr[slot]+=self.m.dV*bins(-mu_r.double()*delta/self.p.D)
        self.mu_lr+=bins(mu_r)
        self.occupancy[slot].scatter_add_(0,labels,torch.ones_like(labels))
        for j,channel in enumerate((0,2)):
            self.local[slot,:,j].scatter_add_(0,labels,self.m.dV*ds[channel].flatten())
            self.mu_local[:,j].scatter_add_(0,labels,mu[channel].double().flatten())

    def finish(self):
        count=self.occupancy.sum(0)
        interval=self.p.dt*self.sample
        mu_lr=torch.where(count[:,None]>0,self.mu_lr/count[:,None],float("nan"))
        mu_local=torch.where(count[:,None]>0,self.mu_local/count[:,None],float("nan"))
        self.observer.result=DecompositionResult(
            metadata=dict(dt=self.p.dt,dV=self.m.dV,area=self.m.area,interval=interval,
                blocks=self.blocks,steps=self.steps,distance="periodic source displacement |x-y|",
                phi="receiving midpoint field phi_mid(x)",local_channels=["local conservative","local active"],
                max_mu_residual=float(self.residual.cpu()),
                phi_tail_fraction=float(self.tail.cpu())/(self.steps*self.p.n**2),backend="torch"),
            phi_edges=self.observer.phi_edges.copy(),distance_edges=self.observer.distance_edges.copy(),
            time=(np.arange(len(self.lr))+.5)*interval,rate_lr=_numpy(self.lr)/interval,
            rate_local=_numpy(self.local)/interval,occupancy=_numpy(self.occupancy),
            mu_lr=_numpy(mu_lr),mu_local=_numpy(mu_local))


def simulate_torch(model,*,burn_steps=40000,production_steps=64000,sample_every=40,
                   frame_every=200,blocks=8,initial=None,progress=True,decomposition=None,
                   device="cpu",dtype="float64",cpu_threads=1):
    """Same Run/observer interface as NumPy; tensors stay resident until collection.

    Dynamics may use float32/64; all entropy and energy accumulators use float64.
    The default float64 is recommended for cancellations between signed shells.
    CPU uses one thread by default for small FFTs, restoring the previous value.
    NumPy and torch RNG streams differ even with the same seed.
    """
    if (burn_steps<0 or production_steps<=0 or sample_every<=0 or frame_every<=0 or blocks<2
            or production_steps%(sample_every*blocks) or burn_steps%frame_every
            or production_steps%frame_every):
        raise ValueError("Use aligned burn/production, sample/frame intervals and blocks >= 2")
    p=model.p
    if decomposition is not None and decomposition.model is not model:
        raise ValueError("The decomposition observer must belong to the simulated model")
    initial=model.initial_field() if initial is None else np.array(initial,dtype=float,copy=True)
    if initial.shape!=(p.n,p.n) or not np.isfinite(initial).all():
        raise ValueError("initial must be a finite (n,n) field")
    device=torch.device(device)
    real_dtype={"float64":torch.float64,"float32":torch.float32}[dtype]
    started=time.perf_counter()
    with _cpu_threads(cpu_threads,device),torch.inference_mode():
        def tensor(a,dt=None):
            return torch.as_tensor(a,dtype=dt or real_dtype,device=device)
        nh=p.n//2+1
        q2=tensor(model.q2[:,:nh]);kernel=tensor(model.K[:,:nh])
        linear=tensor(model.linear_mu[:,:nh]);noise_scale=tensor(model.noise_scale[:,:nh])
        multiplicity=torch.full((nh,),2.,dtype=torch.float64,device=device)
        multiplicity[0]=1
        if p.n%2==0:
            multiplicity[-1]=1
        qlabels=tensor(model.q_labels.reshape(p.n,p.n)[:,:nh].flatten(),torch.int64)
        qcount=tensor(model.q_count,torch.float64)
        rm=tensor(model.r_multipliers[:,:,:nh],torch.float64).reshape(len(model.r_centres),-1)
        rng=torch.Generator(device=device).manual_seed(p.seed)
        phi=tensor(initial);ph=torch.fft.rfft2(phi,norm="ortho")
        target_mean=float(initial.mean())
        records={k:[] for k in ("fields","energy","energy_q","energy_r","structure","variance","mass")}
        times=[];rates=[];prod_q=[];prod_r=[]
        maps=torch.zeros((3,p.n,p.n),dtype=torch.float64,device=device)
        block_maps=torch.zeros((blocks,3,p.n,p.n),dtype=torch.float64,device=device)
        structure_2d=torch.zeros((p.n,nh),dtype=torch.float64,device=device)
        interval=torch.zeros(3,dtype=torch.float64,device=device)
        max_balance=torch.zeros((),dtype=torch.float64,device=device)
        def energy_channels():
            power=ph.abs().double().square()
            weighted=power*multiplicity
            local=phi.double()
            bulk=model.dV*(p.a/2*local.square()+p.b/4*local.pow(4)).sum()
            gradient=model.dV*p.kappa/2*(q2.double()*weighted).sum()
            lr_modes=model.dV/2*model.kernel.strength*kernel.double()*weighted
            byq=torch.zeros(len(model.q_centres),dtype=torch.float64,device=device)
            byq.scatter_add_(0,qlabels,lr_modes.flatten())
            byr=model.dV/2*model.kernel.strength*(rm@weighted.flatten())
            return torch.stack([bulk,gradient,lr_modes.sum()]),byq,byr
        def record(step):
            energy,eq,er=energy_channels()
            power=model.dV*ph.abs().double().square();power[0,0]=0
            structure=torch.zeros_like(qcount)
            structure.scatter_add_(0,qlabels,(power*multiplicity).flatten())
            structure=torch.where(qcount>0,structure/qcount,0.)
            values=dict(fields=phi.float().clone(),energy=energy,energy_q=eq,energy_r=er,
                        structure=structure,variance=phi.double().var(correction=0),mass=phi.double().mean())
            for k,v in values.items():
                records[k].append(v)
            times.append(step*p.dt)
        observer=None if decomposition is None else _DeviceDecomposition(
            decomposition,device,real_dtype,production_steps,sample_every,blocks)
        record(0)
        iterator=range(1,burn_steps+production_steps+1)
        if progress:
            from tqdm.auto import tqdm
            iterator=tqdm(iterator,desc=f"{model.kernel.kind} torch {device} {p.n}x{p.n}",unit="step")
        for step in iterator:
            normal=torch.randn((p.n,p.n),dtype=real_dtype,device=device,generator=rng)
            new,new_hat=_step(phi,ph,normal,linear,q2,noise_scale,p.dt*p.mobility,p.b,p.lam,p.dx)
            if step==burn_steps+1:
                previous_energy,eq,er=energy_channels()
                prod_q.append(eq);prod_r.append(er)
            if step>burn_steps:
                ds,mu=_midpoint(phi,new,ph,new_hat,kernel,p.a,p.b,p.kappa,p.lam,
                                 model.kernel.strength,p.dx,p.D)
                maps+=ds
                block=(step-burn_steps-1)*blocks//production_steps
                block_maps[block]+=ds
                interval+=model.dV*ds.sum(dim=(1,2))
                if observer is not None:
                    observer.update(step-burn_steps-1,phi,new,ph,new_hat,ds,mu)
            phi,ph=new,new_hat
            if step%frame_every==0:
                record(step)
            if step>burn_steps and (step-burn_steps)%sample_every==0:
                energy,eq,er=energy_channels()
                exact=-(energy.sum()-previous_energy.sum())/p.D
                max_balance=torch.maximum(max_balance,(interval[:2].sum()-exact).abs())
                rates.append(interval.clone()/(sample_every*p.dt))
                prod_q.append(eq);prod_r.append(er)
                power=model.dV*ph.abs().double().square();power[0,0]=0
                structure_2d+=power
                interval.zero_();previous_energy=energy
            # Infrequent synchronization also reports explicit-Euler blow-ups.
            if step%max(frame_every,1000)==0:
                if not bool(torch.isfinite(phi).all()) or float(phi.abs().max())>30:
                    raise FloatingPointError(f"Unstable trajectory at step {step}; reduce dt")
        if not bool(torch.isfinite(phi).all()):
            raise FloatingPointError("Nonfinite trajectory; reduce dt")
        mass_error=abs(float(phi.double().mean())-target_mean)
        if mass_error>(1e-10 if dtype=="float64" else 1e-5):
            raise FloatingPointError("Mass conservation was lost")
        if observer is not None:
            observer.finish()
        duration=production_steps*p.dt
        saved={k:_numpy(torch.stack(v)) for k,v in records.items()}
        run=Run(metadata=dict(parameters=asdict(p),kernel=asdict(model.kernel),burn_steps=burn_steps,
            production_steps=production_steps,sample_every=sample_every,frame_every=frame_every,blocks=blocks,
            initial_mean=target_mean,custom_multiplier=model.custom_multiplier,integrator="Euler-Maruyama",
            entropy_convention="midpoint for all terms",backend="torch",device=str(device),dtype=dtype,
            statistics_dtype="float64",torch_version=str(torch.__version__),cpu_threads=cpu_threads,
            random_stream="torch.Generator; same-seed NumPy trajectory is different"),
            time=np.asarray(times),**saved,
            prod_time=(np.arange(len(rates))+.5)*sample_every*p.dt,
            ep_rate=_numpy(torch.stack(rates)),ep_map=_numpy(maps)/duration,
            ep_map_blocks=_numpy(block_maps)/(duration/blocks),prod_energy_q=_numpy(torch.stack(prod_q)),
            prod_energy_r=_numpy(torch.stack(prod_r)),mean_structure_2d=_full_power(_numpy(structure_2d)/len(rates),p.n),
            balance_error=float(max_balance.cpu()))
        run.metadata["elapsed_seconds"]=time.perf_counter()-started
        return run
