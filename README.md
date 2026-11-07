# LRAMB

64 × 64 periodic **Active Model B** baseline with interchangeable reciprocal
long-range conservative interactions. Start with one of the executed notebooks:

| Notebook | Nonzero-mode multiplier | Default exponent |
| --- | --- | --- |
| [riesz.ipynb](riesz.ipynb) | `(ell*q)^(-alpha)` | `alpha=1` |
| [poisson.ipynb](poisson.ipynb) | `exp(-ell*q)` | Poisson semigroup |
| [linnik.ipynb](linnik.ipynb) | `1/(1+(ell*q)^alpha)` | `alpha=1.5` |

Each notebook contains the momentum/real-space kernel, a simulation, numerical
stationarity diagnostics, an animated density/profile movie, time averaged EP
maps, evolving structure factors, chemical-potential/EP decomposition by the
receiving field `phi(x)` and source distance `|x-y|`, plus circle and planar-wall stability.
Explanatory cells are in Korean.

For the space-summed EP fluctuation question, open
[linnik_ep.ipynb](linnik_ep.ipynb), [poisson_ep.ipynb](poisson_ep.ipynb), or
[riesz_ep.ipynb](riesz_ep.ipynb). These sum signed EP over all locations and phi
at fixed source distance rho **before** computing temporal variance/std/PSD.
Local conservative/active EP is assigned to rho=0, so summing rho recovers the
full medium EP. The original notebooks also have a matching section 7b.

**Interaction range means the kernel scale parameter `ell`.**
[poisson_ep.ipynb](poisson_ep.ipynb) sweeps `exp(-ell*q)` and
[linnik_ep.ipynb](linnik_ep.ipynb) sweeps `1/(1+(ell*q)^alpha)` with fixed alpha.
Both integrate **a fresh 64x64 trajectory at each ell**, using defaults
`ELL_VALUES=(1,2,4,8)`, fixed coupling `g=0.35`, identical initial fields and a
common noise seed. Landau coefficient `p.a=-1` remains fixed; the `a` in
`exp(-a*q)` is named `kernel.ell` to distinguish it from that coefficient.
Every member has 40,000 burn-in and 64,000 production steps at `dt=0.01`.
There is no additional ell-dependent kernel normalization or trajectory
reweighting. These notebooks default to the PyTorch backend on CPU/CUDA.

The range sweep compares rho-dependent variance, variance/rho squared,
fixed-rho variance versus ell, global medium/kernel EP variance, temporal
spectra, and each run's density correlation / structure factor. Signed global
rates are summed before taking variance, retaining all shell covariances.
The default is one common seed; shaded 95% intervals are time-block bootstrap
intervals. Change `p.seed` to repeat the sweep with another realization.
Final fields and structural drift reports are shown for every ell; equal
simulation times do not guarantee equal equilibration across kernels.

`results/<kernel>_ep/range_sweep/ell_<ell>/` contains the separately cached runs.
Solver source, parameters, initialization, backend and run protocol are in each
cache signature. The sweep folder holds `range_sweep.npz/json` and four PNGs:
`range_ep_fluctuations`, `range_kernel_and_correlations`,
`range_scaling_comparison`, and `range_states_and_drift`.

[riesz_ep.ipynb](riesz_ep.ipynb) keeps the fine dx-width rho analysis at one
kernel. Riesz's `(ell*q)^(-alpha)` only changes amplitude when ell varies, so
Riesz is excluded from physical range sweeps. Its cumulative source-distance
cutoff `j_(<=R)=j_local+sum_(rho<=R) j_kernel,rho` remains an auxiliary observable
on one trajectory. This cutoff R is distinct from the kernel scale ell.

The EP notebooks also plot `Var(j_rho)/rho^2` against the connected equal-time
density two-point function `C(rho)=<delta_phi(x) delta_phi(x+r)>`. C is a pair
average over each displacement shell, reconstructed from structure power at
every production EP endpoint. The plots show signed C, normalized log-log
shapes, and five-shell local slopes. A shared fitting window ends before C's
first sign change. EP slopes have time-block bootstrap intervals; C slope
uncertainty cannot be recovered from the saved time-averaged structure alone.
These are finite-window effective slopes, not established asymptotic exponents.
The rho-squared division approximates squared shell-size growth in 2D; actual
lattice displacement counts are exported alongside the correlation.
The shell EP is `-Delta F_kernel,rho/(D*sampling_interval)`, so its variance
contains time differences of four-point density correlations. Dividing by rho
squared leaves kernel weights and spatial/angular covariances in place. The
zero-mean periodic kernel can cross zero and create pronounced dips; a single
fit over these features is a descriptive summary, not a universal power law.

## PyTorch CPU / CUDA backend

Install the optional backend with `python -m pip install -e ".[torch]"`.
For CUDA, install the wheel matching your driver using the
[official PyTorch installer](https://pytorch.org/get-started/locally/).
The notebook parameter cell selects the implementation:

```python
SIMULATION = backend_options('auto', device='auto', dtype='float64')
# Force PyTorch even on CPU:
SIMULATION = backend_options('torch', device='auto', dtype='float64')
```

`auto` uses PyTorch when CUDA is available and NumPy otherwise. The main run and
full-step receiving-phi/source-distance observer use `SIMULATION`. Corrected
circle and wall experiments have separate `CIRCLE_SIMULATION` and
`WALL_SIMULATION` controls and default to PyTorch; install the torch extra
for those notebook cells. Backend/device/dtype
are included in notebook cache signatures.
The NumPy-compatible `Run` files work with every existing plot and EP analysis.

```python
from lramb import simulate
run = simulate(model, backend='torch', device='auto', dtype='float64',
               decomposition=observer, cpu_threads=1)
```

The torch loop uses real FFTs with orthonormal normalization, scripted step
functions, device-resident state and entropy accumulation, and infrequent host
synchronization. Array collection happens at the end. On CPU, small FFTs use
one thread by default and restore the previous thread setting afterwards.
The dynamics is still explicit Euler–Maruyama; all entropy uses midpoint.
Float64 remains the default because signed shell cancellations matter. An
optional float32 dynamics mode keeps float64 observable accumulators, but has
larger kernel-energy/midpoint residuals; inspect these before quantitative use.
NumPy and PyTorch use different random streams at the same seed. Paired circles
share a stream within their selected backend.

```powershell
# Save a new torch run separately from the cached notebook trajectory:
python scripts/run_simulation.py riesz --backend torch --device auto
# Include the additional full-step phi-conditioned decomposition:
python scripts/run_simulation.py riesz --backend torch --phi-decomposition
```

Space-summed rho EP needs only quadratic shell energies, so the command-line
runner omits the more expensive phi-conditioned observer unless requested.
Speed depends on hardware and grid size; CUDA was not available in the local
environment used to generate this revision.

## Run

From this directory, using Python 3.10 or newer:

```powershell
python -m pip install -e ".[notebook,torch]"
python -m jupyterlab
```

Open a notebook and **Run All**. Each default run integrates 40,000 burn-in steps
and 64,000 production steps with `dt=0.01`, using a fixed seed. Runtime depends on
the machine. Circle/perturbed-circle pairs at radii 10 and 16, plus named local
controls, run for 16,000 steps each. Flat/perturbed-wall pairs also run for
16,000 steps each. Circle and wall controls share their respective caches across notebooks.
Trajectories and standalone PNG/GIF outputs are saved in
`results/<kernel>/`. These generated files are ignored by Git. Embedded notebook
outputs remain available without opening those archives.

```powershell
# Execute and save every notebook, or just one:
python scripts/run_notebooks.py
python scripts/run_notebooks.py riesz
python scripts/run_notebooks.py linnik_ep poisson_ep riesz_ep

# Create the dedicated EP notebook sources, or refresh just baseline section 7b:
python scripts/make_ep_notebooks.py
python scripts/make_range_ep_notebooks.py
python scripts/refresh_rho_cells.py
python scripts/refresh_circle_cells.py
python scripts/plot_circle_summary.py
python scripts/refresh_wall_cells.py
python scripts/plot_ep_scaling.py

# After executing both physical range-sweep notebooks:
python scripts/plot_range_sweeps.py

# Recreate notebook source cells (clears their outputs):
python scripts/make_notebooks.py

# Check physical identities and entropy bookkeeping:
python -m unittest discover -s tests -v
```

Caches are reused only when parameters, initialization, run lengths, actual
multiplier array, and model source hash agree. `REUSE_CACHE=False` forces a fresh
trajectory. Changing the seed gives an independent noise realization.
Saved fields use `float32` for visualization; energy and entropy statistics are
accumulated in `float64`. Reconstructing entropy from sparse saved fields loses
the full integration-step quadrature used by the reported maps.

## Model and kernel convention

With mobility `M` and noise temperature `D`,

```text
dphi = M Laplacian(mu_C + mu_A) dt + sqrt(2 M D) div(dW)
mu_C = a phi + b phi^3 - kappa Laplacian(phi) + g K*phi
mu_A = lambda |grad(phi)|^2
F_C  = integral [a phi^2/2 + b phi^4/4 + kappa |grad(phi)|^2/2]
       + (g/2) integral phi (K*phi)
```

The same nearest-neighbour lattice symbol
`q² = 4/dx² [sin²(kx dx/2) + sin²(ky dx/2)]` is used for the Laplacian,
kernel multiplier and noise covariance. `|grad(phi)|²` is the site-centred
average of squared forward/backward bond gradients in both directions. The
noise covariance is `2 M D (-Laplacian)/dV`, with `dV=dx²`. FFTs are orthonormal.

The Fourier zero mode is fixed. All kernels use `Khat(0)=0`; the displayed
periodic real-space kernel consequently has a negative constant background and
zero spatial mean. The discrete kernel is the inverse transform of the **lattice**
multiplier, with finite box and lattice cutoffs. It is not an unregularized
continuum singular function sampled at the origin. Convolution weights include
the integration volume: `w=ifft2(Khat)` and `K_eff=w/dV`.

Here “Poisson” means the Poisson semigroup `exp(-ell*q)`. The inverse Poisson
operator `q^-2` can instead be supplied through the custom multiplier API.
Riesz uses `0<alpha<2` in two dimensions. Linnik uses `0<alpha<=2`; at `alpha=2`
it becomes the screened resolvent, so use `alpha<2` for an algebraic tail.
Positive `g` penalizes modes and negative `g` favors them.

```python
from lramb import Parameters, KernelSpec, ActiveModelB, simulate

p = Parameters(n=64, lam=0.8, D=0.05, seed=17)
kernel = KernelSpec("riesz", strength=0.35, ell=2, alpha=1)
model = ActiveModelB(p, kernel)

# Local AMB baseline: KernelSpec(strength=0)
# Passive baseline: Parameters(lam=0)

# A custom real reciprocal multiplier; arrays of shape (n,n) also work:
custom = ActiveModelB(p, kernel,
    multiplier=lambda qx, qy: 1/(1 + 4*qx**2 + qy**2))

run = simulate(model, burn_steps=40000, production_steps=64000)
run.save("results/my_run.npz")
```

For a custom kernel, the supplied array defines the shape and `kernel.strength`
still sets its overall coupling. Nonfinite, complex, or inversion-asymmetric
arrays are rejected because they do not define the reciprocal quadratic energy
used here. Anisotropy is allowed. The default initialization is random for Riesz
and a noisy stripe for Poisson/Linnik to reduce initial two-phase coarsening.

## Entropy and decomposition

The **field-level medium entropy** uses the chemical-potential convention,
in Stratonovich form:

```text
dS_medium = -(1/D) integral (mu_C + mu_A) circ dphi
          = -dF_C/D + dS_active
dS_active = -(1/D) integral lambda |grad(phi)|² circ dphi
```

The decomposition is applied to the **chemical potential and its convolution**:

```text
mu_C(x)       = mu_C,local(x) + sum_rho mu_K,rho(x)
mu_C,local(x) = a phi(x) + b phi(x)^3 - kappa Laplacian(phi)(x)
mu_K,rho(x)   = g sum_(y: |x-y| in shell rho) w_(x-y) phi(y)
ds_K,rho(x)   = -mu_K,rho(phi_mid)(x) Delta_phi(x) / D
```

`rho` is the periodic distance **from kernel centre x to source y**. The
conditioning variable is the receiving midpoint field `phi_mid(x)`. The active
gradient term remains a separate local channel. The full local EP is the sum of
local conservative and local active contributions. The on-site `rho=0` weight is
retained in the kernel sum, using the same zero-mode gauge as the original kernel.
There is no extra factor of one half in the chemical-potential EP contribution.

`KernelDecomposition` observes every production solver step and forms interval
rates `J(phi_bin,rho_bin)` by binning `dV * ds_K,rho(x)` on `phi_mid(x)`. Summing
all phi and source-distance bins reconstructs the full long-range EP rate.
The observer does not change the dynamics. Tail observations are included in
the outer phi bins, and their fraction is reported. Default bins use 32 phi
intervals and 12 source-distance shells, including a separate on-site shell.

The notebooks distinguish additive bin contributions from conditional means
per occupied site. Conditional means require occupancy weights to recover a
global average. Signed phi-resolved means can remain nonzero at steady state;
only their sum over receiving phi becomes the conservative shell boundary term.

The integrator is **Euler–Maruyama**: every drift term is evaluated at the
current field, followed by one additive conserved Gaussian increment.

```text
phi_hat_next = phi_hat - dt M q² mu_hat(phi) + sqrt(2 M D dt / dV) q xi_hat
phi_mid = (phi_next + phi) / 2
ds_medium(x) = -[mu_C(phi_mid)(x) + mu_A(phi_mid)(x)] [phi_next(x)-phi(x)] / D
```

All entropy channels use this same midpoint field at **every solver step**.
For the quadratic long-range interaction its spatial integral equals
`-Delta F_K/D`. For the quartic bulk the midpoint has a finite quadrature
residual: `ds_C_integrated + Delta F_C/D = b dV sum(phi_mid * Delta_phi³)/(4 D)`.
`run.balance_error` records the maximum absolute accumulated residual over a
sampling interval. It is not expected to be at roundoff level. The step size
must resolve explicit drift stability and the physical Stratonovich EP limit.

The maps show local conservative, long-range conservative, local active, and
total medium EP, plus block uncertainty. Spatial EP density depends on the
chosen divergence/boundary convention. This observable describes field-level
irreversibility; no fuel coordinate or system stochastic entropy is included.

`phi_distance_decomposition.npz` stores the complete interval-rate tensor,
phi occupancy, conditional chemical potentials and bin edges. Figures show:

- `kernel_phi_distance_mu.png`: conditional chemical potentials and EP by phi.
- `kernel_phi_distance_fluctuations.png`: mean/RMS of `J(phi,rho)`, marginal
  temporal spectra and the full local/LR/active/medium rates.
- `kernel_phi_resolved_psd.png`: joint `P(phi,rho,f)` at selected receiving phi bins.

Full rate spectra are computed after summing channels, retaining cross
correlations. Summing bin PSDs alone would omit those correlations. Rates are
interval averages, with Nyquist `1/(2*sample_every*dt)`; decrease the sampling
interval to resolve faster fluctuations.

```python
from lramb.decomposition import KernelDecomposition

observer = KernelDecomposition(model, shells=12)
run = simulate(model, decomposition=observer)
decomposition = observer.result
decomposition.save("results/phi_distance_decomposition.npz")
```

## Circular-cluster stability

Notebook section 8 initializes radii **10 and 16** from the local quartic bulk
minima, `phi_inside=+sqrt(-a/b)`, `phi_outside=-sqrt(-a/b)`. Defaults `a=-1,b=1`
give **+1 and -1**. `CircleSpec.from_bulk(p)` also sets the passive planar tanh
width `sqrt(2*kappa/(-a))`. The earlier `phi_inside=0.8` circle results are
superseded and their caches are not reused by the corrected section.

An equal-mass `R(theta)=Rb*(1+epsilon*cos(m*theta))` perturbation uses
`epsilon=0.05,m=2`. **A scalar radius root solve matches the reference mean;
no uniform field shift changes the plateau densities.** Each radius has the
mean implied by its own initial field, separate from the random/stripe run.
The pairs share a conserved noise realization. The integrator remains
Euler–Maruyama and all EP uses midpoint quadrature.

The notebook compares explicitly labeled local passive (`lambda=g=0`), local
active (`g=0`, original lambda), and the actual active+kernel model. The latter
keeps all requested physical parameters. Local bulk minima initialize these
runs; they need not equal the final coexistence densities of the active,
nonlocal, curved interface. Active coexistence conditions depend on interface
terms ([Wittkowski et al.](https://www.nature.com/articles/ncomms5351)).

Periodic connected components track breakup, dissolution and the largest dense
cluster's area radius. Radial threshold crossings about its tracked centre give
shape Fourier modes. Each mode is normalized by its **instantaneous mean
interface radius** before taking the paired shape response. Reference relative
angular roughness is also plotted. Radius evolution, circularity, and paired
shape amplification are separate diagnostics; shrinking alone is not classified
as shape instability. Centre frames separate translation. Absolute displacement
fits, field response and radial profiles are retained. Fits stop at breakup,
loss of interface, underresolved radius or large relative deformation. The
regression interval is diagnostic because time samples are correlated.

`circle_bulk_minima/radius_<R>/reference.npz` and `perturbed.npz` store the
corrected active+kernel trajectories. Named local controls share
`results/circle_controls_bulk_minima/{passive,active}/radius_<R>/` caches.
`circle_initialization.png`, `circle_comparison.png`, `circle_stability.png`
and initialization/stability JSON reports are saved in `circle_bulk_minima/`.
Change `CIRCLE_RADII`, `CIRCLE_STEPS`, `CIRCLE_FIT_UNTIL` and the seed for
radius/noise/time comparisons. A surviving
circle in one finite stochastic window does not establish a stability boundary.
Distance to the tracked cluster centre is used only for the shape/radial-profile
diagnostic; the EP decomposition uses the kernel's source distance `|x-y|`.

## Planar-wall stability (circle experiments remain in section 8)

Notebook section 9 adds a flat wall at **x=L/2**, with a periodic companion
at **x=0**. A single sign-changing planar interface cannot stand alone on the
existing periodic lattice. Both interfaces are evolved with the same periodic
solver and tracked independently. The wall initial condition has approximately
zero mean and bulk plateaus `+sqrt(-a/b)` and `-sqrt(-a/b)`, independently of the
mean used for circles or the main random/stripe run.

`WALL_MODE=2`, `WALL_AMPLITUDE=1` inject
`h(y)=A*cos(2*pi*m*y/L)` into the central interface. The companion starts flat.
`WALL_PERTURBATION='bend'` moves both interfaces together; `'thickness'` moves
them in opposite directions. Equal mass is obtained by adjusting a constant
central-wall position, without shifting the bulk phi values. The flat and
perturbed trajectories share their conserved noise, using Euler–Maruyama and
full-step midpoint EP, on either the NumPy or PyTorch backend.

The analysis measures oriented threshold crossings with temporal unwrapping,
central/companion Fourier responses, wall roughness, uniform translation, and
the positive-phase slab width. Centre-bending and thickness responses are also
saved. An early-time log-amplitude fit concerns only the injected nonzero
Fourier mode. It stops at large slopes, inadequate wall separation, or loss of
the two-interface graph in more than 5% of rows. Missing or multiple crossings
are displayed; fragmented domains are not assigned spurious wall growth rates.
The regression CI is a finite-window diagnostic for correlated snapshots.

Named local passive, local active and active+kernel cases are compared, sharing
control caches across the three notebooks. `WALL_STEPS`, `WALL_FIT_UNTIL`,
`wall_spec` and `WALL_SIMULATION` expose the experiment settings. Outputs are
in `results/<kernel>/wall_bulk_minima/mode_<m>_<signature>/` and shared
`results/wall_controls_bulk_minima/{passive,active}/...`. Each contains the
reference/perturbed trajectories, mode/height/graph diagnostics, and growth-fit
JSON. Notebook figures `wall_initialization.png`, `wall_comparison.png` and
`wall_stability.png` are saved beside the per-case folders. The circle cells and
their executed outputs are retained.

`python scripts/plot_wall_summary.py` compares all five completed cases in
`results/wall_stability_comparison.png` and `wall_response_comparison.png`.
In the default seed-217 window to `t=160`, Poisson and Linnik retain both
walls. Riesz loses the two-interface graph at `t=44` (flat reference) / `t=38`
(perturbed pair), despite decay of the initial injected mode. The notebook
therefore reports mode response and interface survival separately.

## Interpreting steady-state diagnostics

The notebooks display mass conservation, free-energy components, field variance,
structure-weighted mean wave number, block EP means, and the SE versus block
duration. They compare production structure factors in four quarters and report
95% slope intervals for block means. These are finite-window drift screens,
not a proof of stationarity. Long-lived coarsening or correlated blocks can
require longer burn-in and production. A local time averaged map in the laboratory
frame can blur moving interfaces.

The bundled default trajectories retain drift in some structural observables
over the production window `t=400..1040`; the notebooks flag it explicitly.
Treat these as finite-window examples and extend equilibration before quoting
their EP means as stationary estimates.

For quantitative estimates, compare independent seeds and `dt` versus `dt/2`
at identical physical durations and sampling intervals. Doubling burn/production
and sampling/frame step counts preserves those durations when halving `dt`.
Inspect a plateau of block uncertainty before interpreting its value as a
standard error. Increase `PRODUCTION_STEPS` in multiples of `SAMPLE_EVERY*BLOCKS`.

## References

- [Wittkowski et al. (2014), Scalar phi4 field theory for active-particle phase separation](https://www.nature.com/articles/ncomms5351)
- [Nardini et al. (2017), Entropy production in field theories without time reversal symmetry](https://arxiv.org/abs/1610.06112)
- [Lim & Teo (2009), Multivariate generalized Linnik densities](https://arxiv.org/abs/0903.5344)
- [Stinga (2018), Fractional Laplacian and the method of semigroups](https://arxiv.org/abs/1808.05159)
