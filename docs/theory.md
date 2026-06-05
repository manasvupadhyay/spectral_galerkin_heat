# Theory & Physics

fastHeatSolv solves the **non-linear transient heat equation** with a *spectral Galerkin*
(SG) method. The idea is to represent the temperature field as a sum of global modes that
satisfy the boundary conditions exactly, so that linear heat diffusion becomes a set of
*decoupled* ordinary differential equations that can be integrated **exactly and
unconditionally stably**. The non-linear sources, laser heating, evaporative cooling,
convection and latent heat, are then handled by a fast, dimensionally-reduced projection.

This page summarises the method. For runnable examples see {doc}`examples`; for the full
derivation and validation see the accompanying article (preprint forthcoming).

## Problem & governing equation

On a fixed cuboid domain $\Omega = [0, L_x] \times [0, L_y] \times [0, L_z]$, the heat
equation reads

$$
\frac{\partial (\rho \, c_{\text{eff}} \, T)}{\partial t} = \nabla \cdot (k \nabla T),
$$

where $T$ is temperature, $\rho = \rho(T)$ the temperature-dependent mass density, and
$k = k(T)$ the temperature-dependent thermal conductivity. Phase change enters through an
**apparent heat capacity** $c_{\text{eff}}$ that bundles the sensible heat of each phase with
the latent heat $L_j$ released across each transition:

$$
c_{\text{eff}}(T) = \frac{\partial h}{\partial T}
  = \sum_i c_i \, \chi_i(T) + \sum_j L_j \frac{\partial f_j}{\partial T},
$$

with $\chi_i$ the mass fraction of phase $i$ and $f_j(T)$ the transformed fraction of the
$j$-th transition (0 below the solidus $T_{s,j}$, 1 above the liquidus $T_{e,j}$, smoothly
varying in between).

The applied heat fluxes are prescribed as **non-linear Neumann boundary conditions** — the
sum of all surface contributions (laser, evaporation, convection):

$$
-k \, \frac{\partial T}{\partial n} = \sum_{m=1}^{N} q_m(\mathbf{x}, t, T) \quad \text{on } \Gamma .
$$

## Model reduction

Two simplifications make the spectral treatment possible while preserving the physics that
matter for melt-pool prediction:

- **Constant thermophysical properties.** $\rho$, $k$ and the sensible $c$ are taken as
  temperature-independent. This deliberately isolates the dominant non-linearity — the
  *latent-heat stiffness* — which prior work identifies as the main source of error in
  melt-pool geometry and cooling rates.
- **Zeroth-order (volume-averaged) properties.** Each spatially varying coefficient is
  replaced by its *zeroth-order term* — its volume average — discarding the fluctuation about
  it. This gives the constants $\bar{c}$, $\bar{k}$, $\bar{\rho}$ for the sensible capacity,
  conductivity and density, valid because the phase-changing melt pool occupies a small
  fraction of the domain: the fluctuations are confined to that small region, so away from it
  each property is dominated by its (effectively constant) solid-phase value. The latent heat
  is *not* averaged away — it is retained as a localized volumetric source.

The simplified equation is then

$$
\bar{\rho} \, c_{\text{eff}}(T) \, \frac{\partial T}{\partial t} = \bar{k} \, \Delta T .
$$

## Spectral (Galerkin) discretisation

For the cuboid with insulating (homogeneous Neumann) boundaries, the Laplacian eigenproblem
$-\Delta \Phi = \lambda \Phi$ separates into three 1-D Sturm–Liouville problems whose
solutions are **cosines**. The orthonormal 3-D eigenbasis is

$$
\Phi_{mnp}(\mathbf{x}) = C_m C_n C_p
  \cos\!\left(\tfrac{m\pi x}{L_x}\right)
  \cos\!\left(\tfrac{n\pi y}{L_y}\right)
  \cos\!\left(\tfrac{p\pi z}{L_z}\right),
\qquad
C_m = \sqrt{\tfrac{2 - \delta_{m0}}{L_x}},
$$

with eigenvalues $\lambda_{mnp} = (\tfrac{m\pi}{L_x})^2 + (\tfrac{n\pi}{L_y})^2 + (\tfrac{p\pi}{L_z})^2$.
The temperature is expanded over these modes,

$$
T(\mathbf{x}, t) \approx \sum_{m,n,p} \Theta_{mnp}(t)\, \Phi_{mnp}(\mathbf{x}),
$$

and the governing equation is projected onto each $\Phi_{mnp}$ (the Galerkin step). Using
orthonormality and Green's identity, diffusion reduces to $-\bar{k}\lambda_{mnp}\Theta_{mnp}$ and
every source — boundary fluxes and the latent-heat volumetric term — collapses into a single
**modal forcing** $F_{mnp}(t)$. The PDE becomes a *decoupled* ODE per mode:

$$
\bar{\rho} \bar{c} \, \dot{\Theta}_{mnp}(t) + \bar{k} \lambda_{mnp}\, \Theta_{mnp}(t) = F_{mnp}(t).
$$

Because the non-linear fluxes must be evaluated in physical space and projected back, the
solver alternates between physical and spectral space — a **pseudo-spectral** strategy.

## Time integration

The modal system is stiff (through diffusion) but linear in $\Theta_{mnp}$. Defining the
decay rate $\gamma_{mnp} = \bar{k}\lambda_{mnp}/(\bar{\rho}\bar{c})$ and freezing the forcing over a step
$\Delta t$, the linear part is integrated **exactly** with a first-order exponential
time-differencing (ETD1) update:

$$
\Theta_{mnp}(t + \Delta t) = E_{mnp}\, \Theta_{mnp}(t) + Q_{mnp}\, F_{mnp}(t),
$$

$$
E_{mnp} = e^{-\gamma_{mnp}\Delta t},
\qquad
Q_{mnp} = \frac{1 - e^{-\gamma_{mnp}\Delta t}}{\gamma_{mnp}\, \bar{\rho} \bar{c}} .
$$

The propagators $E_{mnp}$ and $Q_{mnp}$ are **precomputed once**, so the time step reduces to
an element-wise array multiply in spectral space — and is unconditionally stable regardless
of $\Delta t$. The scheme extends to higher order (ETD2, ETD-RK4) at no structural cost. The
non-linear fluxes within each step are resolved by a relaxed fixed-point iteration to a
tolerance $\epsilon$.

## Dimensional reduction — why it's fast

Naively, evaluating the non-linear sources would need a full 3-D transform every iteration,
costing $\mathcal{O}(N_{\text{vol}}\log N_{\text{vol}})$ with $N_{\text{vol}} = N_x N_y N_z$.
fastHeatSolv avoids this by exploiting the separability of the basis and the *locality* of
the physics:

- **Surface fluxes → 2-D transform.** The vertical factor of $\Phi_{mnp}$ evaluates to a
  scalar at the boundary, so a surface-flux projection factorizes into a **2-D DCT** over the
  boundary plus a multiply by the out-of-plane mode. Cost drops to
  $\mathcal{O}(N_{\text{surf}}\log N_{\text{surf}})$ with $N_{\text{surf}} = N_x N_y$.
- **Latent heat → localized tensor contraction.** The latent-heat source is non-zero only in
  the small melt-pool region $\Omega_{\text{active}}$. Its projection is computed by a direct
  contraction over those $M \ll N_{\text{vol}}$ points against precomputed 1-D eigenfunction
  values, costing $\mathcal{O}(M \cdot N_{\text{modes}})$.

By confining the dominant work to a surface and a small active sub-volume, the solver reaches
finite-element accuracy at a fraction of the cost — the bridge to the measured speedups
reported on the Validation page.

```{admonition} Assumptions & limitations
:class: warning

- **Domain:** cuboid only — the cosine eigenbasis requires boundaries aligned with the
  Cartesian axes. (Other boundary types, e.g. Dirichlet → sine basis, are supported by the
  framework but not used here.)
- **Properties:** density, conductivity and sensible heat are reduced to constants — their
  volume-averaged (zeroth-order) values $\bar{\rho}$, $\bar{k}$, $\bar{c}$ — which keeps their
  bulk effect but discards the temperature and spatial variation about the average. This holds
  while the melt pool stays a small fraction of the domain. Only the latent heat is kept as a
  non-linear source.
- **Captures:** transient conduction, latent heat of fusion, non-linear surface fluxes
  (laser, evaporation, convection), finite-domain boundary effects.
- **Does not capture:** fluid flow / Marangoni convection, vapour recoil mechanics, or the
  in-domain variation of $k$ and $\rho$ about their averaged values.
```

## Notation & units

| Symbol | Meaning | Units | Config key |
|---|---|---|---|
| $T$ | Temperature | K | — |
| $\bar{\rho}$ | (Volume-averaged) density | kg·m⁻³ | `material.rho` |
| $\bar{k}$ | (Volume-averaged) thermal conductivity | W·m⁻¹·K⁻¹ | `material.k` |
| $\bar{c}$ | (Volume-averaged) sensible heat capacity | J·kg⁻¹·K⁻¹ | `material.Cp` |
| $L_j$ | Latent heat of fusion | J·kg⁻¹ | `material.L_f` |
| $T_{s,j},\,T_{e,j}$ | Solidus / liquidus temperatures | K | `material.T_solidus`, `material.T_liquidus` |
| $h$ | Specific enthalpy | J·kg⁻¹ | — |
| $c_{\text{eff}}$ | Apparent (effective) heat capacity | J·kg⁻¹·K⁻¹ | — |
| $q_m$ | Boundary heat flux ($m$-th contribution) | W·m⁻² | — |
| $L_x,L_y,L_z$ | Domain dimensions | m | `domain.size` |
| $N_x,N_y,N_z$ | Grid resolution / number of modes per axis | — | `domain.mesh` |
| $\Phi_{mnp}$ | Spatial eigenmode | — | — |
| $\lambda_{mnp}$ | Laplacian eigenvalue | m⁻² | — |
| $\Theta_{mnp}$ | Modal temperature coefficient | K·m³ᐟ² | — |
| $F_{mnp}$ | Modal forcing (projected sources) | — | — |
| $\Delta t$ | Time step | s | `simulation.dt` |
| $\gamma_{mnp}$ | Modal decay rate | s⁻¹ | — |
| $\epsilon$ | Fixed-point tolerance | — | — |
