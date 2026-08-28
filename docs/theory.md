# Theory & Physics

spectral_galerkin_heat solves the **non-linear transient heat equation** with a *semi-analytical
spectral Galerkin* (SG) method. Following reference-medium methods from computational
mechanics, the problem is split into a linear, constant-coefficient reference operator and a
collection of non-linear forcing terms. The reference operator is represented in an eigenbasis
that satisfies the boundary conditions exactly, so linear diffusion becomes a set of
*decoupled* ordinary differential equations integrated **exactly and unconditionally stably**.
Everything else — temperature-dependent properties, latent heat, and the non-linear surface
fluxes (laser, evaporation, convection) — is carried as a forcing term and resolved by a
relaxed fixed-point iteration.

This page summarises the method. For runnable examples see {doc}`examples`; for the full
derivation and validation see the accompanying article (preprint forthcoming).

## Problem & governing equation

On a fixed cuboid domain $\Omega = [0, L_x] \times [0, L_y] \times [0, L_z]$, the heat
equation reads

$$
\rho \left[ c + \sum_j L_j \frac{\partial f_j}{\partial T} \right] \frac{\partial T}{\partial t}
  = \nabla \cdot (\boldsymbol{K} \cdot \nabla T) + S,
$$

where $T$ is temperature, $\rho = \rho(T)$ the temperature-dependent mass density,
$\boldsymbol{K} = \boldsymbol{K}(T)$ the (generally anisotropic) thermal conductivity tensor,
and $c = \sum_i c_i \chi_i$ the sensible heat capacity summed over the phase mass fractions
$\chi_i$. Phase change enters through the latent heat $L_j$ released across each transition,
weighted by $\partial f_j / \partial T$, where $f_j(T)$ is the transformed fraction of the
$j$-th transition (0 below the solidus $T_{s,j}$, 1 above the liquidus $T_{e,j}$, smoothly
varying in between). $S$ collects any additional volumetric source.

The applied heat fluxes are prescribed as **non-linear Neumann boundary conditions** — the
sum of all surface contributions (laser, evaporation, convection):

$$
-(\boldsymbol{K} \cdot \nabla T) \cdot \boldsymbol{n} = \sum_{s=1}^{N_s} q_s(t, T)
  \quad \text{on } \partial\Omega .
$$

## Reference-medium decomposition

The spectral treatment rests on a *reference-medium* splitting, standard in reference-medium
methods for heterogeneous mechanics. Each temperature-dependent property is written as a
constant reference value plus a fluctuation about it:

$$
\rho(T) = \bar{\rho} + \tilde{\rho}(T), \qquad
c(T) = \bar{c} + \tilde{c}(T), \qquad
\boldsymbol{K}(T) = \bar{k}\,\mathbb{I} + \tilde{\boldsymbol{K}}(T).
$$

This defines a **linear, isotropic, constant-coefficient reference operator**
$\mathcal{L} = \bar{\rho}\bar{c}\,\partial_t - \bar{k}\nabla^2$ whose eigensystem depends only
on the geometry and the (homogeneous Neumann) boundaries. The heat equation is then rewritten
*exactly* as this reference operator acting on $T$ equal to a bulk forcing $R(T)$, with a
matching surface relation:

$$
R(T) = \nabla \cdot (\tilde{\boldsymbol{K}} \cdot \nabla T)
  - \rho \sum_j L_j \frac{\partial f_j}{\partial T}\,\dot{T}
  - \left( \bar{\rho}\,\tilde{c} + \tilde{\rho}\,\bar{c} + \tilde{\rho}\,\tilde{c} \right)\dot{T}
  + S,
\qquad
R_s(T) = \sum_{s=1}^{N_s} q_s + (\tilde{\boldsymbol{K}} \cdot \nabla T) \cdot \boldsymbol{n}.
$$

No physics is discarded: the property fluctuations $\tilde{\boldsymbol{K}}$, $\tilde{\rho}$,
$\tilde{c}$, the latent heat, and the boundary fluxes are all carried in $R$ and $R_s$. The
reference constants $\bar{\rho}, \bar{c}, \bar{k}$ only set the operator that is integrated
analytically; the fixed-point iteration that resolves the boundary and latent-heat
non-linearities resolves the property fluctuation at the same time, so at convergence the
**full variable-coefficient (temperature-dependent) solution** is recovered. In the code this
fluctuation term is the *property correction*, applied whenever `k`, `rho` or `Cp` are given as
temperature-dependent branches (see {doc}`examples`, example 3).

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

and the reference operator is projected onto each $\Phi_{mnp}$ (the Galerkin step). Using
orthonormality and Green's identity, the reference diffusion reduces to
$-\bar{k}\lambda_{mnp}\Theta_{mnp}$ and every forcing contribution — the boundary fluxes, the
latent-heat volumetric term, and the property-fluctuation terms of $R$ and $R_s$ — collapses
into a single **modal forcing** $F_{mnp}(t)$. The PDE becomes a *decoupled* ODE per mode:

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
non-linear forcing within each step is resolved by a Picard fixed-point iteration: each pass
reconstructs $T$, re-evaluates $F_{mnp}$, and blends the new modal iterate with the previous
one through a relaxation factor $\omega \in (0,1]$ (config key `picard_omega`) until the
relative change of the modes falls below a tolerance $\epsilon$ (`picard_tol`).

## Dimensional reduction — why it's fast

Naively, evaluating the non-linear sources would need a full 3-D transform every iteration,
costing $\mathcal{O}(N_{\text{vol}}\log N_{\text{vol}})$ with $N_{\text{vol}} = N_x N_y N_z$.
spectral_galerkin_heat avoids this by exploiting the separability of the basis and the *locality* of
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
  Cartesian axes. (Other boundary types, e.g. Dirichlet → sine basis, follow the same
  procedure but are not used here.)
- **Reference constants:** the split into $\bar{\rho}, \bar{c}, \bar{k}$ plus a fluctuation is
  exact; the constants only set the operator integrated analytically. Their choice affects the
  *conditioning* of the fixed-point iteration (how large a fluctuation it must resolve), not
  the converged solution. The iteration converges fastest when the melt pool, where the
  fluctuation is largest, stays a modest fraction of the domain.
- **Captures:** transient conduction, temperature-dependent $\rho$, $c$ and $k$, latent heat of
  fusion, non-linear surface fluxes (laser, evaporation, convection), finite-domain boundary
  effects.
- **Does not capture:** fluid flow / Marangoni convection or vapour recoil mechanics — the
  model is conduction-only.
```

## Notation & units

| Symbol | Meaning | Units | Config key |
|---|---|---|---|
| $T$ | Temperature | K | — |
| $\bar{\rho}$ | Reference density | kg·m⁻³ | `material.rho` |
| $\bar{k}$ | Reference thermal conductivity | W·m⁻¹·K⁻¹ | `material.k` |
| $\bar{c}$ | Reference sensible heat capacity | J·kg⁻¹·K⁻¹ | `material.Cp` |
| $\tilde{\rho},\tilde{c},\tilde{\boldsymbol{K}}$ | Property fluctuations about the reference | (as above) | — |
| $L_j$ | Latent heat of fusion | J·kg⁻¹ | `material.L_f` |
| $T_{s,j},\,T_{e,j}$ | Solidus / liquidus temperatures | K | `material.T_solidus`, `material.T_liquidus` |
| $f_j$ | Transformed (liquid) fraction | — | — |
| $q_s$ | Boundary heat flux ($s$-th contribution) | W·m⁻² | — |
| $R,R_s$ | Bulk / surface forcing | — | — |
| $L_x,L_y,L_z$ | Domain dimensions | m | `domain.size` |
| $N_x,N_y,N_z$ | Grid resolution / number of modes per axis | — | `domain.mesh` |
| $\Phi_{mnp}$ | Spatial eigenmode | — | — |
| $\lambda_{mnp}$ | Laplacian eigenvalue | m⁻² | — |
| $\Theta_{mnp}$ | Modal temperature coefficient | K·m³ᐟ² | — |
| $F_{mnp}$ | Modal forcing (projected sources) | — | — |
| $\Delta t$ | Time step | s | `simulation.dt` |
| $\gamma_{mnp}$ | Modal decay rate | s⁻¹ | — |
| $\omega$ | Picard relaxation factor | — | `picard_omega` |
| $\epsilon$ | Fixed-point tolerance | — | `picard_tol` |
