# Theory & Physics

spectral_galerkin_heat solves the **non-linear transient heat equation** with a *semi-analytical
spectral Galerkin* (SG) method. Following reference-medium methods from computational
mechanics, the problem is split into a linear, constant-coefficient reference operator and a
collection of non-linear forcing terms. The reference operator is represented in an eigenbasis
whose eigenfunctions are known analytically, so the reference problem reduces to one ordinary
differential equation per mode, integrated analytically in time.
Everything else, including temperature-dependent properties, latent heat, and the non-linear
surface fluxes (laser, evaporation, convection), is carried as a forcing term and resolved by a
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

The applied heat fluxes are prescribed as **non-linear Neumann boundary conditions**, the
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
$$

$$
R_s(T) = \sum_{s=1}^{N_s} q_s + (\tilde{\boldsymbol{K}} \cdot \nabla T) \cdot \boldsymbol{n}.
$$

The property fluctuations $\tilde{\boldsymbol{K}}$, $\tilde{\rho}$,
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
$$

with normalisation $C_m = \sqrt{\tfrac{2 - \delta_{m0}}{L_x}}$ (and likewise for $C_n$, $C_p$) and
eigenvalues $\lambda_{mnp} = (\tfrac{m\pi}{L_x})^2 + (\tfrac{n\pi}{L_y})^2 + (\tfrac{p\pi}{L_z})^2$.
The temperature is expanded over these modes,

$$
T(\mathbf{x}, t) \approx \sum_{m,n,p} \Theta_{mnp}(t)\, \Phi_{mnp}(\mathbf{x}),
$$

and the reference operator is projected onto each $\Phi_{mnp}$ (the Galerkin step). Using
orthonormality and Green's identity, the reference diffusion reduces to
$-\bar{k}\lambda_{mnp}\Theta_{mnp}$, and every forcing contribution (the boundary fluxes, the
latent-heat volumetric term, and the property-fluctuation terms of $R$ and $R_s$) collapses
into a single **modal forcing** $F_{mnp}(t)$. The PDE becomes a system of **modal ODEs** in time,
one per mode:

$$
\bar{\rho} \bar{c} \, \dot{\Theta}_{mnp}(t) + \bar{k} \lambda_{mnp}\, \Theta_{mnp}(t) = F_{mnp}(t).
$$

The left-hand side carries no mode-to-mode coupling because the isotropic reference operator has no
cross-derivative terms; the modes are still tied together through $F_{mnp}(t)$, which is evaluated
from the full temperature field and therefore depends on every mode. What the eigenbasis buys is a
reference operator that is diagonal, not a set of independent scalar problems.

Because the non-linear fluxes must be evaluated in physical space and projected back, the
solver alternates between physical and spectral space: a **pseudo-spectral** strategy.

## Time integration

The modal system is stiff but linear in $\Theta_{mnp}$. Defining the
decay rate $\mu_{mnp} = \bar{k}\lambda_{mnp}/(\bar{\rho}\bar{c})$ and freezing the forcing over a step
$\Delta t$, the linear part is integrated **exactly** with a first-order exponential
time-differencing (ETD1) update:

$$
\Theta_{mnp}(t + \Delta t) = E_{mnp}\, \Theta_{mnp}(t) + Q_{mnp}\, F_{mnp}(t),
$$

$$
E_{mnp} = e^{-\mu_{mnp}\Delta t},
\qquad
Q_{mnp} =
\begin{cases}
\dfrac{1 - e^{-\mu_{mnp}\Delta t}}{\mu_{mnp}\, \bar{\rho} \bar{c}}, & \mu_{mnp} \neq 0, \\[2ex]
\dfrac{\Delta t}{\bar{\rho} \bar{c}}, & \mu_{mnp} = 0 .
\end{cases}
$$

The second branch is the $\mu_{mnp} \to 0$ limit of the first and applies to the mean mode
$(0,0,0)$, which does not decay and simply accumulates its forcing.

The propagators $E_{mnp}$ and $Q_{mnp}$ are **precomputed once**, so advancing the linear part
reduces to an element-wise array multiply in spectral space. Since $E_{mnp} = e^{-\mu_{mnp}\Delta t}$
never exceeds 1, this part introduces no step-size limit of its own.

The non-linear forcing within each step is resolved by a Picard fixed-point iteration: each pass
reconstructs $T$, re-evaluates $F_{mnp}$, and blends the new modal iterate with the previous
one through a relaxation factor $\omega \in (0,1]$ (config key `picard_omega`) until the
relative change of the modes falls below a tolerance $\epsilon$ (`picard_tol`). The step size is
set by this iteration rather than by the linear part: lagging the forcing across a step imposes a
convergence restriction that tightens as the non-linearity strengthens, and the Picard iteration
converges linearly, so strongly temperature-dependent regimes need more iterations or a smaller
$\Delta t$.

## Scope

The formulation is posed on a fixed cuboid with Neumann boundary conditions; the same derivation
carries over to cylindrical and spherical domains and to Dirichlet or Robin conditions, but those
are not implemented here. The domain does not evolve, so material addition during a build is out
of scope. The model resolves heat transfer only: melt-pool dynamics such as convection, Marangoni
flow, recoil pressure and keyhole formation would require solving the flow problem alongside it
and are not represented.

## Notation & units

| Symbol | Meaning | Units | Config key |
|---|---|---|---|
| $T$ | Temperature | K | - |
| $\bar{\rho}$ | Reference density | kg·m⁻³ | `material.rho` |
| $\bar{k}$ | Reference thermal conductivity | W·m⁻¹·K⁻¹ | `material.k` |
| $\bar{c}$ | Reference sensible heat capacity | J·kg⁻¹·K⁻¹ | `material.Cp` |
| $\tilde{\rho},\tilde{c},\tilde{\boldsymbol{K}}$ | Property fluctuations about the reference | (as above) | - |
| $L_j$ | Latent heat of fusion | J·kg⁻¹ | `material.L_f` |
| $T_{s,j},\,T_{e,j}$ | Solidus / liquidus temperatures | K | `material.T_solidus`, `material.T_liquidus` |
| $f_j$ | Transformed (liquid) fraction | - | - |
| $q_s$ | Boundary heat flux ($s$-th contribution) | W·m⁻² | - |
| $R,R_s$ | Bulk / surface forcing | - | - |
| $L_x,L_y,L_z$ | Domain dimensions | m | `domain.size` |
| $N_x,N_y,N_z$ | Grid resolution / number of modes per axis | - | `domain.mesh` |
| $\Phi_{mnp}$ | Spatial eigenmode | - | - |
| $\lambda_{mnp}$ | Laplacian eigenvalue | m⁻² | - |
| $\Theta_{mnp}$ | Modal temperature coefficient | K·m³ᐟ² | - |
| $F_{mnp}$ | Modal forcing (projected sources) | - | - |
| $\Delta t$ | Time step | s | `simulation.dt` |
| $\mu_{mnp}$ | Modal decay rate | s⁻¹ | - |
| $\omega$ | Picard relaxation factor | - | `picard_omega` |
| $\epsilon$ | Fixed-point tolerance | - | `picard_tol` |
