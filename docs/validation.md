# Validation & Results

fastHeatSolv has been checked against three references: closed-form analytical solutions in the
linear regime, a finite-element (FE) model in the non-linear regime, and a published case study.
In each case it reproduces the reference temperature field at a lower computational cost than an
implicit FE solver.

## Against analytical solutions (linear regime)

In the linear regime the moving-source solutions of Rosenthal (point source) and Eagar–Tsai
(distributed Gaussian beam) provide exact references. The spectral solver is analytically exact
here up to truncation, and its centreline temperature on the top surface follows the Eagar–Tsai
solution.

```{figure} _images/linear_lines.png
:alt: Centreline temperature for Rosenthal, Eagar–Tsai and spectral solutions
:width: 80%
:align: center

Temperature along the centreline on the top surface, comparing the Rosenthal, Eagar–Tsai and
spectral (SG) solutions.
```

Refining the spectral resolution lowers the relative $L^2$ error, with exponential convergence
in the in-plane directions ($x$, $y$). Convergence in the build direction ($z$) is slower,
reflecting the steep gradients of the localized surface heat source.

```{figure} _images/error_analytical_three_plots.png
:alt: Relative L2 error versus number of modes per direction
:width: 55%
:align: center

Relative $L^2$ error between the spectral and analytical solutions versus the number of modes
in each direction (log–log).
```

## Against a finite-element reference (non-linear regime)

Once latent heat of fusion and evaporative cooling enter, no closed form exists, so the spectral
solver is compared against an FE model that solves the same governing equations on the same
domain with the same properties. The cut views show the spectral solver resolving the near-source
gradients and the melt-pool shape, in agreement with FE and distinct from the analytical
solution, which omits latent heat and evaporation.

```{figure} _images/cut_stacked_y.png
:alt: xz-plane cut views for analytical, FE and spectral models
:width: 95%
:align: center

Cut views of the temperature field in the $xz$ plane: (a) Eagar–Tsai analytical, (b) finite
element, (c) spectral.
```

```{figure} _images/nonlinear_lines.png
:alt: Centreline temperature for analytical, FE and spectral models (non-linear)
:width: 80%
:align: center

Centreline temperature on the top surface comparing the Eagar–Tsai, FE and spectral solutions
in the non-linear regime.
```

As in the analytical case, refining the modal resolution lowers the error of the spectral
solution against the FE reference, again exponentially.

```{figure} _images/error_FE_spectral_modes.png
:alt: Relative L2 error between spectral and FE solutions versus modes
:width: 55%
:align: center

Relative $L^2$ error between the spectral solver and the FE reference versus the number of modes
per direction (log–log).
```

## Performance

The solver advances linear diffusion by element-wise spectral updates, projects surface fluxes
through a 2-D transform on the boundary, and contracts the latent-heat source over the melt-pool
region alone, so it performs no global 3-D transforms (see {doc}`theory`). On a single CPU core
it reaches FE-level accuracy at a fraction of the FE runtime, and the runtime-versus-DoF curve
approaches a log–log slope near 3, consistent with the asymptotic
$\mathcal{O}(N_{\text{vol}} \log N_{\text{vol}})$ cost.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
```{figure} _images/runtime_vs_dof.png
:alt: Runtime versus degrees of freedom for spectral and FE
:width: 100%

Execution time versus number of degrees of freedom, spectral and FE (single core, Ryzen 9
5900X).
```
:::

:::{grid-item}
```{figure} _images/runtime_vs_error.png
:alt: Work-precision diagram, runtime versus error
:width: 100%

Work–precision diagram: runtime as a function of numerical error. Lower-left is better.
```
:::
::::

## Application to a literature case study

As an independent check, the solver reproduces the multi-island scan-strategy study of
Ramani et al. (2022) for powder-bed fusion of 316L stainless steel, using the same process
parameters and scan strategies. Evaluated at the same collocation points, the spectral model
reproduces the published ranking of strategies through the thermal-uniformity metric $R(t)$ and
the per-strategy temperature fields.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
```{figure} _images/sg_thermal_uniformity.png
:alt: Thermal uniformity metric R(t) computed by the spectral model
:width: 100%

Thermal-uniformity metric $R(t)$ per scan strategy, computed with the spectral model.
```
:::

:::{grid-item}
```{figure} _images/sg_thermal_maps.png
:alt: End-of-scan temperature fields per strategy, spectral model
:width: 100%

End-of-scan temperature fields for each strategy (spectral model).
```
:::
::::

```{admonition} Reproducibility
:class: note

The timings above were measured on a single core of an AMD Ryzen 9 5900X CPU; GPU runs used an
NVIDIA GeForce RTX 4070. The configuration files and scripts that generate each figure will be
released with the code — see {doc}`status`.
```
