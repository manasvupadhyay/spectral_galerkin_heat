spectral_galerkin_heat Documentation
====================================

**A semi-analytical Spectral Galerkin model for efficiently solving the nonlinear heat equation on a fixed cuboid domain.**

spectral_galerkin_heat resolves the transient thermal field of a scanning laser including phase change,
evaporative cooling, and convection on cuboid domains. It drives the laser heat source directly from G-code.
Linear diffusion is integrated analytically in a spectral eigenbasis, which removes the global
algebraic solve of an implicit finite-element step and is implemented for CPU and GPU. On a 
single-pass laser scan it agrees with a high-fidelity finite-element reference
to within 0.67% in relative :math:`L^2` error (see :doc:`validation`).

.. figure:: _images/fig_lines_2.png
   :alt: Simulation of a laser path with spectral_galerkin_heat
   :width: 500px
   :align: center

   *Simulation of a laser path with spectral_galerkin_heat.*

.. admonition:: How to get started?
   :class: tip

   Start with the :doc:`theory` (how the method works) and the :doc:`validation`
   (how it compares against analytical, finite-element, and existing work). The
   :doc:`examples` then proceed from a short tutorial to a full non-linear run. See
   :doc:`installation` to get the code.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   theory
   validation
   installation
   configuration
   outputs
   examples

.. toctree::
   :maxdepth: 2
   :caption: About:

   status
   citing

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:

   ARCHITECTURE
