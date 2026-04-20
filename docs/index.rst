fastHeatSolv Documentation
==========================

**A semi-analytical, modular solution for the heat equation with support for CPU/GPU backends and G-code-driven laser paths.**

fastHeatSolv is a modular framework designed for simulating heat transfer in additive manufacturing. It uses semi-analytical spectral methods to achieve high performance on both CPU and GPU hardware, and fully supports complex laser trajectories parsed directly from G-code.

Quickstart
----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/TheoADX/fastHeatSolv.git
   cd fastHeatSolv

   # Install the standard CPU environment
   uv sync

   # Run the standard test simulation
   uv run python simulations/main.py simulations/config/standard_test.yaml

Outputs will be saved in unstructured formats ready for ParaView.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   configuration

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:

   ARCHITECTURE
