"""Physics kernels — CPU and GPU spectral method implementations.

**Public API:**
- ``spectral_cpu_kernels``: High-level solver functions for CPU backend
- ``spectral_gpu_kernels``: High-level solver functions for GPU backend (requires CuPy)
- ``spectral_helpers``: Utility functions for spectral computations

Public symbols from each kernel module are the high-level functions
called by the solver (e.g. update_modes_etd1, compute_latent_heat_source).
Internal helpers (prefixed with ``_``) are not part of the public API.

**Internal helpers in spectral_helpers:**
- ``_C_coef``: Normalization coefficients (private)
- ``_calculate_subgrid_indices``: Fine-mesh box centering (private)
"""

__all__ = [
    "spectral_cpu_kernels",
    "spectral_gpu_kernels",
    "spectral_helpers",
]
