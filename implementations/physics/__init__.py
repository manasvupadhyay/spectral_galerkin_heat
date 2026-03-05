"""Physics kernels — CPU and GPU spectral method implementations.

Public symbols from each kernel module are the high-level functions
called by the solver (e.g. update_modes_etd1, compute_latent_heat_source).
Internal helpers (prefixed with ``_``) are not part of the public API.

Note: GPU kernels require CuPy and are not imported eagerly.
Import ``spectral_gpu_kernels`` explicitly when a GPU backend is needed.
"""

__all__ = [
    "spectral_cpu_kernels",
    "spectral_gpu_kernels",
]
