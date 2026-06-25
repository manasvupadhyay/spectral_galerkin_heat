"""Backend-parametrized spectral-solver state (shared by CPU and GPU kernels).

These classes hold the precomputed grid, fine-mesh bases, working buffers and
spectral propagators for :class:`~fast_heat_solv.solvers.spectral.SpectralSolver`,
parametrized by the array module ``xp`` (NumPy or CuPy).

Each kernel module exposes a thin :class:`SpectralSolverState` subclass that
binds its own ``xp``, so callers construct it as
``SpectralSolverState(phys, geom, num, fine)`` regardless of backend.

The state is built at the Python layer only: a dataclass cannot enter the
``@njit`` / ``@cuda.jit`` kernels, so values are unpacked to scalars/arrays at
the kernel boundary.
"""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique
#
# Author: Théo Andrieux
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

import logging
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

import numpy as np

from fast_heat_solv.physics import spectral_helpers as spec_hp

# Backend-agnostic array type: a ``numpy.ndarray`` or a ``cupy.ndarray``.
# Aliased to ``Any`` rather than a union so importing this module never pulls in
# CuPy (an optional dependency); the name documents intent at the call sites.
NDArray = Any

__all__ = [
    "SpectralGrid",
    "FineMeshState",
    "SolverBuffers",
    "BackendHooks",
    "SpectralSolverState",
    "precompute_K_KK",
]

logger = logging.getLogger(__name__)


def _asnumpy(arr):
    """Return a host NumPy view of *arr* (handles CuPy arrays via ``.get()``)."""
    get = getattr(arr, "get", None)
    return get() if callable(get) else np.asarray(arr)


class SpectralGrid:
    """Immutable grid definitions and reconstruction bases.

    Plain class (not a ``@dataclass``): the fields are computed from
    ``(geom, xp)`` in :meth:`__init__`, not passed in, so the generated
    dataclass ``__init__`` would not fit. The annotated declarations below
    document the schema and provide defaults for the lazily-built fields
    (``B_recon`` / ``coords_rec``).
    """
    # Global coordinates (Cell-Centered)
    x: NDArray = None
    y: NDArray = None
    z: NDArray = None

    # Reconstruction constants
    recon_scale: float = 0.0
    Cp32_broadcast: NDArray = None
    Cp32_broadcast_bottom: NDArray = None
    dct_scale: float = 0.0

    # Normalization coefficients (indexed by axis: 0=x, 1=y, 2=z)
    C: tuple = None

    # Lazy-loaded full reconstruction bases and node-centered coords (indexed by axis)
    B_recon: list = None
    coords_rec: list = None

    def __init__(self, geom, xp, dtype=np.float32):
        # Floating-point precision for all arrays built here (np.float32 /
        # np.float64); ``np.float32`` keeps the historical default.
        self.dtype = dtype

        # Global mesh coordinates (cell-centered), one array per axis.
        self.x, self.y, self.z = [((xp.arange(n) + 0.5) * d).astype(dtype)
                                   for n, d in zip(geom.n, geom.d)]

        # Normalization coefficients (C[0]=x, C[1]=y, C[2]=z).
        self.C = tuple(xp.asarray(spec_hp._C_coef(n, L), dtype=dtype)
                       for n, L in zip(geom.n, geom.size))

        # Scaling factors (scalar math stays on the host via np.sqrt).
        dx, dy = geom.d.x, geom.d.y
        nx, ny = geom.n.x, geom.n.y
        Lx, Ly = geom.size.x, geom.size.y
        self.dct_scale = dtype((dx * dy) * np.sqrt((nx * ny) / (Lx * Ly)))
        self.recon_scale = dtype(np.sqrt(nx * ny) / np.sqrt(Lx * Ly))

        # Compute top-surface weighting for projection
        sign = xp.power(-1.0, xp.arange(geom.n.z, dtype=dtype)).astype(dtype)
        self.Cp32_broadcast = (self.C[2].astype(dtype) * sign)[:, None, None]

        # Bottom-surface weighting: cos(p*pi*0/Lz) = 1, so no sign alternation
        self.Cp32_broadcast_bottom = self.C[2].astype(dtype)[:, None, None]

    def prepare_full_reconstruction(self, geom):
        """Compute node-centered grids and full-domain reconstruction bases on demand.

        The bases are kept on the host (NumPy) on both backends: they are used
        only for final output, so this avoids spending GPU memory when full
        reconstruction is never requested.
        """
        if self.B_recon is not None:
            return

        nx, ny, nz = geom.n
        Lx, Ly, Lz = geom.size
        dx, dy, dz = geom.d

        self.coords_rec = [((np.arange(n + 1)) * d).astype(self.dtype)
                           for n, d in zip((nx, ny, nz), (dx, dy, dz))]
        C_host = [_asnumpy(self.C[a]) for a in range(3)]
        self.B_recon = [(C_host[a][:, None] * np.cos(np.pi * np.arange(n)[:, None] * self.coords_rec[a][None, :] / L)).astype(self.dtype)
                        for a, (n, L) in enumerate(zip((nx, ny, nz), (Lx, Ly, Lz)))]


class FineMeshState:
    """Handles the moving fine mesh for latent heat/nonlinearities.

    Plain class (not a ``@dataclass``): fields are computed in
    :meth:`__init__`. The annotated declarations document the schema and
    default the late-set ``T_prev`` / ``Q_prev`` to ``None``.
    """
    refinement: int = 4
    nx_box: int = 0
    ny_box: int = 0
    nz_box: int = 0
    n_fine_totals: list = None  # [nx_total, ny_total, nz_total]

    coords_fine: list = None    # [x_fine, y_fine, z_fine]
    dx_fine: float = 0.0
    dy_fine: float = 0.0
    dz_fine: float = 0.0
    dV_fine: float = 0.0

    B_fine_full: list = None    # cosine bases indexed by axis; z uses domain-shifted coords
    B_fine: list = None         # active window bases; z does not slide

    T_prev: NDArray = None
    Q_prev: NDArray = None

    def __init__(self, geom, grid: SpectralGrid, fine, xp, dtype=np.float32):
        self.dtype = dtype
        dx, dy, dz = geom.d
        Lx, Ly, Lz = geom.size

        # Refinement factor and extents of the refined, laser-following sub-box
        # come from FineMeshParams
        # (configurable via the ``fine_mesh`` config section).
        self.refinement = fine.refinement
        Lx_box, Ly_box, Lz_box = fine.box_size

        self.dx_fine, self.dy_fine, self.dz_fine = dx/self.refinement, dy/self.refinement, dz/self.refinement
        d_fine = [self.dx_fine, self.dy_fine, self.dz_fine]

        self.n_fine_totals = [int(np.ceil(L / d)) for L, d in zip([Lx, Ly, Lz_box], d_fine)]
        self.coords_fine = [((xp.arange(n) + 0.5) * d).astype(dtype)
                            for n, d in zip(self.n_fine_totals, d_fine)]

        z_fine_global = (Lz - Lz_box) + self.coords_fine[2]

        logger.info("Precomputing fine cosine bases...")
        mode_counts = list(geom.n)
        L_domain = [Lx, Ly, Lz]
        fine_coords = [self.coords_fine[0], self.coords_fine[1], z_fine_global]
        self.B_fine_full = [
            (grid.C[a][:, None] * xp.cos(np.pi * xp.arange(mode_counts[a])[:, None] * fine_coords[a][None, :] / L_domain[a])).astype(dtype)
            for a in range(3)
        ]

        self.nx_box = min(int(np.ceil(Lx_box / self.dx_fine)), self.n_fine_totals[0])
        self.ny_box = min(int(np.ceil(Ly_box / self.dy_fine)), self.n_fine_totals[1])
        self.nz_box = min(int(np.ceil(Lz_box / self.dz_fine)), self.n_fine_totals[2])

        self.B_fine = [
            xp.zeros((geom.n.x, self.nx_box), dtype=dtype),
            xp.zeros((geom.n.y, self.ny_box), dtype=dtype),
            self.B_fine_full[2][:, :self.nz_box],  # z: static full-depth slice
        ]
        self.dV_fine = self.dx_fine * self.dy_fine * self.dz_fine

    def update(self, laser_state):
        """Update fine mesh basis subsets for x and y axes."""
        for a, (pos, d, n_box) in enumerate([
            (laser_state.x, self.dx_fine, self.nx_box),
            (laser_state.y, self.dy_fine, self.ny_box),
        ]):
            i_start, i_end, _ = spec_hp._calculate_subgrid_indices(pos, d, self.n_fine_totals[a], n_box)
            self.B_fine[a][:, :] = self.B_fine_full[a][:, i_start:i_end]
        # z does not slide


class SolverBuffers:
    """Reusable working arrays.

    Plain class (not a ``@dataclass``): all buffers are allocated from
    ``(num, fine_mesh, xp)`` in :meth:`__init__`.
    """
    a_temp: NDArray = None      # (nz, ny, nx)
    q_evap_old: NDArray = None  # (ny, nx)
    q_evap_buffer: NDArray = None
    Q_latent_buffer: NDArray = None

    def __init__(self, num, fine_mesh: FineMeshState, xp, dtype=np.float32):
        nx, ny, nz = num.nx, num.ny, num.nz
        self.a_temp = xp.empty((nz, ny, nx), dtype=dtype)
        self.q_evap_old = xp.zeros((ny, nx), dtype=dtype)
        self.q_evap_buffer = xp.zeros((ny, nx), dtype=dtype)
        self.Q_latent_buffer = xp.zeros((fine_mesh.nz_box, fine_mesh.ny_box, fine_mesh.nx_box), dtype=dtype)


@dataclass(frozen=True)
class BackendHooks:
    """Backend primitives that the shared free functions in :mod:`spectral_ops`
    need but cannot express with ``xp`` alone.

    Each kernel module builds one and attaches it to its
    :class:`SpectralSolverState` subclass, so a function like
    ``reconstruct_surface_temperature(a, SsState)`` reaches the right FFT /
    ndimage / source-term implementation through ``SsState.hooks``.

    Attributes
    ----------
    idct : callable
        Inverse (type-III, ortho) DCT — ``IDCT_II`` in the kernel modules.
    ndshift : callable
        ``(field, shift_pixels, order, mode, cval) -> shifted_field`` —
        scipy.ndimage on CPU, cupyx.scipy.ndimage on GPU.
    source_term : callable
        Latent-heat source kernel with signature
        ``(T_curr, T_prev, T_S, T_L, rho, L, dt, out)`` — a plain numba ``@njit``
        call on CPU, a ``@cuda.jit`` launch on GPU.
    """
    idct: Callable
    ndshift: Callable
    source_term: Callable


class SpectralSolverState:
    """Coordinator class for the spectral method state.

    Backend-parametrized: pass ``xp=numpy`` or ``xp=cupy``. The kernel modules
    expose thin subclasses that bind their own ``xp`` (and attach
    :class:`BackendHooks`) so callers use the
    ``SpectralSolverState(phys, geom, num, fine)`` signature.

    Plain class (not a ``@dataclass``): the sub-components are built in
    :meth:`__init__`. The annotated declarations document the schema and
    default the late-set ``a`` / ``hooks`` (the latter attached by the
    per-backend subclass) to ``None``.
    """
    # 1. Components
    grid: SpectralGrid = None
    buffers: SolverBuffers = None
    fine_mesh: FineMeshState = None

    # 2. Primary State
    a: NDArray = None       # Current temperature modes (nz, ny, nx)

    # 3. Spectral Propagators
    K: NDArray = None
    KK: NDArray = None

    # 4. Bound array module (numpy or cupy) — lets backend-agnostic free
    # functions in ``spectral_ops`` recover ``xp`` from the state object.
    xp: ModuleType = None

    # 5. Floating-point precision (np.float32 / np.float64) used for every
    # array built below — recoverable by free functions alongside ``xp``.
    dtype: Any = None

    # 6. Backend primitive hooks (FFT / ndimage shift / source-term launch),
    # attached by the per-backend subclass; see ``BackendHooks``.
    hooks: "BackendHooks" = None

    def __init__(self, phys, geom, num, fine, xp):
        self.xp = xp
        # Precision for all solver arrays; defaults to float32 if ``num`` has
        # no ``dtype`` (e.g. a hand-built NumParams from before this field).
        self.dtype = getattr(num, "dtype", np.float32)
        # Initialize sub-components
        self.grid = SpectralGrid(geom, xp, self.dtype)
        self.fine_mesh = FineMeshState(geom, self.grid, fine, xp, self.dtype)
        self.buffers = SolverBuffers(num, self.fine_mesh, xp, self.dtype)

        # Precompute propagators
        self.K, self.KK = precompute_K_KK(phys, num, geom, xp, self.dtype)


def precompute_K_KK(phys, num, geom, xp, dtype=np.float32):
    """
    Compute spectral Propagators (K, KK) based on grid and time step.
    K = exp(-alpha * k^2 * dt) for ETD1 (Exact integration of linear part)
    KK = phi_1 / (rho * Cp), where phi_1(z) = (exp(z) - 1) / z, z = -alpha * k^2 * dt
    """
    # Per-axis wavenumbers in array-index order [z, y, x].
    k = [np.pi * xp.arange(n) / L for n, L in zip(geom.n.zyx(), geom.size.zyx())]
    k_grids = xp.meshgrid(*k, indexing='ij')  # shape (nz, ny, nx) each
    denom = phys.k / (phys.rho * phys.Cp) * sum(kg**2 for kg in k_grids)
    K = xp.exp(-denom * num.dt).astype(dtype)
    mask_zero = (denom == 0)
    # Avoid div by zero
    denom[mask_zero] = 1.0

    phi_1 = (K - 1.0) / (-denom)
    phi_1[mask_zero] = num.dt

    KK = (phi_1 / (phys.rho * phys.Cp)).astype(dtype)
    return K, KK
