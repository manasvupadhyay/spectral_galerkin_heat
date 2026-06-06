"""Backend-parametrized spectral-solver state (shared by CPU and GPU kernels).

These classes hold the precomputed grid, fine-mesh bases, working buffers and
spectral propagators for :class:`~fast_heat_solv.solvers.spectral.SpectralSolver`.
They used to exist as near-identical copies in ``spectral_cpu_kernels`` and
``spectral_gpu_kernels`` (the only difference being ``np.`` vs ``cp.``), so they
are unified here and parametrized by the array module ``xp``.

Each kernel module exposes a thin :class:`SpectralSolverState` subclass that
binds its own ``xp`` (NumPy or CuPy), so the public construction signature
``SpectralSolverState(phys, geom, num)`` is unchanged.

Like :class:`~fast_heat_solv.core.vector.Vec3`, this lives at the Python
state-construction layer only: a dataclass cannot enter the numba
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
from typing import Any

import numpy as np

from fast_heat_solv.physics import spectral_helpers as spec_hp

__all__ = [
    "SpectralGrid",
    "FineMeshState",
    "SolverBuffers",
    "SpectralSolverState",
    "precompute_K_KK",
]

logger = logging.getLogger(__name__)


def _asnumpy(arr):
    """Return a host NumPy view of *arr* (handles CuPy arrays via ``.get()``)."""
    get = getattr(arr, "get", None)
    return get() if callable(get) else np.asarray(arr)


@dataclass
class SpectralGrid:
    """Immutable grid definitions and reconstruction bases."""
    # Global coordinates (Cell-Centered)
    x: Any = None
    y: Any = None
    z: Any = None

    # Reconstruction constants
    recon_scale: float = 0.0
    Cp32_broadcast: Any = None
    Cp32_broadcast_bottom: Any = None
    dct_scale: float = 0.0

    # Normalization coefficients (indexed by axis: 0=x, 1=y, 2=z)
    C: tuple = None

    # Lazy-loaded full reconstruction bases and node-centered coords (indexed by axis)
    B_recon: list = None
    coords_rec: list = None

    def __init__(self, geom, xp):
        # Global mesh coordinates (Cell-Centered). Coordinate arrays stay
        # per-axis; Vec3 groups only the scalar triples (n, d, size) since a
        # dataclass cannot enter the numba kernels downstream.
        self.x, self.y, self.z = [((xp.arange(n) + 0.5) * d).astype(xp.float32)
                                   for n, d in zip(geom.n, geom.d)]

        # Normalization coefficients (C[0]=x, C[1]=y, C[2]=z), float32 on both
        # backends.
        self.C = tuple(xp.asarray(spec_hp._C_coef(n, L), dtype=xp.float32)
                       for n, L in zip(geom.n, geom.size))

        # Scaling factors (scalar math stays on the host via np.sqrt).
        dx, dy = geom.dx, geom.dy
        nx, ny = geom.nx, geom.ny
        Lx, Ly = geom.Lx, geom.Ly
        self.dct_scale = xp.float32((dx * dy) * np.sqrt((nx * ny) / (Lx * Ly)))
        self.recon_scale = xp.float32(np.sqrt(nx * ny) / np.sqrt(Lx * Ly))

        # Compute top-surface weighting for projection
        sign = xp.power(-1.0, xp.arange(geom.nz, dtype=xp.float32)).astype(xp.float32)
        self.Cp32_broadcast = (self.C[2].astype(xp.float32) * sign)[:, None, None]

        # Bottom-surface weighting: cos(p*pi*0/Lz) = 1, so no sign alternation
        self.Cp32_broadcast_bottom = self.C[2].astype(xp.float32)[:, None, None]

    def prepare_full_reconstruction(self, geom):
        """Compute node-centered grids and full-domain reconstruction bases on demand.

        The bases are kept on the host (NumPy) on both backends: they are used
        only for final output, so this avoids spending GPU memory when full
        reconstruction is never requested.
        """
        if self.B_recon is not None:
            return

        nx, ny, nz = geom.nx, geom.ny, geom.nz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
        dx, dy, dz = geom.dx, geom.dy, geom.dz

        self.coords_rec = [((np.arange(n + 1)) * d).astype(np.float32)
                           for n, d in zip((nx, ny, nz), (dx, dy, dz))]
        C_host = [_asnumpy(self.C[a]) for a in range(3)]
        self.B_recon = [(C_host[a][:, None] * np.cos(np.pi * np.arange(n)[:, None] * self.coords_rec[a][None, :] / L)).astype(np.float32)
                        for a, (n, L) in enumerate(zip((nx, ny, nz), (Lx, Ly, Lz)))]


@dataclass
class FineMeshState:
    """Handles the moving fine mesh for latent heat/nonlinearities."""
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

    T_prev: Any = None
    Q_prev: Any = None

    def __init__(self, geom, grid: SpectralGrid, xp):
        dx, dy, dz = geom.dx, geom.dy, geom.dz
        Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz

        self.refinement = 4
        # TODO: these fine-mesh box extents are hard-coded; they should become
        # configurable parameters (or be sized dynamically from the domain /
        # laser footprint) rather than baked-in constants.
        Lx_box, Ly_box, Lz_box = 0.9e-3, 0.9e-3, 0.04e-3

        self.dx_fine, self.dy_fine, self.dz_fine = dx/self.refinement, dy/self.refinement, dz/self.refinement
        d_fine = [self.dx_fine, self.dy_fine, self.dz_fine]

        self.n_fine_totals = [int(np.ceil(L / d)) for L, d in zip([Lx, Ly, Lz_box], d_fine)]
        self.coords_fine = [((xp.arange(n) + 0.5) * d).astype(xp.float32)
                            for n, d in zip(self.n_fine_totals, d_fine)]

        z_fine_global = (Lz - Lz_box) + self.coords_fine[2]

        logger.info("Precomputing fine cosine bases...")
        mode_counts = [geom.nx, geom.ny, geom.nz]
        L_domain = [Lx, Ly, Lz]
        fine_coords = [self.coords_fine[0], self.coords_fine[1], z_fine_global]
        self.B_fine_full = [
            (grid.C[a][:, None] * xp.cos(np.pi * xp.arange(mode_counts[a])[:, None] * fine_coords[a][None, :] / L_domain[a])).astype(xp.float32)
            for a in range(3)
        ]

        self.nx_box = min(int(np.ceil(Lx_box / self.dx_fine)), self.n_fine_totals[0])
        self.ny_box = min(int(np.ceil(Ly_box / self.dy_fine)), self.n_fine_totals[1])
        self.nz_box = min(int(np.ceil(Lz_box / self.dz_fine)), self.n_fine_totals[2])

        self.B_fine = [
            xp.zeros((geom.nx, self.nx_box), dtype=xp.float32),
            xp.zeros((geom.ny, self.ny_box), dtype=xp.float32),
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


@dataclass
class SolverBuffers:
    """Reusable working arrays."""
    a_temp: Any = None      # (nz, ny, nx)
    q_evap_old: Any = None  # (ny, nx)
    q_evap_buffer: Any = None
    Q_latent_buffer: Any = None

    def __init__(self, num, fine_mesh: FineMeshState, xp):
        nx, ny, nz = num.nx, num.ny, num.nz
        self.a_temp = xp.empty((nz, ny, nx), dtype=xp.float32)
        self.q_evap_old = xp.zeros((ny, nx), dtype=xp.float32)
        self.q_evap_buffer = xp.zeros((ny, nx), dtype=xp.float32)
        if fine_mesh:
            self.Q_latent_buffer = xp.zeros((fine_mesh.nz_box, fine_mesh.ny_box, fine_mesh.nx_box), dtype=xp.float32)


@dataclass
class SpectralSolverState:
    """Coordinator class for the spectral method state.

    Backend-parametrized: pass ``xp=numpy`` or ``xp=cupy``. The kernel modules
    expose thin subclasses that bind their own ``xp`` so callers can keep using
    the ``SpectralSolverState(phys, geom, num)`` signature.
    """
    # 1. Components
    grid: SpectralGrid = None
    buffers: SolverBuffers = None
    fine_mesh: FineMeshState = None

    # 2. Primary State
    a: Any = None       # Current temperature modes (nz, ny, nx)

    # 3. Spectral Propagators
    K: Any = None
    KK: Any = None

    def __init__(self, phys, geom, num, xp):
        # Initialize sub-components
        self.grid = SpectralGrid(geom, xp)
        self.fine_mesh = FineMeshState(geom, self.grid, xp)
        self.buffers = SolverBuffers(num, self.fine_mesh, xp)

        # Precompute propagators
        self.K, self.KK = precompute_K_KK(phys, num, geom, xp)


def precompute_K_KK(phys, num, geom, xp):
    """
    Compute spectral Propagators (K, KK) based on grid and time step.
    K = exp(-alpha * k^2 * dt) for ETD1 (Exact integration of linear part)
    KK = phi_1 / (rho * Cp), where phi_1(z) = (exp(z) - 1) / z, z = -alpha * k^2 * dt
    """
    # Per-axis wavenumbers in array-index order [z, y, x]; .zyx() makes the
    # reversal from (x, y, z) explicit instead of an implicit reversed literal.
    k = [np.pi * xp.arange(n) / L for n, L in zip(geom.n.zyx(), geom.size.zyx())]
    k_grids = xp.meshgrid(*k, indexing='ij')  # shape (nz, ny, nx) each
    denom = phys.k / (phys.rho * phys.Cp) * sum(kg**2 for kg in k_grids)
    K = xp.exp(-denom * num.dt).astype(xp.float32)
    mask_zero = (denom == 0)
    # Avoid div by zero
    denom[mask_zero] = 1.0

    phi_1 = (K - 1.0) / (-denom)
    phi_1[mask_zero] = num.dt

    KK = (phi_1 / (phys.rho * phys.Cp)).astype(xp.float32)
    return K, KK
