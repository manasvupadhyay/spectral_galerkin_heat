#!/usr/bin/env python3
"""
eagar_tsai.py
=============
Eagar-Tsai analytical solution for a moving Gaussian heat source on a
semi-infinite substrate, with an optional method-of-images correction for the
finite-domain insulated walls the spectral solver enforces.
Reference : T. W. Eagar, N. S. Tsai, Temperature fields produced by traveling 
distributed heat sources, Welding Journal 62 (12) (1983) 346s–355s.


This module is **both** an importable reference (used by the linear analytical
validation test, plan §2.1) and a stand-alone script that writes the full 3-D
field to HDF5 / XDMF.  The physics lives in :func:`eagar_tsai_field`; the
``__main__`` block is a wrapper that fixes the config parameters and
saves the result.

Equation (laser frame, ξ = x - x_laser):

    T(ξ,y,z,t) - T₀ = AP / [π ρCp √(πα)] ∫₀ᵗ  1 / [√τ (4ατ + σ²)]
        × exp[ -((ξ+vτ)²+y²) / (4ατ+σ²)  -  z²/(4ατ) ] dτ

with σ² = r_b² / 2  (r_b is the 1/e² beam radius used by the solver; σ is the
1/e radius that appears in the convolution).  Exponentially-spaced (geometric)
quadrature concentrates points near τ→0 where the 1/√τ singularity lives.

Usage
-----
    python tests/eagar_tsai.py            # direct
    python -m tests.eagar_tsai            # as module
"""

import os

import numpy as np


def eagar_tsai_field(
    *,
    rho: float,
    k: float,
    Cp: float,
    T0: float,
    A: float,
    P: float,
    r_b: float,
    Lx: float,
    Ly: float,
    Lz: float,
    nx: int,
    ny: int,
    nz: int,
    x_end: float,
    y_laser: float,
    v: float,
    t_total: float,
    n_tau: int = 400,
    tau_min: float = 1e-10,
    images: bool = True,
) -> np.ndarray:
    """Return the node-centred analytical Eagar-Tsai temperature field.

    The field is evaluated at the instant the laser sits at ``(x_end, y_laser)``
    on the top surface ``z = Lz``, having travelled at constant signed velocity
    ``v`` for ``t_total`` seconds.

    Parameters
    ----------
    rho, k, Cp, T0 : float
        Material density, conductivity, heat capacity, ambient temperature.
    A, P, r_b : float
        Absorptivity, laser power, and the 1/e² beam radius (metres).  The beam
        convention matches the solver: ``I(r) = 2AP/(π r_b²)·exp(−2r²/r_b²)``,
        i.e. the convolution variance is ``σ² = r_b²/2``.
    Lx, Ly, Lz, nx, ny, nz : float / int
        Domain extent and mesh counts.  The returned grid is node-centred with
        ``n+1`` points per axis (``x_j = j·Lx/nx`` etc.), matching
        ``reconstruct_temperature_DCT``.
    x_end, y_laser : float
        Laser position (metres) at the evaluation instant.
    v : float
        Signed laser velocity along x (m/s).
    t_total : float
        Elapsed travel time (s); integration upper bound.
    n_tau, tau_min : int / float
        Geometric τ-quadrature resolution and lower cut-off (avoids the 1/√τ
        singularity at τ = 0).
    images : bool, default True
        When True, add the five image sources (reflections across x=0, x=Lx,
        y=0, y=Ly, and the z=0 bottom) so the field satisfies the insulated
        (zero-flux) walls the spectral solver's cosine basis enforces on all
        faces.  When False, return the bare semi-infinite solution.

    Returns
    -------
    ndarray, shape (nz+1, ny+1, nx+1), float64
        Node-centred temperature field in ``[z, y, x]`` index order.
    """
    alpha = k / (rho * Cp)

    # Geometric τ-grid — dense near τ→0, we take aadvantage of 
    # Exponential convergence of the series solution. 
    tau_pts = np.geomspace(tau_min, t_total, n_tau)
    dtau = np.diff(tau_pts)

    # Node-centred grid: n+1 points per axis, includes boundary nodes.
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    z = np.linspace(0.0, Lz, nz + 1)

    xi = x - x_end               # ξ = x_lab − x_laser
    eta = y - y_laser            # η = y_lab − y_laser
    depth = Lz - z               # surface = 0 at z = Lz, bottom = Lz at z = 0

    # Method-of-images η offsets for reflections across y = 0 and y = Ly.
    eta_img_y0 = eta + 2 * y_laser
    eta_img_yLy = eta - 2 * (Ly - y_laser)

    # σ² = r_b²/2 (1/e radius) — the solver's r_b is the 1/e² radius.
    sigma_sq = r_b**2 / 2.0

    # Pre-factor C = AP / [π ρ Cp √(π α)], from convolving the Gaussian source
    # with the semi-infinite Green's function and integrating over x',y'.
    C = A * P / (np.pi * rho * Cp * np.sqrt(np.pi * alpha))

    T_field = np.zeros((nz + 1, ny + 1, nx + 1), dtype=np.float64)
    T_images = np.zeros_like(T_field)
    buf = np.empty_like(T_field)
    buf_img = np.empty_like(T_field)

    for i in range(n_tau - 1):
        tau = 0.5 * (tau_pts[i] + tau_pts[i + 1])   # midpoint of interval
        w = dtau[i]                                  # interval width

        denom_xy = 4.0 * alpha * tau + sigma_sq      # 4ατ + σ²
        denom_z = 4.0 * alpha * tau                  # 4ατ (z-decay)
        scalar = w / (np.sqrt(tau) * denom_xy)       # × dτ folded in

        # Separable 1-D exponentials.
        xi_shifted = xi + v * tau
        exp_x = np.exp(-xi_shifted * xi_shifted / denom_xy)
        exp_y = np.exp(-eta * eta / denom_xy)
        exp_z = np.exp(-depth * depth / denom_z)

        f_xy = exp_y[:, None] * exp_x[None, :]
        np.multiply(exp_z[:, None, None], f_xy[None, :, :], out=buf)
        buf *= scalar
        T_field += buf

        if images:
            # In-plane reflections across the four x/y side walls.
            # Image 1: across x = 0 (velocity −v).
            xi_img_x0 = xi + 2 * x_end - v * tau
            exp_x_img_x0 = np.exp(-xi_img_x0 * xi_img_x0 / denom_xy)
            # Image 2: across x = Lx.
            xi_img_xLx = xi + 2 * x_end - 2 * Lx - v * tau
            exp_x_img_xLx = np.exp(-xi_img_xLx * xi_img_xLx / denom_xy)
            # Images 3 & 4: across y = 0 and y = Ly.
            exp_y_img_y0 = np.exp(-eta_img_y0 * eta_img_y0 / denom_xy)
            exp_y_img_yLy = np.exp(-eta_img_yLy * eta_img_yLy / denom_xy)

            f_xy_img = exp_y[:, None] * exp_x_img_x0[None, :]
            f_xy_img += exp_y[:, None] * exp_x_img_xLx[None, :]
            f_xy_img += exp_y_img_y0[:, None] * exp_x[None, :]
            f_xy_img += exp_y_img_yLy[:, None] * exp_x[None, :]
            np.multiply(exp_z[:, None, None], f_xy_img[None, :, :], out=buf_img)
            buf_img *= scalar
            T_images += buf_img

            # Image 5: reflection across the z = 0 bottom wall (same xy source
            # profile, mirrored depth) — closes the insulated bottom face.
            depth_img_z0 = 2.0 * Lz - depth
            exp_z_img_z0 = np.exp(-depth_img_z0 * depth_img_z0 / denom_z)
            np.multiply(exp_z_img_z0[:, None, None], f_xy[None, :, :], out=buf_img)
            buf_img *= scalar
            T_images += buf_img

    if images:
        return (T_field + T_images) * C + T0
    return T_field * C + T0


# ─────────────────────────────────────────────────────────────────────────────
#  Stand-alone script: fixed config-grade parameters → HDF5 / XDMF
# ─────────────────────────────────────────────────────────────────────────────

# Parameters from config/fast_test.yaml.
_PARAMS = {
    "rho": 7850.0, "k": 15.0, "Cp": 500.0, "T0": 293.0,
    "A": 0.30, "P": 200.0, "r_b": 60.0e-6,
    "Lx": 0.005, "Ly": 0.0025, "Lz": 0.00125,
    "nx": 512, "ny": 256, "nz": 50,
}
# Laser goes x_start → x_end = domain centre; y fixed at Ly/2; v from F48000.
_V_MAG = 0.8
_X_START = 0.0


def _save_field(T_data, basename, *, x, y, z, t_total, v, x_laser, y_laser, out_dir):
    """Write one temperature field to HDF5 + XDMF (ParaView-compatible)."""
    import h5py

    nz1, ny1, nx1 = T_data.shape
    base = os.path.join(out_dir, basename)
    h5_path = base + ".h5"
    xmf_path = base + ".xmf"

    T32 = T_data.astype(np.float32)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=x.astype(np.float32))
        f.create_dataset("Y", data=y.astype(np.float32))
        f.create_dataset("Z", data=z.astype(np.float32))
        ds = f.create_dataset("temperature", data=T32)
        ds.attrs["time"] = t_total
        ds.attrs["v"] = v
        ds.attrs["x_laser"] = x_laser
        ds.attrs["y_laser"] = y_laser

    from spectral_galerkin_heat.io_utils import XdmfBuilder

    builder = XdmfBuilder(version="2.0")
    builder.add_structured_grid(
        name="EagarTsai",
        dims=(nz1, ny1, nx1),
        h5_ref=os.path.basename(h5_path),
        attributes={"temperature": "temperature"},
        time=t_total,
    )
    builder.write(xmf_path)
    print(f"Saved:  {h5_path}")
    print(f"Saved:  {xmf_path}")


def main() -> None:
    import time as wall_clock

    p = _PARAMS
    x_end = p["Lx"] / 2
    y_laser = p["Ly"] / 2
    v = _V_MAG * np.sign(x_end - _X_START)
    t_total = abs(x_end - _X_START) / _V_MAG

    print("Eagar–Tsai analytical solution (with method of images)")
    print(f"  Domain  : {p['Lx']*1e3:.1f} × {p['Ly']*1e3:.1f} × {p['Lz']*1e3:.1f} mm")
    print(f"  Mesh    : {p['nx']} × {p['ny']} × {p['nz']}")
    print(f"  Laser   : P={p['P']} W, A={p['A']}, r_b={p['r_b']*1e6:.0f} µm, v={v:.2f} m/s")
    print(f"  t_total : {t_total*1e6:.1f} µs")

    t0 = wall_clock.perf_counter()
    T_uncorrected = eagar_tsai_field(
        **p, x_end=x_end, y_laser=y_laser, v=v, t_total=t_total, images=False
    )
    T_corrected = eagar_tsai_field(
        **p, x_end=x_end, y_laser=y_laser, v=v, t_total=t_total, images=True
    )
    print(f"\nDone in {wall_clock.perf_counter() - t0:.1f} s")
    print(f"  Uncorrected : T_max = {T_uncorrected.max():.1f} K  T_min = {T_uncorrected.min():.1f} K")
    print(f"  Corrected   : T_max = {T_corrected.max():.1f} K  T_min = {T_corrected.min():.1f} K")

    x = np.linspace(0.0, p["Lx"], p["nx"] + 1)
    y = np.linspace(0.0, p["Ly"], p["ny"] + 1)
    z = np.linspace(0.0, p["Lz"], p["nz"] + 1)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation_results")
    os.makedirs(out_dir, exist_ok=True)
    save_kw = {"x": x, "y": y, "z": z, "t_total": t_total, "v": v,
               "x_laser": x_end, "y_laser": y_laser, "out_dir": out_dir}
    _save_field(T_uncorrected, "eagar_tsai", **save_kw)
    _save_field(T_corrected, "eagar_tsai_corrected", **save_kw)


if __name__ == "__main__":
    main()
