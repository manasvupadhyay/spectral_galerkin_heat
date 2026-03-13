#!/usr/bin/env python3
"""
eagar_tsai.py
=============
Compute the Eagar-Tsai analytical solution for a moving Gaussian heat source
on a semi-infinite substrate and write the full 3-D temperature field to
HDF5 / XDMF (ParaView-compatible).

Equation (laser frame, ξ = x - x_laser):

    T(ξ,y,z,t) - T₀ = AP / [π ρCp √(πα)] ∫₀ᵗ  1 / [√τ (4ατ + σ²)]
        × exp[ -(ξ+vτ)²+y²) / (4ατ+σ²)  -  z²/(4ατ) ] dτ

with σ² = r_b² / 2  (r_b is the 1/e² beam radius used by the solver;
                      σ is the 1/e radius that appears in the convolution)

Uses exponentially-spaced (geometric) quadrature points so that the
dense sampling concentrates near τ→0 where the 1/√τ singularity lives.

Parameters are taken from config/fast_test.yaml.
The laser travels from x_start to x_end = Lx/2 (domain centre).

Usage
-----
    python utils/eagar_tsai.py                  # direct
    python -m utils.eagar_tsai                  # as module
"""

import numpy as np
import h5py
import os
import time as wall_clock

# ────────────────────────────────────────────────────────────────────────────
#  Parameters  (config/fast_test.yaml)
# ────────────────────────────────────────────────────────────────────────────

# -- Material --
rho   = 7850.0        # kg/m³
k     = 15.0          # W/(m·K)
Cp    = 500.0         # J/(kg·K)
T0    = 293.0         # K  (ambient / initial)
alpha = k / (rho * Cp)  # thermal diffusivity  [m²/s]

# -- Domain --
Lx, Ly, Lz = 0.005, 0.0025, 0.00125          # [m]
nx, ny, nz  = 512,  256,   50            # mesh points

# -- Laser --
A   = 0.30              # absorptivity
P   = 200.0             # W
r_b = 60.0e-6           # beam radius  [m]
# -- Velocity  (G-code: F48000 mm/min = 800 mm/s = 0.8 m/s) --
v_mag = 0.8             # m/s

# -- Trajectory --
# Laser goes from x_start to x_end = middle of the domain.
# y stays at domain centre.
x_start = 0.0                    # [m]  starting X position
x_end   = Lx / 2                       # [m]  = 0.005  (domain centre)
y_laser = Ly / 2                       # [m]  = 0.0025

# Signed velocity  (direction from start → end)
v = v_mag * np.sign(x_end - x_start)   # negative if moving in -x
t_total = abs(x_end - x_start) / v_mag # integration upper bound  [s]

# ────────────────────────────────────────────────────────────────────────────
#  Quadrature  (exponentially-spaced τ)
# ────────────────────────────────────────────────────────────────────────────
N_TAU   = 400 
TAU_MIN = 1e-10           # avoid 1/√τ singularity at τ=0
tau_pts = np.geomspace(TAU_MIN, t_total, N_TAU)
dtau    = np.diff(tau_pts)  # interval widths  (N_TAU - 1)

# ────────────────────────────────────────────────────────────────────────────
#  Node-centred grid  (n+1 points per axis, includes boundary nodes)
# ────────────────────────────────────────────────────────────────────────────
dx_g = Lx / nx;  dy_g = Ly / ny;  dz_g = Lz / nz
x = np.linspace(0, Lx, nx+1)
y = np.linspace(0, Ly, ny+1)
z = np.linspace(0, Lz, nz+1)

# Relative coordinates (current laser at x_end, y_laser, z-surface = Lz)
xi    = x - x_end          # (nx+1,)  ξ = x_lab − x_laser
eta   = y - y_laser         # (ny+1,)  η = y_lab − y_laser
depth = Lz - z              # (nz+1,)  surface = 0, bottom = Lz

# Method-of-images: η offsets for reflections across y = 0 and y = Ly
eta_img_y0  = eta + 2 * y_laser           # reflected η for y = 0 plane
eta_img_yLy = eta - 2 * (Ly - y_laser)    # reflected η for y = Ly plane

# ────────────────────────────────────────────────────────────────────────────
#  Beam convention
# ────────────────────────────────────────────────────────────────────────────
# The solver (and YAML config) defines r_b as the 1/e² beam radius:
#   I(r) = (2AP)/(π r_b²) exp(−2r²/r_b²)
#
# The Eagar-Tsai convolution uses the 1/e radius σ = r_b/√2, giving
#   σ² = r_b² / 2   →  denominator  4ατ + σ²
sigma_sq = r_b**2 / 2.0

# ────────────────────────────────────────────────────────────────────────────
#  Pre-factor   C = AP / [π ρ Cp √(π α)]
# ────────────────────────────────────────────────────────────────────────────
# Derived from convolving the Gaussian source with the semi-infinite
# Green's function (factor 2 from image source) and integrating over
# the surface x',y'  →  yields  1/(π^{3/2} √α)  =  1/(π √(πα))
C = A * P / (np.pi * rho * Cp * np.sqrt(np.pi * alpha))

# ────────────────────────────────────────────────────────────────────────────
#  Numerical integration  (midpoint rule on geometric grid)
# ────────────────────────────────────────────────────────────────────────────
T_field  = np.zeros((nz+1, ny+1, nx+1), dtype=np.float64)
T_images = np.zeros_like(T_field)   # accumulates image-source contributions
buf      = np.empty_like(T_field)
buf_img  = np.empty_like(T_field)

print("Eagar–Tsai analytical solution (with method of images)")
print(f"  Domain  : {Lx*1e3:.1f} × {Ly*1e3:.1f} × {Lz*1e3:.1f} mm")
print(f"  Mesh    : {nx} × {ny} × {nz}  ({nx*ny*nz/1e6:.1f} M pts)")
print(f"  alpha   : {alpha:.4e} m²/s")
print(f"  Laser   : P={P} W,  A={A},  r_b={r_b*1e6:.0f} µm,  v={v:.2f} m/s")
print(f"  Travel  : x_start={x_start*1e3:.2f} mm → x_end={x_end*1e3:.2f} mm")
print(f"  t_total : {t_total*1e6:.1f} µs  ({N_TAU} τ-steps, geomspace)")
print()

t0_wall = wall_clock.perf_counter()

for i in range(N_TAU - 1):
    tau = 0.5 * (tau_pts[i] + tau_pts[i + 1])   # midpoint of interval
    w   = dtau[i]                                 # interval width

    denom_xy = 4.0 * alpha * tau + sigma_sq       # 4ατ + σ²
    denom_z  = 4.0 * alpha * tau                  # 4ατ   (for z-decay)

    # Scalar part of integrand  (× dτ already folded in)
    scalar = w / (np.sqrt(tau) * denom_xy)

    # --- Separable 1-D exponentials ------------------------------------------
    xi_shifted = xi + v * tau                      # (nx,)
    exp_x = np.exp(-xi_shifted * xi_shifted / denom_xy)   # (nx,)
    exp_y = np.exp(-eta * eta / denom_xy)                  # (ny,)
    exp_z = np.exp(-depth * depth / denom_z)               # (nz,)

    # --- 3-D outer product into pre-allocated buffer -------------------------
    f_xy = exp_y[:, None] * exp_x[None, :]         # (ny, nx)
    np.multiply(exp_z[:, None, None],
                f_xy[None, :, :],
                out=buf)
    buf *= scalar
    T_field += buf

    # --- Method of images (3 image sources for finite-domain correction) ---
    # Image 1: reflection across x = 0  (velocity −v)
    xi_img_x0 = xi + 2 * x_end - v * tau
    exp_x_img = np.exp(-xi_img_x0 * xi_img_x0 / denom_xy)
    f_xy_img  = exp_y[:, None] * exp_x_img[None, :]

    # Image 2: reflection across y = 0
    exp_y_img_y0 = np.exp(-eta_img_y0 * eta_img_y0 / denom_xy)
    f_xy_img += exp_y_img_y0[:, None] * exp_x[None, :]

    # Image 3: reflection across y = Ly  (= 2.5 mm)
    exp_y_img_yLy = np.exp(-eta_img_yLy * eta_img_yLy / denom_xy)
    f_xy_img += exp_y_img_yLy[:, None] * exp_x[None, :]

    np.multiply(exp_z[:, None, None], f_xy_img[None, :, :], out=buf_img)
    buf_img *= scalar
    T_images += buf_img

    # Progress
    if (i + 1) % 500 == 0 or i == 0:
        elapsed = wall_clock.perf_counter() - t0_wall
        print(f"  step {i+1:>5d}/{N_TAU-1}  "
              f"τ={tau:.3e} s  elapsed={elapsed:.1f} s")

# ── Build both uncorrected and corrected fields ──
T_uncorrected = T_field * C + T0             # pure Eagar-Tsai (semi-infinite)
T_corrected   = (T_field + T_images) * C + T0  # with method-of-images correction

elapsed = wall_clock.perf_counter() - t0_wall
print(f"\nDone in {elapsed:.1f} s")
print(f"  Uncorrected : T_max = {T_uncorrected.max():.1f} K   T_min = {T_uncorrected.min():.1f} K")
print(f"  Corrected   : T_max = {T_corrected.max():.1f} K   T_min = {T_corrected.min():.1f} K")

# ────────────────────────────────────────────────────────────────────────────
#  Save HDF5 + XDMF
# ────────────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'validation_results')
os.makedirs(out_dir, exist_ok=True)


def _save_field(T_data, basename):
    """Write one temperature field to HDF5 + XDMF."""
    nz1, ny1, nx1 = T_data.shape
    base     = os.path.join(out_dir, basename)
    h5_path  = base + '.h5'
    xmf_path = base + '.xmf'
    h5_ref   = os.path.basename(h5_path)

    T32 = T_data.astype(np.float32)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('X', data=x.astype(np.float32))
        f.create_dataset('Y', data=y.astype(np.float32))
        f.create_dataset('Z', data=z.astype(np.float32))
        ds = f.create_dataset('temperature', data=T32)
        ds.attrs['time']    = t_total
        ds.attrs['v']       = v
        ds.attrs['x_laser'] = x_end
        ds.attrs['y_laser'] = y_laser

    xmf = f"""\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
   <Grid Name="EagarTsai" GridType="Uniform" Time="{t_total}">
     <Topology TopologyType="3DRectMesh" Dimensions="{nz1} {ny1} {nx1}"/>
     <Geometry GeometryType="VXVYVZ">
       <DataItem Dimensions="{nx1}" NumberType="Float" Precision="4" Format="HDF">
          {h5_ref}:/X
       </DataItem>
       <DataItem Dimensions="{ny1}" NumberType="Float" Precision="4" Format="HDF">
          {h5_ref}:/Y
       </DataItem>
       <DataItem Dimensions="{nz1}" NumberType="Float" Precision="4" Format="HDF">
          {h5_ref}:/Z
       </DataItem>
     </Geometry>
     <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="{nz1} {ny1} {nx1}" NumberType="Float" Precision="4" Format="HDF">
          {h5_ref}:/temperature
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
"""
    with open(xmf_path, 'w') as f:
        f.write(xmf)

    print(f"Saved:  {h5_path}")
    print(f"Saved:  {xmf_path}")


# ── Save uncorrected (pure Eagar-Tsai, semi-infinite) ──
_save_field(T_uncorrected, 'eagar_tsai')

# ── Save corrected (method of images for finite domain) ──
_save_field(T_corrected, 'eagar_tsai_corrected')
