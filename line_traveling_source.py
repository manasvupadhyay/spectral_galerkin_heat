import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# --- 1. Material & Process Parameters ---
rho = 0.00785        # Density (g/mm^3)
cp = 0.5             # Specific heat (J/g/K)
lamb = 0.015         # Thermal conductivity (W/mm/K)
D = lamb / (rho * cp) # Thermal diffusivity (mm^2/s)
V = 800.0            # Laser velocity (mm/s)

# Source Definition
Lf = 267.7           # Specific Latent Heat (J/g)
l_m = 0.2            # Length of mushy zone (mm)
dS  = 0.01           # Cross-section area (mm^2)
z_source = -0.05     # Source depth (mm)
q_line = (rho * Lf * V * dS) / l_m # Linear Power Density (W/mm)

# --- 2. Green's Function ---
def integrand(u, x, y, z, z_src, V, D, q_lin, lamb):
    # Relative distance in scanning direction (xi = x - u)
    xi_rel = x - u 
    
    # Distance to Real Source (z_src) and Image Source (-z_src)
    R1 = np.sqrt(xi_rel**2 + y**2 + (z - z_src)**2)
    R2 = np.sqrt(xi_rel**2 + y**2 + (z + z_src)**2)
    
    # Rosenthal Green's Functions
    # Prevent division by zero if R is very small
    R1 = np.maximum(R1, 1e-6)
    R2 = np.maximum(R2, 1e-6)

    G1 = (np.exp(-V * (xi_rel + R1) / (2 * D))) / R1
    G2 = (np.exp(-V * (xi_rel + R2) / (2 * D))) / R2
    
    return (q_lin / (4 * np.pi * lamb)) * (G1 + G2)

def compute_map(x_arr, y_arr, z_fixed, axis_mode):
    """
    Computes 2D Temperature map. 
    axis_mode 'xy', 'xz', or 'yz' determines which coordinates vary.
    """
    T_map = np.zeros((len(y_arr), len(x_arr)))
    
    # Simple loop integration (vectorized integration is complex for quad)
    for i, crd2 in enumerate(y_arr): # Rows (Y or Z)
        for j, crd1 in enumerate(x_arr): # Cols (X or Y)
            
            # Map grid coordinates to physical x,y,z based on mode
            if axis_mode == 'xy':
                px, py, pz = crd1, crd2, z_fixed
            elif axis_mode == 'xz':
                px, py, pz = crd1, 0, crd2 # y is fixed at 0
            elif axis_mode == 'yz':
                px, py, pz = 0, crd1, crd2 # x is fixed at 0 (laser center)
            
            # Integrate line source from -l_m to 0
            val, _ = integrate.quad(integrand, -l_m, 0, 
                                    args=(px, py, pz, z_source, V, D, q_line, lamb))
            T_map[i, j] = val
    return T_map

# --- 3. Grid Generation (Centered on Laser at 0,0,0) ---
res = 200 # Resolution (keep low for speed in this example)
range_L = 0.6 # mm (Scan length to view)

# Grids
x_grid = np.linspace(-range_L/2, range_L/2, res) # X: Along scan
y_grid = np.linspace(-range_L/4, range_L/4, res) # Y: Transverse
z_grid = np.linspace(-0.3, 0, res)               # Z: Depth (negative)

print("Computing XY View...")
Map_XY = compute_map(x_grid, y_grid, z_fixed=0, axis_mode='xy')

print("Computing XZ View...")
Map_XZ = compute_map(x_grid, z_grid, z_fixed=0, axis_mode='xz')

print("Computing YZ View...")
# Note: For YZ, X is fixed at 0. Horizontal axis is Y, Vertical is Z
Map_YZ = compute_map(y_grid, z_grid, z_fixed=0, axis_mode='yz') 


# --- 4. Plotting ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
cmap = 'inferno'

# XY View (Top)
im1 = ax[0].imshow(Map_XY, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
                   origin='lower', cmap=cmap, aspect='equal')
ax[0].set_title("Top View (XY) @ Surface")
ax[0].set_xlabel("x (mm) [Scan Dir]")
ax[0].set_ylabel("y (mm)")
ax[0].axvline(0, color='w', ls='--', lw=0.8) # Laser Center
plt.colorbar(im1, ax=ax[0], fraction=0.04, pad=0.04, label="Temp Rise (K)")

# XZ View (Side)
im2 = ax[1].imshow(Map_XZ, extent=[x_grid.min(), x_grid.max(), z_grid.min(), z_grid.max()], 
                   origin='lower', cmap=cmap, aspect='auto')
ax[1].set_title("Side View (XZ) @ Center")
ax[1].set_xlabel("x (mm) [Scan Dir]")
ax[1].set_ylabel("z (mm) [Depth]")
ax[1].axvline(0, color='w', ls='--', lw=0.8)
plt.colorbar(im2, ax=ax[1], fraction=0.04, pad=0.04, label="Temp Rise (K)")

# YZ View (Front)
im3 = ax[2].imshow(Map_YZ, extent=[y_grid.min(), y_grid.max(), z_grid.min(), z_grid.max()], 
                   origin='lower', cmap=cmap, aspect='auto')
ax[2].set_title("Front View (YZ) @ x=0")
ax[2].set_xlabel("y (mm)")
ax[2].set_ylabel("z (mm) [Depth]")
ax[2].axvline(0, color='w', ls='--', lw=0.8)
plt.colorbar(im3, ax=ax[2], fraction=0.04, pad=0.04, label="Temp Rise (K)")

plt.suptitle(f"Latent Heat Line Source (l_m={l_m}mm, z_src={z_source}mm)", fontsize=14)
plt.show()