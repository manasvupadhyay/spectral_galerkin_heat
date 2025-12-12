import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# ============================================================
#   PARAMETERS (MATCHING YOUR SPECTRAL CODE)
# ============================================================

class PhysParams:
    def __init__(self):
        self.rho = 7850.0          # Density [kg/m^3]
        self.Cp_base = 500.0       # Base Specific Heat [J/kgK]
        self.k = 15.0              # Thermal Conductivity [W/mK] (Constant as requested)
        self.T0 = 293.0            # Ambient Temperature [K]
        
        # Laser
        self.P = 200.0             # Power [W]
        self.Absorptivity = 0.30   # Absorptivity
        self.r_b = 6e-5            # Beam radius [m] (Not used in point-source Rosenthal, but noted)
        self.vx = 0.8              # Scan speed [m/s]
        
        # Geometry / Initial position
        self.x0 = 0.0
        self.y0 = 0.0025
        
        # Phase Change
        self.L_f = 267700.0        # Latent Heat of Fusion [J/kg]
        self.T_solidus = 1700.0    # Solidus Temperature [K]
        self.T_liquidus = 1800.0   # Liquidus Temperature [K]
        
        # Effective Power (Q = eta * P)
        self.Q = self.Absorptivity * self.P / 2

class GeomParams:
    def __init__(self, phys):
        # Domain size (from your test.py)
        self.Lx = 0.01
        self.Ly = 0.005
        self.Lz = 0.0025
        
        # Resolution 
        # (Reduced slightly from spectral code for fast iterative execution in Python)
        # Spectral code had 512x256x1200. We use a coarser mesh for the demo 
        # to prevent memory overflow in standard environments, but enough for accuracy.
        self.nx = 512
        self.ny = 256
        self.nz = 200
        
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        
        # Coordinates
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.z = np.linspace(0, self.Lz, self.nz)
        
        # 3D Meshgrid (Memory intensive, be careful with high N)
        # Indexing 'ij' to match matrix notation (nz, ny, nx) if needed, 
        # but 'xy' is standard for meshgrid. We will use (ny, nx) for 2D and broadcasting for 3D.
        self.X_2d, self.Y_2d = np.meshgrid(self.x, self.y)
        
        # We will compute Z broadcasting on the fly to save RAM if possible, 
        # or build full 3D if RAM allows. For 256x128x200 = 6.5M points, full 3D is fine (~50MB per array).
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='xy')

# ============================================================
#   AUGMENTED ROSENTHAL IMPLEMENTATION
# ============================================================

def get_Ceff(T, phys):
    """
    Computes temperature-dependent heat capacity.
    Accounts for Latent Heat in the range [T_solidus, T_liquidus].
    """
    C_eff = np.full_like(T, phys.Cp_base)
    
    # Identify mushy zone
    mask_mushy = (T >= phys.T_solidus) & (T <= phys.T_liquidus)
    
    # Add latent heat component: L_f / DeltaT
    C_mushy = phys.Cp_base + phys.L_f / (phys.T_liquidus - phys.T_solidus)
    
    C_eff[mask_mushy] = C_mushy
    return C_eff

def solve_augmented_rosenthal(phys, geom, t_snapshot, iterations=20, store_history=False):
    """
    Solves the implicit Rosenthal equation: T(x) = Rosenthal(x, alpha(T(x)))
    Returns T_field, T_0, laser center xc,yc, index of max-difference location,
    and optionally history of T profiles along x for animation.
    """
    print(f"Solving Augmented Rosenthal at t = {t_snapshot:.4f} s")
    
    # 1. Current Laser Position
    xc = phys.x0 + phys.vx * t_snapshot
    yc = phys.y0
    zc = 0.0
    
    # 2. Coordinate Transformation (Moving Frame relative to laser)
    Xi = geom.X - xc
    Yi = geom.Y - yc
    Zi = geom.Z - zc
    
    # Radius from source (avoid singularity)
    R = np.sqrt(Xi**2 + Yi**2 + Zi**2) + 1e-9
    
    # Base alpha
    alpha_base = phys.k / (phys.rho * phys.Cp_base)
    prefactor_const = phys.Q / (2 * np.pi * phys.k)
    
    # Initial Guess
    print("  Iter 0: Constant properties initialization...")
    exponent = -phys.vx * (R + Xi) / (2 * alpha_base)
    T = phys.T0 + (prefactor_const / R) * np.exp(exponent)
    T_0 = T.copy()
    maxdiff_idx = (0,0,0)
    
    # Find y index closest to laser center for profile extraction
    idx_y_center = np.argmin(np.abs(geom.y - yc))
    
    # Store history for animation
    T_history = []
    if store_history:
        T_profile_init = np.clip(T[idx_y_center, :, 0], None, 3600.0)
        T_history.append(('Iter 0 (Initial)', T_profile_init.copy()))
    
    # Iterations
    for i in range(iterations):
        T_old = T.copy()
        
        C_eff_field = get_Ceff(T, phys)
        alpha_field = phys.k / (phys.rho * C_eff_field)
        
        exponent_new = -phys.vx * (R + Xi) / (2 * alpha_field)
        T_new = phys.T0 + (prefactor_const / R) * np.exp(exponent_new)
        
        # Under-relaxation for stability (blend old and new)
        relaxation = 0.1
        T = relaxation * T_new + (1 - relaxation) * T_old
        
        # Store profile for animation
        if store_history:
            T_profile = np.clip(T[idx_y_center, :, 0], None, 3600.0)
            T_history.append((f'Iter {i+1}', T_profile.copy()))
        
        # Convergence check (compare unclipped new vs old for meaningful diff)
        diff = np.abs(T - T_old)
        max_diff = np.max(diff)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        maxdiff_idx = max_idx
        
        mean_diff = np.mean(diff)
        print(f"  Iter {i+1}: Max Diff = {max_diff:.2f} K, Mean Diff = {mean_diff:.4f} K at idx {max_idx}")
        
        if max_diff < 0.5:
            print("  Converged.")
            break
    
    T = np.clip(T, None, 3600.0)
    T_0 = np.clip(T_0, None, 3600.0)
    
    if store_history:
        return T, T_0, xc, yc, maxdiff_idx, T_history, idx_y_center
    return T, T_0, xc, yc, maxdiff_idx

def create_iteration_animation(geom, T_history, idx_y, output_file='rosenthal_iteration.gif'):
    """
    Create an animation showing T profile along x evolving with iterations.
    
    Args:
        geom: GeomParams object
        T_history: List of (label, T_profile) tuples
        idx_y: y index used for profile extraction
        output_file: Output filename for animation
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x_mm = geom.x * 1000  # Convert to mm
    
    # Find global min/max for consistent y-axis
    all_temps = np.concatenate([t[1] for t in T_history])
    T_min = max(0, np.min(all_temps) - 100)
    T_max = min(4000, np.max(all_temps) + 100)
    
    # Initial plot
    line_aug, = ax.plot([], [], 'b-', linewidth=2, label='Augmented Rosenthal')
    line_init, = ax.plot(x_mm, T_history[0][1], 'orange', linestyle='--', 
                          linewidth=1.5, alpha=0.7, label='Classical Rosenthal (Iter 0)')
    
    ax.set_xlim(x_mm[350], x_mm[-10])
    ax.set_ylim(T_min, T_max)
    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    title = ax.set_title('', fontsize=14)
    
    # Add horizontal lines for solidus/liquidus
    ax.axhline(y=1700, color='red', linestyle=':', alpha=0.5, label='Solidus')
    ax.axhline(y=1800, color='darkred', linestyle=':', alpha=0.5, label='Liquidus')
    ax.axhline(y=3600, color='purple', linestyle=':', alpha=0.5, label='Clip (3600K)')
    
    def init():
        line_aug.set_data([], [])
        title.set_text('')
        return line_aug, title
    
    def animate(frame):
        label, T_profile = T_history[frame]
        line_aug.set_data(x_mm, T_profile)
        title.set_text(f'Temperature Profile along x (y=center, z=0) - {label}')
        return line_aug, title
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(T_history), interval=300, blit=True)
    
    # Save animation
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer='pillow', fps=3)
    print(f"Animation saved!")
    
    plt.close(fig)
    return anim

# ============================================================
#   MAIN EXECUTION
# ============================================================

def main():
    phys = PhysParams()
    geom = GeomParams(phys)
    
    t_snapshot = 0.012 
    
    start_time = time.time()
    result = solve_augmented_rosenthal(phys, geom, t_snapshot, iterations=20, store_history=True)
    T_field, T_0, xc, yc, maxdiff_idx, T_history, idx_y_center = result
    print(f"Calculation Time: {time.time() - start_time:.2f} s")
    
    # Create iteration animation
    create_iteration_animation(geom, T_history, idx_y_center, output_file='rosenthal_iteration.gif')
    
    # Find indices of maximum temperature on the surface (z=0)
    T_surface = T_field[:, :, 0]
    idx_max = np.unravel_index(np.argmax(T_surface), T_surface.shape)
    idx_y = idx_max[0]
    idx_x = idx_max[1]
    idx_z = 0  # Surface
    
    # Extract Profiles
    T_profile_x = T_field[idx_y, :, 0] 
    T_profile_x_0 = T_0[idx_y, :, 0]
    T_profile_y = T_field[:, idx_x, 0]
    T_profile_z = T_field[idx_y, idx_x, :]
    
    # Plot profiles x
    plt.figure(figsize=(20,4))
    plt.plot(geom.x[400:-1] * 1000, T_profile_x[400:-1], label='Augmented Rosenthal', color='blue')
    plt.plot(geom.x[400:-1] * 1000, T_profile_x_0[400:-1], label='Classical Rosenthal', color='orange', linestyle='--')
    plt.xlabel('x (mm)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Profile along x at y=max, z=0')
    plt.legend()
    plt.show()
    # Plot profiles y

    # Save profiles to text files (coordinates in m, temperature in K)
    np.savetxt('x_rosenthal_latent_heat.txt',
               np.column_stack((geom.x , T_profile_x)),
               header='x_m T_K', fmt='%.6e')
    np.savetxt('y_rosenthal_latent_heat.txt',
               np.column_stack((geom.y, T_profile_y)),
               header='y_m T_K', fmt='%.6e')
    np.savetxt('z_rosenthal_latent_heat.txt',
               np.column_stack((geom.z, T_profile_z)),
               header='z_m T_K', fmt='%.6e')
    print("Saved profiles to x_rosenthal_latent_heat.txt, y_rosenthal_latent_heat.txt, z_rosenthal_latent_heat.txt")
    
    # --- Visualization: zoom around melt pool (1 mm radius) and mark max-diff ---
    # center coordinates (m)
    center_x = geom.x[idx_x]
    center_y = geom.y[idx_y]
    radius_m = 1e-3  # 1 mm
    
    # index half-widths
    half_nx = int(np.ceil(radius_m / geom.dx))
    half_ny = int(np.ceil(radius_m / geom.dy))
    xmin = max(0, idx_x - half_nx)
    xmax = min(geom.nx, idx_x + half_nx + 1)
    ymin = max(0, idx_y - half_ny)
    ymax = min(geom.ny, idx_y + half_ny + 1)
    
    # Subset arrays
    xs = geom.x[xmin:xmax] * 1000  # mm
    ys = geom.y[ymin:ymax] * 1000  # mm
    T_sub = T_field[ymin:ymax, xmin:xmax, 0]
    
    plt.figure(figsize=(6,5))
    cs = plt.contourf(xs, ys, T_sub, levels=50, cmap='inferno')
    plt.colorbar(label='Temperature (K)')
    plt.title('Zoom: Surface Temperature around Melt Pool (±1 mm)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.axis('equal')
    
    # mark temperature maximum (center) and max-diff location
    plt.scatter([center_x*1000], [center_y*1000], c='cyan', marker='+', s=80, label='Temp max (center)')
    
    # convert maxdiff_idx to coordinates (only if inside domain)
    md_y, md_x, md_z = maxdiff_idx
    md_x_m = geom.x[md_x]
    md_y_m = geom.y[md_y]
    plt.scatter([md_x_m*1000], [md_y_m*1000], c='white', marker='x', s=100, linewidths=2, label='Max diff')
    
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()