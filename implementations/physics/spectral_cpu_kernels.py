import numpy as np
from numba import njit, prange
from scipy.ndimage import shift as scipy_shift

# ======================================
# Spectral Method CPU Kernels
# ======================================


@njit(parallel=True, fastmath=True)
def update_modes_etd1(aK, KK_by_Cp, B_scaled, a_temp_out):
    """
    Update spectral coefficients for ETD1 scheme.
    Calculates: a_out = aK + (KK/Cp) * B_scaled
    """
    nz = aK.shape[0]
    for p in prange(nz):
        a_temp_out[p, :, :] = aK[p, :, :] + KK_by_Cp[p, :, :] * B_scaled

@njit(parallel=True, fastmath=True)
def add_source_term_modes(a_temp, KK, Q_modes):
    """
    Accumulate volumetric source term into temperature modes.
    a_temp += KK * Q_modes
    """
    nz = a_temp.shape[0]
    for p in prange(nz):
        for i in range(a_temp.shape[1]):
            for j in range(a_temp.shape[2]):
                a_temp[p, i, j] += KK[p, i, j] * Q_modes[p, i, j]

@njit(parallel=True, fastmath=True)
def compute_source_term_from_temperature(T_curr, T_prev, T_S, T_L, rho, L, dt, out):
    """
    Compute Q = - rho * L * (1 / (TL - TS)) * (dT/dt) * Indicator(TS <= T <= TL)
    Used for latent heat calculation.
    """
    factor = - rho * L / ( (T_L - T_S) * dt )
    nz, ny, nx = T_curr.shape
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                T = T_curr[k, j, i]
                # Indicator function for mushy zone (inclusive)
                if T >= T_S and T <= T_L:
                    dT = T - T_prev[k, j, i]
                    out[k, j, i] = factor * dT
                else:
                    out[k, j, i] = 0.0

@njit(parallel=True, fastmath=True)
def compute_evaporation_flux(T_surface, q_out, P0, R, T_boil, DeltaH_LV, R_v, T_liquidus):
    """
    Compute evaporative heat flux based on surface temperature using Arrhenius law.
    q_out is updated in-place.
    """
    ny, nx = T_surface.shape
    factor1 = 0.82 * DeltaH_LV * P0 / np.sqrt(2 * np.pi * R_v)
    factor2 = DeltaH_LV / (R_v * T_boil)
    
    for j in prange(ny):
        for i in range(nx):
            T = T_surface[j, i]
            if T < T_liquidus:
                q_out[j, i] = 0.0
            else:
                # 1/sqrt(T) * exp(...)
                term = (1.0 / np.sqrt(T)) * np.exp(factor2 * (1.0 - T_boil / T))
                q_out[j, i] = factor1 * term

def reconstruct_temperature_box(a, SsState):
    """
    Reconstructs temperature in a small ROI around the laser.
    Pure Python function using tensordot (optimized in numpy).
    """
    if not SsState.fine_mesh_initialized:
        raise RuntimeError("Fine mesh not initialized.")
    
    # Tensor Contraction: Modes -> Physical Space
    # T(x,y,z) = sum_p sum_n sum_m  a[p,n,m] * Bz[p,z] * By[n,y] * Bx[m,x]
    
    T_step1 = np.tensordot(a, SsState.Bz_fine, axes=(0, 0)) # Contraction over Z
    T_step2 = np.tensordot(T_step1, SsState.By_fine, axes=(0, 0)) # Contraction over Y
    T_box = np.tensordot(T_step2, SsState.Bx_fine, axes=(0, 0)) # Contraction over X
    
    return T_box.astype(np.float32), (SsState.box_x, SsState.box_y, SsState.box_z)

def compute_latent_heat_source(Q_buffer, phys, laser, geom, num, SsState, alpha=0.4):
    """
    Compute volumetric latent heat source Q (W/m^3).
    HIGH-LEVEL ORCHESTRATOR (Runs in Python, calls Kernels).
    """
    # 1. Reconstruct Temperature on Fine Mesh
    T_box, _ = reconstruct_temperature_box(SsState.a_temp, SsState)
    
    # 2. Initialize/Retrieve State buffers
    if not hasattr(SsState, 'T_prev') or SsState.T_prev is None:
        SsState.T_prev = np.zeros_like(T_box)
        SsState.T_prev[:] = T_box[:] 
        SsState.Q_prev = np.zeros_like(Q_buffer)
        SsState.laser_x_prev = laser.x
        SsState.laser_y_prev = laser.y
        Q_buffer.fill(0.0)
        return

    # 3. Shift Previous Fields to Current Frame
    shift_x = laser.x - SsState.laser_x_prev
    shift_y = laser.y - SsState.laser_y_prev
    
    shift_pixels = (0, -shift_y / SsState.dy_fine, -shift_x / SsState.dx_fine)
    
    # Order=1 (Linear) usually sufficient for smooth fields like T
    T_prev_aligned = scipy_shift(SsState.T_prev, shift_pixels, order=1, mode='nearest')
    Q_prev_aligned = scipy_shift(SsState.Q_prev, shift_pixels, order=1, mode='constant', cval=0.0)
    
    # 4. Compute Source Term (Calls Numba Kernel)
    compute_source_term_from_temperature(T_box, T_prev_aligned, 
                                          phys.T_solidus, phys.T_liquidus, 
                                          phys.rho, phys.L_f, num.dt, 
                                          Q_buffer)
    
    # 5. Apply Relaxation
    if alpha < 1.0:
        Q_buffer[:] = alpha * Q_buffer + (1.0 - alpha) * Q_prev_aligned
    
    # 6. Update History
    SsState.T_prev[:] = T_box[:]
    SsState.Q_prev[:] = Q_buffer[:] 
    SsState.laser_x_prev = laser.x
    SsState.laser_y_prev = laser.y


