import h5py
import os
import numpy as np
from scipy.ndimage import shift as scipy_shift

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
except ImportError:
    cp = None
    cupy_fft = None
    cupy_ndimage = None

def get_array_module(arr):
    if cp is not None:
        return cp.get_array_module(arr)
    return np

OUT_DIR = "out"

def save_field_to_hdf5(filename_base, field, grid_coords, value_name="Field", geom=None, verbose=False):
        """Serialize a 3D scalar field to HDF5 with an accompanying XDMF wrapper."""
        h5_name = f"{filename_base}.h5"
        xmf_name = f"{filename_base}.xmf"
        h5_ref = os.path.basename(h5_name)

        x_coords, y_coords, z_coords = grid_coords
        nz, ny, nx = field.shape

        if verbose and geom is not None:
                print(f"Exporting HDF5/XDMF. Domain Size: {geom.Lx:.2e} x {geom.Ly:.2e} x {geom.Lz:.2e}")

        with h5py.File(h5_name, "w") as f:
                f.create_dataset("X", data=x_coords)
                f.create_dataset("Y", data=y_coords)
                f.create_dataset("Z", data=z_coords)
                f.create_dataset(value_name, data=field)

        xmf_content = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
     <Grid Name="Mesh" GridType="Uniform">
         <Topology TopologyType="3DRectMesh" Dimensions="{nz} {ny} {nx}"/>
         <Geometry GeometryType="VXVYVZ">
             <DataItem Dimensions="{nx}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/X
             </DataItem>
             <DataItem Dimensions="{ny}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/Y
             </DataItem>
             <DataItem Dimensions="{nz}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/Z
             </DataItem>
         </Geometry>
         <Attribute Name="{value_name}" AttributeType="Scalar" Center="Node">
             <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">
                {h5_ref}:/{value_name}
             </DataItem>
         </Attribute>
     </Grid>
 </Domain>
</Xdmf>
"""
        with open(xmf_name, "w") as f:
                f.write(xmf_content)

        if verbose:
                print(f"Saved debug files: {xmf_name} (Open this in Paraview)")

# ============================================================
#   PHYSICAL HELPERS COMMON
# ============================================================

def q_laser(SsState, laser):
    """Gaussian laser heat flux using current Laser position (laser.x, laser.y)."""
    xp = get_array_module(SsState.X)
    r_sq = (SsState.X - laser.x) ** 2 + (SsState.Y - laser.y) ** 2
    return (laser.laser_coef * xp.exp(-2.0 * r_sq / laser.r_b ** 2)).astype(np.float32)


def compute_evaporation_flux(T_surface, q_out, P0, R, T_boil, DeltaH_LV, R_v, T_liquidus) -> np.ndarray:
    """Evaporative heat flux."""
    xp = get_array_module(T_surface)
    q = 0.82 * DeltaH_LV * P0 / xp.sqrt(2 * np.pi * R_v * T_surface) * \
        xp.exp((DeltaH_LV / (R_v * T_boil)) * (1.0 - T_boil / T_surface))
    q[T_surface < T_liquidus] = 0.0
    return q.astype(np.float32)


def check_resolution(laser, num, geom):
    """Check if spatial and temporal resolutions are sufficient."""
    dx_rb, dy_rb = geom.dx / laser.r_b, geom.dy / laser.r_b
    v_mag = np.linalg.norm(laser.v)
    v_crit = v_mag / (10 * geom.dx / num.dt) if v_mag > 0 else 0
    
    print(f"Resolution: dx/rb={dx_rb:.2f}, dy/rb={dy_rb:.2f}, v_crit={v_crit:.2f}")
    if dx_rb > 0.4 or dy_rb > 0.4: print("WARNING: Spatial resolution insufficient!")
    if v_crit > 1.0: print("WARNING: Laser moves too fast for time step!")


