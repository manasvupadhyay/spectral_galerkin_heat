import h5py
import os
import numpy as np

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