import h5py
import os

def T_to_HDF5(filename_base, T_box, box_coords, geom=None):
    """
    Saves the 3D temperature box to HDF5 and creates an XDMF wrapper 
    for easy opening in Paraview.
    
    Args:
        filename_base: Filename without extension (e.g., "debug_step_10")
        T_box: 3D numpy array of temperature (z, y, x)
        box_coords: Tuple (x_coords, y_coords, z_coords) 1D arrays
        geom: Optional geometry object for metadata
    """
    h5_name = f"{filename_base}.h5"
    xmf_name = f"{filename_base}.xmf"
    
    # Use basename for the reference inside XDMF to avoid double directory paths
    # when ParaView resolves relative paths.
    h5_ref = os.path.basename(h5_name)
    
    x, y, z = box_coords
    nz, ny, nx = T_box.shape
    
    if geom is not None:
        print(f"Exporting HDF5/XDMF. Domain Size: {geom.Lx:.2e} x {geom.Ly:.2e} x {geom.Lz:.2e}")

    # 1. Save Data to HDF5
    with h5py.File(h5_name, "w") as f:
        # Save geometry
        f.create_dataset("X", data=x)
        f.create_dataset("Y", data=y)
        f.create_dataset("Z", data=z)
        # Save attributes
        f.create_dataset("Temperature", data=T_box)

    # 2. Write XDMF File (XML description for Paraview)
    # This maps the raw H5 data to a 3D Rectilinear Grid
    # Topology Dimensions are K J I (Z Y X) for C-order arrays
    # Geometry VXVYVZ expects DataItems in order X, Y, Z
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
     <Attribute Name="Temperature" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">
        {h5_ref}:/Temperature
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
"""
    with open(xmf_name, "w") as f:
        f.write(xmf_content)
    
    print(f"Saved debug files: {xmf_name} (Open this in Paraview)")
