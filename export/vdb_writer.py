import openvdb
import numpy as np

def write_density_to_vdb(density, cell_centers, voxel_size=0.1, filename="out.vdb"):
    grid = openvdb.FloatGrid()
    grid.transform = openvdb.createLinearTransform(voxel_size)
    accessor = grid.getAccessor()

    for value, pos in zip(density, cell_centers):
        ijk = tuple((pos / voxel_size).astype(int))
        accessor.setValueOn(ijk, float(value))

    grid.name = "density"
    openvdb.write(filename, grids=[grid])

