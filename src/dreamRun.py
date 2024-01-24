import numpy as np
import h5py


import matplotlib.pyplot as plt

import model 
import option_mod
import utility
import loss



h = h5py.File('/Users/anayakhan/Downloads/5842WCu.dream3d', 'r')  # dream3d is a file format that is identical to an HDF5 file, just with a different name

# Get the data
volume = np.squeeze(h['DataContainers/ImageDataContainer/CellData/BSE'][...])  # we use np.squeeze in order to remove the extra dimension that Dream3D adds to the data
resolution = h['DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING'][...][::-1]  # Dream3D stores the resolution backwards so we need to flip it

stack = np.stack(volume, axis = 1)

print("Resolution: z: {:.3f} µm, y: {:.3f} µm, x: {:.3f} µm".format(*resolution))
print("Dimensions: z: {:.3f} µm, y: {:.3f} µm, x: {:.3f} µm".format(*np.array(volume.shape) * np.array(resolution)))  # Note that we have much higher resolution in the XY plane than in the Z plane
print("Dimensions: z: {} voxels, y: {} voxels, x: {} voxels".format(*volume.shape))

# calling the model
