# Testing the normalize and unnormalize function
import utility
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
import torch


# taking the first image of the dataset
img = io.imread("F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/W_phase_only_(greyscale)/1.tif")[:1000, :1000]

# covert to tensor
img = np.expand_dims(img, axis=2)
np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))).copy()
img = torch.from_numpy(np_transpose).float()

# normalizing the image
img = utility.normalize(img)
#img = utility.unnormalize(img)

# converting the tensor back to numpy
img = img.numpy()
io.imsave("F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/W_phase_only_(greyscale)/normTest/norm2_test.tif", img)