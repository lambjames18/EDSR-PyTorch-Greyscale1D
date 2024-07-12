import os
from skimage import io
import numpy as np
from tqdm.auto import tqdm

# adding 1 of every 11 of the images to stack
folder = "F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/W_phase_only_(greyscale)/"
paths = [folder + file for file in os.listdir(folder) if file.endswith('.tif') or file.endswith('.tiff')]
paths = sorted(paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
volume = np.array([io.imread(path) for path in tqdm(paths)])
volume = volume[:250, 1000:2000, 1000:2000]
resolution = np.array([0.5, 0.081, 0.081])


stackArr = []
subtract = 2
for i in range(volume.shape[0]):
    if i > 0 and (i-subtract) % 11 == 0:
        stackArr.append(volume[i])
    if i > 0 and i-2 % 100 == 0:
        subtract = i

for i in range(len(stackArr)):
    io.imsave(f"F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/W_phase_only_(greyscale)/misaligntest/{i}.tiff", stackArr[i])