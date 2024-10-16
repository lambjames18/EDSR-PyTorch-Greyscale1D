from skimage import io
import os
import numpy as np 

filePath = "F:/WCu-Data-SR/5842WCu_Spall/"
appendList =  ["first_set", "second_set"]


for append in appendList: 
    print("Changing axis in: " + append)
    sr_X = filePath + append + "/images/SR_X/"
    paths = [f for f in os.listdir(sr_X) if f.endswith(".tif")]

    paths = sorted(paths, key=lambda x: int(x.split('.')[0]))
    imgs = np.array([io.imread(sr_X + p) for p in paths])
    print("Paths: ", paths)
    print("Images shape: ", imgs.shape)

    #output = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]), dtype=np.uint8)
    #print("Output shape: ", output.shape)
    swapped = np.swapaxes(imgs, 0, 1)
    swapped = np.swapaxes(swapped, 1, 2)
    print("Swapped shape: ", swapped.shape)
    
    # saving images in the new axis
    savedSR = filePath + append + "/images/SR/"
    for i, img in enumerate(swapped):
        io.imsave(savedSR + str(i) + 'tif', img)