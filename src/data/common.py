import random

import numpy as np
import skimage.color as sc

import torch

# assuming low res patch size 
# will scale upward for high res
def get_patch(*args, patch_size=96, scale=2):
    # consider decreasing the patch size, not too much, test with different patch sizes, end result
    ih, iw = args[0].shape[-2:]

    # for testing 
    # ih2, iw2 = args[1].shape[:2]

    if type(scale) is str:
        scale = int(scale)

    iy = random.randrange(0, ih - patch_size + 1)
    ty = iy * scale 

    ix = random.randrange(0, iw - (patch_size * scale) + 1)

    ret = [
        args[0][iy:iy + patch_size, ix:ix + (patch_size * scale)],
        *[a[ty:ty + (patch_size * scale), ix:ix + (patch_size * scale)] for a in args[1:]]
    ]

    return ret
   
def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))).copy()
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1]
        
        return img

    return [_augment(a) for a in args]

