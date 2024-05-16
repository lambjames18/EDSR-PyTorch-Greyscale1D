import random

import numpy as np
import skimage.color as sc

import torch

# grabs a patch of the data for testing based off of args
def get_patch(*args, patch_size=96, scale=2):

    ih, iw = args[0].shape[-2:]

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