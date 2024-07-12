import os
import numpy as np
import h5py
import torch
from skimage import io
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.widgets

import model 
import option_mod
import utility
import loss
import trainer_pollock
import InteractiveView

folder = "F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/W_phase_only_(greyscale)/"
paths = [folder + file for file in os.listdir(folder) if file.endswith('.tif') or file.endswith('.tiff')]
paths = sorted(paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
print(*paths, sep="\n")

