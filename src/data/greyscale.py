import os

from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data


class GreyScale(data.Dataset):
    def __init__(self, args, name='GreyScale', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        
        # fix this
        #for f in os.listdir(args.dir_demo):

        # goes through all of the tiff files in the HR folder
        new_path = os.path.join(args.dir_demo, 'HR')
        for f in os.listdir(new_path):
            if f.find('.tiff') >= 0:
                self.filelist.append(os.path.join(new_path, f))
        self.filelist.sort()
        # starting with fewer to see if it runs
        self.filelist = self.filelist[:1]
        print(self.filelist)

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        lr = imageio.imread(self.filelist[idx])
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)

        return lr_t, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale



