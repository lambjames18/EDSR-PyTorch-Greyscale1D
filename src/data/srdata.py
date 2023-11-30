import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
from skimage import io, transform

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        self.apath = os.path.abspath(args.dir_data)
        self.ext = ('.tiff', '.tiff')
        
        self.images_hr, self.images_lr = self.fill_HR_LR()

        
    def set_as_training(self):
        self.train = True
        n_patches = self.args.batch_size * self.args.test_every
        n_images = len(self.args.data_train) * len(self.images_hr)
        if n_images == 0:
            self.repeat = 0
        else:
            self.repeat = max(n_patches // n_images, 1)

    def set_as_testing(self):
        self.train = False
        self.repeat = 1

    # fills the high res and low res folders from the raw
    # save to the high res and low res, dont save
    def fill_HR_LR(self):
        # list of images 
        imgs = []
        hr_list = []
        lr_list = []

        print(self.apath)
        for file in os.listdir(self.apath):
            if file.endswith('.tif'):
                imgs.append(io.imread(os.path.join(self.apath, file)))
            if len(imgs) > self.args.imageLim and self.args.imageLim != 0:
                break

        # fill the high res and low res
        for i in range(len(imgs)):
            img = imgs[i]
            img_split = [img[:2000, :2000], img[:2000, 2000:4000], img[2000:4000, :2000], img[2000:4000, 2000:4000]]
            for j in range(4):
                img_temp = img_split[j]
                img_temp_down = transform.downscale_local_mean(img_temp, (6,1))
                img_temp_down = (255 * img_temp_down/img_temp_down.max()).astype('uint8')
                hr_list.append(img_temp)
                lr_list.append(img_temp_down)
        return hr_list, lr_list

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        hr = self.images_hr[idx]
        lr = self.images_lr[idx]

        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        return pair_t[0], pair_t[1]

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]

        if type(scale) is str:
            scale = int(scale)

        # print("Training?: ", self.train)
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale
            )
            if not self.args.no_augment: 
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr


    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)