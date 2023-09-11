import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data

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
        
    
        self._set_filesystem(args.dir_data)

        # testing
        # print("List of files in hr ", os.listdir(self.dir_hr))
        
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            #path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)
        
        list_hr, list_lr = self._scan()
        
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            os.makedirs(
                self.dir_lr.replace(self.apath, path_bin),
                exist_ok=True
            )

            '''for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )'''
            
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) 
            for l in list_lr:
                b = l.replace(self.apath, path_bin)
                b = b.replace(self.ext[1], '.pt')
                self.images_lr.append(b)
                self._check_and_load(args.ext, l, b, verbose=True) 
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )

        names_lr = [[] for _ in self.scale]
        
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0]))
        )
        
        #return names_hr[:1], [lr_list[:1] for lr_list in names_lr]
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.abspath(dir_data)
        self.dir_hr = os.path.join(self.apath, self.split, 'HR')
        # changed from LR_bicubic
        self.dir_lr = os.path.join(self.apath, self.split, 'LR')
        self.ext = ('.tiff', '.tiff')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        print("Shape of Pair[0]: ", pair[0].shape, "Length of Pair[1]: ", pair[1].shape)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        return pair_t[0], pair_t[1], filename

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

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        # f_lr = self.images_lr[self.idx_scale][idx]
        f_lr = self.images_lr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            hr = hr.reshape(hr.shape[0], hr.shape[1], 1)
            lr = imageio.imread(f_lr)
            lr = lr.reshape(lr.shape[0], lr.shape[1], 1)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

