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
        
    
        self._set_filesystem(args.dir_data)

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
            
            self.images_hr, self.images_lr = [], []
            for h in list_hr:
                if(h != ''):
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(args.ext, h, b, verbose=True)
            for l in list_lr:
                if(l != ''):
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
        
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1]))
        )
        
        #return names_hr[:1], [lr_list[:1] for lr_list in names_lr]
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.abspath(dir_data)
        self.raw = os.path.join(self.apath, 'raw_2')

        # high resolution
        self.dir_hr = os.path.join(self.apath, 'HR_2')
        os.makedirs(self.dir_hr, exist_ok=True)

        self.dir_lr = os.path.join(self.apath, 'LR_2')
        os.makedirs(self.dir_lr, exist_ok=True)

        # fill the high res and low res
        for file in os.listdir(self.raw):
            img = io.imread(self.raw + "/" + file)

            # high
            top_left = img[:2000, :2000]
            io.imsave(os.path.join(self.dir_hr, file + '_tl.tiff'), top_left)
            # low
            top_leftLow = transform.downscale_local_mean(top_left, (4,1))
            top_leftLow = (255 * top_leftLow/top_leftLow.max()).astype('uint8')
            io.imsave(os.path.join(self.dir_lr, file + '_tlLow.tiff'), top_leftLow)

            # high
            top_right = img[:2000, 2000:4000]
            io.imsave(os.path.join(self.dir_hr, file + '_tr.tiff'), top_right)
            # low
            top_rightLow = transform.downscale_local_mean(top_right, (4,1))
            top_rightLow = (255 * top_rightLow/top_rightLow.max()).astype('uint8')
            io.imsave(os.path.join(self.dir_lr, file + '_trLow.tiff'), top_rightLow)

            # high
            bot_left = img[2000:4000, :2000]
            io.imsave(os.path.join(self.dir_hr, file + '_bl.tiff'), bot_left)
            # low
            bot_leftLow = transform.downscale_local_mean(bot_left, (4,1))
            bot_leftLow = (255 * bot_leftLow/bot_leftLow.max()).astype('uint8')
            io.imsave(os.path.join(self.dir_lr, file + '_blLow.tiff'), bot_leftLow)

            # high
            bot_right = img[2000:4000, 2000:4000]
            io.imsave(os.path.join(self.dir_hr, file + '_br.tiff'), bot_right)
            # low 
            bot_rightLow = transform.downscale_local_mean(bot_right, (4,1))
            bot_rightLow = (255 * bot_rightLow/bot_rightLow.max()).astype('uint8')
            io.imsave(os.path.join(self.dir_lr, file + '_brLow.tiff'), bot_rightLow)

        print("Made")

        # filling the low 
        

        #self.dir_lr = os.path.join(self.apath, 'LR')
        #os.makedirs(self.dir_lr, exist_ok=True)
        # self.dir_hr = os.path.join(self.apath, self.split, 'HR')
        #dir_hr = os.path.join(self.apath, dir_data)
        
        #self.dir_hr = os.path.join(self.apath, self.split, 'HR')
        # changed from LR_bicubic
        #self.dir_lr = os.path.join(self.apath, self.split, 'LR')
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
        # print("Shape of Pair[0]: ", pair[0].shape, "Length of Pair[1]: ", pair[1].shape)
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
            # print("Filename to be loaded: ", f_lr)
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
            #if not self.args.no_augment: 
            #    lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr


    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

