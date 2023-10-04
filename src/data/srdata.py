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
        
        # list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = self.fill_HR_LR()
        # print(self.images_hr)
        # print(self.images_lr)
        
    def set_as_training(self):
        self.train = True
        n_patches = self.args.batch_size * self.args.test_every
        n_images = len(self.args.data_train) * len(self.images_hr)
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
        self.ext = ('.tiff', '.tiff')

        # high resolution
        # self.dir_hr = os.path.join(self.apath, 'HR_2')
        #os.makedirs(self.dir_hr, exist_ok=True)

        # self.dir_lr = os.path.join(self.apath, 'LR_2')
        # os.makedirs(self.dir_lr, exist_ok=True)

        # self.fill_HR_LR()
    
        #self.dir_lr = os.path.join(self.apath, 'LR')
        #os.makedirs(self.dir_lr, exist_ok=True)
        # self.dir_hr = os.path.join(self.apath, self.split, 'HR')
        #dir_hr = os.path.join(self.apath, dir_data)
        
        #self.dir_hr = os.path.join(self.apath, self.split, 'HR')
        # changed from LR_bicubic
        #self.dir_lr = os.path.join(self.apath, self.split, 'LR')

    # fills the high res and low res folders from the raw
    # save to the high res and low res, dont save
    def fill_HR_LR(self):
        # list of images 
        imgs = []
        hr_list = []
        lr_list = []

        for file in os.listdir(self.raw):
            if file.endswith('.tif'):
                imgs.append(io.imread(os.path.join(self.raw, file)))

        # fill the high res and low res
        # count_train = 0
        for i in range(len(imgs)):
            img = imgs[i]
            img_split = [img[:2000, :2000], img[:2000, 2000:4000], img[2000:4000, :2000], img[2000:4000, 2000:4000]]
            for j in range(4):
                img_temp = img_split[j]
                img_temp_down = transform.downscale_local_mean(img_temp, (4,1))
                img_temp_down = (255 * img_temp_down/img_temp_down.max()).astype('uint8')
                hr_list.append(img_temp)
                lr_list.append(img_temp_down)
                '''io.imsave(os.path.join(self.dir_hr, f'{count_train}.tiff'), img_temp)
                io.imsave(os.path.join(self.dir_lr, f'{count_train}.tiff'), img_temp_down)'''
                # count_train += 1
        return hr_list, lr_list

    # dont care 
    '''def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)'''

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        # print("Lr for patch: ", lr)
        print("LR Shape: ", lr.shape)
        
        # print("Hr for patch: ", hr)
        print("HR Shape: ", hr.shape)
        pair = self.get_patch(lr, hr)
        print("Shape of Pair[0]: ", pair[0].shape, "Length of Pair[1]: ", pair[1].shape)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        # return pair_t[0], pair_t[1], filename
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

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        print("f_hr shape: ", f_hr.shape)
        # f_lr = self.images_lr[self.idx_scale][idx]
        
        f_lr = self.images_lr[idx]
        print("f_lr shape: ", f_lr.shape)

        # filename, _ = os.path.splitext(os.path.basename(f_hr))
        # if self.args.ext == 'img' or self.benchmark:


        # hr = imageio.imread(f_hr)
        f_hr = f_hr.reshape(f_hr.shape[0], f_hr.shape[1], 1)
        # lr = imageio.imread(f_lr)
        f_lr = f_lr.reshape(f_lr.shape[0], f_lr.shape[1], 1)


        '''elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            # print("Filename to be loaded: ", f_lr)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)'''

        return f_lr, f_hr

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]

        if type(scale) is str:
            scale = int(scale)

        print("Training?: ", self.train)
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

