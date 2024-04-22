import os
import time
import torch
import numpy as np
from skimage import io


# tracks how long the training and testing take
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff


def normalize(image, improve_contrast=True):
    if improve_contrast:
        return 2*((image - image.min())/(image.max() - image.min())) - 1
    else:
        return 2*(image/image.max()) - 1


def unnormalize(image, bit_depth=8):
    image = (image - image.min())/(image.max() - image.min())
    if bit_depth == 8:
        return torch.round(image*255).to(torch.uint8)
    elif bit_depth == 16:
        return torch.round(image*65535).to(torch.uint16)
    else:
        raise Exception("Invalid bit depth")


class log():
    def __init__(self, args):
        # create the log file 
        self.args = args
    
    # saves the images to the test folder
    def save_results(self, save_list, index, loss):
        if self.args.save_results:
            # save list format: sr, lr, hr
            postfix = ('SR', 'LR', 'HR')

            for i in postfix:
                path = os.path.join(self.args.dir_data, 'test', i)
                os.makedirs(path, exist_ok = True)
            
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):

                filename = 'results-{}'.format(index)

                if p == 'SR': 
                    filename += '_loss-{}'.format(np.round(loss.cpu(),3))

                normalized = v[0].mul(255 / self.args.rgb_range)

                image_array = np.squeeze(normalized.cpu().numpy())

                io.imsave(os.path.join(self.args.dir_data, 'test', p,  '{}.tiff'.format(filename)), image_array.astype(np.uint8))

    # split the high and low res images into 4 to make them smaller
    def test_split(self, hr, lr):
        scale = int(self.args.scale)
        test = hr[:, :, :1000, :1000]
        hr_split = [hr[:, :, :1000, :1000], hr[:, :, :1000, 1000:2000], hr[:, :, 1000:2000, :1000], hr[:, :, 1000:2000, 1000:2000]]
        lowResLim = (2000//scale)//2
        lr_split = [lr[:, :, :lowResLim, :1000], lr[:, :, :lowResLim, 1000:2000], lr[:, :, lowResLim:2000, :1000], lr[:, :, lowResLim:2000, 1000:2000]]
        return hr_split, lr_split
    