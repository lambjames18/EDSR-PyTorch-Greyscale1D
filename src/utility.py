import os
import math
import time
import datetime
from skimage import io
#from multiprocessing import Process
#from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from data import common


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

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

def bg_target(queue):
    while True:
        if not queue.empty():
            filename, tensor = queue.get()
            if filename is None: break
            imageio.imwrite(filename, np.squeeze(tensor.numpy()))

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            # make a test folder with the SR, HR, LR
            self.dir = os.path.join(self.args.dir_data, args.save)
            os.makedirs(self.dir, exist_ok = True)
            #self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join(self.args.dir_data, args.load)
            #self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        #os.makedirs(self.dir, exist_ok=True)
        #os.makedirs(self.get_path('model'), exist_ok=True)
        #for d in args.data_test:
        #    os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'

        with open(self.get_path('log.txt'), open_type) as f:
            f.write(now + '\n')
            f.write("Batch size: {}\n".format(args.batch_size))
            f.write("Loss Type: {}\n".format(args.loss.split('*')[1]))

        self.log_file = open(self.get_path('log.txt'), 'a')

        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        # changed from 8 to 1
        self.n_processes = 1

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)
    
    # change this so that the average loss is plotted
    def save(self, trainer, epoch):
        trainer.loss.save(self.dir)

        #self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        #print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, pnsrList, epochLim):
        axis = np.arange(epochLim) + 1
        
        label = 'SR for Epochs'
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, pnsrList, label='Scale {}'.format(self.args.scale), marker = 'o', color = 'pink')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(os.path.join(self.args.dir_data, 'test', 'psnr_log.png'))
        plt.close(fig)


    def save_results(self, save_list, index, loss=0, testOnly = False):
        if self.args.save_results:
            # save_list = common.get_patch(*save_list, patch_size=self.args.patch_size, scale=self.args.scale)
            # save list format: sr, lr, hr

            postfix = ['LR', 'SR']

            if not testOnly:
                postfix.append('HR')

            for i in postfix:
                path = os.path.join(self.args.dir_data, 'test', i)
                os.makedirs(path, exist_ok=True)
            
            for v, p in zip(save_list, postfix):
                
                filename = p + f'{index}'

                # unnormalize the image
                image_array = unnormalize(v).cpu().numpy()

                # tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                io.imsave(os.path.join(self.args.dir_data, 'test', p,  '{}.tiff'.format(filename)), image_array.astype(np.uint8))
                #self.queue.put(('{}{}.tiff'.format(filename, p), tensor_cpu))
            
            if not testOnly:
                with open(os.path.join(self.args.dir_data, 'test', 'SR_loss.txt'), 'a') as f:
                    f.write(f'{index} {np.round(loss.cpu(),3)}\n')


    # split the high and low res images into 4 to make them smaller
    def test_split(self, hr, lr):
        scale = int(self.args.scale)
        hr_split = [hr[:, :, :1000, :1000], hr[:, :, :1000, 1000:2000], hr[:, :, 1000:2000, :1000], hr[:, :, 1000:2000, 1000:2000]]
        lowResLim = (2000//scale)//2
        lr_split = [lr[:, :, :lowResLim, :1000], lr[:, :, :lowResLim, 1000:2000], lr[:, :, lowResLim:2000, :1000], lr[:, :, lowResLim:2000, 1000:2000]]
        return hr_split, lr_split
    
        
    # look into the conversion for greyscale
    def calc_psnr(self, sr, hr, scale, rgb_range, dataset=None):
        if type(scale) is not int:
            scale = int(scale)

        if hr.nelement() == 1: return 0

        sr = sr.mul(255 / self.args.rgb_range)

        diff = (sr - hr) / rgb_range
        if dataset and dataset.dataset.benchmark:
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        else:
            shave = scale + 6

        valid = diff[..., shave:-shave, shave:-shave]
        mse = valid.pow(2).mean()

        return -10 * math.log10(mse)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

# adding 1e-6 to all 0 values
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

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

