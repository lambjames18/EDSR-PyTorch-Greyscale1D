import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_list = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == "GV":
                gv = import_module('loss.gradient_variance_loss')
                loss_function = gv.GradientVariance(args.patch_size, args.cpu)
            elif loss_type == 'G':
                g = import_module('loss.g_loss')
                loss_function = g.GLoss(args.cpu)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        #print("Losses: ", losses)
        #print("Loss sum: ", loss_sum)

        return loss_sum
    
    # will plot the loss for each epoch, as well as the total
    def saveLoss(self, epoch, loss_path, trainLoss, validateLoss = [], isTot = False, trainlength = 0):
        fig = plt.figure()

        # if it is the totalLoss, plot both training and validation
        # if not, just plot the training loss
        if isTot:
            plt.title(f"Total epochs")
            fileName = 'totalLoss.pdf'
            xLabel = 'Epochs'
            xRange =  np.arange(epoch) + 1
        else:
            plt.title(f"Loss Function epoch {epoch}")
            fileName = 'Epoch_{}.pdf'.format(epoch)
            xLabel = 'Batches'
            xRange = np.arange(0, trainlength)

        plt.plot(xRange, trainLoss, marker = 'o', color = 'blue', label = "Training")
        if isTot:
            plt.plot(xRange, validateLoss, marker = 'o', color = 'red', label = "Validation")
        plt.legend()
        plt.xlabel(xLabel)
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(loss_path, fileName))
        plt.close(fig)

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def get_loss(self):
        return self.loss_list

    def get_last_loss(self):
        return self.loss_list[-1]

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
    
        for l, c in zip(self.loss, self.log[-1]):
            self.loss_list.append((c / n_samples).numpy())
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    # plotting the batch vs the loss instead of the epoch (which is consistently 1)
    def plot_loss(self, apath, batch_idx):
    # def plot_loss(self, epoch):
        x_values = np.arange(1, len(self.log) + 1)
        loss_values = self.log.numpy()

        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            print(loss_values[:, i])
            plt.plot(x_values, loss_values[:, i], label=label, marker = 'o')
            plt.legend()
            plt.xlabel('Batches')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    # loss display adjusted for this data 
    #def plot_loss2(self, apath, batch_idx):
     #   axis
        

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

