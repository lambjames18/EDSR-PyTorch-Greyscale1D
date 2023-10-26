import os
import utility

import torch
import torch.nn.utils as utils
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from data import common
import numpy as np

import random
from tqdm import tqdm

# steps: 
# fix loader reading 
# kfold validation holdout 
# for loops for 10 epochs
# separate into validation and train, assuming its already in the test and train
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, trainInd, testInd, epoch_limit):
        self.args = args
        self.scale = args.scale
        self.loss = my_loss
        self.model = my_model
        self.loaderTot = loader.total_loader
        self.trainTnd = trainInd
        self.testInd = testInd
        self.optimizer = utility.make_optimizer(self.args, self.model)
        self.checkpoint = utility.checkpoint(self.args)
        self.idx_scale = 0
        self.epoch_limit = epoch_limit
        self.epoch_averages = []
    
    # splits the training and testing based off of the indices 
    # runs the training and testing accordingly
    def run(self): 
        train_files = []
        test_files = []

        for batch_ind in range(len(self.loaderTot)):
            if batch_ind in self.trainTnd:
                self.loaderTot.dataset.set_as_training()
                (lr,hr) = self.loaderTot.dataset[batch_ind]
                train_files.append((lr,hr))
            elif batch_ind in self.testInd:
                (lr,hr) = self.loaderTot.dataset[batch_ind]
                test_files.append((lr,hr))
        
        self.trainTot = train_files
        self.testTot = test_files

        self.train()

    def train(self):
        # loop over 2 epochs
        epoch = self.optimizer.get_last_epoch() + 1

        # runs the test when the epoch has run as many times as needed 
        if(epoch > self.epoch_limit): 
            print("Made to the testing")
            # save the graph of the total epochs 
            fig = plt.figure()
            plt.title(f"Loss Function Total")
            plt.plot(epoch, self.epoch_averages, marker = 'o')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(self.args.loss_path, f'totalLoss.pdf'))
            plt.close(fig)
            exit()
            self.test()

        # taking the first ten percent of the training images as validation 
        # after validating in the validation function, shuffles and takes another 
        validate_ind = (len(self.trainTot)) // 10
        random.shuffle(self.trainTot)

        validation_data = self.trainTot[:validate_ind]
        train_data = self.trainTot[validate_ind:]

        self.loss.step()

        # getting the learning rate 
        lr_rate = self.optimizer.get_lr()

        # initialization of loss log
        self.loss.start_log()
        self.trainLoss = []
        print("Loss Log started")
                
        # set model to train where there is possibility of test
        self.model.train()
        print("Model to train reached")

        timer_data, timer_model = utility.timer(), utility.timer()
        print("Timer set")


        # batch_idx, (lr, hr, _,) = next(enumerate(loaderTrain))
        pbar = tqdm(train_data, total=len(train_data), desc=f"Epoch {epoch}", unit="batch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for batch_idx, (lr, hr) in enumerate(pbar):
        # for batch_idx, (lr, hr) in enumerate(train_data):
            lr = torch.unsqueeze(lr,0)
            hr = torch.unsqueeze(hr,0)

            lr, hr = self.prepare(lr,hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            #print("Optimizer zero_grad acheived")

            # forward pass with low res input and scale factor of zero
            # currently too large for the GPU to handle, potentially convert to binary
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
        
            self.optimizer.step()
            timer_model.hold()

            # logging every training status currently
            self.checkpoint.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
            (batch_idx + 1) * self.args.batch_size,
            len(train_data),
            self.loss.display_loss(batch_idx),
            timer_model.release(),
            timer_data.release()))

            # loss_list.append(self.loss.get_loss())
            pbar.set_postfix({"Loss": self.loss.get_last_loss()})
            self.trainLoss.append(self.loss.get_last_loss())

        #print("Train status ", batch_idx + 1, " logged")
        timer_data.tic()

        self.loss.end_log(len(train_data))
        error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        # will be one point on the graph of total
        self.epoch_averages.append(np.average(self.trainLoss))
        self.validate_train(validation_data, len(train_data), epoch)

    # validation in the training
    def validate_train(self, validate_data, trainlength, epoch):
        # complete validation on the 10%
        self.loss.start_log()
        print("Validation Loss Log started")
        self.validateLoss = []

        # the weights wont be updated 
        self.model.eval()

        timer_data, timer_model = utility.timer(), utility.timer()

        # looping through the validation 
        for batch_idx, (lr,hr) in enumerate(validate_data):
            lr = torch.unsqueeze(lr,0)
            hr = torch.unsqueeze(hr,0)

            lr, hr = self.prepare(lr,hr)

            with torch.no_grad():
                timer_data.hold()
                timer_model.tic()

                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)

                # logging the validation
                self.checkpoint.write_log('Validation: [{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch_idx+1),
                    len(validate_data),
                    self.loss.display_loss(batch_idx),
                    timer_model.release(),
                    timer_data.release()
                ))
                self.validateLoss.append(self.loss.get_last_loss())

        self.loss.end_log(len(validate_data))  # End loss logging for validation
        self.model.train()  # Set the model back to training mode
        self.optimizer.schedule()

        # for plotting the loss with the validation

        # printing both the validation and the training loss
        '''### Save the loss function
        # the path to where to save the loss function
        apath = "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR/loss/"
        # self.loss.plot_loss(apath, batch_idx + 1)
        # print("Made to plot")
        #print(self.loss.get_loss())
        x_values = np.arange(1, batch_idx + 2)
        y_values = self.loss.get_loss()
        #print("Y_values: ", y_values)
        # makeshift loss function save
        fig = plt.figure()
        plt.title(f"Loss Function epoch {epoch}")
        plt.plot(x_values, y_values, marker = 'o')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(apath, f'loss_{epoch}.pdf'))
        plt.close(fig)'''

        # saves the validation loss and training loss for epoch graph
        x_trainLoss = np.arange(0, trainlength)
        y_trainLoss = self.trainLoss

        x_validationLoss = np.arange(trainlength, trainlength + len(validate_data))
        y_validateLoss = self.validateLoss


        # Plot for one epoch
        fig = plt.figure()
        plt.plot(x_trainLoss, y_trainLoss, marker = 'o', color = 'red')
        plt.plot(x_validationLoss, y_validateLoss, marker = 'o', color = 'blue')
        plt.title(f"Loss Function epoch 1")
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.args.loss_path, f'loss_{epoch}.pdf'))
        plt.close(fig)

        # saves the average validation loss as the point for this epoch


        # train at the end of the validation
        self.train()

    def prepare(self, lr, hr):
         # defining the device without the parallel processing in the given function
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            #print("CUDA Available: ", torch.cuda.is_available())
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        
        lr = lr.to(device)
        hr = hr.to(device)

        return lr, hr 

    # test