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
from decimal import Decimal

import random
from tqdm import tqdm

# steps: 
# fix loader reading 
# kfold validation holdout 
# for loops for 10 epochs
# separate into validation and train, assuming its already in the test and train
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, trainInd, testInd, epoch_limit, checkpoint):
        self.args = args
        self.scale = args.scale
        self.loss = my_loss
        self.model = my_model
        self.ckp = checkpoint
        self.loaderTot = loader.total_loader
        self.trainInd = trainInd
        self.testInd = testInd
        self.optimizer = utility.make_optimizer(self.args, self.model)
        self.checkpoint = utility.checkpoint(self.args)
        self.idx_scale = 0
        self.epoch_limit = epoch_limit
        self.epoch_averages_validation = []
        self.epoch_averages_train = []
    
    # splits the training and testing based off of the indices 
    # runs the training and testing accordingly
    def run(self): 
        train_files = []
        test_files = []

        # Get the training data
        self.loaderTot.dataset.set_as_training()
        for i in self.trainInd // self.args.batch_size:
            lr_batch = []
            hr_batch = []
            for j in range(self.args.batch_size):
                lr, hr = self.loaderTot.dataset[i*self.args.batch_size+j]
                lr_batch.append(lr)
                hr_batch.append(hr)
            lr_stack = torch.stack(lr_batch)
            hr_stack = torch.stack(hr_batch)
            train_files.append((lr_stack, hr_stack))

        # Get the testing data, but dont implement a batch size
        self.loaderTot.dataset.set_as_testing()
        for i in self.testInd // self.args.batch_size:
            lr, hr = self.loaderTot.dataset[i]
            lr = torch.unsqueeze(lr,0)
            hr = torch.unsqueeze(hr,0)
            test_files.append((lr,hr))

        # for batch_ind in range(len(self.loaderTot)):
        #     if batch_ind in self.trainTnd:
        #         self.loaderTot.dataset.set_as_training()
        #         (lr,hr) = self.loaderTot.dataset[batch_ind]
        #         train_files.append((lr,hr))
        #     elif batch_ind in self.testInd:
        #         (lr,hr) = self.loaderTot.dataset[batch_ind]
        #         test_files.append((lr,hr))
        
        self.trainTot = train_files
        self.testTot = test_files

        self.train()

    def train(self):
        # loop over 2 epochs
        for epoch in range(1, self.epoch_limit + 1):
            # epoch = self.optimizer.get_last_epoch() +1

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
            # train at the end of the validation
            torch.set_grad_enabled(True)
            self.model.train()  # Set the model back to training mode
            self.optimizer.schedule()
            print("Model to train reached")

            timer_data, timer_model = utility.timer(), utility.timer()
            print("Timer set")

            pbar = tqdm(train_data, total=len(train_data), desc=f"Epoch {epoch}", unit="batch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            for batch_idx, (lr, hr) in enumerate(pbar):
            # for batch_idx, (lr, hr) in enumerate(train_data):
                # lr = torch.unsqueeze(lr,0)
                # hr = torch.unsqueeze(hr,0)

                lr, hr = self.prepare(lr,hr)
                timer_data.hold()
                timer_model.tic()

                self.optimizer.zero_grad()

                # forward pass with low res input and scale factor of zero
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

                #print("Loss after callin the model: ", self.loss.get_last_loss())

                # loss_list.append(self.loss.get_loss())
                pbar.set_postfix({"Loss": self.loss.get_last_loss()})
                self.trainLoss.append(self.loss.get_last_loss())
                #print("Train status ", batch_idx + 1, " logged")
            
            timer_data.tic()
            self.epoch_averages_train.append(self.loss.get_last_loss())
            self.loss.end_log(len(train_data))
            error_last = self.loss.log[-1, -1]
            
            # will be one point on the graph of total
            self.validate_train(validation_data, len(train_data), epoch)
        
        # Training finished
        # save the graph of the total epochs 
        fig = plt.figure()
        plt.title(f"Total epochs")
        epoch_range = np.arange(self.epoch_limit) + 1
        plt.plot(epoch_range, self.epoch_averages_validation, marker = 'o', color = 'red', label = "Validation")
        plt.plot(epoch_range, self.epoch_averages_train, marker = 'o', color = 'blue', label = "Training")
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.args.loss_path, f'totalLoss.pdf'))
        plt.close(fig)

        # saving the model 
        self.model.save(self.args.dir_data, epoch)
        self.test()

    # validation in the training
    def validate_train(self, validate_data, trainlength, epoch):
        torch.set_grad_enabled(False)

        # complete validation on the 10%
        self.loss.start_log()
        print("Validation Loss Log started")
        self.validateLoss = []

        # the weights wont be updated 
        self.model.eval()

        timer_data, timer_model = utility.timer(), utility.timer()

        # looping through the validation 
        for batch_idx, (lr,hr) in enumerate(validate_data):
            # lr = torch.unsqueeze(lr,0)
            # hr = torch.unsqueeze(hr,0)

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

        # for plotting the loss with the validation

        # printing both the validation and the training loss

        # saves the validation loss and training loss for epoch graph
        x_trainLoss = np.arange(0, trainlength)
        y_trainLoss = self.trainLoss

        #x_validationLoss = np.arange(trainlength, trainlength + len(validate_data))
        #y_validateLoss = self.validateLoss

        # adding the average 
        self.epoch_averages_validation.append(np.average(self.validateLoss))

        # Plot for one epoch, plotting one every 10
        
        if(epoch) % self.args.print_every == 0:
            fig = plt.figure()
            plt.plot(x_trainLoss, y_trainLoss, marker = 'o', color = 'red')
            plt.title(f"Loss Function epoch {epoch}")
            plt.xlabel('Batches')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(self.args.loss_path, f'Epoch_{epoch}.pdf'))
            plt.close(fig)
    
    # this will both save the model and test on the test images
    def test(self):
        print("Testing starting...")
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.model.eval()

        timer_test = utility.timer()

        # multiprocessing 
        #if self.args.save_results: self.ckp.begin_background()

        test_data = self.testTot
        pbar = tqdm(test_data, total=len(test_data), desc=f"Testing {epoch}", unit="batch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        scale = self.args.scale
        self.loaderTot.dataset.set_scale(scale)
        for idx_data, (lr, hr) in enumerate(pbar): 
            # lr = torch.unsqueeze(lr,0)
            # hr = torch.unsqueeze(hr,0)

            lr, hr = self.prepare(lr,hr)
            sr = self.model(lr, scale)
            sr = utility.quantize(sr, self.args.rgb_range)

            save_list = [sr]
            #self.ckp.log[-1, idx_data, scale] += utility.calc_psnr(
            #    sr, hr, scale, self.args.rgb_range, dataset=dataset
            #)
            if self.args.save_gt:
                save_list.extend([lr, hr])

            #if self.args.save_results:
            #    self.ckp.save_results(save_list)
        
        #self.ckp.log[-1, idx_data, scale] /= len(dataset)
        #best = self.ckp.log.max(0)
        #self.ckp.write_log(
        #    '[{:.3f} (Best: {:.3f} @epoch {})'.format(
        #        self.ckp.log[-1, idx_data, scale],
        #        best[0][idx_data, scale],
        #        best[1][idx_data, scale] + 1
        #    )
        #)

        self.ckp.write_log('Saving...')

        if not self.args.test_only:
            self.ckp.save(self, epoch)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

        

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