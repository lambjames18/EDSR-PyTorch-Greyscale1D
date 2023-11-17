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

        self.error_last = 1e8
    
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
            #lr = torch.unsqueeze(lr,0)
            #hr = torch.unsqueeze(hr,0)
            test_files.append((lr,hr))
        
        self.trainTot = train_files
        self.testTot = test_files

        self.train()
        self.test()

# training and validation log output
    def train(self):
        self.best_validation_average = 1e8
        # loop over 2 epochs
        for epoch in range(1, self.epoch_limit + 1):
            # taking the first ten percent of the training images as validation 
            # after validating in the validation function, shuffles and takes another 
            validate_ind = (len(self.trainTot)) // 10
            random.shuffle(self.trainTot)

            validation_data = self.trainTot[:validate_ind]
            train_data = self.trainTot[validate_ind:]

            self.loss.step()
            # getting the learning rate 
            lr_rate = self.optimizer.get_lr()

            # writing to the text file
            self.ckp.write_log(
                '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr_rate))
            )

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
                #lr = torch.unsqueeze(lr,0)
                #hr = torch.unsqueeze(hr,0)

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

                
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch_idx + 1) * self.args.batch_size,
                    len(train_data),
                    self.loss.display_loss(batch_idx),
                    timer_model.release(),
                    timer_data.release()))
                    
                timer_data.tic()

                #print("Loss after callin the model: ", self.loss.get_last_loss())

                # loss_list.append(self.loss.get_loss())
                pbar.set_postfix({"Loss": self.loss.get_last_loss()})
                self.trainLoss.append(self.loss.get_last_loss())
                #print("Train status ", batch_idx + 1, " logged")
            
            self.epoch_averages_train.append(self.loss.get_last_loss())
            self.loss.end_log(len(train_data))
            self.error_last = self.loss.log[-1, -1]
            
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

    # validation in the training
    def validate_train(self, validate_data, trainlength, epoch):
        torch.set_grad_enabled(False)

        # complete validation on the 10%
        self.loss.start_log()
        print("Validation Loss Log started")
        self.validateLoss = []

        self.ckp.write_log('\nValidation:')
        self.ckp.add_log(
            torch.zeros(1, len(validate_data), len(self.scale))
        )

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
                self.checkpoint.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch_idx+1),
                    len(validate_data),
                    self.loss.display_loss(batch_idx),
                    timer_model.release(),
                    timer_data.release()
                ))
                self.validateLoss.append(self.loss.get_last_loss())

        self.loss.end_log(len(validate_data))  # End loss logging for validation

        # saves training loss for epoch graph
        x_trainLoss = np.arange(0, trainlength)
        y_trainLoss = self.trainLoss

        # adding the average 
        self.epoch_averages_validation.append(np.average(self.validateLoss))

        # Plot for one epoch, plotting one every 10
        # will also save the model 
        if(epoch) % self.args.print_every == 0:
            fig = plt.figure()
            plt.plot(x_trainLoss, y_trainLoss, marker = 'o', color = 'red')
            plt.title(f"Loss Function epoch {epoch}")
            plt.xlabel('Batches')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(self.args.loss_path, f'Epoch_{epoch}.pdf'))
            plt.close(fig)

            # save the model checkpoint
            self.model.save(self.args.dir_data, epoch)

        # check for best model
        if self.epoch_averages_validation[-1] < self.best_validation_average:
            self.best_average = self.epoch_averages_validation[-1]
            self.model.save(self.args.dir_data, epoch, is_best=True)



    # this will both save the model and test on the test images
    def test(self):
        print("Testing starting...")

        # load in the best model
        # currently loads in the latest model
        # change to load in the best model
        self.model.load(self.args.dir_data)

        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.model.eval()

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.testTot), len(self.scale))
        )

        timer_test = utility.timer()

        test_data = self.testTot
        # only taking the first 2 training imagesß
        pbar = tqdm(test_data[:2], total=len(test_data), desc=f"Testing {epoch}", unit="batch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        scale = self.args.scale
        self.loaderTot.dataset.set_scale(scale)
        test_lossList = []
        for idx_data, (lr, hr) in enumerate(pbar): 
            # lr = torch.unsqueeze(lr,0)
            # hr = torch.unsqueeze(hr,0)

            lr, hr = self.prepare(lr,hr)
            sr = self.model(lr, scale)
            sr = utility.quantize(sr, self.args.rgb_range)
            loss = self.loss(sr, hr)
            test_lossList.append(loss) 

            save_list = [sr]
            
            # logging the psnr for one image
            self.ckp.log[-1, idx_data] += utility.calc_psnr(
                sr, hr, scale, self.args.rgb_range
            )

            # this adds the low resolution and high resolution images to the list
            if self.args.save_gt:
                save_list.extend([lr, hr])

            # saves the results in the designated folders
            if self.args.save_results:
                self.ckp.save_results(save_list, idx_data, loss)
            
        


        #self.ckp.write_log('Saving...')

        # saves the model, loss, and the pnsr model
        if not self.args.test_only:
            self.ckp.save(self, epoch)

        #self.ckp.write_log(
        #    'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        #)

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