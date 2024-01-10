import os
import utility
import utility2

from skimage import io

import torch
import torch.nn.utils as utils
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        self.idx_scale = 0
        self.epoch_limit = epoch_limit
        self.epoch_validationLoss = []
        self.epoch_validationPSNR = []
        self.epoch_trainLoss = []

        self.error_last = 1e8
    
    # splits the training and testing based off of the indices 
    # runs the training and testing accordingly
    def run(self): 
        train_files = []
        test_files = []

        # Get the training data
        self.loaderTot.dataset.set_as_training()

        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        for i in range(self.trainInd.shape[0] // self.args.batch_size):
            lr_batch = []
            hr_batch = []
            for j in range(self.args.batch_size):
                lr, hr = self.loaderTot.dataset[self.trainInd[i*self.args.batch_size+j]]

                lr = utility2.normalize(lr)
                hr = utility2.normalize(hr)

                lr_batch.append(lr)
                hr_batch.append(hr)
            lr_stack = torch.stack(lr_batch)
            hr_stack = torch.stack(hr_batch)
            train_files.append((lr_stack, hr_stack))

        # Get the testing data, but dont implement a batch size
        self.loaderTot.dataset.set_as_testing()
        for i in range(self.testInd.shape[0]):
            lr, hr = self.loaderTot.dataset[self.testInd[i]]
            lr = torch.unsqueeze(lr,0)
            hr = torch.unsqueeze(hr,0)

            lr = utility2.normalize(lr)
            hr = utility2.normalize(hr)
            
            test_files.append((lr,hr))
        
        self.trainTot = train_files
        self.testTot = test_files

        # for testing purposes
        '''for batch_idx, (lr,hr) in enumerate(self.trainTot):
            filenameLr = 'LR' + str(batch_idx)
            filenameHr = 'HR' + str(batch_idx)

            # saving the images
            normalizedLr = lr.mul(255 / self.args.rgb_range)
            normalizedHr = hr.mul(255 / self.args.rgb_range)

            image_arrayLr = np.squeeze(normalizedLr.cpu().numpy())
            image_arrayHr = np.squeeze(normalizedHr.cpu().numpy())

            # tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            io.imsave(os.path.join(self.args.dir_data, 'test', '{}.tiff'.format(filenameLr)), image_arrayLr.astype(np.uint8))
            io.imsave(os.path.join(self.args.dir_data, 'test', '{}.tiff'.format(filenameHr)), image_arrayHr.astype(np.uint8))'''

        self.train()
        self.test()

# training and validation log output
    def train(self):
        self.best_validation_average = 1e8
        print("Training starting...")
        self.ckp.write_log('\nTraining:')

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

            # initialization of loss log
            self.loss.start_log()
            self.trainLoss = []
                    
            # set model to train where there is possibility of test
            # train at the end of the validation
            torch.set_grad_enabled(True)
            self.model.train()  # Set the model back to training mode
            
            # test
            self.optimizer.schedule()
            
            timer_data, timer_model = utility.timer(), utility.timer()

            # potentially write this to a txt file
            pbar = tqdm(train_data, total=len(train_data), desc=f"Epoch {epoch}", unit="batch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            for batch_idx, (lr, hr) in enumerate(pbar):
                
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
                    
                timer_data.tic()

                # logging the training
                pbar.set_postfix({"Loss": loss.cpu().detach().numpy()})
                self.trainLoss.append(loss.cpu().detach().numpy())
            
            self.epoch_trainLoss.append(loss.cpu().detach().numpy())
            # self.loss.end_log(len(train_data))
            # what we want 
            self.error_last = self.loss.log[-1, -1]

            # first instance of saving the loss
            self.loss_path = os.path.join(self.args.dir_data, 'loss')
            os.makedirs(self.loss_path, exist_ok=True)
            
            # will be one point on the graph of total
            self.validate_train(validation_data, len(train_data), epoch)
            self.ckp.write_log(f'Epoch {epoch} -> Validation Loss: {np.average(self.validateLossTot):.4f}, Training Loss: {np.average(self.epoch_trainLoss):.4f}, PSNR: {np.average(self.validatePSNRtot):.4f}')

            # potential save depending on how we want 
            '''loss_to_save = np.around(np.vstack((self.epoch_trainLoss, self.epoch_validationLoss)).T, 4)
            header = "Train,Validation"

            np.savetxt(os.path.join(self.loss_path, f'lossLog.csv'), loss_to_save, delimiter=",", header=header, fmt="%.4f")'''
        
        # update the csv/txt file with the average loss for each epoch
        # Training finished
        # save the graph of the total epochs 
            
        self.loss.saveLoss(self.epoch_limit, self.loss_path, self.epoch_trainLoss, self.epoch_validationLoss, True)
        self.ckp.plot_psnr(self.epoch_validationPSNR, self.epoch_limit)


    # validation in the training
    def validate_train(self, validate_data, trainlength, epoch):
        torch.set_grad_enabled(False)

        # complete validation on the 10%
        # self.loss.start_log()
        print("Validation Loss Log Reached")
        self.validateLossTot = []
        self.validatePSNRtot = []

        # the weights wont be updated 
        self.model.eval()

        timer_data, timer_model = utility.timer(), utility.timer()

        # looping through the validation 
        for batch_idx, (lr,hr) in enumerate(validate_data):

            lr, hr = self.prepare(lr,hr)

            with torch.no_grad():
                timer_data.hold()
                timer_model.tic()

                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)
                psnr = self.ckp.calc_psnr(sr, hr, self.args.scale, 255)

                self.validatePSNRtot.append(psnr)
                self.validateLossTot.append(loss.cpu().numpy())

        # adding the average 
        self.epoch_validationLoss.append(np.average(self.validateLossTot))
        self.epoch_validationPSNR.append(np.average(self.validatePSNRtot))

        # Plot for one epoch, plotting one for every printevery, as well as the first one
        # will also save the model and plot psnr
        if((epoch) % self.args.print_every == 0) or (epoch == 1):
            self.loss.saveLoss(epoch, self.loss_path, self.trainLoss, self.validateLossTot, False, trainlength)
            self.model.save(self.args.dir_data, epoch)

        # check for best model
        if self.epoch_validationLoss[-1] < self.best_validation_average:
            self.best_average = self.epoch_validationLoss[-1]
            self.model.save(self.args.dir_data, epoch, is_best=True)


    # this will both save the model and test on the test images
    def test(self):
        print("Testing starting...")
        self.ckp.write_log('\nTesting:')

        # load in the best model
        # if loading in pretrained model, set pre_train to model path
        modelPath = os.path.join(self.args.dir_data, 'model')
        self.model.load(modelPath)

        torch.set_grad_enabled(False)
        self.model.eval()

        timer_test = utility.timer()

        # initialization of loss log
        self.loss.start_log()

        test_data = self.testTot
        pbar = tqdm(test_data, total=len(test_data), desc=f"Testing", unit="batch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        scale = self.args.scale
        self.loaderTot.dataset.set_scale(scale)
        test_lossList = []
        for idx_data, (lr, hr) in enumerate(pbar): 

            lr, hr = self.prepare(lr,hr)

            # split the images into 4 and test on each of them
            hrList, lrList = self.ckp.test_split(hr, lr)

            # srList has each of the 4 images 
            # will stitch back together for saving as full image
            sr_list = []
            testLossTot = []

            for i in range(4):
                sr = self.model(lrList[i], scale)
                sr_list.append(sr)
                loss = self.loss(sr, hrList[i])
                testLossTot.append(loss) 

            # combine the sr list back into 1 before adding to savelist
            #srConcate = torch.cat(sr_list, dim=2)
            srConcate = torch.cat([torch.cat(sr_list[:2], dim=3), torch.cat(sr_list[2:], dim=3)], dim=2)
            # Resize the concatenated HR image to the original size
            srConcate = F.interpolate(srConcate, size=(hr.size(2), hr.size(3)), mode='bicubic', align_corners=True)
            
            losses = [loss.cpu().numpy() for loss in testLossTot]
            test_lossList.append(np.average(losses))
            #save_list = [utility2.unnormalize(srConcate),
            #             utility2.unnormalize(lr),
            #             utility2.unnormalize(hr)]
            save_list = [srConcate, lr, hr]

            # saves the results in the designated folders
            if self.args.save_results:
                self.ckp.save_results(save_list, idx_data, loss)
        
        elapsed_time = timer_test.toc()

        self.ckp.write_log(f"Test Loss: {np.average(test_lossList):.4f}, Batch Size: {self.args.batch_size}, Time Taken: {elapsed_time:.2f} seconds" + '\n')
        '''saveTest = np.around(np.vstack((np.average(test_lossList), self.args.batch_size, elapsed_time)).T, 4)
        header = "Test Loss, Batch Size, Time Taken"

        results_file_path = os.path.join(self.args.dir_data, 'test', f'results.csv')

        # Check if the file already exists
        if os.path.exists(results_file_path):
            with open(results_file_path, 'a') as file:
                np.savetxt(file, saveTest, delimiter=",", fmt="%.4f")
        else:
            np.savetxt(results_file_path, saveTest, delimiter=",", header=header, comments='', fmt="%.4f")
        
        # saves the model, loss, and the pnsr model
        #if not self.args.test_only:
        #    self.ckp.save(self, epoch)'''

        torch.set_grad_enabled(True)


    def prepare(self, lr, hr):
         # defining the device without the parallel processing in the given function
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        
        lr = lr.to(device)
        hr = hr.to(device)



        return lr, hr  
