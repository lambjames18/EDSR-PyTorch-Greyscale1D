import os
import utility

import torch
import torch.nn.utils as utils
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

# steps: 
# fix loader reading 
# kfold validation holdout 
# for loops for 10 epochs
# separate into validation and train, assuming its already in the test and train
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, trainInd, testInd):
        self.args = args
        self.loss = my_loss
        self.model = my_model
        self.loaderTot = loader.total_loader
        self.trainTnd = trainInd
        self.testInd = testInd
        self.optimizer = utility.make_optimizer(self.args, self.model)
        self.checkpoint = utility.checkpoint(self.args)
    
    # splits the training and testing based off of the indices 
    # runs the training and testing accordingly
    def run(self): 
        train_files = []
        test_files = []

        for batch_ind in self.trainTnd:
            data_train = self.loaderTot.dataset[batch_ind]
            train_files.append(data_train)

        for batch_ind in self.testInd: 
            data_test = self.loaderTot.dataset[batch_ind]
            test_files.append(data_test)
        
        self.trainTot = train_files
        self.testTot = test_files

        self.train()
        # eventually self.test()
    

    def train(self):
        self.loss.step()

        # logs current epoch number and learning rate from optimizer
        epoch = self.optimizer.get_last_epoch() + 1

        # getting the learning rate 
        lr_rate = self.optimizer.get_lr()

        # initialization of loss log
        self.loss.start_log()
        print("Loss Log started")
                
        # set model to train where there is possibility of test
        self.model.train()
        print("Model to train reached")

        timer_data, timer_model = utility.timer(), utility.timer()
        print("Timer set")

        # setting scale
        # self.loaderTot.testset.set_scale(0)
        loss_list = []

        # batch_idx, (lr, hr, _,) = next(enumerate(loaderTrain))
        self.loaderTot.dataset.set_as_training()
        for batch_idx, (lr, hr) in enumerate(self.trainTot):
            print("Lr shape: ", lr.shape)
            print("Hr shape: ", hr.shape)
            print("Epoch num: ", epoch)
            
            #print("LR Shape (Batch {}): {}".format(batch_idx, lr.shape))
            #print("HR Shape (Batch {}): {}".format(batch_idx, hr.shape))

            # defining the device without the parallel processing in the given function
            if self.args.cpu:
                device = torch.device('cpu')
            else:
                print("CUDA Available: ", torch.cuda.is_available())
                if torch.backends.mps.is_available():
                    device = torch.device('mps')
                elif torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')


            # processed with the determined device, in this case cpu 
            lr = lr.to(device)
            hr = hr.to(device)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            print("Optimizer zero_grad acheived")

            # forward pass with low res input and scale factor of zero
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
        
            self.optimizer.step()
            timer_model.hold()

            # logging every training status currently
            self.checkpoint.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
            (batch_idx + 1) * self.args.batch_size,
            len(self.loaderTrain.dataset),
            self.loss.display_loss(batch_idx),
            timer_model.release(),
            timer_data.release()))

            loss_list.append(self.loss.get_loss())
            # the path to where to save the loss function
            apath = "C:/Users/Pollock-GPU/Documents/jlamb_code/SR-Data/loss"
            # self.loss.plot_loss(apath, batch_idx + 1)
            # print("Made to plot")

            if(batch_idx == 20):
                print(self.loss.get_loss())
                x_values = np.arange(1, batch_idx + 2)
                y_values = self.loss.get_loss()

                # makeshift loss function save
                fig = plt.figure()
                plt.title("Loss Function epoch 1")
                plt.plot(x_values, y_values, marker = 'o')
                plt.xlabel('Batches')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.savefig(os.path.join(apath, 'loss_1.pdf'))
                plt.close(fig)


                exit()

        print("Train status ", batch_idx + 1, " logged")
        timer_data.tic()


        self.loss.end_log(len(self.loaderTrain))
        error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
