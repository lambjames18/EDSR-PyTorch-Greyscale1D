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
    def __init__(self, args, loader, my_model, my_loss):
        self.args = args
        self.loss = my_loss
        self.model = my_model
        # includes the total list of hr and low res
        self.loaderTot = loader.total_loader
        #self.loaderTest = loader.loader_test
        self.optimizer = utility.make_optimizer(self.args, self.model)
        self.checkpoint = utility.checkpoint(self.args)
    
    def train(self):
        # there are 4 loops for each image
        self.kFold = 4
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

        # testing the kfold
        # train_indices, test_indices = KFold(n_splits = self.kFold, shuffle=True, random_state=42).split(self.loaderTot)
        # print("KFold split", train_indices, test_indices)

        for batch_idx, (lr, hr, _,) in enumerate(self.loaderTot):
            print("Epoch num: ", epoch)
            exit()
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
