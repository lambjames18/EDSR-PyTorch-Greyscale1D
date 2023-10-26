
import torch
import os

import numpy as np

import utility
import data
import model
import loss
import option_mod

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# for testing the Train
from decimal import Decimal
import torch.nn.utils as utils
import matplotlib.pyplot as plt

from trainer_pollock import Trainer
from sklearn.model_selection import KFold

# Act like this is the command line but bypass the commandline version so we can use a python script
# args = option_mod.parser.parse_args(["--dir_data", "/Users/anayakhan/Desktop/Pollock/dataset/pollockData", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "4"])
# args = option_mod.parser.parse_args(["--dir_data", "C:/Users/Pollock-GPU/Documents/jlamb_code/SR-Data", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48"])
args = option_mod.parser.parse_args(["--dir_data", "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48", "--loss_path", "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR/loss/"])
#args = option_mod.parser.parse_args(["--dir_data", "/Users/anayakhan/Desktop/Pollock/dataset/pollockData", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48"])
args = option_mod.format_args(args)
if not args.cpu and torch.cuda.is_available():
    USE_GPU = True
    torch.cuda.empty_cache()

# Just setting the seed for random variables
torch.manual_seed(args.seed)
# This is just a class that creates a checkpoint directory and saves the args to a config file
# Theres more to this class but I don't think it's important for now
checkpoint = utility.checkpoint(args)

# Make sure there's nothing wrong with the arguments and run if there isn't
if not checkpoint.ok:
    raise Exception("Something is wrong with the arguments")

# This is a class that loads the data
loader = data.Data(args)  # loader needs to have two attributes: loader_train and loader_test

# This is a class that loads the model
_model = model.Model(args, checkpoint)

# This is a class that loads the loss function
# we wont use test only
'''if args.test_only:
    _loss = None
else:
    _loss = loss.Loss(args, checkpoint)'''

_loss = loss.Loss(args, checkpoint)
epoch_limit = 5

# seeing if the old works 
trainer = Trainer(args, loader, _model, _loss)
exit()

# creating the training object
# kfold here, before running train class 

# runs a new trainer for each set of indices returned
splits = 5
kf = KFold(n_splits=splits)
print(len(loader.total_loader))
#loader.total_loader.dataset.set_as_training()
# list of 0s 
#X = [(lr, hr) for (lr, hr) in loader.total_loader]
X = np.zeros(len(loader.total_loader))


for fold, (trainInd, testInd) in enumerate(kf.split(X)):
    print(f'Fold {fold}')
    print("-----------------------------------")

    print("Test indices: ", testInd)
    print("Train indices: ", trainInd)
    
    trainer = Trainer(args, loader, _model, _loss, trainInd, testInd, epoch_limit)
    # beginning with running the training
    trainer.run()

# train_indices, test_indices = KFold(n_splits = kFold, shuffle=True, random_state=42).split(loader.total_loader.dataset)
# print("KFold split", train_indices, test_indices)
# print( KFold(n_splits = kFold, shuffle=True, random_state=42).split(loader.total_loader.dataset))

#trainer = Trainer(args, loader, _model, _loss)
# training the data
# trainer.train()
