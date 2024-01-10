
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
# this command line uses a batch size of 8 
args = option_mod.parser.parse_args(["--dir_data", "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR", "--scale", "4", "--save_results" ,"--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48"])
#args = option_mod.parser.parse_args(["--dir_data", "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "1", "--n_GPUs", "1", "--patch_size", "48", "--loss_path", "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR/loss/"])
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
loader = data.Data(args)

# This is a class that loads the model
_model = model.Model(args, checkpoint)

_loss = loss.Loss(args, checkpoint) 

# seeing if the older version works
epoch_limit = 700

# runs a new trainer for each set of indices returned
splits = 5
kf = KFold(n_splits=splits)

#loader.total_loader.dataset.set_as_training()
# list of 0s 
#X = [(lr, hr) for (lr, hr) in loader.total_loader]
X = np.zeros(len(loader.total_loader.dataset.images_hr))


for fold, (trainInd, testInd) in enumerate(kf.split(X)):
    checkpoint.write_log(f"Fold {fold}" + '\n' + "-----------------------------------")
    print(f'Fold {fold}')
    print("-----------------------------------")

    print("Test indices: ", testInd)
    print("Train indices: ", trainInd)
    
    trainer = Trainer(args, loader, _model, _loss, trainInd, testInd, epoch_limit, checkpoint)
    # beginning with running the training
    trainer.run()
    exit()
    """
    ### testing the model and getting the output
    trainer.test() # output one image and get loss values for all testin images
    loss_values = trainer.test_loss_values
    print("Average loss: ", np.mean(loss_values), np.std(loss_values))
    with open("loss.txt", "a") as output_file:
        output_file.write(f"{fold} {np.mean(loss_values)} {np.std(loss_values)}\n")
    """

    

# train_indices, test_indices = KFold(n_splits = kFold, shuffle=True, random_state=42).split(loader.total_loader.dataset)
# print("KFold split", train_indices, test_indices)
# print( KFold(n_splits = kFold, shuffle=True, random_state=42).split(loader.total_loader.dataset))

#trainer = Trainer(args, loader, _model, _loss)
# training the data
# trainer.train()
