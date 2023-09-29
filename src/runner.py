
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

# Act like this is the command line but bypass the commandline version so we can use a python script
args = option_mod.parser.parse_args(["--dir_data", "/Users/anayakhan/Desktop/Pollock/dataset/pollockData", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "4"])
#args = option_mod.parser.parse_args(["--dir_data", "C:/Users/Pollock-GPU/Documents/jlamb_code/SR-Data", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48"])
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

# creating the training object
trainer = Trainer(args, loader, _model, _loss)
# training the data
trainer.train()
