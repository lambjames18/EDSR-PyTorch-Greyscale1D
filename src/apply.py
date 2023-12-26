import torch

import numpy as np

import utility
import data
import model
import loss
import option_mod

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trainer_pollock import Trainer
from sklearn.model_selection import KFold

# this script allows the user to run the program with specific parameters
# ask about batch size 
# ask for the file to the images 
# ask for number of epochs
# ask for number of splits 

print("Welcome to the SR program")
print("This program will allow you to run SR on your data with parameters of your choice")


