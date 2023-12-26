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

print()
print("Welcome to the SR program!")
print("This program will allow you to run SR on your data with parameters of your choice")
print()

params = []

print("Enter parameters below, D for default")
scale = input("Scale (default - 4): ")
batch = input("Batch Size (default - 4): ")
epoch = input("Epochs Run (default - 15): ")
split = input("Kfold split (default - 5): ")
path = input("Path to your greyscale images (required): ")

print("Are there any other parameters you would like to set? (Refer to the option_mod file for options)")
rest_args = input("Format Example -> '--scale,4' : ")

args = option_mod.parser.parse_args('''Enter the arguements''')

def parseArgs(input):
    print("Parsing arg: ", input)
    