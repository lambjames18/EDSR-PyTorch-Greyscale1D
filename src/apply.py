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

# dictionary: {name, param}
params = {}
command = ""

print("Enter parameters one at a time below, enter 'run' to run the program. Refer to option_mod for options and their defaults")
print("Required: path to images , Recommended: Batch Size, Scale, ")
print("Format example -> batch size, 4")


while True:
    print("Current Arguements: ", params)
    command = input("Insert: ")
    print()

    if command == "run": 
        break
    
    command_split = command.split(",")

    if len(command_split) != 2:
        print("This input is invalid. Please try again")
        continue
    
    name = "--" + command_split[0].replace(" ", "_")
    value = command_split[1]

    params[name] = value

    

