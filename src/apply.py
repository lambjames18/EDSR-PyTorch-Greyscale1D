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
print("Please read directions carefully")
print()

# dictionary: {name, param}
params = {}
command = ""

print("Enter parameters one at a time below, enter 'run' to run the program, 'restart' to refill the arguments.")
print("Refer to option_mod for options and their defaults. WARNING: Case Sensitive")
print("Required: dir data (path to images), Recommended: batch size, scale, epochs")
print("Format example -> batch size, 4")
print()

# collecting the arguements from the user
while True:
    print("Current Arguements:", params)
    command = input("Insert: ")
    print()

    if command.strip() == "run": 
        if "dir data" not in params:
            print("Enter the path to the images to run, command: dir data")
            continue
        else:
            break

    if command.strip() == 'restart':
        params = {}
        continue
    
    command_split = command.split(",")

    if len(command_split) != 2:
        print("This input is invalid. Please try again")
        continue
    
    name = command_split[0].strip().replace(" ", "_")
    value = command_split[1].strip()

    params[name] = value

argsTot = []

for name in params:
    argsTot.append(f"--{name}")
    argsTot.append(params[name])

# parsing the arguements
args = option_mod.parser.parse_args(argsTot)
args = option_mod.format_args(args)

if not args.cpu and torch.cuda.is_available():
    USE_GPU = True
    torch.cuda.empty_cache()

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if not checkpoint.ok:
    raise Exception("Something is wrong with the arguments")

# defining data loader, model, loss
loader = data.Data(args)
_model = model.Model(args, checkpoint)
_loss = loss.Loss(args, checkpoint) 

# epoch and kfold
epoch_limit = args.epochs
splits = args.k_fold

kf = KFold(n_splits=splits)
X = np.zeros(len(loader.total_loader.dataset.images_hr))

# running the training
for fold, (trainInd, testInd) in enumerate(kf.split(X)):
    print(f'Fold {fold}')
    print("-----------------------------------")

    print("Test indices: ", testInd)
    print("Train indices: ", trainInd)
    
    trainer = Trainer(args, loader, _model, _loss, trainInd, testInd, epoch_limit, checkpoint)
    trainer.run()
    exit()