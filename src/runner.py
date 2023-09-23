
import torch

import utility
import data
import model
import loss
import option_mod

# for testing the Train
from decimal import Decimal
import torch.nn.utils as utils
import matplotlib.pyplot as plt

from trainer_pollock import Trainer

# Act like this is the command line but bypass the commandline version so we can use a python script
#args = option_mod.parser.parse_args(["--dir_data", "/Users/anayakhan/Desktop/Pollock/dataset/pollockData", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "4"])
args = option_mod.parser.parse_args(["--dir_data", "C:/Users/Pollock-GPU/Documents/jlamb_code/SR-Data", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48"])
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
print("Test: ", loader.loader_test)
print("Train: ", loader.loader_train)
# This is a class that loads the model
_model = model.Model(args, checkpoint)
# This is a class that loads the loss function
if args.test_only:
    _loss = None
else:
    _loss = loss.Loss(args, checkpoint)

trainer = Trainer(args, loader, _model, _loss)
trainer.train()


exit()
# below is the working training loop 
# Lets just run the model once to get a loss value
# training
loaderTrain = loader.loader_train

# update learning rate at the beginning of new epoch
_loss.step()

# creating optimizer
optimizer = utility.make_optimizer(args, _model)

# logs current epoch number and learning rate from optimizer
epoch = optimizer.get_last_epoch() + 1

# getting the learning rate 
lr_rate = optimizer.get_lr()

# initialization of loss log
_loss.start_log()
print("Loss Log started")
        
# set model to train where there is possibility of test
_model.train()
print("Model to train reached")

timer_data, timer_model = utility.timer(), utility.timer()
print("Timer set")

# setting scale
loaderTrain.dataset.set_scale(0)
loss_list = []

# batch_idx, (lr, hr, _,) = next(enumerate(loaderTrain))
for batch_idx, (lr, hr, _,) in enumerate(loaderTrain):
    print("Epoch num: ", epoch)
    #print("LR Shape (Batch {}): {}".format(batch_idx, lr.shape))
    #print("HR Shape (Batch {}): {}".format(batch_idx, hr.shape))

    # defining the device without the parallel processing in the given function
    if args.cpu:
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

    optimizer.zero_grad()
    print("Optimizer zero_grad acheived")

    # forward pass with low res input and scale factor of zero
    sr = _model(lr, 0)
    loss = _loss(sr, hr)
    loss.backward()
 
    optimizer.step()
    timer_model.hold()

    # logging every training status currently
    checkpoint.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
    (batch_idx + 1) * args.batch_size,
    len(loaderTrain.dataset),
    _loss.display_loss(batch_idx),
    timer_model.release(),
    timer_data.release()))

    loss_list.append(_loss.get_loss())
    # the path to where to save the loss function
    apath = "C:/Users/Pollock-GPU/Documents/jlamb_code/SR-Data/loss"
    _loss.plot_loss(apath, batch_idx + 1)
    print("Made to plot")

    if(batch_idx == 5):
        print(loss_list)
        exit()

    print("Train status ", batch_idx + 1, " logged")

    timer_data.tic()

_loss.end_log(len(loaderTrain))
error_last = _loss.log[-1, -1]
optimizer.schedule()




