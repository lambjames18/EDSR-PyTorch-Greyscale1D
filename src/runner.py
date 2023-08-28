import torch

import utility
import data
import model
import loss
import option_mod
from trainer import Trainer

# Act like this is the command line but bypass the commandline version so we can use a python script
args = option_mod.parser.parse_args(["--data_test", "GreyScale", "--scale", "4", "--test_only", "--save_results", "--n_colors", "1", "--n_axis", "1"])
args = option_mod.format_args(args)

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
if args.test_only:
    _loss = None
else:
    _loss = loss.Loss(args, checkpoint)

# Lets just run the model once to get a loss value
exit()
# Now we can train and test the model
# t = Trainer(args, loader, _model, _loss, checkpoint)
# while not t.terminate():
#     t.train()
#     t.test()
#
# checkpoint.done()

