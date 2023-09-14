
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

from trainer import Trainer

# Act like this is the command line but bypass the commandline version so we can use a python script
# args = option_mod.parser.parse_args(["--dir_data", "/Users/anayakhan/Desktop/Pollock/dataset/pollockData", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "4"])
args = option_mod.parser.parse_args(["--dir_data", "C:/Users/Pollock-GPU/Documents/jlamb_code/SR-Data", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48"])
# args = option_mod.parser.parse_args(["--dir_data", "/Users/anayakhan/Desktop/Pollock/dataset/pollockData", "--scale", "4", "--save_results", "--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48"])
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
count = 0

# batch_idx, (lr, hr, _,) = next(enumerate(loaderTrain))
for batch_idx, (lr, hr, _,) in enumerate(loaderTrain):
    print("LR Shape (Batch {}): {}".format(batch_idx, lr.shape))
    print("HR Shape (Batch {}): {}".format(batch_idx, hr.shape))

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
 
    '''if args.gclip > 0:
        utils.clip_grad_value_(
            _model.parameters(),
            args.gclip
            )'''
    optimizer.step()
    timer_model.hold()

    # logging every training status currently
    checkpoint.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
    (batch_idx + 1) * args.batch_size,
    len(loaderTrain.dataset),
    _loss.display_loss(batch_idx),
    timer_model.release(),
    timer_data.release()))

    print("Train status ", count, " logged")

    timer_data.tic()
    count+=1

_loss.end_log(len(loaderTrain))
error_last = _loss.log[-1, -1]
optimizer.schedule()


# Now we can train and test the model
# t = Trainer(args, loader, _model, _loss, checkpoint)

'''class Train:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
    
    def train(self): # goal: plot the loss function
        # update learning rate at the beginning of new epoch
        self.loss.step()
        # logs current epoch number and learning rate from optimizer
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        # Printing learning rate
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        # initialization of loss log
        self.loss.start_log()
        print("Loss Log started")
        # set model to train where there is possibility of test
        self.model.train()
        print("Model to train reached")

        # times the process
        timer_data, timer_model = utility.timer(), utility.timer()
        print("Timer set")
        # TEMP
        self.loader_train.dataset.set_scale(0)
        print("loader_train rescaled to 0")
        count = 0
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            # to keep track of it running
            print("Loop number: ", count)
            
            # use of prepare function  for parallel processing 
            lr, hr = self.prepare(lr, hr)
            print("LR: ", lr)
            print("HR: ", hr)
            print()

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            print("Optimizer zero_grad acheived")

            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            # how many batches to wait before logging training status
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
            count += 1

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
    
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
    
    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]


# _loss = my_loss
# loss function for training
t = Trainer(args, loader, _model, _loss, checkpoint)

while not t.terminate():
    # beginning with training
    t.train()

checkpoint.done()'''

