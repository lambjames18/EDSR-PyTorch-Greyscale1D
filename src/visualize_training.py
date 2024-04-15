import torch
import os

import numpy as np
from skimage import io, transform

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

def prepare(lr, hr, args):
    lr = torch.from_numpy(lr.reshape((1, 1,) + lr.shape)).float()
    hr = torch.from_numpy(hr.reshape((1, 1,) + hr.shape)).float()
    if args.cpu:
        device = torch.device('cpu')
    else:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    lr = lr.to(device)
    hr = hr.to(device)
    return lr, hr

args = option_mod.parser.parse_args(["--dir_data", "F:/WCu-Data-SR", "--scale", "4", "--save_results" ,"--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48"])
args = option_mod.format_args(args)
if not args.cpu and torch.cuda.is_available():
    USE_GPU = True
    torch.cuda.empty_cache()

checkpoint = utility.checkpoint(args)
model = model.Model(args, checkpoint)
loss = loss.Loss(args, checkpoint)
loss.start_log()

directory = "F:/WCu-Data-SR/Results/Trained_Model/model/"
model_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f != "model_best.pt"]
model_paths = sorted(model_paths, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))

HR = io.imread("F:/WCu-Data-SR/Slice000.tif")[500:1500, 500:1500].astype(np.float32)
HR = (HR - np.min(HR)) / (np.max(HR) - np.min(HR))
LR = transform.downscale_local_mean(HR, (int(args.scale),1))
LR, HR = prepare(LR, HR, args)

print(LR.shape, HR.shape)

for model_path in model_paths:
    name = model_path.split("/")[-1].split(".")[0]
    print(name)
    model.load("", pre_train=model_path)
    torch.set_grad_enabled(False)
    model.eval()

    SR = model(LR, args.scale)
    loss_val = np.around(loss(SR, HR).cpu().detach().numpy(), 4)

    _hr = HR[0, 0, :, :].detach().cpu().numpy()
    _lr = LR[0, 0, :, :].detach().cpu().numpy()
    _lr = np.repeat(_lr, args.scale, axis=0)
    _sr = SR[0, 0, :, :].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].imshow(_hr, cmap='gray')
    ax[1].imshow(_lr, cmap='gray')
    ax[2].imshow(_sr, cmap='gray')
    ax[0].set_title("HR")
    ax[1].set_title("LR")
    ax[2].set_title("SR")
    fig.suptitle(f"{name} ({str(loss_val)})")
    plt.tight_layout()
    plt.savefig(f"F:/WCu-Data-SR/visualize_training/{name}.png")
    plt.close()
