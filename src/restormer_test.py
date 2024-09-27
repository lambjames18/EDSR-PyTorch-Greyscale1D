import utility
import data
import model
import loss
import option_mod

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

import os


def prepare(lr, hr, args):
    # defining the device without the parallel processing in the given function
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


def calc_psnr(sr, hr, scale=4, data_range=None):
    if data_range is None:
        data_range = hr.max() - hr.min()
    diff = (sr - hr) / data_range
    shave = scale + 6
    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


# Get the image to look at
img = io.imread("F:/WCu-Data-SR/8119WCu/BSE/images/Raw/8.tif")[300:700, 300:700]
#filePath1 = "F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/all_phases_greyscale/"
#img = io.imread(img + "3.tif")[100:1100, 100:1100]
img_down = transform.downscale_local_mean(img, (4, 1))
img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
img_down = np.expand_dims(np.expand_dims(img_down, axis=0), axis=0)
img = torch.from_numpy(img).float()
img_down = torch.from_numpy(img_down).float()
img = utility.normalize(img)
img_down = utility.normalize(img_down)

filePath = "F:/WCu-Data-SR/8119WCu/BSE/"
#filePath2 = "F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/W_phase_only_(greyscale)/"
#"F:/WCu-Data-SR/5842WCu_Images/MaxPool_MSE_100E_GLoss/model/model_best.pt" 
edsrPath = "F:/WCu-Data-SR/8119WCu/BSE/train/EDSR/model/model_best.pt" 
#edsrPath = filePath2 + "/pretrained/restormer/model/model_best.pt"
# "F:/WCu-Data-SR/5842WCu_Images/model/model_best.pt"
#preTrain = edsrPath
preTrain = "F:/WCu-Data-SR/8119WCu/BSE/train/Restormer/model/model_best.pt"
#preTrain = filePath + "/pretrained/improved_contrast/model/model_best.pt"

args = option_mod.parser.parse_args(["--dir_data", filePath, "--scale", "4", "--save_results" ,"--n_colors", "1", "--n_axis", "1", "--imageLim", "10",
                                     "--batch_size", "2", "--n_GPUs", "1", "--patch_size", "48", "--loss", "1*G", "--model", "Restormer",
                                     "--EDSR_path", edsrPath, "--pre_train", preTrain])
args = option_mod.format_args(args)
if not args.cpu and torch.cuda.is_available():
    USE_GPU = True
    torch.cuda.empty_cache()

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
restormer = model.Model(args, checkpoint)
torch.set_grad_enabled(False)
restormer.eval()

lr, hr = prepare(img_down, img, args)
sr = restormer(lr, int(args.scale))
print("Restormer PSNR:", calc_psnr(sr, hr).item())

io.imsave(filePath + "Restormer_Test/SR-Restormer.png", utility.unnormalize(sr[0, 0]).cpu().numpy())
#io.imsave(filePath + "/restormerTest/HR-Restormer.png", utility.unnormalize(hr[0, 0]).cpu().numpy())


#filePath = "F:/WCu-Data-SR/5842WCu_Images/"
args = option_mod.parser.parse_args(["--dir_data", filePath, "--scale", "4", "--save_results" ,"--n_colors", "1", "--n_axis", "1", "--imageLim", "10",
                                     "--batch_size", "2", "--n_GPUs", "1", "--patch_size", "48", "--loss", "1*G", "--model", "EDSR",
                                     "--pre_train", edsrPath])
args = option_mod.format_args(args)
if not args.cpu and torch.cuda.is_available():
    USE_GPU = True
    torch.cuda.empty_cache()

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
edsr = model.Model(args, checkpoint)
torch.set_grad_enabled(False)
edsr.eval()

lr, hr = prepare(img_down, img, args)
sr = edsr(lr, int(args.scale))
print("EDSR PSNR:", calc_psnr(sr, hr).item())

io.imsave(filePath + "/Restormer_Test/SR-EDSR.png", utility.unnormalize(sr[0, 0]).cpu().numpy())
io.imsave(filePath + "/Restormer_Test/HR.png", utility.unnormalize(hr[0, 0]).cpu().numpy())

