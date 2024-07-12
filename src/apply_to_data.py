import os
import utility
import data
import model
import loss
import option_mod
import utility2
from tqdm.auto import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import InteractiveView

def prepare(im):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    im = im.to(device)
    return im


# Get the images
paths = [f for f in os.listdir("D:/Research/WCu/Data/3D/5842WCu_BSE/") if f.endswith(".tif")]
paths = sorted(paths)[:100]
imgs = []
imgs = np.array([io.imread("D:/Research/WCu/Data/3D/5842WCu_BSE/" + p)[100:1100, 100:1100] for p in paths])
print(imgs.shape)

# Get the model
filePath = ""
args = option_mod.parser.parse_args([
    "--dir_data", filePath,
    "--scale", "4",
    "--n_colors", "1",
    "--n_axis", "1",
    "--imageLim", "10",
    "--batch_size", "2",
    "--n_GPUs", "1",
    "--patch_size", "48",
    "--loss", "1*G",
    "--model", "Restormer",
    "--EDSR_path", "D:/Research/WCu/Data/SuperRes/model_files/edsr.pt",
    "--pre_train", "D:/Research/WCu/Data/SuperRes/model_files/restormer.pt",
    "--save_results",
])
args = option_mod.format_args(args)
if not args.cpu and torch.cuda.is_available():
    USE_GPU = True
    torch.cuda.empty_cache()

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
restormer = model.Model(args, checkpoint)
torch.set_grad_enabled(False)
restormer.eval()

# Perform the super resolution
output = np.zeros((imgs.shape[0] * int(args.scale), imgs.shape[1], imgs.shape[2]), dtype=np.uint8)
print(output.shape)
for i in tqdm(range(imgs.shape[1])):
    im_to_view = imgs[:, i]
    im = np.expand_dims(np.expand_dims(imgs[:, i], axis=0), axis=0)
    im = torch.from_numpy(im).float()
    im = utility2.normalize(im)
    im = prepare(im)
    sr = restormer(im, int(args.scale))
    sr = np.around(utility2.unnormalize(sr[0, 0]).cpu().numpy(), 0).astype(np.uint8)
    output[:, i] = sr

InteractiveView.Viewer(output, "binary_r")

# Save the images
for i in range(output.shape[0]):
    io.imsave(f"D:/Research/WCu/Data/SuperRes/SR-output/{i}.tif", output[i])
