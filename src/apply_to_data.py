import os
import utility
import model
import option_mod
import utility2
from tqdm.auto import tqdm
import torch
import numpy as np
from skimage import io
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


raw_imgs_folder = "D:/Research/WCu/Data/3D/5842WCu_BSE/"
output_folder = "E:/WCu/5842/"
data_slice = (slice(0, 400), slice(48, 48+4000), slice(1072, 1072+4000))
edsr_path = "D:/Research/WCu/Data/SuperRes/model_files/edsr.pt"
restormer_path = "D:/Research/WCu/Data/SuperRes/model_files/restormer.pt"

# Get the images
paths = [f for f in os.listdir(raw_imgs_folder) if f.endswith(".tif")]
paths = sorted(paths)
imgs = []
imgs = np.array([io.imread(raw_imgs_folder + p) for p in paths])
imgs = imgs[data_slice]

# Get the model
args = option_mod.parser.parse_args([
    "--dir_data", "",
    "--scale", "4",
    "--n_colors", "1",
    "--n_axis", "1",
    "--batch_size", "2",
    "--n_GPUs", "1",
    "--patch_size", "48",
    "--loss", "1*G",
    "--model", "Restormer",
    "--EDSR_path", edsr_path,
    "--pre_train", restormer_path,
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
for i in tqdm(range(imgs.shape[1])):
    im = np.expand_dims(np.expand_dims(imgs[:, i], axis=0), axis=0)
    im = torch.from_numpy(im).float()
    im = utility2.normalize(im)
    im = prepare(im)
    sr = restormer(im, int(args.scale))
    sr = np.around(utility2.unnormalize(sr[0, 0]).cpu().numpy(), 0).astype(np.uint8)
    output[:, i] = sr

# InteractiveView.Viewer(output, "binary_r")

# Save the images
for i in range(output.shape[0]):
    io.imsave(f"{output_folder}{i}.tif", output[i])
