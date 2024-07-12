import os
import numpy as np
import h5py
import torch
from skimage import io
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.widgets

import model 
import option_mod
import utility
import loss
import trainer_pollock

def prepare(lr, args):
    lr = torch.from_numpy(lr.reshape((1, 1,) + lr.shape)).float()
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
    return lr

def normalize(image):
    minVal = np.min(image, axis=1)
    maxVal = np.max(image, axis=1)
    return (image - minVal)/(maxVal - minVal)

folder = "F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/all_phases_greyscale/"
paths = [folder + file for file in os.listdir(folder) if file.endswith('.tif') or file.endswith('.tiff')]
paths = sorted(paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
volume = np.array([io.imread(path) for path in tqdm(paths)])
# volume = volume[:250, 1000:2000, 1000:2000]

# saving these images to folder 
# for i in range(len(volume)):
#     io.imsave(f"F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/all_phases_greyscale/dreamTest/{i}.tiff", volume[i])
# exit()

print("Dimensions: z: {} voxels, y: {} voxels, x: {} voxels".format(*volume.shape))

axis = 1
#stack = np.stack(volume, axis = axis)

args = option_mod.parser.parse_args(["--dir_data", "F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/all_phases_greyscale/", 
                                     "--scale", "4", "--save_results" ,"--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48",
                                     "--model", "Restormer", 
                                     "--EDSR_path", "F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/all_phases_greyscale/pretrained/restormer/model/model_best.pt"])
args = option_mod.parser.parse_args(["--dir_data", "F:/WCu-Data-SR/5842WCu_Images/", 
                                     "--scale", "4", "--save_results" ,"--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48",
                                     "--model", "Restormer", 
                                     "--EDSR_path", "F:/WCu-Data-SR/5842WCu_Images/pretrained/restormer/model/model_best.pt"])
args = option_mod.format_args(args)
if not args.cpu and torch.cuda.is_available():
    USE_GPU = True
    torch.cuda.empty_cache()

checkpoint = utility.checkpoint(args)
model = model.Model(args, checkpoint)

#modelPath = "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR/Results/Trained_Model/model/model_best.pt"
model.load(args.EDSR_path)
torch.set_grad_enabled(False)
model.eval()

srImages = []

# swapping the axis
print("Old Dimensions: z: {} voxels, y: {} voxels, x: {} voxels".format(*volume.shape))
swapped_vol = np.swapaxes(volume, 0, 1)  # Shape (250, 1000, 1000) -> (1000, 250, 1000)
print("New Dimensions: z: {} voxels, y: {} voxels, x: {} voxels".format(*swapped_vol.shape))

for image_ind in tqdm(range(swapped_vol.shape[0])):
    name = f"Axis1_{image_ind}"
    
    #x = volume[:, image_ind, :]
    x = swapped_vol[image_ind]

    LR = utility.normalize(x)
    LR = prepare(LR, args)

    SR = model(LR, args.scale)

    SR_array = np.squeeze(SR.cpu().numpy())
    SR_array = np.around(255 * (SR_array - SR_array.min()) / (SR_array.max() - SR_array.min())).astype(np.uint8)

    #SR_array = utility.unnormalize(SR_array).cpu().numpy()

    #_lr = LR[0, 0, :, :].detach().cpu().numpy()
    #_lr = np.repeat(_lr, args.scale, axis=0)
    #_sr = SR_array[0, 0, :, :].detach().cpu().numpy()

    srImages.append(SR_array)

    # io.imsave(f"F:/WCu-Data-SR/dreamResults/{name}.tiff", SR_array)

    '''fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(_sr, cmap='gray')
    #ax[1].imshow(_lr, cmap='gray')
    ax[0].set_title("SR")
    ax[1].set_title("LR")

    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"F:/WCu-Data-SR/dreamResults/{name}.png")
    plt.close()'''

#srImages = np.array(srImages)
# swapping the axis back
srImages = np.swapaxes(srImages, 0, 1)  # Shape (1000, 400, 1000) -> (400, 1000, 1000)

print("Dimensions: z: {} voxels, y: {} voxels, x: {} voxels".format(*srImages.shape))

for i in range(len(srImages)):
    io.imsave(f"F:/WCu-Data-SR/8119WCu/8119WCusegmenteddatasets/all_phases_greyscale/SR_imageStack/{i}.tiff", srImages[i])

# graphing srImages with a slider 
'''fig = plt.figure(81234, figsize=(12, 8))
ax = fig.add_subplot(111)

vmin = np.amin(srImages)
vmax = np.amax(srImages)

im = ax.imshow(srImages[0], cmap='gray', vmin = vmin, vmax = vmax)

# Put slider on
plt.subplots_adjust(left=0.15, bottom=0.15)
left = ax.get_position().x0
bot = ax.get_position().y0
height = ax.get_position().height
right = ax.get_position().x1
axslice = plt.axes([left - 0.15, bot, 0.05, height])
slice_slider = matplotlib.widgets.Slider(
    ax=axslice,
    label="Slice #",
    valmin=0,
    valmax=len(srImages) - 1,
    valinit=0,
    valstep=1,
    orientation="vertical",
)'''

