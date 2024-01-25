import numpy as np
import h5py
import torch

import matplotlib.pyplot as plt

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

h = h5py.File('C:/Users/PollockGroup/Documents/coding/DreamData/5842WCu.dream3d', 'r')  # dream3d is a file format that is identical to an HDF5 file, just with a different name

# Get the data
volume = np.squeeze(h['DataContainers/ImageDataContainer/CellData/BSE'][...])  # we use np.squeeze in order to remove the extra dimension that Dream3D adds to the data
resolution = h['DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING'][...][::-1]  # Dream3D stores the resolution backwards so we need to flip it

print("Resolution: z: {:.3f} µm, y: {:.3f} µm, x: {:.3f} µm".format(*resolution))
print("Dimensions: z: {:.3f} µm, y: {:.3f} µm, x: {:.3f} µm".format(*np.array(volume.shape) * np.array(resolution)))  # Note that we have much higher resolution in the XY plane than in the Z plane
print("Dimensions: z: {} voxels, y: {} voxels, x: {} voxels".format(*volume.shape))

axis = 1
#stack = np.stack(volume, axis = axis)

args = option_mod.parser.parse_args(["--dir_data", "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR", "--scale", "4", "--save_results" ,"--n_colors", "1", "--n_axis", "1", "--batch_size", "8", "--n_GPUs", "1", "--patch_size", "48", "--pre_train", "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR/Results/Trained_Model/model/model_best.pt"])
args = option_mod.format_args(args)
if not args.cpu and torch.cuda.is_available():
    USE_GPU = True
    torch.cuda.empty_cache()

checkpoint = utility.checkpoint(args)
model = model.Model(args, checkpoint)

#modelPath = "C:/Users/PollockGroup/Documents/coding/WCu-Data-SR/Results/Trained_Model/model/model_best.pt"
model.load("", pre_train=args.pre_train)
torch.set_grad_enabled(False)
model.eval()

#for image_ind in range(volume.shape[axis]):
for image_ind in range(100):
    name = f"Axis{axis}_{image_ind}"
    
    x = volume[:, image_ind, :]

    LR = checkpoint.normalize(x)
    LR = prepare(LR, args)

    SR = model(LR, args.scale)

    SR_array = np.squeeze(SR.cpu().numpy())
    SR_array = np.around(255 * SR_array).astype(np.uint8)

    _lr = LR[0, 0, :, :].detach().cpu().numpy()
    #_lr = np.repeat(_lr, args.scale, axis=0)
    _sr = SR[0, 0, :, :].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(_sr, cmap='gray')
    ax[1].imshow(_lr, cmap='gray')
    ax[0].set_title("SR")
    ax[1].set_title("LR")

    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"C:/Users/PollockGroup/Documents/coding/WCu-Data-SR/dreamResults/{name}.png")
    plt.close()
