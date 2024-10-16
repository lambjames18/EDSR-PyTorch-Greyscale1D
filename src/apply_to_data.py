import os
import model
import option_mod
import utility
from tqdm.auto import tqdm
import torch
import numpy as np
from skimage import io
import support as sp
import matplotlib.pyplot as plt
import matplotlib

def prepare(im):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    im = im.to(device)
    return im

filePath = "F:/WCu-Data-SR/5842WCu_Spall/"
appendList =  ["first_set", "second_set"]
#data_slice = (slice(None), slice(48, 48+400), slice(1072, 1072+400))

# visualizing the sr 
# srPathse = filePath + "SE/images/SR/"
# srPathbse = filePath + "BSE/images/SR/"
# stackedse = np.stack([io.imread(srPathse + f) for f in os.listdir(srPathse) if f.endswith(".tif")])
# stackedbse = np.stack([io.imread(srPathbse + f) for f in os.listdir(srPathbse) if f.endswith(".tif")])
# print("Stacked SE shape: ", stackedse.shape)
# print("Stacked BSE shape: ", stackedbse.shape)
# slice = 500 
# slicese = stackedse[slice]
# slicebse = stackedbse[slice]
# stackedTot = np.stack((slicese, slicese, slicebse), axis=-1)
# print("Stacked Total shape: ", stackedTot.shape)
# sp.Viewer(stackedTot, cmap="gray")
# exit()


edsr_path = filePath + "/train/EDSR/model/model_best.pt"
restormer_path = filePath + "/train/Restormer/model/model_best.pt"

# running the SR 
for append in appendList:
    print("Running SR on: " + append)
    raw_imgs_folder = filePath + append + "/images/Raw/"
    output_folder = filePath + append + "/images/SR/"

    # Get the images
    paths = [f for f in os.listdir(raw_imgs_folder) if f.endswith(".tif")]
    print("Number of images: ", len(paths))
    paths = sorted(paths, key=lambda x: int(x.split('Slice')[1].split('.')[0]))
    paths = paths[:250]
    print("Paths: ", paths) 

    imgs = []
    imgs = np.array([io.imread(raw_imgs_folder + p) for p in paths])
    #imgs = imgs[data_slice]
    print("Original image stack shape: ", imgs.shape)
    #plt.imshow(imgs[:, 10], cmap='gray')
    #plt.savefig("./temp.png")

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
    print("Target output shape: ", output.shape)
    for i in tqdm(range(imgs.shape[2])):
        im = np.expand_dims(np.expand_dims(imgs[:, :, i], axis=0), axis=0)
        im = torch.from_numpy(im).float()
        im = utility.normalize(im)
        im = prepare(im)
        sr = restormer(im, int(args.scale))
        sr = np.around(utility.unnormalize(sr[0, 0]).cpu().numpy(), 0).astype(np.uint8)
        io.imsave(f"{output_folder}{i}.tif", sr)
        #output[:, :, i] = sr

    # Save the images
    #print("Saving images: " + append)
    #for i in range(output.shape[0]):
    #    io.imsave(f"{output_folder}{i}.tif", output[i])
