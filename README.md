Attempting to do two things with this fork:

- Make the model greyscale for electron microscopy images
- Convert the 2D super resolution to be 1D (only one axis of the image is upsampled, the other remains constant). This is of interest for certain 3D microscopy experiments where the voxels are anisotropic (higher resolution in plane, lower resolution along the z axis).

CONDA ENV: `conda create -n edsr python numpy scikit-image scikit-learn imageio matplotlib tqdm opencv pytorch torchvision torchaudio pytorch-cuda=[CUDA VERSION] -c pytorch -c nvidia`

Note that in the conda environment above, `[CUDA VERSION]` must be determined by you. [This link](https://developer.nvidia.com/cuda-downloads) will tkae you to the nvidia website that allows you to install the CUDA toolkit.
