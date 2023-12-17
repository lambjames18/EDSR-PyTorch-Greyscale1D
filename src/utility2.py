import os
import numpy as np
from skimage import io

# new class
# For Loss: 
#    - plots the loss
#    - logs the loss for training and validation 
# For Model:
#    - saves the model

class log():
    def __init__(self, args):
        # create the log file 
        self.args = args
    

    # saves the images to the test folder
    def save_results(self, save_list, index, loss):
        if self.args.save_results:
            # save list format: sr, lr, hr
            postfix = ('SR', 'LR', 'HR')

            for i in postfix:
                path = os.path.join(self.args.dir_data, 'test', i)
                os.makedirs(path, exist_ok = True)
            
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):

                filename = 'results-{}'.format(index)

                if p == 'SR': 
                    filename += '_loss-{}'.format(np.round(loss.cpu(),3))

                normalized = v[0].mul(255 / self.args.rgb_range)

                image_array = np.squeeze(normalized.cpu().numpy())

                io.imsave(os.path.join(self.args.dir_data, 'test', p,  '{}.tiff'.format(filename)), image_array.astype(np.uint8))

    # split the high and low res images into 4 to make them smaller
    def test_split(hr, lr):
        hr_split = [hr[:2000, :2000], hr[:2000, 2000:4000], hr[2000:4000, :2000], hr[2000:4000, 2000:4000]]
        lr_split = [lr[:2000, :2000], lr[:2000, 2000:4000], lr[2000:4000, :2000], lr[2000:4000, 2000:4000]]
        return hr_split, lr_split