from torch import nn
import torch
import torch.nn.functional as F


class GLoss(nn.Module):
    """Class for calculating GV loss between to RGB images
       :parameter
       patch_size : int, scalar, size of the patches extracted from the gt and predicted images
       cpu : bool,  whether to run calculation on cpu or gpu
        """
    def __init__(self, cpu=False, pooling="max", loss_type="mse"):
        super(GLoss, self).__init__()
        # Sobel kernel for the gradient map calculation
        self.kernel_0 = torch.FloatTensor([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_1 = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_2 = torch.FloatTensor([[0, 0, -1], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_3 = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_4 = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_5 = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_6 = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.kernel_7 = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, -1]]).unsqueeze(0).unsqueeze(0)
        if not cpu:
            self.kernel_0 = self.kernel_0.cuda()
            self.kernel_1 = self.kernel_1.cuda()
            self.kernel_2 = self.kernel_2.cuda()
            self.kernel_3 = self.kernel_3.cuda()
            self.kernel_4 = self.kernel_4.cuda()
            self.kernel_5 = self.kernel_5.cuda()
            self.kernel_6 = self.kernel_6.cuda()
            self.kernel_7 = self.kernel_7.cuda()
        # Set the pooling
        if pooling == "avg":
            self.pool = self.avg_pool
        elif pooling == "max":
            self.pool = self.max_pool
        else:
            raise Exception("Invalid pooling type")
        # Set the loss type
        if loss_type == "mae":
            self.loss = F.l1_loss
        elif loss_type == "mse":
            self.loss = F.mse_loss
        else:
            raise Exception("Invalid loss type")

    def max_pool(self, img):
        return F.max_pool2d(img, 2, stride=2)

    def avg_pool(self, img):
        return F.max_pool2d(img, 2, stride=2)

    def forward(self, output, target):
        # calculation of the gradient maps
        g_target = []
        g_output = []
        for i in range(8):
            g_target.append(F.conv2d(target, getattr(self, f'kernel_{i}'), stride=1, padding=1))
            g_output.append(F.conv2d(output, getattr(self, f'kernel_{i}'), stride=1, padding=1))
        g_target = torch.stack(g_target, dim=1)
        g_output = torch.stack(g_output, dim=1)
        g_loss = self.loss(g_target, g_output)

        # calculate the pixel loss
        p_loss = self.loss(output, target)

        # downsample the images
        target_down = self.pool(target)
        output_down = self.pool(output)

        # calculate the gradient loss for the downsampled images
        g_target_down = []
        g_output_down = []
        for i in range(8):
            g_target_down.append(F.conv2d(target_down, getattr(self, f'kernel_{i}'), stride=1, padding=1))
            g_output_down.append(F.conv2d(output_down, getattr(self, f'kernel_{i}'), stride=1, padding=1))
        g_target_down = torch.stack(g_target_down, dim=1)
        g_output_down = torch.stack(g_output_down, dim=1)
        g_d_loss = self.loss(g_target_down, g_output_down)

        # calculate the total loss
        total_loss = g_loss + p_loss + g_d_loss

        return total_loss