import torch.nn as nn
from PIL.GimpGradientFile import linear
# from caffe2.python.operator_test.elementwise_logical_ops_test import rowmux
from torchvision import datasets, models, transforms
import torch
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from mytransforms import *
from mytransforms import mytransforms
# from skimage.filters import threshold_otsu
# from skimage import feature
# from skimage.color import rgb2gray
from numpy import matlib, character
import cv2
import os,sys
import numpy as np
from numpy import *
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
import random
from fuction import argsoftmax_grad, cargsoftmax, num_2_index

def gray_to_rgb(gray):
    h,w = gray.shape
    rgb=np.zeros((h,w,3))
    rgb[:,:,0]=gray;    rgb[:,:,1]=gray;    rgb[:,:,2]=gray;
    return rgb


# class dataload(Dataset):
#     def __init__(self,  path='train', H=600,W=480,pow_n=3, aug=True):
#
#         self.dinfo = torchvision.datasets.ImageFolder(root=path)
#         self.mask_num = int(len(self.dinfo.classes))## 20
#         self.data_num = int(len(self.dinfo.targets) / self.mask_num) ##150
#         self.path_mtx = np.array(self.dinfo.samples)[:, :1].reshape(self.mask_num,
#                                                                     self.data_num)  ## all data path loading  [ masks  20 , samples 150]
#         #self.images = [Image.open(path) for path in self.path_mtx.reshape(-1)]  # all image loading
#         #self.path1D = self.path_mtx.reshape(-1)  # all image path list
#
#         self.aug=aug
#         self.pow_n = pow_n
#         self.task = path
#         self.H = H
#         self.W = W
#         #self.mask_num=len(self.datainfo.classes)-1
#         #self.data_num = int(len(self.datainfo)/len(self.datainfo.classes))
#
#     def __len__(self):
#         return self.data_num
#
#     def __getitem__(self, idx):
#         # augmenation of img and masks
#         self.mask_trans = mytransforms.Compose([transforms.Resize((self.H, self.W)),
#                                                 transforms.Grayscale(1),
#                                                 mytransforms.Affine(random.randrange(-25, 25), #(-25, 25)
#                                                                     translate=[random.uniform(0, 0.05),
#                                                                                random.uniform(0, 0.05)],
#                                                                     scale=random.uniform(0.9, 1.1),
#                                                                     fillcolor=0),
#                                                 mytransforms.ToTensor(),
#
#                                                 ])
#         self.col_trans = mytransforms.Compose([mytransforms.ColorJitter(brightness=random.random(),
#                                                                         # contrast=random.random(),
#                                                                         # saturation=random.random(),
#                                                                         # hue=random.random() / 2
#                                                                         ),
#                                                ])
#         mask = torch.empty(self.mask_num, self.H, self.W, dtype=torch.float)  # 150 * H * W
#
#
#         for k in range(0, self.mask_num):
#             X = Image.open(self.path_mtx[k, idx])
#             if k==0: X=self.col_trans(X)
#             #now = time.time()
#             mask[k] = self.mask_trans(X)
#             #print(time.time() - now)
#
#         input, heat = mask[0:1],mask[1:20]
#         heat = torch.pow(heat, self.pow_n)
#         heat = heat / heat.max()
#
#         # plt.imshow(input[0], cmap='gray');
#         # plt.show()
#         # plt.imshow(heat[0], cmap='gray');
#         # plt.show()
#         # print("idx :", idx, "path ", self.task)
#
#         return [input, heat]
#

class dataload(Dataset):
    def __init__(self,  path='train', H=600,W=480,pow_n=3, aug=True):

        init_trans = transforms.Compose([transforms.Resize((H, W)),
                                         transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         ])
        self.datainfo = torchvision.datasets.ImageFolder(root=path, transform=init_trans)
        # print(self.datainfo)
        self.mask_num=len(self.datainfo.classes)-1
        self.data_num = int(len(self.datainfo)/len(self.datainfo.classes))
        self.aug=aug
        self.pow_n = pow_n
        self.task = path
        self.H = H
        self.W = W

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        if self.aug == True:
            self.rv = random.random()
            # print('True')
        else:
            self.rv=-1
            # print('False')
        if self.rv>=.1:
            # augmenation of img and masks
            angle = random.randrange(-25, 25)
            trans_rand = [random.uniform(0, 0.05) , random.uniform(0, 0.05)]
            scale_rand = random.uniform(0.9, 1.1)
            # trans img with masks
            self.input_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                     mytransforms.Affine(angle,
                                                                         translate=trans_rand,
                                                                         scale=scale_rand,
                                                                         fillcolor=0),
                                                     mytransforms.ToTensor(),
                                                     ])
            self.mask_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                    mytransforms.Affine(angle,
                                                                        translate=trans_rand,
                                                                        scale=scale_rand,
                                                                        fillcolor=0),
                                                    mytransforms.ToTensor(),
                                                    ])

            self.col_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                   mytransforms.ColorJitter(brightness=random.random(),
                                                                            contrast=random.random(),
                                                                            saturation=random.random(),
                                                                            hue=random.random() / 2
                                                                            ),
                                                   mytransforms.ToTensor(),
                                                   ])

            #print("angle:", angle, "vfilp:", vfilp)
            image, _ = self.datainfo.__getitem__(idx)

            #plt.imshow(image[0], cmap='gray');plt.show()
            image = self.col_trans(image)
            image = self.input_trans(image)

            #plt.imshow(image[0], cmap= 'gray' ); plt.show()
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)

            for k in range(0, self.mask_num):
                X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = self.mask_trans(X)

########################################################################################################################

            # print('aug on!')
        else:
            image, _ = self.datainfo.__getitem__(idx)
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
            for k in range(0, self.mask_num):
                X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = X
            # print('aug off!')

        mask = torch.pow(mask, self.pow_n)
        mask = mask / mask.max()
        #print("idx :", idx, "path ", self.task)

        #plt.imshow(image[0], cmap='gray');   plt.show()
        # print(image)
        # print(mask)
        return [image, mask ]
# # conv_block 원본
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 첫 번째 Conv2d 층에서만 dilation
class conv_block_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2(ch_in, ch_out, kernel_size=3, stride=1, padding=2, bias=True, dilation=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=2, bias=True, dilation=2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# up_conv 원본
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class up_conv_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_1, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=2, bias=True, dilation=2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class single_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, dila=1):
        super(single_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, dilation=dila, stride=1, padding=dila, bias=True),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Original_SE_Block(nn.Module):
    def __init__(self, c, r=8):
        super(Original_SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class cartesian_SE_Block(nn.Module):
    def __init__(self, c, r=4):
        super(cartesian_SE_Block, self).__init__()
        self.excitation = nn.Sequential(
            nn.Linear(2 * c, 2 * c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x, beta=1e-0):
        print('x: {}' .format(x.shape))
        x_ = x.view(x.shape[0], x.shape[1], -1)
        numerator = torch.exp(-torch.abs(x_ - x_.max(dim=2)[0].unsqueeze(dim=-1)) / (beta))
        denominator = torch.sum(numerator, dim=-1).unsqueeze(dim=-1)
        softmax = numerator / denominator
        index = torch.range(0, x_.shape[-1] - 1).to('cuda:0')
        identity = torch.matmul(softmax, index)
        col_points = ((identity % x.shape[-1]) + 1e-5) / x.shape[-1]  # 열, x축
        row_points = ((identity / x.shape[-1]) + 1e-5) / x.shape[-2]  # 행, y축
        cartesian = torch.cat((col_points, row_points), dim=-1)
        y = self.excitation(cartesian).view(x.shape[0], x.shape[1], 1, 1)
        return x * y.expand_as(x)


class cartesian_SE_Block_1(nn.Module):
    def __init__(self, c, r=4):
        super(cartesian_SE_Block_1, self).__init__()
        self.excitation = nn.Sequential(
            nn.Linear(2 * c, 2 * c // r, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        index = num_2_index(x.shape)
        y = cargsoftmax(
                x.view(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3]),
                index.to('cuda:0')
            ).view(1, -1)
        y = self.excitation(y).view(x.shape[0], x.shape[1], 1, 1)
        return x * y.expand_as(x)

class cartesian_SE_Block_2(nn.Module):
    def __init__(self, c, r=4):
        super(cartesian_SE_Block_2, self).__init__()
        # bs, c, h, w = x_shape
        # a, b = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        # a = (a / (h-1)).view(h, w, 1) ; b = (b / (w-1)).view(h, w, 1)
        # self.index = torch.cat((a, b), dim=2).view(h * w, 2).to(torch.float)
        self.excitation = nn.Sequential(
            nn.Linear(2 * c, 2 * c // r, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        index = num_2_index(x.shape)
        y = cargsoftmax(
                x.view(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3]),
                index.to('cuda:1')
            ).view(1, -1)
        y = self.excitation(y).view(x.shape[0], x.shape[1], 1, 1)
        return x * y.expand_as(x)


class SE_Block_1(nn.Module):
    def __init__(self, c, r=4):
        super(SE_Block_1, self).__init__()
        self.excitation = nn.Sequential(
            nn.Linear(2 * c, 2 * c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x, beta):
        # print('x: {}' .format(x.shape))
        x_ = x.view(x.shape[0], x.shape[1], -1)
        a = torch.exp(-torch.abs(x_ - x_.max(dim=2)[0].unsqueeze(dim=-1)) / (beta))
        b = torch.sum(a, dim=-1).unsqueeze(dim=-1)
        softmax = a / b
        identity = torch.sum(softmax * torch.arange(0, x_.shape[-1], dtype=torch.float), dim=-1)
        col_point = (identity % x.shape[-2])  # 열, x축
        row_point = ((identity / x.shape[-2]).trunc())  # 행, y축
        rho = torch.sqrt(torch.pow(col_point, 2) + torch.pow(row_point, 2))
        theta = torch.atan2(row_point, col_point) + 1e-10
        identity = torch.cat((rho, theta), dim=-1)
        points = x.shape[1]
        y = self.excitation(identity).view(1, points, 1, 1)
        return x * y.expand_as(x)

class SE_Block_3(nn.Module):
    def __init__(self, channel_in, channel_out, r=16):
        super(SE_Block_3, self).__init__()
            # self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.hiddenLinear = channel_in * channel_out
        self.excitation = nn.Sequential(
            # 4
            nn.Conv1d(channel_in, channel_in, kernel_size=3, stride=2, padding=3, bias=True, dilation=3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(channel_out, channel_out // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_out // r, channel_out, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x, maxpool=0):
        # print('x: {}' .format(x.shape))
        x_ = x.view(x.shape[0], x.shape[1], -1)
        identity = argsoftmax_dim2(x_,
                                   x_.max(dim=2)[0].unsqueeze(dim=-1),
                                   x_.max(dim=2)[1],
                                   H=x.shape[-2],
                                   W=x.shape[-1])
        channel, _, points = identity.shape
        y = self.excitation(identity).view(1, points, 1, 1)
        return x * y.expand_as(x)

class my_Net_01(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(my_Net_01, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)

        self.enSEblock1 = cartesian_SE_Block_1(c=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)

        self.enSEblock2 = cartesian_SE_Block_1(c=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)

        self.enSEblock3 = cartesian_SE_Block_1(c=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.deSEblock3 = cartesian_SE_Block_1(c=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.deSEblock2 = cartesian_SE_Block_1(c=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.deSEblock1 = cartesian_SE_Block_1(c=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.enSEblock1(x2)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.enSEblock2(x3)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.enSEblock3(x4)
        x4 = self.Conv4(x4)

        # decoding
        d4 = self.Up4(x4)
        d4 = self.deSEblock3(d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.deSEblock2(d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.deSEblock1(d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class my_Net_02(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(my_Net_02, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)

        self.enSEblock1 = cartesian_SE_Block_2(c=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)

        self.enSEblock2 = cartesian_SE_Block_2(c=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)

        self.enSEblock3 = cartesian_SE_Block_2(c=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.deSEblock3 = cartesian_SE_Block_2(c=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.deSEblock2 = cartesian_SE_Block_2(c=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.deSEblock1 = cartesian_SE_Block_2(c=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.enSEblock1(x2)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.enSEblock2(x3)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.enSEblock3(x4)
        x4 = self.Conv4(x4)

        # decoding
        d4 = self.Up4(x4)
        d4 = self.deSEblock3(d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.deSEblock2(d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.deSEblock1(d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class my_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(my_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.enSEblock1 = Original_SE_Block(c=64)

        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.enSEblock2 = Original_SE_Block(c=128)

        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.enSEblock3 = Original_SE_Block(c=256)

        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.enSEblock4 = Original_SE_Block(c=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.deSEblock3 = Original_SE_Block(c=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.deSEblock2 = Original_SE_Block(c=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.deSEblock1 = Original_SE_Block(c=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x1 = self.enSEblock1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.enSEblock2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.enSEblock3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.enSEblock4(x4)

        # decoding
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.deSEblock3(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.deSEblock2(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.deSEblock1(d2)

        d1 = self.Conv_1x1(d2)

        return d1

# U_Net 원본
class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.enSEblock1 = Original_SE_Block(c=64)
        self.enSEblock2 = Original_SE_Block(c=128)
        self.enSEblock3 = Original_SE_Block(c=256)
        self.enSEblock4 = Original_SE_Block(c=512)

        self.deSEblock4 = Original_SE_Block(c=512)
        self.deSEblock3 = Original_SE_Block(c=256)
        self.deSEblock2 = Original_SE_Block(c=128)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x1 = self.enSEblock1(x1)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x2 = self.enSEblock2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x3 = self.enSEblock3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        x4 = self.enSEblock4(x4)

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.deSEblock4(d4)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.deSEblock3(d3)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.deSEblock2(d2)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class U1_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U1_Net, self).__init__()

        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Seblock1 = Original_SE_Block(c=64)
        self.Seblock2 = Original_SE_Block(c=128)
        self.Seblock3 = Original_SE_Block(c=256)
        self.Seblock4 = Original_SE_Block(c=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x1 = self.Seblock1(x1)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x2 = self.Seblock2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x3 = self.Seblock3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        x4 = self.Seblock4(x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class U_Net_1(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_1, self).__init__()

        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.betaLinear1 = nn.Linear(64, 1)
        self.Seblock1 = SE_Block(c=64)

        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.betaLinear2 = nn.Linear(128, 1)
        self.Seblock2 = SE_Block(c=128)

        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.betaLinear3 = nn.Linear(256, 1)
        self.Seblock3 = SE_Block(c=256)

        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        beta1 = self.avg(x1)
        beta1 = self.betaLinear1(beta1.squeeze())
        x1 = self.Seblock1(x1, beta1)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        beta2 = self.avg(x2)
        beta2 = self.betaLinear2(beta2.squeeze())
        x2 = self.Seblock2(x2, beta2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        beta3 = self.avg(x3)
        beta3 = self.betaLinear3(beta3.squeeze())
        x3 = self.Seblock3(x3, beta3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        print('beta1 : {0}, beta2 : {1}, beta3 : {2}'.format(beta1.shape, beta2.shape, beta3.shape))

        return d1

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        # self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        #
        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)
        #
        # # decoding + concat path
        # d5 = self.Up5(x5)
        # x4 = self.Att5(g=d5, x=x4)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class U_Net_11(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_11, self).__init__()

        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.betaLinear1 = nn.Linear(64, 1)
        self.Seblock1 = SE_Block_1(c=64)

        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.betaLinear2 = nn.Linear(128, 1)
        self.Seblock2 = SE_Block_1(c=128)

        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.betaLinear3 = nn.Linear(256, 1)
        self.Seblock3 = SE_Block_1(c=256)

        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        beta1 = self.avg(x1)
        beta1 = self.betaLinear1(beta1.squeeze())
        x1 = self.Seblock1(x1, beta1)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        beta2 = self.avg(x2)
        beta2 = self.betaLinear2(beta2.squeeze())
        x2 = self.Seblock2(x2, beta2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        beta3 = self.avg(x3)
        beta3 = self.betaLinear3(beta3.squeeze())
        x3 = self.Seblock3(x3, beta3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        print('beta1 : {0}, beta2 : {1}, beta3 : {2}'.format(beta1, beta2, beta3))

        return d1

class U_Net_13(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_13, self).__init__()
        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class U_Net_12(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_12, self).__init__()
        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Seblock1 = SE_Block_1(c=64)
        self.Seblock2 = SE_Block_1(c=128)
        self.Seblock3 = SE_Block_1(c=256)
        self.Seblock4 = SE_Block_1(c=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Seblock1(x2)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Seblock2(x3)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Seblock3(x4)

        x4 = self.Conv4(x4)
        x4 = self.Seblock4(x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class U_Net_2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_2, self).__init__()

        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Seblock4 = SE_Block(c=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Seblock3 = SE_Block(c=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Seblock2 = SE_Block(c=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)
        d4 = self.Seblock4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.Up_conv3(d3)
        d3 = self.Seblock3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.Up_conv2(d2)
        d2 = self.Seblock2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class U_Net_3(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_3, self).__init__()
        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Seblock4 = SE_Block(c=512)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Seblock3 = SE_Block(c=256)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Seblock2 = SE_Block(c=128)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Seblock4(d4, maxpool=0)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Seblock3(d3, maxpool=0)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Seblock2(d2, maxpool=0)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# class AttU_Net(nn.Module):
#     def __init__(self, img_ch=3, output_ch=1):
#         super(AttU_Net, self).__init__()
#
#         #
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
#
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)
#
#         # self.Conv5 = conv_block(ch_in=512, ch_out=1024)
#         #
#         # self.Up5 = up_conv(ch_in=1024, ch_out=512)
#         # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
#         # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
#
#         self.Up4 = up_conv(ch_in=512, ch_out=256)
#
#         self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
#
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
#
#         self.Up3 = up_conv(ch_in=256, ch_out=128)
#
#         self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
#
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
#
#         self.Up2 = up_conv(ch_in=128, ch_out=64)
#
#         self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
#
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
#
#
#         self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
#
#         def initialize_weights(m):
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias.data, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight.data, 1)
#                 nn.init.constant_(m.bias.data, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight.data)
#                 nn.init.constant_(m.bias.data, 0)
#
#
#
#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)
#
#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
#
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)
#
#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)
#
#         # x5 = self.Maxpool(x4)
#         # x5 = self.Conv5(x5)
#         #
#         # # decoding + concat path
#         # d5 = self.Up5(x5)
#         # x4 = self.Att5(g=d5, x=x4)
#         # d5 = torch.cat((x4, d5), dim=1)
#         # d5 = self.Up_conv5(d5)
#
#         d4 = self.Up4(x4)
#         x3 = self.Att4(g=d4, x=x3)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)
#
#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3, x=x2)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)
#
#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2, x=x1)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)
#
#         d1 = self.Conv_1x1(d2)
#
#         return d1

# student U-Net - 1
class U_Net_std_1(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_std_1, self).__init__()

        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        # self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1

# student U-Net - 2
class U_Net_std_2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_std_2, self).__init__()

        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        # self.Conv4 = conv_block(ch_in=256, ch_out=512)
        # self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        # self.Up4 = up_conv(ch_in=512, ch_out=256)
        # self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1
