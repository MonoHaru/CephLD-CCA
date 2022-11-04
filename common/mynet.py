import torch.nn as nn
from PIL.GimpGradientFile import linear
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
from numpy import matlib
import cv2
import os,sys
import numpy as np
from numpy import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random

def gray_to_rgb(gray):
    h,w = gray.shape
    rgb=np.zeros((h,w,3))
    rgb[:,:,0]=gray;    rgb[:,:,1]=gray;    rgb[:,:,2]=gray;
    return rgb

class dataload(Dataset):
    def __init__(self,  path='train', H=600,W=480,pow_n=3, aug=True):

        init_trans = transforms.Compose([transforms.Resize((H, W)),
                                         transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         ])
        self.datainfo = torchvision.datasets.ImageFolder(root=path, transform=init_trans)
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

        if self.aug == True: self.rv = random.random()
        else: self.rv=-1
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
####################################################
        else:
            image, _ = self.datainfo.__getitem__(idx)
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
            for k in range(0, self.mask_num):
                X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = X

        mask = torch.pow(mask, self.pow_n)
        mask = mask / mask.max()
        #print("idx :", idx, "path ", self.task)

        #plt.imshow(image[0], cmap='gray');   plt.show()
        return [image, mask ]

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

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        #
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

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
        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
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

class dilNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(dilNet, self).__init__()
        layernum = 32
        self.Conv1 = self.conv = nn.Sequential(
            nn.Conv2d(img_ch, layernum, kernel_size=3, stride=1, dilation=1, padding=1,  bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(layernum, layernum, kernel_size=3, stride=1,dilation=1, padding=1,  bias=True),
            nn.LeakyReLU(inplace=True),
        )
        self.Conv2 = self.conv = nn.Sequential(
            nn.Conv2d(layernum, layernum, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(layernum, layernum, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),
            nn.LeakyReLU(inplace=True),
        )
        self.Conv3 = self.conv = nn.Sequential(
            nn.Conv2d(layernum, layernum, kernel_size=3, stride=1, dilation=4, padding=4, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(layernum, layernum, kernel_size=3, stride=1, dilation=4, padding=4, bias=True),
            nn.LeakyReLU(inplace=True),
        )
        self.Conv4 = self.conv = nn.Sequential(
            nn.Conv2d(layernum, layernum, kernel_size=3, stride=1, dilation=8, padding=8, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(layernum, layernum, kernel_size=3, stride=1, dilation=8, padding=8, bias=True),
            nn.LeakyReLU(inplace=True),
        )
        self.Conv5 = self.conv = nn.Sequential(
            nn.Conv2d(layernum, layernum, kernel_size=3, stride=1, dilation=16, padding=16, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(layernum, layernum, kernel_size=3, stride=1, dilation=16, padding=16, bias=True),
            nn.LeakyReLU(inplace=True),
        )
        self.Convfinal = self.conv = nn.Sequential(
            nn.Conv2d(layernum*5, layernum*5, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(layernum*5, layernum*5, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
        )

        self.Conv_1x1 = nn.Conv2d(layernum*5, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        # decoding path
        final = self.Convfinal(torch.cat((x1,x2,x3, x4,x5), dim=1))
        out = self.Conv_1x1(final)

        return out