import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
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

    def forward(self, x):
        x = self.up(x)
        return x
class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1
class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1
class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
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

a1=80;a2=160;a3=320;a4=640
class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        #super(U_Net, self).__init__()
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=a1)
        self.Conv2 = conv_block(ch_in=a1, ch_out=a2)
        self.Conv3 = conv_block(ch_in=a2, ch_out=a3)
        self.Conv4 = conv_block(ch_in=a3, ch_out=a4)
        self.Conv5 = conv_block(ch_in=a4, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=a4)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=a4)

        self.Up4 = up_conv(ch_in=a4, ch_out=a3)
        self.Up_conv4 = conv_block(ch_in=a4, ch_out=a3)

        self.Up3 = up_conv(ch_in=a3, ch_out=a2)
        self.Up_conv3 = conv_block(ch_in=a3, ch_out=a2)

        self.Up2 = up_conv(ch_in=a2, ch_out=a1)
        self.Up_conv2 = conv_block(ch_in=a2, ch_out=a1)

        self.Conv_1x1 = nn.Conv2d(a1, output_ch, kernel_size=1, stride=1, padding=0)

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
        # d5 = torch.cat([x4, d5], dim=1)
        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
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
class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        # self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        #
        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        # self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.RRCNN5(x5)
        #
        # # decoding + concat path
        # d5 = self.Up5(x5)
        # x4 = self.Att5(g=d5, x=x4)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

def double_conv(in_channels, out_channels,d ):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,3, dilation=d, padding=d),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,3, dilation=d, padding=d),
        nn.LeakyReLU(inplace=True),
        # nn.Conv2d(out_channels, out_channels, 3, dilation=5, padding=5),
        # nn.LeakyReLU(inplace=True)
    )
def double_3dconv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True)
    )
di =[1,1,1,1];
class FCN_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = models.segmentation.fcn_resnet101(pretrained=True)
        self.fcn.classifier[4] = nn.Conv2d(512, 19, kernel_size=(1, 1), stride=(1, 1))
        self.fcn.aux_classifier[4] = nn.Conv2d(256, 19, kernel_size=(1, 1), stride=(1, 1))

    # def forward(self, x):
    #     network = self.fcn(x)
    #     return network

class UNet(nn.Module):
    def __init__(self, n_input, n_class, ln):
        super().__init__()

        self.dconv_down1 = double_conv(n_input, ln[0],di[0])
        self.dconv_down2 = double_conv(ln[0], ln[1], di[1])
        self.dconv_down3 = double_conv(ln[1], ln[2], di[2])
        self.dconv_down4 = double_conv(ln[2], ln[3], di[3])

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(ln[2] + ln[3], ln[2] , di[2])
        self.dconv_up2 = double_conv(ln[1] + ln[2], ln[1],di[1])
        self.dconv_up1 = double_conv(ln[1] + ln[0], ln[0],di[0])

        self.conv_last = nn.Conv2d(ln[0], n_class, 1)

    def forward(self, x):
        # Local
        conv1 = self.dconv_down1(x); x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x); x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x); x = self.maxpool(conv3)

        x = self.dconv_down4(x);x = self.upsample(x); x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x); x = self.upsample(x); x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x); x = self.upsample(x); x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x);  Lout = self.conv_last(x)

        return Lout
#############################################################################################################
#############################################################################################################

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random
from mytransforms import *
from mytransforms import mytransforms
import random
import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage import feature
from skimage.color import rgb2gray
from numpy import matlib
import cv2

def comb_black_rec(image, trimage ,DIS, ratio, H, W):

    image = image.numpy()
    trimage = trimage.numpy()
    mrs = 90  # half of 200
    minv= 30
    rl = random.sample([0,1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], ratio )

    for landmark in rl:
        gaussian_mean = DIS[landmark, 0]
        gaussian_cov = DIS[landmark, 1:3]

        random_P = np.random.multivariate_normal(gaussian_mean, gaussian_cov)
        random_P = np.flip(random_P)

        h_ = int(round((random_P[0] / 2400) * H))
        w_ = int(round((random_P[1] / 1935) * W))
        r_image = cv2.resize(trimage[0], ( round(W*random.uniform(.05,.1))  ,round(H*random.uniform(.05,.1))))
        r_image = cv2.resize(r_image, ( W , H))

        r_image = np.expand_dims(r_image, axis=0)

        # plt.imshow(image[0], cmap='gray');plt.show()
        h1=random.random();h2=random.random();w1=random.random();w2=random.random()

        image[:, h_ - max(round(mrs*h1+ minv) ,0): \
        h_ + min(round(mrs*h2 +minv) ,H), \
        w_ - max(round(mrs*w1+minv) ,0): \
        w_ + min(round(mrs*w2+minv) ,W)] =  r_image[:, h_ - max(round(mrs*h1+ minv) ,0): \
        h_ + min(round(mrs*h2 +minv) ,H), \
        w_ - max(round(mrs*w1+minv) ,0): \
        w_ + min(round(mrs*w2+minv) ,W)]

    image = torch.from_numpy(image)

    return image


class MD(Dataset):
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
        #self.SJ_gaussian = np.load('SJ_gt_gaussian.npy')
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

            image = comb_black_rec(image, self.col_trans(image) ,  self.SJ_gaussian, 1 ,self.H, self.W)
            image = self.col_trans(image)
            image = self.input_trans(image)

            #plt.imshow(image[0], cmap= 'gray' ); plt.show()
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)

            for k in range(0, self.mask_num):
                X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = self.mask_trans(X)

            #plt.imshow(mask[0], cmap='gray'); plt.show()

####################################################
        else:

            image, _ = self.datainfo.__getitem__(idx)
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
            for k in range(0, self.mask_num):
                X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = X

        mask = torch.pow(mask, self.pow_n)
        mask = mask / mask.max()

        return [image, mask]

