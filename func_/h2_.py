import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

pyra_f = 960; encoded_h=100; encoded_w=80
class conv_(nn.Module):
    def __init__(self, ch_in, ch_out, dila=1):
        super(conv_, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, dilation=dila, stride=1, padding=dila, bias=True),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class sub_classifier(nn.Module):
    def __init__(self):
        super(sub_classifier, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1=conv_(ch_in=pyra_f , ch_out=256)
        self.d5c1=conv_(ch_in=pyra_f , ch_out=128, dila=5)
        self.d10c1 = conv_(ch_in=pyra_f , ch_out=128, dila=10)
        self.fc = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128,3, bias=False),
            nn.Softmax()
        )
        self.SE = SE3way()
    def forward(self, x):
        x = self.SE(x); attx=x  # Attetion module
        f=torch.cat([self.c1(x), self.d5c1(x), self.d10c1(x)], dim=1)
        b, c, _, _ = f.size()
        f=self.avg_pool(f).view(b,c)
        f=self.fc(f)
        return F.softmax(f), attx
class MT_Net(nn.Module):
    def __init__(self):
        super(MT_Net, self).__init__()
        locnet = torch.load('BEST.pt');

        self.Maxpool = nn.MaxPool2d(2)
        self.Maxpool2 = nn.MaxPool2d(4)
        self.Maxpool3 = nn.MaxPool2d(8)
        self.Up=nn.Upsample(scale_factor=2)
        self.Up2 = nn.Upsample(scale_factor=4)
        self.Up3 = nn.Upsample(scale_factor=8)
        self.locnet=list( locnet.children() )
        # Encoding
        self.e1 = self.locnet[1]; self.e2 = self.locnet[2]
        self.e3 = self.locnet[3]; self.e4 = self.locnet[4]

        # Decoding
        self.de1 = self.locnet[5];self.de1_att = self.locnet[6]; self.de1_up = self.locnet[7]
        self.de2 = self.locnet[8];self.de2_att = self.locnet[9];self.de2_up = self.locnet[10]
        self.de3 = self.locnet[11];self.de3_att = self.locnet[12];self.de3_up = self.locnet[13]
        self.de4=self.locnet[14]

        # Sub classifier
        self.anb = sub_classifier();self.snb = sub_classifier()
        self.sna = sub_classifier();self.odi =sub_classifier()
        self.apdi = sub_classifier(); self.fhi=sub_classifier()
        self.fha =sub_classifier(); self.mw=sub_classifier()


    def forward(self, x):
        #1 Encoder
        x1 = self.e1(x);x2 = self.Maxpool(x1)
        x2 = self.e2(x2);x3 = self.Maxpool(x2)
        x3 = self.e3(x3);x4 = self.Maxpool(x3)
        endocded = self.e4(x4)

        #2 Sub network
        sub_input=torch.cat((self.Maxpool3(x1),self.Maxpool2(x2)\
                             ,self.Maxpool(x3), endocded),dim=1)

        anb_ , anb_att = self.anb(sub_input); snb_, snb_att = self.snb(sub_input)
        sna_, sna_att = self.sna(sub_input);odi_, odi_att = self.odi(sub_input)
        apdi_, apdi_att = self.apdi(sub_input);fhi_, fhi_att = self.fhi(sub_input)
        fha_, fha_att = self.fha(sub_input);mw_, mw_att = self.mw(sub_input)

        att_sum= (anb_att+snb_att+sna_att+odi_att \
                  +apdi_att+fhi_att+fha_att+mw_att)/8  # combination of att featuremaps

        endocded=att_sum[:,448:960,:,:]; x3=self.Up(att_sum[:,192:448,:,:])
        x2=self.Up2(att_sum[:,64:192,:, :]); x1=self.Up3(att_sum[:,0:64,:,:])

        #print("anb" , anb_.shape, "anb_att" , anb_att.shape)
        # print("encoded", endocded.shape, "x3", x3.shape, "x1" , x1.shape)

        #3 Decoder
        d4 = self.de1(endocded)
        x3 = self.de1_att(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.de1_up(d4)
        d3 = self.de2(d4)
        x2 = self.de2_att(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.de2_up(d3)
        d2 = self.de3(d3)
        x1 = self.de3_att(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.de3_up(d2)
        locnet = self.de4(d2)

        return locnet, torch.cat((anb_, snb_, sna_,odi_,apdi_,fhi_,fha_,mw_), dim=0)

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
    mrs = 60  # half of 200
    minv= 5
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
        self.task = path

        self.SJ_gaussian = np.load('SJ_gt_gaussian.npy')
        self.class_label = torch.tensor(np.load('class_label.npy'), dtype=torch.long)#.unsqueeze(2) # 차원 유지/증가
        self.train_c=self.class_label[0:150]
        self.val_c = self.class_label[150:300]
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
            #image = comb_black_rec(image, self.col_trans(image) ,  self.SJ_gaussian, 3 ,self.H, self.W)
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


#### AC LOSS ###########################################
########################################################
import os,sys
import numpy as np
from numpy import *
import os,sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import *
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random


H=800;W=640;batch_size=1
def t_angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
def angle_mat(v1, v2):
    v1 = np.array(v1);v2 = np.array(v2)
    v1=v1-[H/2, W/2]; v2=v2-[H/2, W/2];
    r =np.arccos(np.dot(v1, v2.transpose()) / (np.linalg.norm(v1, axis=1).reshape(v1.shape[0],1)\
                                       * np.linalg.norm(v2, axis=1)))
    r[isnan(r)]=0
    return r
def dist_mat(v1, v2):
    v1 = np.array(v1); v2= np.array(v2);
    y1, y2 = np.meshgrid(v1[:, 0], v2[:, 0])
    x1, x2 = np.meshgrid(v1[:, 1], v2[:, 1])
    dist_ = sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2);
    #dist_ = dist_ / dist_.max()
    return dist_
def s_to_m(output,label): # tensor(1,19,600,480)
    pred_mtx = []; gt_mtx = [];
    for k in range(0, 19):
        A = output[0][k];A = A.cpu()
        B = label[0][k]; B=B.cpu()
        amax = np.array(np.where(A == A.max()))
        bmax = np.array(np.where(B == B.max()))
        pred_mtx.append(amax[:, 0])
        gt_mtx.append(bmax[:, 0])
    return pred_mtx, gt_mtx

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


class SELayer(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE3way(nn.Module):
    def __init__(self, cc=pyra_f, yy=encoded_h, xx=encoded_w):
        super(SE3way, self).__init__()
        self.se_c = SELayer(channel=cc)
        self.se_y = SELayer(channel=yy)
        self.se_x = SELayer(channel=xx)

    def forward(self, x):
        f_c = self.se_c(x)
        f_y = self.se_y(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        f_x = self.se_x(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return f_c+f_y+f_x


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

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

        self.SE1 = SE3way(cc=64, yy=400, xx=320)
        self.SE2 = SE3way(cc=128, yy=200, xx=160)
        self.SE3 = SE3way(cc=256, yy=100, xx=80)
        self.SE4 = SE3way(cc=512, yy=100, xx=80)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.SE1(x2)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.SE2(x3)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.SE3(x4)

        x4 = self.Conv4(x4)
        x4 = self.SE4(x4)
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
