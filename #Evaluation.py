import os,sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random
import torch
import torch.nn as nn
import time
from collections import defaultdict
import torch.nn.functional as F
import cv2 as cv
from numpy import *
from scipy.spatial import distance


device_txt = 'cuda:1'
device = torch.device( device_txt if torch.cuda.is_available() else "cpu")
num_class = 19

print(device)
### parameter modification ################################ A2
H=800; W=640;
from A1_ import *
import A1_ as network

model = network.AttU_Net(1,num_class).to(device)
#model.load_state_dict(torch.load('model/bestloc/bestloc.pth',map_location=device_txt))
model.load_state_dict(torch.load(r'E:\X-Ray\model\22_02_16_3py\Network_0.005422673188149929_E_1899.pth', map_location=device_txt))
#model=model.eval()
print(model)
#torch.save(model, 'BEST.pt')

########################

val_data = 'test1' #######

if val_data =='test1':
    data = MD(path='data/test1', H=H, W=W, aug=False);
    num_land = 150
    #testgt = test1gt
if val_data =='test2':
    data = MD(path='data/test2', H=H, W=W, aug=False);
    num_land = 100
    #testgt=test2gt

Ymap, Xmap = np.mgrid[0:H:1, 0:W:1]
Ymap, Xmap = Ymap.flatten(), Xmap.flatten()
Ymap_, Xmap_ = np.tile(Ymap,(19,1)).transpose(1,0) , np.tile(Xmap,(19,1)).transpose(1,0)

Ymap_1, Xmap_1 = np.mgrid[0:H:1, 0:W:1]
Ymap_1, Xmap_1 = torch.tensor(Ymap_1.flatten(), dtype=torch.float).unsqueeze(1).to(device), torch.tensor(Xmap_1.flatten(),dtype=torch.float).unsqueeze(1).to(device)

count =0
ed=[]
T= .9
def Process_Heat(x):  #Heat to Axis
    x = x[0].view(19, -1).cpu()
    x = x / x.max(dim=1).values.unsqueeze(1)
    x = x > T  # 계단함수
    sum_x = torch.sum(x, dim=1)
    return x ,sum_x
def HtoA(x):
    x=x.view(x.shape[1], -1)  # Tensor : size 19 * [H*W]
    x = x/x.max(dim=1).values.unsqueeze(1) # x.max(dim=1) == size 19 * 1
    x = torch.pow(x,5)
    # x[x<=T],x[x>T] =0,1 # Step Function
    sum_x = torch.sum(x, dim=1).unsqueeze(1) #sum_x = size 19*1
    return x, sum_x   # x= size 19*(H*W) , sum_x = size 19*1

for k in range(0, 150):

    print('=== num :', k)
    x = data.__getitem__(k)

    inputs = x[0].unsqueeze(0).to(device)  # INPUT
    label = x[1].unsqueeze(0).to(device)  # GT

    output = model(inputs)
    output = torch.sigmoid(output)


    # #OLD
    # Heat,sum_Heat=Process_Heat(output) # Size of input batch*19*800*640
    # amax = np.diag(np.dot(Heat, Ymap_))/sum_Heat, np.diag(np.dot(Heat, Xmap_)) / sum_Heat
    #
    # Heat, sum_Heat = Process_Heat(label)  # Size of input batch*19*800*640
    # bmax = np.diag(np.dot(Heat, Ymap_)) / sum_Heat, np.diag(np.dot(Heat, Xmap_)) / sum_Heat


    # NEW
    #output = torch.pow(output, 10)
    Heat_, sum_Heat_ = HtoA(output)  # in: 19*H*w  // out:19*1
    Axis_Pred = torch.cat((torch.mm(Heat_, Ymap_1) / sum_Heat_,
                           torch.mm(Heat_, Xmap_1) / sum_Heat_), 1)  # size = 19*2

    label = torch.pow(label, 10)
    Heat_, sum_Heat_ = HtoA(label)  # in: 19*H*w  // out:19*1
    Axis_GT = torch.cat((torch.mm(Heat_, Ymap_1) / sum_Heat_,
                           torch.mm(Heat_, Xmap_1) / sum_Heat_), 1)  # size = 19*2

    plt.imshow(output[0][random.randint(0,18)].detach().cpu());
    plt.show()
    plt.imshow(Heat_[5].detach().cpu().reshape([800, 640]));
    plt.show()

    #plt.imshow(Heat_[5].detach().cpu().reshape([800, 640]));

    #dst = distance.euclidean([amax[0] * (2400 / H), amax[1] * (1935 / W)], [bmax[0] * (2400 / H), bmax[1] * (1935 / W)])
    dst = distance.euclidean([Axis_Pred[:, 0].detach().cpu() * (2400 / H), Axis_Pred[:, 1].detach().cpu() * (1935 / W)] ,[Axis_GT[:, 0].detach().cpu() * (2400 / H), Axis_GT[:, 1].detach().cpu() * (1935 / W)])
    print("num : ", k, "dst : ", dst)
    ed.append(np.array(dst))


mm = 20
mtx = np.array(ed)
print("2mm:", np.mean(mtx < mm), "2.5mm:", np.mean(mtx < mm * 1.25), \
      "3mm:", np.mean(mtx < mm * 1.5), \
      "4mm:", np.mean(mtx < mm * 2), "ave:", np.mean(mtx))

