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

device_txt = 'cuda:1'
device = torch.device( device_txt if torch.cuda.is_available() else "cpu")
num_class = 19


### parameter modification ################################ A2
H=800; W=640;
from A1_ import *
import A1_ as network


model = network.AttU_Net(1,num_class).to(device)
#model.load_state_dict(torch.load('model/bestloc/bestloc.pth',map_location=device_txt))
model.load_state_dict(torch.load('model/Network0816_0.0006922060856595635_E_89.pth', map_location=device_txt))
model=model.eval()
print(model)
#torch.save(model, 'BEST.pt')



########################
print(model)


test1gt = np.load('test1.npy')
test2gt = np.load('test2.npy')


val_data = 'test1' #######

if val_data =='test1':
    data = MD(path='data/test1', H=H, W=W, aug=False);
    num_land = 150
    testgt = test1gt
if val_data =='test2':
    data = MD(path='data/test2', H=H, W=W, aug=False);
    num_land = 100
    testgt=test2gt


count =0; T= .9
ed=[]
for k in range(0,1):
    now = time.time()
    print('=== num :' , k)
    x = data.__getitem__(k)
    # plt.imshow(x[0][0]); plt.show()
    inputs = x[0] # pred
    inputs = inputs.unsqueeze(0)
    label = x[1]  # GT
    label = label.unsqueeze(0)

    inputs = inputs.to(device)

    outputs = model(inputs.data)
    outputs = torch.sigmoid(outputs)
    output = outputs.data

    from scipy.spatial import distance

    for jj in range(0, 19):
        A = output[0][jj];
        A = A.cpu(); A=A/A.max()
        B = label[0][jj]
        B = B.cpu(); B=B/B.max()
        amax = np.array(np.where(A ==A.max()))
        amax = np.array(np.where(A > T)); amax =amax.mean(axis=1)
        bmax = np.array(np.where(B == B.max()))
        bmax = np.array(np.where(B >  T));bmax = bmax.mean(axis=1)

        dst = distance.euclidean([amax[0] * (2400 / H), amax[1] * (1935 / W)], [bmax[0]* (2400 / H), bmax[1]* (1935 / W)])
        #dst = distance.euclidean([amax[0]*(2400/H) , amax[1]*(1935/W)]  , [bmax2[1], bmax2[0]])
        ed.append(dst)
        print("xx : " ,k,jj,"dst : " ,   dst)
        if dst >=20:
            count=count+1
            print(count)

    print(time.time() - now)

mm=20
mtx=np.array(ed)
print("2mm:", np.mean(mtx < mm), "2.5mm:", np.mean(mtx <mm * 1.25),\
      "3mm:", np.mean(mtx <mm * 1.5), \
         "4mm:", np.mean(mtx < mm * 2), "ave:", np.mean(mtx))



