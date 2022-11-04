import os,sys
#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from collections import defaultdict
import torch.nn.functional as F
import cv2 as cv
from numpy import *
from scipy.spatial import distance

device_txt = 'cuda:0'
device = torch.device( device_txt if torch.cuda.is_available() else "cpu")

### parameter modification ##########
from net import *
import net as network

num_class = 19
H=600; W=480;

model = network.U_Net(img_ch=1, output_ch=num_class).to(device);
model.load_state_dict(torch.load(r'E:\X-Ray\model\22_02_16_3py\Network_0.005422673188149929_E_1899.pth',map_location=device_txt)) # model loading
model=model.eval()

# Data loading with preprocessing
preprocess = transforms.Compose([transforms.Resize((H, W)),
                                 transforms.Grayscale(1),
                                 transforms.ToTensor(),
                                 ])
datainfo = torchvision.datasets.ImageFolder(root='data/test1', transform=preprocess)

## predict landmarks
subjects=[]
for k in range(0,10):
    x = datainfo.__getitem__(k)
    x = x[0].unsqueeze(0).to(device)
    outputs = model(x)

    subject = []
    for land in range(0, 19):
        A = outputs[0][land]
        A = A.cpu()
        pred = np.array(np.where(A > (A.max() * 0.95)))
        pred = pred.mean(axis=1);
        pred = np.round(pred)
        print(pred)
        subject.append(pred)
    subjects.append(subject)

subjects=np.array(subjects)


## 시각화
k = 0
pred_mtx = []
x = datainfo.__getitem__(k)
x = x[0].unsqueeze(0).to(device)
outputs = model(x)
for m in range(0, 19):
    mtx = gray_to_rgb(x[0][m].cpu())
    pred_mtx.append(mtx)
    # pred=subjects[k]
    print("Pred:", mtx)
    for k in range(0, num_class):
        cv.circle(mtx, (int(pred[k][1]), int(pred[k][0])), 2, (0, 0, 1), 3);  # pred
plt.imshow(mtx);plt.show()


