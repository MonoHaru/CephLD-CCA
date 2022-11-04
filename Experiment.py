import os,sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
from net import *
import net as network
import numpy as np


#os.chdir("/content/drive/MyDrive/ISBI_pytorch/ISBI_pytorch")
batch_size = 1
H=800; W=640;

from collections import defaultdict
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
def HtoA(x):
    x=x.view(num_class, -1)
    return x
# Accuracy = Accuracy_n(outputs, labels, 1e-3)
def Accuracy_n(outputs, labels, n):
    B, C, H, W = labels.shape
    # over = outputs > labels - torch.abs(0.1)
    # under = outputs < labels + torch.abs(0.1)
    over = outputs > labels - n
    under = outputs < labels + n
    gap = over * under
    accuracy = torch.count_nonzero(gap) / (B * C * H * W) * 100
    return accuracy


device_txt = "cuda:1"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
num_class = 19

if __name__ == '__main__':

    #model=torch.load('BEST.pt').to(device)
    model=network.U_Net(1, num_class).to(device)
    model.load_state_dict(torch.load(r'E:\X-Ray\model\Best_Val_Network_0.0002603263419587165_E_1009.pth',map_location=device_txt))

    # Observe that all parameters are being optimized
    num_epochs = 2500
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)