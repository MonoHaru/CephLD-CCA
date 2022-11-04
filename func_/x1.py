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

from x1_ import *
import x1_ as network

batch_size = 1
H=800; W=640;
dataloaders = {
    'train': DataLoader(MD(path='data/train', H=H, W=W,pow_n=10, aug=True) , batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(MD(path='data/test1', H=H, W=W, pow_n=10, aug=False), batch_size=batch_size, shuffle=False, num_workers=3)
}

import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F

############ 2019.05.07

def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss

criterion = nn.CrossEntropyLoss()

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

device_txt = "cuda:0"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    #model = network.AttU_Net(img_ch=1, output_ch=19).to(device)
    #model=torch.load('BEST.pt').to(device)
    model=network.U_Net(img_ch=1, output_ch=19).to(device)
    #model.load_state_dict(torch.load('model/bestloc/bestloc.pth',map_location=device_txt))


    # Observe that all parameters are being optimized
    num_epochs = 1000
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 500)
    print("****************************GPU : ", device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    valtest = 10
    for epoch in range(num_epochs):
        print('========================' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('------------------------' * 10)
        now = time.time()

        if (epoch + 1) % valtest == 0:
            uu = ['train', 'val']
        else:
            uu = ['train']

        for phase in uu:
            # since = time.time()
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float) # 성능 값 중첩
            epoch_samples = 0

            num_ = 0
            for inputs, labels, cls_label in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                cls_label=cls_label.to(device)
                num_ = num_ + 1
                #print("cls label :" ,cls_label.shape , "cls" , cls_label[:,2])

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward computation
                   #outputs, anb,snb,sna,odi,apdi,fhi,fha,mw = model(inputs)
                    outputs = model(inputs) #clsout : 8by3  // cls_label : 1 by 8
                    #print("shape :" , clsout.shape, "label", cls_label.shape,"value :", cls_label+1)
                    acloss = L2_loss(outputs, labels)

                    #l1 = criterion(clsout[0:1, :], cls_label[:, 0])
                    loss=acloss
                    metrics['Jointloss'] += loss
                    #print("loss :" , loss, "integ loss", metrics['Jointloss'])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)


                # Each epoch has a training and validation phase

            # print_metrics(metrics, epoch_samples, phase)

            epoch_Jointloss = metrics['Jointloss'] / epoch_samples
            epoch_loss = metrics['reconloss'] / epoch_samples
            epoch_L12loss = metrics['L12loss'] / epoch_samples
            epoch_dist_loss = metrics['distloss'] / epoch_samples
            epoch_cls_loss = metrics['clsloss'] / epoch_samples
            print(phase,"Joint loss :", epoch_Jointloss , "epoch reconloss : ", epoch_loss)
            print("L12loss :", epoch_L12loss, "dist_loss:", epoch_dist_loss, "cls_loss", epoch_cls_loss)

            # deep copy the model

            savepath = 'model/h1/new_{}_L1_{}_E_{}.pth'
            if phase == 'val' and epoch_Jointloss < best_loss:
                print("saving best model")
                best_loss = epoch_Jointloss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss, epoch_cls_loss, epoch))

            if (epoch + 1) % 100 == 0:
                print("saving best model")
                best_loss = epoch_Jointloss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss, epoch_cls_loss, epoch))

        print(time.time() - now)
