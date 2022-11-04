import os,sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import *
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random
from scipy.spatial import distance
from net import *
import net as network
from collections import defaultdict
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

# 이미지 크기
H=800; W=640;

## Tensor Create
datapath = ['train', 'test1' , 'test2']

# 사용 gpu number
device_txt = "cuda:1"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")

class customdataset(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(customdataset, self).__init__()
        self.x = x_tensor
        self.y = y_tensor
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)

class Context_Net(nn.Module):
    def __init__(self, inch=38, reduction=4):
        super(Context_Net, self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(inch, inch, bias=True),
            nn.Dropout(0.05),
            nn.SELU(inplace=True),
            nn.Linear(inch, inch//reduction, bias=True),
            nn.Dropout(0.05),
            nn.SELU(inplace=True),
            nn.Linear(inch//reduction, inch, bias=True),
            #nn.Sigmoid()
        )
    def forward(self, x):
        h,w = x.size()
        y= x.view(1,h*w)
        y= self.fc(y)
        y = y.view(h,w)
        return y


train_data, train_gt = torch.load('axisdata/train_pred.pt'),torch.load('axisdata/train_GT.pt')
val_data, val_gt = torch.load('axisdata/test1_pred.pt'),torch.load('axisdata/test1_GT.pt')

dataloader = {
    'train': DataLoader(customdataset(train_data,train_gt) , batch_size=1, shuffle=True  ),
    'val': DataLoader(customdataset(val_data, val_gt), batch_size=1, shuffle=False)
}

def L2_loss(pred, target):
    return torch.mean(torch.pow((pred - target), 2))
def edist_loss(pred, target):  # input size 19*2
    return torch.mean(torch.sqrt(torch.sum(torch.pow((pred - target), 2),1)))

if __name__ == '__main__':

    model = Context_Net(reduction=1).to(device)
    # Observe that all parameters are being optimized
    valtest = 10
    num_epochs =50000
    best_epoch=1e10
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.1)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)


    for epoch in range(num_epochs + 1):

        phase = ['train', 'val'] if (epoch + 1) % valtest == 0 else ['train']
        for p_ in phase:

            cnt=0; batch_loss=0
            metrics = defaultdict(float)  # 성능 값 중첩

            if p_ == 'train': model.train()  # Set model to training mode
            else: model.eval()  # Set model to evaluate mode

            for batch_idx, samples in enumerate(dataloader[p_]):
                x_, y_ = samples
                x, y = x_[0].to(device), y_[0].to(device)
                optimizer.zero_grad()

                out = model(x)
                LOSS = L2_loss(out, y) * 1e-4
                if p_=='train':
                    LOSS.backward()
                    optimizer.step()

                cnt+=x_.size(0)

                metrics['batchloss'] += LOSS.item()
                metrics['Edist'] += edist_loss(out, y).item()

            # performance print
            epoch_loss = metrics['batchloss'] / cnt
            epoch_edist = metrics['Edist'] / cnt

            if p_ == 'val':
                if best_epoch > epoch_edist:
                    best_epoch = epoch_edist

            print("epoch ", epoch, " ", p_, "loss", epoch_loss , 'EDIST ' , epoch_edist,'LR',optimizer.param_groups[0]['lr'],'best' ,  best_epoch)

            # deep copy the model
            savepath = 'model/0908/Network0908_{}_E_{}.pth'
            if phase == 'val' and epoch_edist == best_epoch:
                print("saving best model")
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_epoch, epoch))
            if (epoch + 1) % 100 == 0:
                print("saving best model")
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_epoch, epoch))

        print("--------------------------------------------------------------------------------- ")


