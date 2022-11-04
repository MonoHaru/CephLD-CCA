import os,sys
import matplotlib.pyplot as plt
import mkl_random
from PIL import Image
import numpy as np
from numpy import *
import torch
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

#os.chdir("/content/drive/MyDrive/ISBI_pytorch/ISBI_pytorch")
batch_size = 1
H=512; W=512;
dataloaders = {
    'train': DataLoader(dataload(path='data/train', H=H, W=W,pow_n=8, aug=True) , batch_size=batch_size, shuffle=True, num_workers=3),
    'val': DataLoader(dataload(path='data/test1', H=H, W=W, pow_n=8, aug=False), batch_size=batch_size, shuffle=False, num_workers=3)
}


def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target))
    #metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
def L2_loss(pred, target):
    torch.mean(torch.pow((pred - target), 2))
    return torch.mean(torch.pow((pred - target), 2))
def edist_loss(pred, target):  # input size 19*2
    return torch.mean(torch.sqrt(torch.sum(torch.pow((pred - target), 2),1)))
# class HtoA(nn.Module):
#     def __init__(self, stepT=0.9):
#         super(HtoA, self).__init__()
#         self.T=stepT
#     def forward(self, x):
#         x = x.view(x.shape[1], -1)  # Tensor : 19 * [H*W]
#         return x

def HtoA(x, T=.9):
    x=x.view(x.shape[1], -1)  # Tensor : size 19 * [H*W]
    # x = x/x.max(dim=1).values.unsqueeze(1) # x.max(dim=1) == size 19 * 1
    # x[x<=T],x[x>T] =0,1 # Step Function
    sum_x = torch.sum(x, dim=1).unsqueeze(1) #sum_x = size 19*1
    return x, sum_x   # x= size 19*(H*W) , sum_x = size 19*1

device_txt = "cuda:0"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
num_class = 19

# Pre-defined coordinates maps    Size = [H*w]*1
Ymap, Xmap = np.mgrid[0:H:1, 0:W:1]
Ymap, Xmap = torch.tensor(Ymap.flatten(), dtype=torch.float).unsqueeze(1).to(device), torch.tensor(Xmap.flatten(),dtype=torch.float).unsqueeze(1).to(device)
#Ymap_, Xmap_ = np.tile(Ymap,(19,1)).transpose(1,0) , np.tile(Xmap,(19,1)).transpose(1,0)


if __name__ == '__main__':
    ### model select

    #model=torch.load('BEST.pt').to(device)
    model=network.AttU_Net(1, num_class).to(device)
    #model.load_state_dict(torch.load('model/bestloc/bestloc.pth',map_location=device_txt))
    #model.load_state_dict(torch.load('model/Network_0.001305533922277391_E_239.pth', map_location=device_txt))


    # Observe that all parameters are being optimized
    num_epochs =1200
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    print("****************************GPU : ", device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    valtest = 10
    tmp = 0
    for epoch in range(num_epochs):
        print('========================' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('------------------------' * 10)
        now = time.time()

        phase_= ['train', 'val'] if (epoch + 1) % valtest == 0 else ['train']

        for phase in phase_:
            # since = time.time()
            if phase == 'train': scheduler.step(); model.train()  # Set model to training mode
            else: model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float) # 성능 값 중첩
            epoch_samples = 0

            num_ = 0
            for inputs, labels in dataloaders[phase]:
                num_ += 1
                inputs, labels = inputs.to(device),labels.to(device)
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward computation

                    # Pred
                    outputs = model(inputs) #
                    outputs = torch.sigmoid(outputs)

                    Heat, sum_Heat=HtoA(outputs) # in: 19*H*w  // out:19*1
                    Axis_Pred = torch.cat((torch.mm(Heat, Ymap)/sum_Heat,
                                       torch.mm(Heat, Xmap)/sum_Heat) , 1) # size = 19*2
                    ## GT
                    Heat_, sum_Heat_ = HtoA(labels)  # in: 19*H*w  // out:19*1
                    Axis_GT = torch.cat((torch.mm(Heat_, Ymap) / sum_Heat_,
                                      torch.mm(Heat_, Xmap) / sum_Heat_), 1)  # size = 19*2

                    ## loss compute
                    regressloss, axisloss  = L2_loss(outputs, labels), L2_loss(Axis_Pred, Axis_GT)*1e-5
                    LOSS = regressloss+axisloss

                   #print('LOSS ', LOSS, 'AxisLoss' , axisloss )

                    # if num_%100==0:
                    #     plt.imshow(outputs[0][random.randint(0,18)].detach().cpu());
                    #     plt.show()
                        # plt.imshow(Heat[5].detach().cpu().reshape([800, 640]));
                        # plt.show()
                        # plt.imshow(Heat_[5].detach().cpu().reshape([800, 640]));
                        # plt.show()

                    metrics['Jointloss'] += LOSS
                    metrics['regressloss'] += regressloss
                    metrics['axisloss'] += axisloss
                    metrics['Edist'] +=  edist_loss(Axis_Pred, Axis_GT)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        LOSS.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)

            # print_metrics(metrics, epoch_samples, phase)
            epoch_Jointloss = metrics['Jointloss'] / epoch_samples
            epoch_regressloss = metrics['regressloss'] / epoch_samples
            epoch_axisloss = metrics['axisloss'] / epoch_samples
            epoch_edist = metrics['Edist'] / epoch_samples

            print(phase,"Joint loss :", epoch_Jointloss, "regress loss :", epoch_regressloss.item())
            print("axis loss :", epoch_axisloss.item(), 'edist', epoch_edist.item() )

            # deep copy the model
            savepath = 'model/0820/LOSS = regressloss+axisloss/Network0820_{}_E_{}.pth'
            if phase == 'val' and epoch_Jointloss < best_loss:
                print("saving best model")
                best_loss = epoch_Jointloss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss,  epoch))
            if (epoch + 1) % 100 == 0:
                print("saving best model")
                best_loss = epoch_Jointloss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss,  epoch))


        print(time.time() - now)
