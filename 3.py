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
dataloaders = {
    'train': DataLoader(dataload(path='data/train_15', H=H, W=W,pow_n=8, aug=True) , batch_size=batch_size, shuffle=True, num_workers=8),
    'val': DataLoader(dataload(path='data/test1', H=H, W=W, pow_n=8, aug=False), batch_size=batch_size, shuffle=False, num_workers=8)
}

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

device_txt = "cuda:0"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
num_class = 19

if __name__ == '__main__':

    #model=torch.load('BEST.pt').to(device)
    model=network.U_Net(1, num_class).to(device)
    # model.load_state_dict(torch.load(r'E:\X-Ray\model\1_3py\Network_0.0016705767484381795_E_259.pth',map_location=device_txt))

    # Observe that all parameters are being optimized
    num_epochs = 1000
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    print("****************************GPU : ", device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    valtest = 10
    train_losses = []
    val_losses = []
    val_loss_ = 0
    train_loss_ = 0

    for epoch in range(num_epochs):
        print('========================' * 9)
        print('Epoch {}/{}, learning_rate {}'.format(epoch, num_epochs - 1, scheduler.get_last_lr()))
        print('------------------------' * 9)
        now = time.time()

        uu= ['train', 'val'] if (epoch + 1) % valtest == 0 else ['train']

        for phase in uu:
            # since = time.time()
            if phase == 'train': scheduler.step(); model.train()  # Set model to training mode
            else: model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float) # 성능 값 중첩
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward computation
                    outputs = model(inputs) #
                    #outputs = torch.sigmoid(outputs)

                    LOSS = L1_loss(outputs, labels)
                    metrics['Jointloss'] += LOSS

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        LOSS.backward()
                        optimizer.step()
                # # statistics
                # if num_ % 10000 == 0:
                #     plt.title("H=200, W=160")
                #     plt.imshow(outputs[0][1].detach().cpu());
                #     plt.show()

                epoch_samples += inputs.size(0)



            # print_metrics(metrics, epoch_samples, phase)

            epoch_Jointloss = metrics['Jointloss'] / epoch_samples
            epoch_Jointloss_cpu = epoch_Jointloss.cpu().detach().numpy()

            if phase == 'train':
                train_loss_ = epoch_Jointloss_cpu
            else:
                val_loss_ = epoch_Jointloss_cpu

            train_losses.append(train_loss_)
            val_losses.append(val_loss_)

            print(phase,"Joint loss :", epoch_Jointloss )
            # deep copy the model

            savepath = r'E:\X-Ray\model\T_1\Network_{}_E_{}.pth'
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

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('encoding network')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train', 'Valid'])
    plt.show()