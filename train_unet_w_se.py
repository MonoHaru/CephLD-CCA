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
from models.unet_w_se import *
# import models.unet_w_se.U_Net_w_se as network
import numpy as np
from collections import defaultdict
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import math

#os.chdir("/content/drive/MyDrive/ISBI_pytorch/ISBI_pytorch")
batch_size = 1
H=800; W=640;

# 데이터 로드하는 코드
dataloaders = {
    'train': DataLoader(dataload(path='data/train', H=H, W=W,pow_n=8, aug=False) , batch_size=batch_size, shuffle=True, num_workers=8),
    'val': DataLoader(dataload(path='data/test1', H=H, W=W, pow_n=8, aug=False), batch_size=batch_size, shuffle=False, num_workers=8),
    'test': DataLoader(dataload(path='data/test2', H=H, W=W, pow_n=8, aug=False), batch_size=batch_size, shuffle=False, num_workers=8)
}

class CosineAnnealingWarmUpRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

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

device_txt = "cuda:1"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
num_class = 19

if __name__ == '__main__':

    #model=torch.load('BEST.pt').to(device)
    model = U_Net_SE(1, num_class).to(device)
    # model.load_state_dict(torch.load(r'E:\X-Ray\model\1_3py\Network_0.0016705767484381795_E_259.pth',map_location=device_txt))

    # Observe that all parameters are being optimized
    num_epochs = 1000
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-10)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=100, T_mult=2, eta_max=1e-5, T_up=1, gamma=0.3)
    print("****************************GPU : ", device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_loss = 1e10
    best_val_loss = 1e10
    valtest = 10
    train_losses = []
    val_losses = []
    test_losses = []
    train_loss_ = 0
    val_loss_ = 0
    test_loss_ = 0


    for epoch in range(num_epochs):
        print('========================' * 9)
        print('Epoch {}/{}, learning_rate {}'.format(epoch, num_epochs - 1, scheduler.get_lr()))
        print('------------------------' * 9)
        now = time.time()

        uu= ['train', 'val', 'test'] if (epoch + 1) % valtest == 0 else ['train']

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
                    LOSS = L2_loss(outputs, labels)
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
                if phase == 'val':
                    val_loss_ = epoch_Jointloss_cpu
                else:
                    test_loss_ = epoch_Jointloss_cpu
            train_losses.append(train_loss_)
            val_losses.append(val_loss_)
            test_losses.append(test_loss_)

            print(phase,"Joint loss :", epoch_Jointloss )
            # deep copy the model

            savepath1 = r'E:\X-Ray\model\Original_SE_block_All_U_Net_005_1\Network_{}_E_{}.pth'
            savepath2 = r'E:\X-Ray\model\Original_SE_block_All_U_Net_005_1\Best_Network_{}_E_{}.pth'
            if phase == 'val' and epoch_Jointloss < best_val_loss:
                print("saving best val model")
                best_val_loss = epoch_Jointloss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath2.format(best_val_loss, epoch))
            if (epoch + 1) % 100 == 0:
                print("saving best model")
                best_train_loss = epoch_Jointloss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath1.format(best_train_loss, epoch))

        print(time.time() - now)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.plot(test_losses)
    plt.title('Original_SE_block_All_U_Net_005_1')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train', 'Valid', 'Test'])
    plt.show()