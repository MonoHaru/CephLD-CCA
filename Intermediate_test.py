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
from net import *
import net as network
import numpy as np

def gray_to_rgb(gray):
    h,w = gray.shape
    rgb=np.zeros((h,w,3))
    rgb[:,:,0]=gray;    rgb[:,:,1]=gray;    rgb[:,:,2]=gray;
    return rgb

batch_size = 1
H=800; W=640;

data = dataload(path=r'E:\X-Ray\data\test1', H=H, W=W, aug=False);

from collections import defaultdict

device_txt = "cuda:0"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
num_class = 19

from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

model = network.U_Net_2(1, num_class).to(device)
model.load_state_dict(torch.load(r'E:\X-Ray\model\SE_T_1_test1\Network_0.06308535486459732_E_929.pth', map_location=device_txt))
model=model.eval()

x = data.__getitem__(2)
x = x[0].unsqueeze(0).to(device)

return_layers = {
    'Conv1': 'Conv1',
    'Conv2': 'Conv2',
    'Conv3': 'Conv3',
    'Conv4': 'Conv4',
    'Up4': 'Up4',
    'Up_conv4': 'Up_conv4',
    'Seblock4' : 'Seblock4',
    'Up3': 'Up3',
    'Up_conv3': 'Up_conv3',
    'Seblock3': 'Seblock3',
    'Up2': 'Up2',
    'Up_conv2': 'Up_conv2',
    'Seblock2': 'Seblock2',
    'Conv_1x1': 'Conv_1x1'
}

trans = transforms.ToPILImage()

mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
mid_outputs, model_output = mid_getter(x)

for i in range(len(mid_outputs['Conv_1x1'][0])):
    tensor_mid = mid_outputs['Conv_1x1'][0][i].cpu().detach()
    tensor_mid = tensor_mid.unsqueeze(0)
    # tensor_mid = tensor_mid.permute(1,2,0)
    # numpy_mid = tensor_mid.cpu().detach().numpy()
    # image_data = trans(numpy_mid)
    plt.imshow(tensor_mid.permute(1,2,0))
    plt.show()
# print(model_output)
# print(mid_outputs)


# model_output is None if keep_ouput is False
# if keep_output is True the model_output contains the final model's output
