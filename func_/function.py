import torch

def argsoftmax( x, index, beta=1e-5):
    a =torch.exp(-torch.abs(x-x.max())/(beta))
    b =torch.sum(a)
    softmax = a / b
    return torch.sum(softmax * index)




import torch.nn as nn
class Gconv_layer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Gconv_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=1, bias=True)
        )

class Gconv_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=19):
        super(Gconv_Net, self).__init__()

        self.Gconv1 = Gconv_layer(ch_in=img_ch, ch_out=19)

    def forward(self, x):
        x1 = self.Gconv1(x)