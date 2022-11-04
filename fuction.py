import torch

def Hargsoftmax(x, index, beta=1e-4):
    c = torch.exp(-torch.abs(x - x.max(dim=1).values.unsqueeze(1)) / (beta))
    d = torch.sum(c, dim=1).unsqueeze(1)
    softmax = c / d
    return torch.mm(softmax, index)

def cargsoftmax(x, index, beta=1):
    c = torch.exp(-torch.abs(x - x.max(dim=1).values.unsqueeze(1)) / (beta))
    d = torch.sum(c, dim=1).unsqueeze(1)
    softmax = c / d
    return torch.mm(softmax, index)

def num_2_index(x_shape):
    # print('x', x_shape)
    bs, c, h, w = x_shape
    a, b = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    a = (a / (h-1)).view(h, w, 1); b = (b / (w-1)).view(h, w, 1)
    return torch.cat((a, b), dim=2).view(h * w, 2).to(torch.float)


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

# def argsoftmax
''' 
    x = tensor.view(batch_size, channel_size, -1)
    x_un = x.max(dim=2)[0].unsqueeze(dim=-1)
    # ㄴㄴ -> index = x.argmax(dim=2).unsqueeze(dim=-1)
'''
def argsoftmax(x, index, beta=1e-5):
    a =torch.exp(-torch.abs(x-x.max())/(beta))
    b =torch.sum(a)
    softmax = a / b
    return torch.sum(softmax * index)
'''
    if
    1. x.shape -> (1,2,3,3)
    c = 
    for i in range(x.shape[1]):
        x[:,i] = x[:, i] - torch.max(x[:, i])
    
    a = torch.exp(-torch.abs(
        
'''

def aaa(x, beta):
    sub_x = x.clone()
    for i in range(sub_x.shape[1]):
        sub_x[:, i] = sub_x[:, i] - torch.max(sub_x[:, i])
    sub_x = torch.exp(-torch.abs(sub_x) / beta)
    sum_x = torch.sum(sub_x.view(sub_x.shape[0], sub_x.shape[1], -1), dim=-1)
    softmax = sub_x / sum_x.unsqueeze(dim=-1).unsqueeze(dim=-1)
    index = torch.arange(1, sub_x.shape[-1] * sub_x.shape[-2] + 1, dtype=float).to('cuda:0')

# beta를 매개변수로 학습 가능 유도해보기
def argsoftmax_grad(x_, x_max, index, beta):
    a = torch.exp(-torch.abs(x_ - x_max) / (beta))
    b = torch.sum(a, dim=-1).unsqueeze(dim=-1)
    softmax = a / b
    c = torch.sum(softmax * index, dim=-1)
    return c

def argsoftmax_e_10(x_, x_max, index, beta=1e+1):
    a = torch.exp(-torch.abs(x_ - x_max) / (beta))
    b = torch.sum(a, dim=-1).unsqueeze(dim=-1)
    softmax = a / b
    c = torch.sum(softmax * index, dim=-1)
    return c

def argsoftmax_e_100(x_, x_max, index, beta=1e+2):
    a = torch.exp(-torch.abs(x_ - x_max) / (beta))
    b = torch.sum(a, dim=-1).unsqueeze(dim=-1)
    softmax = a / b
    c = torch.sum(softmax * index, dim=-1)
    return c
# def argsoftmax(x, x_un, index, H, W, beta=1e-5):
#     a = torch.exp(-torch.abs(x-x_un)/(beta))
#     b = torch.sum(a, axis=2)
#     c = b * index
#     # return torch.sum(b * index)
#     return c

# def argsoftmax_dim2
'''
    hyperparameter:
    x = tensor.view(batch_size, channel_size, -1)
    x_un = x.max(dim=2)[0].unsqueeze(dim=-1)
    index = x.max(dim=2)[1]
    # index = x.max(dim=2).unsqueeze(dim=-1)
    return -> x_point, y_point
'''
def argsoftmax_dim2(x, x_un, index, H, W, beta=1e-10):
    a = torch.exp(-torch.abs(x-x_un) / (beta))
    b = torch.sum(a, axis=2)
    c = b * index
    d = torch.cat(((c / W).trunc(), c % W), dim=0) # x, y points
    d = d.unsqueeze(dim=0)
    # return torch.sum(b * index)
    return d

def argsoftmax_dim2as(x, x_un, index, H, W, beta=1e-10):
    a = torch.exp(-torch.abs(x-x_un) / (beta))
    b = torch.sum(a, axis=2)
    c = a / b.unsqueeze(dim=-1)
    c = c * index.unsqueeze(dim=-1)
    d = torch.cat(((c / W).trunc(), c % W), dim=0) # x, y points
    d = d.unsqueeze(dim=0)
    # return torch.sum(b * index)
    return d

# argsoftmax_dim에서 x_point만 반환
def argsoftmax_dim2x(x, x_un, index, H, W, beta=1e-10):
    a = torch.exp(-torch.abs(x-x_un) / (beta))
    b = torch.sum(a, axis=2)
    c = b * index
    d = torch.cat(((c / W).trunc(), c % W), dim=0) # x, y points
    x_point = (c / W).trunc()
    # return torch.sum(b * index)
    return x_point

# argsoftmax_dim2x return 값을 정규화함(-1~+1까지)
def argsoftmax_dim2x_regular_1(x_, x_un, index, H, W, beta=1e-10):
    a = torch.exp(-torch.abs(x_-x_un) / (beta))
    b = torch.sum(a, dim=-1).unsqueeze(dim=-1)
    c = b * index
    d = torch.cat(((c / W).trunc(), c % W), dim=0) # x, y points
    x_point = (c / W).trunc()
    x_point = x_point - ((H-1)/2)
    x_point = x_point / ((H-1)/2)
    return x_point
'''
    x_ = x.view(x.shape[0],x.shape[1],-1)
    x_max = x_.max(dim=2)[0].unsqueeze(dim=-1)
    index = torch.arange(0,x_.shape[-1])
    H=x.shape[-2]
    W=x.shape[-1])
'''
# argsoftmax_dim2x return 값을 정규화함(0~+1까지)
def argsoftmax_dim2x_regular_2(x_, x_max, index, H, W, beta=1e-10):
    a = torch.exp(-torch.abs(x_-x_max) / (beta))
    b = torch.sum(a, dim=-1).unsqueeze(dim=-1)
    softmax = a / b
    c = torch.sum(softmax * index, dim=-1)
    x_point = (c / W).trunc()
    x_point = x_point / (H-1)
    return x_point

def softargPolarCoordinate(x_, x_max, index, H, W, beta=1e-0):
    a = torch.exp(-torch.abs(x_-x_max) / (beta))
    b = torch.sum(a, dim=-1).unsqueeze(dim=-1)
    softmax = a / b
    c = torch.sum(softmax * index, dim=-1)
    col_point = (c % W)  # 열, x축
    row_point = ((c / W).trunc()) # 행, y축
    rho = torch.sqrt(torch.pow(col_point, 2) + torch.pow(row_point, 2))
    theta = torch.atan2(row_point, col_point) + 1e-10 # angle = phi * 180 / 3.14
    # torch.cat((rho, theta), dim=-1)
    return rho, theta

'''
    해볼 것(1)
    !!좀 더 확인해보고 해보기!!
    a = torch.exp(-torch.abs(x_ - x_max) / (1))
    b = torch.sum(a, dim=-1).unsqueeze(dim=-1)
    softmax = a / b
    c = torch.sum(softmax * index, dim=-1)
    col_point = (c % W)  # 열
    row_point = c / W  # 행
    rho = torch.sqrt(torch.pow(col_point, 2) + torch.pow(row_point, 2))
    theta = torch.atan2(row_point, row_point)  # angle = phi * 180 / 3.14
    
    해볼 것(2)
    !!좀 더 확인해보고 해보기!! 중심점이 (0,0)
    a = torch.exp(-torch.abs(x_ - x_max) / (1))
    b = torch.sum(a, dim=-1).unsqueeze(dim=-1)
    softmax = a / b
    c = torch.sum(softmax * index, dim=-1)
    col_point = (c % W) - ((W-1)/2)  # 열
    row_point = -(c / W) + ((H-1)/2) # 행
    rho = torch.sqrt(torch.pow(col_point, 2) + torch.pow(row_point, 2))
    theta = torch.atan2(col_point-(, row_point)  # angle = phi * 180 / 3.14
'''

# Tif x > x - torch.abs(x*0.05) or x < x + torch.abs(x*0.05):
#     True_ += 1
''' polar argmaxsoft
x_ = x.view(x.shape[0], x.shape[1], -1)
numerator = torch.exp(-torch.abs(x_ - x_.max(dim=2)[0].unsqueeze(dim=-1)) / (beta))
denominator = torch.sum(numerator, dim=-1).unsqueeze(dim=-1)
softmax = numerator / denominator
identity = torch.sum(softmax * torch.arange(1, x_.shape[-1] + 1).to('cuda:1'), dim=-1)
col_point = (identity % x.shape[-2])  # 열, x축
row_point = ((identity / x.shape[-2]).trunc())  # 행, y축
rho = torch.sqrt(torch.pow(col_point, 2) + torch.pow(row_point, 2))
theta = torch.atan2(row_point, col_point)
identity = torch.cat((rho, theta), dim=-1)
points = x.shape[1]
y = self.excitation(identity).view(1, points, 1, 1)
return x * y.expand_as(x)
'''