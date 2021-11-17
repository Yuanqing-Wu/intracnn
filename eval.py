import torch
from model import Net
from torchvision.transforms import Compose, ToTensor
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
#import torchsnooper

def transform():
    return Compose([
        ToTensor(),
    ])

def read_yuv(yuv_path, w, h):
    fp = open(yuv_path, 'rb')
    Y_data = fp.read(w*h)
    Y = np.reshape(np.fromstring(Y_data,'B'),(h, w, 1))
    fp.close()
    return Y

model_path = './model/model_epoch600.pth'
org_path = '/home/wgq/research/bs/VVCSoftware_VTM/video/BasketballPass_416x240_50_8bit/00000000org.yuv'
pre_path = '/home/wgq/research/bs/VVCSoftware_VTM/video/BasketballPass_416x240_50_8bit/00000000pre.yuv'
qp = 32

org = read_yuv(org_path, 32, 32)
pre = read_yuv(pre_path, 32, 32)

org = (np.array(org)).astype(np.float32)
pre = (np.array(pre)).astype(np.float32)

org = org / 127.5 - 1.
pre = pre / 127.5 - 1.

org = transform()(org)
pre = transform()(pre)
org = org.unsqueeze(1)
pre = pre.unsqueeze(1)

qp = torch.ones(1, 1)*qp

model = Net()
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
model = model.cuda(0)

model.eval()
with torch.no_grad():

    org = Variable(org).cuda(0)
    pre = Variable(pre).cuda(0)
    qp = Variable(qp).cuda(0)

    print(org)

    predict= model(org, pre, qp)
    predict = F.softmax(predict, dim=1)
    print(predict)
