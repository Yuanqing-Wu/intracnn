import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import Net
from dataset import DatasetFromFolder
import time
import os
import numpy as np
import pandas as pd
#import torchsnooper

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


## 参数
threads = 8 #数据加载线程
data_dir = '/home/wgq/research/bs/dataset/32x32'
test_list = 'test_list.txt'
batchSize = 32
model_path = './model/model_epoch30.pth'

print('===> Loading datasets')
test_set = DatasetFromFolder(data_dir, test_list, 32, 32)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=batchSize, shuffle=True)

print('===> Building model ')
model = Net()
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
model = model.cuda(0)

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')


TP = 0
FP = 0
FN = 0
TN = 0

model.eval()
with torch.no_grad():
    t0 = time.time()
    for iteration, batch in enumerate(testing_data_loader, 1):

        org, pre, qp, target= batch[0], batch[1], batch[2], batch[3]

        org = Variable(org).cuda(0)
        pre = Variable(pre).cuda(0)
        qp = Variable(qp).cuda(0)
        target = Variable(target).cuda(0)

        predict= model(org, pre, qp)
        predict = F.softmax(predict, dim=1)
        for i in range(predict.size()[0]):
            if target[i] == 0:
                if predict[i][0] > 0.5:
                    TP += 1
                else:
                    FN += 1
            else:
                if predict[i][0] > 0.5:
                    FP += 1
                else:
                    TN += 1
        # break
        print("===> ({}/{})".format(iteration, len(testing_data_loader)))
    t1 = time.time()

print('准确度:', (TP+TN)/(TP+FN+FP+TN))
print('不划分预测准确度:', TP/(TP+FN))
print('划分预测准确度:', TN/(FP+TN))



# 准确度: 0.8218418644706389
# 不划分预测准确度: 0.8184509392324668
# 划分预测准确度: 0.825232789708811