import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import net
from dataset import DatasetFromFolder
import time
import os
import numpy as np
import pandas as pd
import argparse
#import torchsnooper

# Testing settings
parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--cu_size', type=str, default='32x32', help="cu size")
parser.add_argument('--data_dir', type=str, default='/home/wgq/research/bs/dataset/allintra', help='data dir')
parser.add_argument('--test_list', type=str, default='train_list_32x32.txt', help='test list file')
parser.add_argument('--batch_size', type=int, default=32, help='training and testing batch size')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--model', default='', help='pretrained base model')

opt = parser.parse_args()

cu_size = opt.cu_size
w = int(cu_size.split('x')[0])
h = int(cu_size.split('x')[1])
data_dir = opt.data_dir
test_list = opt.test_list
batchSize = opt.batch_size
threads = opt.threads
model_path = opt.model

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



# print('===> Loading datasets')
test_set = DatasetFromFolder(data_dir, test_list, w, h)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=batchSize, shuffle=True)

# print('===> Building model ')
if cu_size == '64x64':
    model = net.Net64x64()
elif cu_size == '32x32':
    model = net.Net32x32()
elif cu_size == '16x16':
    model = net.Net16x16()
elif cu_size == '8x8':
    model = net.Net8x8()
elif cu_size == '32x16':
    model = net.Net32x16()
elif cu_size == '32x8':
    model = net.Net32x8()
elif cu_size == '16x8':
    model = net.Net16x8()

model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
model = model.cuda(0)

# print('---------- Networks architecture -------------')
# print_network(model)
# print('----------------------------------------------')


TP_sns, FP_sns, FN_sns, TN_sns = 0,0,0,0
TP_hsvs, FP_hsvs, FN_hsvs, TN_hsvs = 0,0,0,0

model.eval()
with torch.no_grad():
    t0 = time.time()
    for iteration, batch in enumerate(testing_data_loader, 1):

        org, res, qp, sns_target, hsvs_target= batch[0], batch[1], batch[2], batch[3], batch[4]

        org = Variable(org).cuda(0)
        res = Variable(res).cuda(0)
        qp = Variable(qp).cuda(0)
        sns_target = Variable(sns_target).cuda(0)
        hsvs_target = Variable(hsvs_target).cuda(0)

        predict= model(org, res, qp)
        if cu_size == '64x64':
            predict = F.softmax(predict[0], dim=1)
            for i in range(predict.size()[0]):
                print('%f,%f,%d'%(float(predict[i][0]), float(predict[i][1]), int(sns_target[i])))
                if sns_target[i] == 0:
                    if predict[i][0] > 0.5:
                        TP_sns += 1
                    else:
                        FN_sns += 1
                else:
                    if predict[i][0] > 0.5:
                        FP_sns += 1
                    else:
                        TN_sns += 1

        else:
            sns = F.softmax(predict[0], dim=1)
            hsns = F.softmax(predict[1], dim=1)
            for i in range(sns.size()[0]):
                print('%f,%f,%d,%f,%f,%d'%(float(sns[i][0]), float(sns[i][1]), int(sns_target[i]),float(hsns[i][0]), float(hsns[i][1]), int(hsvs_target[i])))
                if sns_target[i] == 0:
                    if sns[i][0] > 0.5:
                        TP_sns += 1
                    else:
                        FN_sns += 1
                else:
                    if sns[i][0] > 0.5:
                        FP_sns += 1
                    else:
                        TN_sns += 1

                if hsvs_target[i] == 0:
                    if hsns[i][0] > 0.5:
                        TP_hsvs += 1
                    else:
                        FN_hsvs += 1
                else:
                    if hsns[i][0] > 0.5:
                        FP_hsvs += 1
                    else:
                        TN_hsvs += 1
        # break
    t1 = time.time()
# print('--------------------------------------------------')
# print("===> Teste finish: Timer: {:.4f} sec.".format((t1 - t0)))
# print('s-ns acc:', (TP_sns+TN_sns)/(TP_sns+FN_sns+FP_sns+TN_sns), str(TP_sns+TN_sns) + '/' + str(TP_sns+FN_sns+FP_sns+TN_sns))
# print('ns acc:', TP_sns/(TP_sns+FN_sns), str(TP_sns) + '/' + str(TP_sns+FN_sns))
# print('s acc:', TN_sns/(FP_sns+TN_sns), str(TN_sns) + '/' + str(FP_sns+TN_sns))
# if cu_size != '64x64':
#     print('hs-vs acc:', (TP_hsvs+TN_hsvs)/(TP_hsvs+FN_hsvs+FP_hsvs+TN_hsvs), str(TP_hsvs+TN_hsvs) + '/' + str(TP_hsvs+FN_hsvs+FP_hsvs+TN_hsvs))
#     print('hs acc:', TP_hsvs/(TP_hsvs+FN_hsvs), str(TP_hsvs) + '/' + str(TP_hsvs+FN_hsvs))
#     print('vs acc:', TN_hsvs/(FP_hsvs+TN_hsvs), str(TN_hsvs) + '/' + str(FP_hsvs+TN_hsvs))
# print('--------------------------------------------------')
