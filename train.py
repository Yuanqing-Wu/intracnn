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
#import torchsnooper

def checkpoint(epoch):
    model_out_path = './model/model_epoch' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def train(epoch):
    epoch_loss = 0
    model.train()
    t0 = time.time()
    for iteration, batch in enumerate(training_data_loader, 1):
        org, pre, qp, target= batch[0], batch[1], batch[2], batch[3]

        org = Variable(org).cuda(0)
        pre = Variable(pre).cuda(0)
        qp = Variable(qp).cuda(0)
        target = Variable(target).cuda(0)

        t2 = time.time()
        optimizer.zero_grad()
        predict= model(org, pre, qp)#train

        loss = criterion(predict, target)

        epoch_loss += loss.data

        loss.backward()
        optimizer.step()
        t3 = time.time()

        # print("===> Epoch[{}]({}/{}): Loss: {:.6f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.item(), (t3 - t2)))
    t1 = time.time()
    print("===> Epoch [{}] Complete: Loss: {:.6f} || Timer: {:.4f} sec.".format(epoch, epoch_loss / len(training_data_loader), (t1 - t0)))


def test():

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
        t1 = time.time()
    print('--------------------------------------------------')
    print('准确度:', (TP+TN)/(TP+FN+FP+TN), str(TP+TN) + '/' + str(TP+FN+FP+TN))
    print('不划分预测准确度:', TP/(TP+FN), str(TP) + '/' + str(TP+FN))
    print('划分预测准确度:', TN/(FP+TN), str(TN) + '/' + str(FP+TN))
    print('--------------------------------------------------')

## 参数
threads = 8 #数据加载线程
data_dir = '/home/wgq/research/bs/dataset/32x32'
train_list = 'train_list.txt'
test_list = 'test_list.txt'
batchSize = 32
lr = 1e-4
nEpochs = 50000
pretrained = 1960
model_path = './model/model_epoch1960.pth'

print('===> Loading datasets')
train_set = DatasetFromFolder(data_dir, train_list, 32, 32)
training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize, shuffle=True)

test_set = DatasetFromFolder(data_dir, test_list, 32, 32)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=batchSize, shuffle=True)

print('===> Building model ')
model = Net()
criterion = nn.CrossEntropyLoss()

if pretrained:
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.')

model = model.cuda(0)
criterion = criterion.cuda(0)

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(1, nEpochs + 1):

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train(epoch+pretrained)

    if (epoch+pretrained)%10 == 0:
        checkpoint(epoch+pretrained)
        test()


    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+pretrained) % 500 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 1.1
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))



