import torch
import torch.nn as nn
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


def test(epoc, max_gain, max_epoch):
    model.eval()
    check = 0
    with torch.no_grad():
        psnr, cnt = 0.,0
        t0 = time.time()
        for iteration, batch in enumerate(testing_data_loader, 1):

            cnt += 1
            im, ref, ref_d, target= batch[0], batch[1], batch[2], batch[3]

            im = Variable(im).cuda(0)
            ref = Variable(ref).cuda(0)
            ref_d = Variable(ref_d).cuda(0)
            target = Variable(target).cuda(0)

            predict= model(im=im, ref=ref, ref_d=ref_d, job='test')#train
            #print(predict.size())
            _psnr= calc_psnr_and_ssim(target.detach(), predict.detach())

            psnr += _psnr
            #print(cnt, _psnr)
        psnr_ave = psnr / cnt
        d_psnr = psnr_ave - org_psnr
        if max_gain < d_psnr:
            max_gain = d_psnr
            max_epoch = epoch
            check = 1
        t1 = time.time()
        print("===> Ref  PSNR: {:.6f} || Gain: {:.6f} ||Epoch[{}] Max Gain: {:.6f}|| Timer: {:.4f} sec.".format(psnr_ave, d_psnr, max_epoch, max_gain, (t1 - t0)))
        return max_gain, max_epoch, check


## 参数
threads = 8 #数据加载线程
data_dir = '/home/wgq/research/bs/dataset/32x32'
train_list = 'train_list.txt'
test_list = 'test_list.txt'
patch_size = 80
batchSize = 8
lr = 1e-4
nEpochs = 500
pretrained = 0
model_path = './model/model_epoch30.pth'

print('===> Loading datasets')
train_set = DatasetFromFolder(data_dir, train_list, 32, 32)
training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize, shuffle=True)

test_set = DatasetFromFolder(data_dir, test_list, 32, 32)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=1, shuffle=True)

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
    train(epoch)
    # test(epoch, max_gain, max_epoch)

    if epoch%10 == 0:
        checkpoint(epoch)


    # learning rate is decayed by a factor of 10 every half of total epochs
    if epoch % 200 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 1.1
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))



