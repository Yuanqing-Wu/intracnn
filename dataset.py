import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor
from os import listdir
from os.path import join
import os.path
import numpy as np

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

def load_data(data_path, index, w, h):
    org = read_yuv(data_path + '/' + index + 'org.yuv', w, h)
    pre = read_yuv(data_path + '/' + index + 'pre.yuv', w, h)
    return org, pre

class DatasetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, file_list, width, height):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(dataset_dir,file_list))]
        self.data_paths = [join(dataset_dir,x.split(',')[0]) for x in alist]
        self.indexs = [x.split(',')[1] for x in alist]
        self.qps = [x.split(',')[2] for x in alist]
        self.targets = [x.split(',')[3] for x in alist]
        self.w = width
        self.h = height


    def __getitem__(self, index):

        # print(self.data_paths[index], self.indexs[index])
        org, pre = load_data(self.data_paths[index], self.indexs[index], self.w, self.h)

        qp = int(self.qps[index])
        target = int(self.targets[index])

        org = (np.array(org)).astype(np.float32)
        pre = (np.array(pre)).astype(np.float32)

        org = org / 127.5 - 1.
        pre = pre / 127.5 - 1.

        org = transform()(org)
        pre = transform()(pre)

        return org, pre, qp, target

    def __len__(self):
        return len(self.data_paths)
