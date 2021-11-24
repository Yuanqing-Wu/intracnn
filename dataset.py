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

def read_yuv(yuv_path, pos, w, h):
    fp = open(yuv_path, 'rb')
    fp.seek(pos)
    Y = fp.read(w*h)
    Y = np.reshape(np.fromstring(Y,'B'), (w, h, 1))
    fp.close()
    return Y

def load_data(data_path, pos, w, h):
    pos = int(pos)
    org = read_yuv(data_path + 'org.yuv', pos, w, h)
    pre = read_yuv(data_path + 'pre.yuv', pos, w, h)
    return org, pre

class DatasetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, file_list, width, height):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(dataset_dir,file_list))]
        self.data_paths = [join(dataset_dir,x.split(',')[0]) for x in alist]
        self.pos = [x.split(',')[1] for x in alist]
        self.qps = [x.split(',')[2] for x in alist]
        self.sns_targets = [x.split(',')[3] for x in alist]
        self.hsvs_targets = [x.split(',')[4] for x in alist]
        self.w = width
        self.h = height


    def __getitem__(self, index):

        # print(self.data_paths[index], self.indexs[index])
        org, pre = load_data(self.data_paths[index], self.pos[index], self.w, self.h)

        qp = int(self.qps[index])
        sns_target = int(self.sns_targets[index])
        hsvs_target = int(self.hsvs_targets[index])

        org = (np.array(org)).astype(np.float32)
        pre = (np.array(pre)).astype(np.float32)

        org = org / 127.5 - 1.
        pre = pre / 127.5 - 1.

        org = transform()(org)
        pre = transform()(pre)

        qp = torch.from_numpy(np.array([qp]).astype(np.float32))

        return org, pre - org, qp, sns_target, hsvs_target

    def __len__(self):
        return len(self.data_paths)
