import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net64x64(torch.nn.Module):
    def __init__(self):
        super(Net64x64, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 4, kernel_size=4, stride=4, padding=0, bias=True) # (1, 64, 64) -> (4, 16, 16)
        self.NC_p1 = nn.Conv2d(2, 4, kernel_size=4, stride=4, padding=0, bias=True) # (2, 64, 64) -> (4, 16, 16)

        self.NC_o2 = nn.Conv2d(4, 4, kernel_size=4, stride=4, padding=0, bias=True) # (4, 16, 16) -> (4, 4, 4)
        self.NC_p2 = nn.Conv2d(4, 4, kernel_size=4, stride=4, padding=0, bias=True) # (4, 16, 16) -> (4, 4, 4)

        self.NC3 = nn.Conv2d(8, 16, kernel_size=4, stride=4, padding=0, bias=True) # (8, 4, 4) -> (32, 1, 1)

        self.FC1 = nn.Linear(17,2) # 33 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        F_o = F.relu(self.NC_o2(F_o))
        F_p = F.relu(self.NC_p2(F_p))

        T = F.relu(self.NC3(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x = self.FC1(torch.cat((T, qp), dim=1))
        return x, x

class Net32x32(torch.nn.Module):
    def __init__(self):
        super(Net32x32, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 4, kernel_size=4, stride=4, padding=0, bias=True) # (1, 32, 32) -> (4, 8, 8)
        self.NC_p1 = nn.Conv2d(2, 4, kernel_size=4, stride=4, padding=0, bias=True) # (2, 32, 32) -> (4, 8, 8)

        self.NC2 = nn.Conv2d(8, 8, kernel_size=4, stride=4, padding=0, bias=True) # (4, 8, 8) -> (4, 2, 2)
        self.NC3 = nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (16, 1, 1)

        self.FC1_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC1_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))
        T = F.relu(self.NC3(T))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1_stage1(torch.cat((T, qp), dim=1))
        x2 = self.FC1_stage2(torch.cat((T, qp), dim=1))

        return x1, x2

class Net16x16(torch.nn.Module):
    def __init__(self):
        super(Net16x16, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 4, kernel_size=4, stride=4, padding=0, bias=True) # (1, 16, 16) -> (4, 4, 4)
        self.NC_p1 = nn.Conv2d(2, 4, kernel_size=4, stride=4, padding=0, bias=True) # (2, 16, 16) -> (4, 4, 4)

        self.NC2 = nn.Conv2d(8, 16, kernel_size=4, stride=4, padding=0, bias=True) # (8, 4, 4) -> (32, 1, 1)

        self.FC1_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC1_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1_stage1(torch.cat((T, qp), dim=1))
        x2 = self.FC1_stage2(torch.cat((T, qp), dim=1))

        return x1, x2

class Net32x16(torch.nn.Module):
    def __init__(self):
        super(Net32x16, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 4, kernel_size=(8, 4), stride=(8, 4), padding=0, bias=True) # (1, 32, 16) -> (4, 4, 4)
        self.NC_p1 = nn.Conv2d(2, 4, kernel_size=(8, 4), stride=(8, 4), padding=0, bias=True) # (2, 32, 16) -> (4, 4, 4)

        self.NC2 = nn.Conv2d(8, 16, kernel_size=4, stride=4, padding=0, bias=True) # (8, 4, 4) -> (32, 1, 1)

        self.FC1_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC1_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1_stage1(torch.cat((T, qp), dim=1))
        x2 = self.FC1_stage2(torch.cat((T, qp), dim=1))

        return x1, x2

class Net8x8(torch.nn.Module):
    def __init__(self):
        super(Net8x8, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 4, kernel_size=4, stride=4, padding=0, bias=True) # (1, 8, 8) -> (4, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 4, kernel_size=4, stride=4, padding=0, bias=True) # (2, 8, 8) -> (4, 2, 2)

        self.NC2 = nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (16, 1, 1)

        self.FC1_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC1_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1_stage1(torch.cat((T, qp), dim=1))
        x2 = self.FC1_stage2(torch.cat((T, qp), dim=1))

        return x1, x2

class Net32x8(torch.nn.Module):
    def __init__(self):
        super(Net32x8, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 4, kernel_size=(16, 4), stride=(16, 4), padding=0, bias=True) # (1, 32, 8) -> (4, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 4, kernel_size=(16, 4), stride=(16, 4), padding=0, bias=True) # (2, 32, 8) -> (4, 2, 2)

        self.NC2 = nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0, bias=True) # (16, 2, 2) -> (32, 1, 1)

        self.FC1_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC1_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))
        T = T.squeeze(3).squeeze(2)

        qp = qp / 64 - 0.5

        x1 = self.FC1_stage1(torch.cat((T, qp), dim=1))
        x2 = self.FC1_stage2(torch.cat((T, qp), dim=1))

        return x1, x2

class Net16x8(torch.nn.Module):
    def __init__(self):
        super(Net16x8, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 4, kernel_size=(8, 4), stride=(8, 4), padding=0, bias=True) # (1, 16, 8) -> (4, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 4, kernel_size=(8, 4), stride=(8, 4), padding=0, bias=True) # (2, 16, 8) -> (4, 2, 2)

        self.NC2 = nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (16, 1, 1)

        self.FC1_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC1_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1_stage1(torch.cat((T, qp), dim=1))
        x2 = self.FC1_stage2(torch.cat((T, qp), dim=1))

        return x1, x2

class Net32x4(torch.nn.Module):
    def __init__(self):
        super(Net32x4, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=(16, 2), stride=(16, 2), padding=0, bias=True) # (1, 32, 4) -> (8, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=(16, 2), stride=(16, 2), padding=0, bias=True) # (2, 32, 4) -> (8, 2, 2)

        self.NC2 = nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=True) # (16, 2, 2) -> (32, 1, 1)

        self.FC1 = nn.Linear(33,2) # 33 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x = self.FC1(torch.cat((T, qp), dim=1))

        return x

class Net16x4(torch.nn.Module):
    def __init__(self):
        super(Net16x4, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=(8, 2), stride=(8, 2), padding=0, bias=True) # (1, 16, 4) -> (8, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=(8, 2), stride=(8, 2), padding=0, bias=True) # (2, 16, 4) -> (8, 2, 2)

        self.NC2 = nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=True) # (16, 2, 2) -> (32, 1, 1)

        self.FC1 = nn.Linear(33,2) # 33 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x = self.FC1(torch.cat((T, qp), dim=1))

        return x

class Net8x4(torch.nn.Module):
    def __init__(self):
        super(Net8x4, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=(4, 2), stride=(4, 2), padding=0, bias=True) # (1, 8, 4) -> (8, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=(4, 2), stride=(4, 2), padding=0, bias=True) # (2, 8, 4) -> (8, 2, 2)

        self.NC2 = nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=True) # (16, 2, 2) -> (32, 1, 1)

        self.FC1 = nn.Linear(33,2) # 33 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x = self.FC1(torch.cat((T, qp), dim=1))

        return x

if __name__ == '__main__':

    model = Net8x8()
    x = torch.ones(1, 1, 8, 8)
    out = model(x, x, torch.ones(1, 1))
    print (out)
