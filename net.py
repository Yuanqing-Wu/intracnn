import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net64x64(torch.nn.Module):
    def __init__(self):
        super(Net64x64, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 4, kernel_size=4, stride=4, padding=0, bias=True) # (1, 64, 64) -> (4, 16, 16)
        self.NC_p1 = nn.Conv2d(2, 4, kernel_size=4, stride=4, padding=0, bias=True) # (2, 64, 64) -> (4, 16, 16)

        self.NC_o2 = nn.Conv2d(4, 16, kernel_size=4, stride=4, padding=0, bias=True) # (4, 16, 16) -> (16, 4, 4)
        self.NC_p2 = nn.Conv2d(4, 16, kernel_size=4, stride=4, padding=0, bias=True) # (4, 16, 16) -> (16, 4, 4)

        self.NC3 = nn.Conv2d(32, 128, kernel_size=4, stride=4, padding=0, bias=True) # (32, 4, 4) -> (128, 1, 1)

        self.FC1 = nn.Linear(129,32) # 129 -> 32
        self.FC2 = nn.Linear(33,2) # 33 -> 2



    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        F_o = F.relu(self.NC_o2(F_o))
        F_p = F.relu(self.NC_p2(F_p))

        T = F.relu(self.NC3(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x = F.relu(self.FC1(torch.cat((T, qp), dim=1)))
        x = self.FC2(torch.cat((x, qp), dim=1))
        return x

class Net32x32(torch.nn.Module):
    def __init__(self):
        super(Net32x32, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=4, stride=4, padding=0, bias=True) # (1, 32, 32) -> (8, 8, 8)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=4, stride=4, padding=0, bias=True) # (2, 32, 32) -> (8, 8, 8)

        self.NC2 = nn.Conv2d(16, 64, kernel_size=4, stride=4, padding=0, bias=True) # (16, 8, 8) -> (64, 2, 2)
        self.NC3 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0, bias=True) # (64, 2, 2) -> (128, 1, 1)

        self.FC1_stage1 = nn.Linear(129,32) # 129 -> 32
        self.FC1_stage2 = nn.Linear(129,32) # 129 -> 32

        self.FC2_stage1 = nn.Linear(33,2) # 33 -> 1
        self.FC2_stage2 = nn.Linear(33,2) # 33 -> 1

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))
        T = F.relu(self.NC3(T))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = F.relu(self.FC1_stage1(torch.cat((T, qp), dim=1)))
        x1 = self.FC2_stage1(torch.cat((x1, qp), dim=1))

        x2 = F.relu(self.FC1_stage2(torch.cat((T, qp), dim=1)))
        x2 = self.FC2_stage2(torch.cat((x2, qp), dim=1))

        return x1, x2

class Net16x16(torch.nn.Module):
    def __init__(self):
        super(Net16x16, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=4, stride=4, padding=0, bias=True) # (1, 16, 16) -> (8, 4, 4)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=4, stride=4, padding=0, bias=True) # (2, 16, 16) -> (8, 4, 4)

        self.NC2 = nn.Conv2d(16, 64, kernel_size=4, stride=4, padding=0, bias=True) # (16, 4, 4) -> (64, 1, 1)

        self.FC1_stage1 = nn.Linear(65,16) # 65 -> 16
        self.FC1_stage2 = nn.Linear(65,16) # 65 -> 16

        self.FC2_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC2_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = F.relu(self.FC1_stage1(torch.cat((T, qp), dim=1)))
        x1 = self.FC2_stage1(torch.cat((x1, qp), dim=1))

        x2 = F.relu(self.FC1_stage2(torch.cat((T, qp), dim=1)))
        x2 = self.FC2_stage2(torch.cat((x2, qp), dim=1))

        return x1, x2

class Net32x16(torch.nn.Module):
    def __init__(self):
        super(Net32x16, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=(4, 8), stride=(4, 8), padding=0, bias=True) # (1, 32, 16) -> (8, 4, 4)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=(4, 8), stride=(4, 8), padding=0, bias=True) # (2, 32, 16) -> (8, 4, 4)

        self.NC2 = nn.Conv2d(16, 64, kernel_size=4, stride=4, padding=0, bias=True) # (16, 4, 4) -> (64, 1, 1)

        self.FC1_stage1 = nn.Linear(65,16) # 65 -> 16
        self.FC1_stage2 = nn.Linear(65,16) # 65 -> 16

        self.FC2_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC2_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = F.relu(self.FC1_stage1(torch.cat((T, qp), dim=1)))
        x1 = self.FC2_stage1(torch.cat((x1, qp), dim=1))

        x2 = F.relu(self.FC1_stage2(torch.cat((T, qp), dim=1)))
        x2 = self.FC2_stage2(torch.cat((x2, qp), dim=1))

        return x1, x2

class Net8x8(torch.nn.Module):
    def __init__(self):
        super(Net8x8, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=4, stride=4, padding=0, bias=True) # (1, 8, 8) -> (8, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=4, stride=4, padding=0, bias=True) # (2, 8, 8) -> (8, 2, 2)

        self.NC2 = nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=True) # (16, 2, 2) -> (32, 1, 1)

        self.FC1_stage1 = nn.Linear(33,16) # 33 -> 16
        self.FC1_stage2 = nn.Linear(33,16) # 33 -> 16

        self.FC2_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC2_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = F.relu(self.FC1_stage1(torch.cat((T, qp), dim=1)))
        x1 = self.FC2_stage1(torch.cat((x1, qp), dim=1))

        x2 = F.relu(self.FC1_stage2(torch.cat((T, qp), dim=1)))
        x2 = self.FC2_stage2(torch.cat((x2, qp), dim=1))

        return x1, x2

class Net32x8(torch.nn.Module):
    def __init__(self):
        super(Net32x8, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=(4, 16), stride=(4, 16), padding=0, bias=True) # (1, 32, 8) -> (8, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=(4, 16), stride=(4, 16), padding=0, bias=True) # (2, 32, 8) -> (8, 2, 2)

        self.NC2 = nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=True) # (16, 2, 2) -> (32, 1, 1)

        self.FC1_stage1 = nn.Linear(33,16) # 33 -> 16
        self.FC1_stage2 = nn.Linear(33,16) # 33 -> 16

        self.FC2_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC2_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))
        T = T.squeeze(3).squeeze(2)

        qp = qp / 64 - 0.5

        x1 = F.relu(self.FC1_stage1(torch.cat((T, qp), dim=1)))
        x1 = self.FC2_stage1(torch.cat((x1, qp), dim=1))

        x2 = F.relu(self.FC1_stage2(torch.cat((T, qp), dim=1)))
        x2 = self.FC2_stage2(torch.cat((x2, qp), dim=1))

        return x1, x2

class Net16x8(torch.nn.Module):
    def __init__(self):
        super(Net16x8, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=(4, 8), stride=(4, 8), padding=0, bias=True) # (1, 16, 8) -> (8, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=(4, 8), stride=(4, 8), padding=0, bias=True) # (2, 16, 8) -> (8, 2, 2)

        self.NC2 = nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=True) # (16, 2, 2) -> (32, 1, 1)

        self.FC1_stage1 = nn.Linear(33,16) # 33 -> 16
        self.FC1_stage2 = nn.Linear(33,16) # 33 -> 16

        self.FC2_stage1 = nn.Linear(17,2) # 17 -> 2
        self.FC2_stage2 = nn.Linear(17,2) # 17 -> 2

    def forward(self, org, pre, qp):

        F_o = F.relu(self.NC_o1(org))
        F_p = F.relu(self.NC_p1(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC2(torch.cat((F_o, F_p), dim=1)))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = F.relu(self.FC1_stage1(torch.cat((T, qp), dim=1)))
        x1 = self.FC2_stage1(torch.cat((x1, qp), dim=1))

        x2 = F.relu(self.FC1_stage2(torch.cat((T, qp), dim=1)))
        x2 = self.FC2_stage2(torch.cat((x2, qp), dim=1))

        return x1, x2

class Net32x4(torch.nn.Module):
    def __init__(self):
        super(Net32x4, self).__init__()

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=(2, 16), stride=(2, 16), padding=0, bias=True) # (1, 32, 4) -> (8, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=(2, 16), stride=(2, 16), padding=0, bias=True) # (2, 32, 4) -> (8, 2, 2)

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

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=(2, 8), stride=(2, 8), padding=0, bias=True) # (1, 16, 4) -> (8, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=(2, 8), stride=(2, 8), padding=0, bias=True) # (2, 16, 4) -> (8, 2, 2)

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

        self.NC_o1 = nn.Conv2d(1, 8, kernel_size=(2, 4), stride=(2, 4), padding=0, bias=True) # (1, 8, 4) -> (8, 2, 2)
        self.NC_p1 = nn.Conv2d(2, 8, kernel_size=(2, 4), stride=(2, 4), padding=0, bias=True) # (2, 8, 4) -> (8, 2, 2)

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

    model = Net64x64()
    x = torch.ones(1, 1, 64, 64)
    out = model(x, x, torch.ones(1, 1))
    print (out)
