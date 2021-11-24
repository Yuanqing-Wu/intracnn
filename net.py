import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net64x64(torch.nn.Module):
    def __init__(self):
        super(Net64x64, self).__init__()

        self.NC1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding =1, bias=True) # (1, 64, 64) -> (8, 64, 64)

        self.NC2 = nn.Conv2d(9, 9, kernel_size=32, stride=32, padding=0, bias=True) # (8, 64, 64) -> (8, 2, 2)
        self.NC3 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)

        self.FC = nn.Linear(10,2) # 33 -> 2

    def forward(self, org, pre, qp):

        T = F.relu(self.NC1(org))
        T = F.relu(self.NC2(torch.cat((T, pre), dim=1)))
        T = F.relu(self.NC3(T))

        T = T.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x = self.FC(torch.cat((T, qp), dim=1))
        return x, x

class Net32x32(torch.nn.Module):
    def __init__(self):
        super(Net32x32, self).__init__()

        self.NC1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding =1, bias=True) # (1, 64, 64) -> (8, 64, 64)
        self.NC2 = nn.Conv2d(9, 9, kernel_size=16, stride=16, padding=0, bias=True) # (8, 64, 64) -> (8, 2, 2)
        self.NC3_1 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)
        self.NC3_2 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)

        self.FC1 = nn.Linear(10,2) # 9 -> 2
        self.FC2 = nn.Linear(10,2) # 9 -> 2

    def forward(self, org, pre, qp):

        T = F.relu(self.NC1(org))
        T = F.relu(self.NC2(torch.cat((T, pre), dim=1)))

        T1 = F.relu(self.NC3_1(T))
        T2 = F.relu(self.NC3_2(T))

        T1 = T1.squeeze(3).squeeze(2)
        T2 = T2.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1(torch.cat((T1, qp), dim=1))
        x2 = self.FC1(torch.cat((T2, qp), dim=1))
        return x1, x2

class Net16x16(torch.nn.Module):
    def __init__(self):
        super(Net16x16, self).__init__()

        self.NC1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding =1, bias=True) # (1, 64, 64) -> (8, 64, 64)
        self.NC2 = nn.Conv2d(9, 9, kernel_size=8, stride=8, padding=0, bias=True) # (8, 64, 64) -> (8, 2, 2)
        self.NC3_1 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)
        self.NC3_2 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)

        self.FC1 = nn.Linear(10,2) # 9 -> 2
        self.FC2 = nn.Linear(10,2) # 9 -> 2

    def forward(self, org, pre, qp):

        T = F.relu(self.NC1(org))
        T = F.relu(self.NC2(torch.cat((T, pre), dim=1)))

        T1 = F.relu(self.NC3_1(T))
        T2 = F.relu(self.NC3_2(T))

        T1 = T1.squeeze(3).squeeze(2)
        T2 = T2.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1(torch.cat((T1, qp), dim=1))
        x2 = self.FC1(torch.cat((T2, qp), dim=1))
        return x1, x2

class Net32x16(torch.nn.Module):
    def __init__(self):
        super(Net32x16, self).__init__()

        self.NC1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding =1, bias=True) # (1, 64, 64) -> (8, 64, 64)
        self.NC2 = nn.Conv2d(9, 9, kernel_size=(16, 8), stride=(16, 8), padding=0, bias=True) # (8, 64, 64) -> (8, 2, 2)
        self.NC3_1 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)
        self.NC3_2 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)

        self.FC1 = nn.Linear(10,2) # 9 -> 2
        self.FC2 = nn.Linear(10,2) # 9 -> 2

    def forward(self, org, pre, qp):

        T = F.relu(self.NC1(org))
        T = F.relu(self.NC2(torch.cat((T, pre), dim=1)))

        T1 = F.relu(self.NC3_1(T))
        T2 = F.relu(self.NC3_2(T))

        T1 = T1.squeeze(3).squeeze(2)
        T2 = T2.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1(torch.cat((T1, qp), dim=1))
        x2 = self.FC1(torch.cat((T2, qp), dim=1))
        return x1, x2

class Net8x8(torch.nn.Module):
    def __init__(self):
        super(Net8x8, self).__init__()

        self.NC1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding =1, bias=True) # (1, 64, 64) -> (8, 64, 64)
        self.NC2 = nn.Conv2d(9, 9, kernel_size=4, stride=4, padding=0, bias=True) # (8, 64, 64) -> (8, 2, 2)
        self.NC3_1 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)
        self.NC3_2 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)

        self.FC1 = nn.Linear(10,2) # 9 -> 2
        self.FC2 = nn.Linear(10,2) # 9 -> 2

    def forward(self, org, pre, qp):

        T = F.relu(self.NC1(org))
        T = F.relu(self.NC2(torch.cat((T, pre), dim=1)))

        T1 = F.relu(self.NC3_1(T))
        T2 = F.relu(self.NC3_2(T))

        T1 = T1.squeeze(3).squeeze(2)
        T2 = T2.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1(torch.cat((T1, qp), dim=1))
        x2 = self.FC1(torch.cat((T2, qp), dim=1))
        return x1, x2

class Net32x8(torch.nn.Module):
    def __init__(self):
        super(Net32x8, self).__init__()

        self.NC1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding =1, bias=True) # (1, 64, 64) -> (8, 64, 64)
        self.NC2 = nn.Conv2d(9, 9, kernel_size=(16, 4), stride=(16, 4), padding=0, bias=True) # (8, 64, 64) -> (8, 2, 2)
        self.NC3_1 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)
        self.NC3_2 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)

        self.FC1 = nn.Linear(10,2) # 9 -> 2
        self.FC2 = nn.Linear(10,2) # 9 -> 2

    def forward(self, org, pre, qp):

        T = F.relu(self.NC1(org))
        T = F.relu(self.NC2(torch.cat((T, pre), dim=1)))

        T1 = F.relu(self.NC3_1(T))
        T2 = F.relu(self.NC3_2(T))

        T1 = T1.squeeze(3).squeeze(2)
        T2 = T2.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1(torch.cat((T1, qp), dim=1))
        x2 = self.FC1(torch.cat((T2, qp), dim=1))
        return x1, x2

class Net16x8(torch.nn.Module):
    def __init__(self):
        super(Net16x8, self).__init__()

        self.NC1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding =1, bias=True) # (1, 64, 64) -> (8, 64, 64)
        self.NC2 = nn.Conv2d(9, 9, kernel_size=(8, 4), stride=(8, 4), padding=0, bias=True) # (8, 64, 64) -> (8, 2, 2)
        self.NC3_1 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)
        self.NC3_2 = nn.Conv2d(9, 9, kernel_size=2, stride=2, padding=0, bias=True) # (8, 2, 2) -> (8, 1, 1)

        self.FC1 = nn.Linear(10,2) # 9 -> 2
        self.FC2 = nn.Linear(10,2) # 9 -> 2

    def forward(self, org, pre, qp):

        T = F.relu(self.NC1(org))
        T = F.relu(self.NC2(torch.cat((T, pre), dim=1)))

        T1 = F.relu(self.NC3_1(T))
        T2 = F.relu(self.NC3_2(T))

        T1 = T1.squeeze(3).squeeze(2)
        T2 = T2.squeeze(3).squeeze(2)
        qp = qp / 64 - 0.5

        x1 = self.FC1(torch.cat((T1, qp), dim=1))
        x2 = self.FC1(torch.cat((T2, qp), dim=1))
        return x1, x2



if __name__ == '__main__':

    model = Net64x64()
    x = torch.ones(1, 1, 64, 64)
    out = model(x, x, torch.ones(1, 1))
    print (out)

    model = Net32x32()
    x = torch.ones(1, 1, 32, 32)
    out = model(x, x, torch.ones(1, 1))
    print (out)

    model = Net16x16()
    x = torch.ones(1, 1, 16, 16)
    out = model(x, x, torch.ones(1, 1))
    print (out)

    model = Net8x8()
    x = torch.ones(1, 1, 8, 8)
    out = model(x, x, torch.ones(1, 1))
    print (out)

    model = Net32x16()
    x = torch.ones(1, 1, 32, 16)
    out = model(x, x, torch.ones(1, 1))
    print (out)

    model = Net32x8()
    x = torch.ones(1, 1, 32, 8)
    out = model(x, x, torch.ones(1, 1))
    print (out)

    model = Net16x8()
    x = torch.ones(1, 1, 16, 8)
    out = model(x, x, torch.ones(1, 1))
    print (out)
