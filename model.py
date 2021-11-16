import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()  #调用父类构造函数

        self.org = nn.Conv2d(1, 16, kernel_size=4, stride=4, padding=0, bias=True) #(8,8,16)
        self.pre = nn.Conv2d(2, 16, kernel_size=4, stride=4, padding=0, bias=True) #(8,8,16)

        self.NC1 = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=0, bias=True) #(2,2,32)
        self.NC2 = nn.Conv2d(32, 128, kernel_size=2, stride=2, padding=0, bias=True) #(1,1,128)

        self.FC1 = nn.Linear(129,64)
        self.FC2 = nn.Linear(65,2)

    def forward(self, org, pre, qp):

        F1 = F.relu(self.org(org))
        F2 = F.relu(self.pre(torch.cat((org, pre), dim=1)))

        T = F.relu(self.NC1(torch.cat((F1, F2), dim=1)))
        T = F.relu(self.NC2(T))

        T = torch.squeeze(T)

        qp = qp/54 - 0.5
        qp = qp.unsqueeze(1)
        # print(T.size(), qp.size())

        x = F.relu(self.FC1(torch.cat((T, qp), dim=1)))
        x = F.relu(self.FC2(torch.cat((x, qp), dim=1)))
        return x

if __name__ == '__main__':
    model = Net()
    x = torch.ones(1, 1, 32, 32)
    out = model(x, x, 22)
    print (out)
