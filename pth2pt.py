import torch
from model import Net

model = Net()
model.load_state_dict(torch.load("./model/model_epoch600.pth"))
model.eval()
traced_script_module = torch.jit.trace(model, (torch.ones(1, 1, 32, 32), torch.ones(1, 1, 32, 32), torch.ones(1, 1)))
traced_script_module.save("./model/model_epoch600.pt")