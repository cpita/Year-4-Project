import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass

model = ConvNet().to(device)