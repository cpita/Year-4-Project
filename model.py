import torch.nn.functional as F
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x