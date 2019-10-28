import torch.nn as nn
import torch.nn.functional as F
import torch


class PyramidRepresentationNN(nn.Module):
    def __init__(self):
        super(PyramidRepresentationNN, self).__init__()
        self.conv1 = nn.Conv2d(10, 32, (2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, (2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 128, (2, 2), stride=(2, 2))
        self.conv4 = nn.Conv2d(128, 256, (8, 8), stride=(8, 8))

    def forward(self, x, v):
        x = torch.cat((x, v.expand(v.size(0), 7, 64, 64)), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x
