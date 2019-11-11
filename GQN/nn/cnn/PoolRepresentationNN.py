import torch.nn as nn
import torch.nn.functional as F
import torch


class PoolRepresentationNN(nn.Module):
    def __init__(self):
        super(PoolRepresentationNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(2, 2), stride=(2, 2))
        self.conv4 = nn.Conv2d(263, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        self.conv_skip_1 = nn.Conv2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
        self.conv_skip_2 = nn.Conv2d(263, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.AvgPool2d(kernel_size=(16, 16))

    def forward(self, x, v):
        skip_1 = F.relu(self.conv1(x))
        x = F.relu(self.conv2(skip_1))
        x = F.relu(self.conv3(x)) + F.relu(self.conv_skip_1(skip_1))
        skip_2 = torch.cat((x, v.expand(v.size(0), 7, 16, 16)), dim=1)
        x = F.relu(self.conv4(skip_2))
        x = F.relu(self.conv5(x)) + F.relu(self.conv_skip_2(skip_2))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        return x
