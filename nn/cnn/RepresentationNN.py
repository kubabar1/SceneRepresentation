import torch.nn as nn
import torch.nn.functional as F
import torch


class RepresentationNN(nn.Module):
    def __init__(self):
        super(RepresentationNN, self).__init__()
        # image 64x64x3
        self.conv1 = nn.Conv2d(3, 256, (2, 2), stride=(2, 2))
        self.conv1_skip = nn.Conv2d(256, 256, (1, 1), stride=(2, 2))
        # image 32x32x256
        self.conv2 = nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        # image 32x32x128
        self.conv3 = nn.Conv2d(128, 256, (2, 2), stride=(2, 2))
        # image 16x16x256
        self.conv4 = nn.Conv2d(256 + 7, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_skip = nn.Conv2d(256 + 7, 256, (1, 1), stride=(1, 1))
        # image 16x16x128
        self.conv5 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        # image 16x16x256
        self.conv6 = nn.Conv2d(256, 256, (1, 1), stride=(1, 1))
        # image 16x16x256
        # self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, v):
        # print(x.size())  # [1, 3, 64, 64]
        x = self.conv1(x)
        skip = self.conv1_skip(x)  # TODO: it is in right place?
        x = F.relu(x)
        # print(x.size())  # [1, 256, 32, 32]
        # x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        # print(skip.size())  # [1, 256, 32, 32]

        x = self.conv3(x)
        # print(x.size())  # [1, 256, 16, 16]
        x = x.add(skip)
        # print(x.size())  # [1, 256, 16, 16]
        # print(x.size())  # [1, 256, 16, 16]
        # x = self.pool(x)

        # print("----------------------------------")

        # v = v.view(7, 1, 1)

        v = v.expand(36, 7, 16, 16)

        # print(v.size()) # [1, 7, 16, 16]
        # print(x.size()) # [1, 256, 16, 16]

        x = torch.cat((x, v), dim=1)
        # print(x.size())  # [1, 263, 16, 16]
        skip2 = self.conv4_skip(x)  # TODO: it is in right place?

        x = F.relu(x)
        x = self.conv4(x)
        # print(x.size())  # [1, 128, 16, 16]
        # print(x_tmp2.size())  # [1, 128, 16, 16]
        x = F.relu(x)

        x = self.conv5(x)
        x = x.add(skip2)
        # print(x.size())  # [1, 256, 16, 16]
        x = F.relu(x)

        x = self.conv6(x)
        # print(x.size())  # [1, 256, 16, 16]
        x = F.relu(x)

        return x
