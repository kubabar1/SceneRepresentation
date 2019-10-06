from torchvision import transforms

from GQN import GQN
from Properties import *
from dataset.ScenesDataset import ScenesDataset, sample_batch

import torch
from torch import nn
from torch import optim

transform = transforms.Compose([transforms.Resize(IMAGE_RESIZE)])
scenes_dataset = ScenesDataset(dataset_root_path=DATA_PATH, json_path=JSON_PATH, transform=transform)

model = GQN()
model.to(DEVICE)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for t in range(S_max):
    D = sample_batch(scenes_dataset, B)
    optimizer.zero_grad()
    ELBO_loss = model.estimate_ELBO(D, 2)
    ELBO_loss.backward()
    optimizer.step()
    print("ELBO_loss=" + str(ELBO_loss))
