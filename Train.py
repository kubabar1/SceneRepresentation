import math
from torchvision import transforms

from GQN import GQN
from Properties import *
from dataset.ScenesDataset import ScenesDataset, sample_batch

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

transform = transforms.Compose([transforms.Resize(IMAGE_RESIZE)])
scenes_dataset = ScenesDataset(dataset_root_path=DATA_PATH, json_path=JSON_PATH, transform=transform)

def test(text: str, align: bool = True) -> str:
    print("test")

mi = lambda t: max(MI_F + (MI_I - MI_F) * (1 - t / MI_N), MI_F)

sigma = lambda t: max(SIGMA_F + (SIGMA_I - SIGMA_F) * (1 - t / SIGMA_N), SIGMA_F)

gamma = lambda t: mi(t) * (math.sqrt(1 - BETA_2)) / (1 - BETA_1)

gamma_NOT_WORKING = lambda t: mi(t) * (math.sqrt(1 - BETA_2 ** t)) / (1 - BETA_1 ** t)

model = GQN()
model.to(DEVICE)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=gamma(0), betas=(BETA_1, BETA_2), eps=EPSILON)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma(0))

mi_t = mi(0)
sigma_t = sigma(0)

for t in range(S_max):  # tqdm(range(S_max)):
    D = sample_batch(scenes_dataset, B)
    optimizer.zero_grad()
    ELBO_loss = model.estimate_ELBO(D, sigma_t)
    ELBO_loss.backward()
    optimizer.step()
    scheduler.step()
    mi_t = mi(t)
    sigma_t = sigma(t)
    print("#" + str(t) + " ELBO_loss=" + str(ELBO_loss.item()) + ", lr=" + str(scheduler.get_lr()[0]))
