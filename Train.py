from torchvision import transforms

from GQN import GQN
from Properties import *
from dataset.ScenesDataset import ScenesDataset, sample_batch

transform = transforms.Compose([transforms.Resize(IMAGE_RESIZE)])
scenes_dataset = ScenesDataset(dataset_root_path=DATA_PATH, json_path=JSON_PATH, transform=transform)

for t in range(S_max):
    D = sample_batch(scenes_dataset, B)
    ELBO = 0
    gqn = GQN()
    gqn.estimate_ELBO(D, 2)