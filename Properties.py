import torch

DATA_PATH = 'data'
JSON_PATH = 'data/observations.json'
IMAGE_RESIZE = (64, 64)
B = 36
L = 12
S_max = 1  # 2000000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
