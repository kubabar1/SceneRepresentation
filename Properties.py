import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATH = 'data'
JSON_PATH = 'data/observations.json'
IMAGE_RESIZE = (64, 64)

L = 12
B = 36
S_max = 50  # 2000000

MI_I = 5 * 10 ** (-4)
MI_F = 5 * 10 ** (-5)
MI_N = 1.6 * 10 ** 6

BETA_1 = 0.9
BETA_2 = 0.999

EPSILON = 10 ** (-8)

SIGMA_I = 2.0
SIGMA_F = 0.7
SIGMA_N = 2 * 10 ** 5

