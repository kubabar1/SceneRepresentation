import torch.distributions
from nn.lstm.GeneratorLSTMcell import GeneratorLSTMcell
from nn.lstm.InterferenceLSTMcell import InterferenceLSTMcell
from nn.cnn.RepresentationNN import *
import torch
import torch.nn.functional as F
from Properties import *


class Encoder:
    def __init__(self):
        self.model