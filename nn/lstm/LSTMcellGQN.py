import torch.nn as nn
from torch.nn.modules.rnn import RNNCellBase


class LSTMcellGQN(nn.Module):
    def __init__(self):
        super(LSTMcellGQN, self).__init__()
        self.forget_layer = nn.Sigmoid()
        self.input_layer = nn.Sigmoid()
        self.output_layer = nn.Sigmoid()
        self.candidate_layer = nn.Tanh()
