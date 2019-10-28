import torch.nn as nn
import torch
from torch.nn.modules.rnn import RNNCellBase


class LSTMcellGQN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMcellGQN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.down_sample = nn.Conv2d(input_size + hidden_size, hidden_size, (5, 5), stride=(1, 1), padding=(2, 2))
        self.forget_layer = nn.Sigmoid()
        self.input_layer = nn.Sigmoid()
        self.output_layer = nn.Sigmoid()
        self.candidate_layer = nn.Tanh()

    def forward(self, input, hx) :
        h = hx[0]
        c = hx[1]
        input = self.down_sample(torch.cat([input, h], dim=1))
        ft = self.forget_layer(input)
        candidates = self.candidate_layer(input)
        it = self.input_layer(input)
        c = c * ft + candidates * it
        ot = self.output_layer(input)
        h = ot * torch.tanh(c)
        return c, h
