import torch.nn as nn
import torch

from nn.lstm.LSTMcellGQN import LSTMcellGQN


class InferenceLSTMcell(LSTMcellGQN):
    def __init__(self, input_size, hidden_size):
        super(InferenceLSTMcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.down_sample = nn.Conv2d(input_size + hidden_size, hidden_size, (5, 5), stride=(1, 1), padding=(2, 2))

    def forward(self, input, hx):
        h_e = hx[0]
        c_e = hx[1]
        input = self.down_sample(torch.cat([input, h_e], dim=1))
        ft = self.forget_layer(input)
        candidates = self.candidate_layer(input)
        it = self.input_layer(input)
        c_e = c_e * ft + candidates * it
        ot = self.output_layer(input)
        h_e = ot * torch.tanh(c_e)
        return c_e, h_e
