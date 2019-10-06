import torch.nn as nn
import torch

from nn.lstm.LSTMcellGQN import LSTMcellGQN


class GeneratorLSTMcell(LSTMcellGQN):
    def __init__(self):
        super(GeneratorLSTMcell, self).__init__()
        self.down_sample = nn.Conv2d(256 + 7 + 128 + 3, 128, (5, 5), stride=(1, 1), padding=(2, 2))

    def forward(self, h_g, v_q, r, z, c_g, u):
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(v_q.size())
        # print(r.size())
        # print(h_g.size())
        # print(z.size())
        # print(self.up_sample_v(v_q).size())
        input = torch.cat([self.up_sample_v(v_q), r, h_g, z], dim=1)
        input = self.down_sample(input)
        ft = self.forget_layer(input)
        candidates = self.candidate_layer(input)
        it = self.input_layer(input)
        c_g = c_g * ft + candidates * it
        ot = self.output_layer(input)
        h_g = ot * torch.tanh(c_g)
        u = self.up_sample_h_g(h_g) + u
        return c_g, h_g, u
