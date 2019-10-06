import torch.nn as nn
import torch

from nn.lstm.LSTMcellGQN import LSTMcellGQN


class InferenceLSTMcell(LSTMcellGQN):
    def __init__(self):
        super(InferenceLSTMcell, self).__init__()
        self.down_sample = nn.Conv2d(256 + 7 + 2 * 128 + 3, 128, (5, 5), stride=(1, 1), padding=(2, 2))
        self.down_sample_x_q = nn.Conv2d(3, 3, (4, 4), stride=(4, 4))

    def forward(self, x_q, v_q, r, c_e, h_e, h_g, u):
        # print("*********************************")
        # print(u.size())
        # print(self.down_sample_x_q(x_q).size())
        input = torch.cat([self.down_sample_x_q(x_q), self.up_sample_v(v_q), h_g, h_e, r], dim=1)
        # print(input.size())

        input = self.down_sample(input)
        # print(input.size())
        ft = self.forget_layer(input)
        candidates = self.candidate_layer(input)
        it = self.input_layer(input)
        c_e = c_e * ft + candidates * it
        ot = self.output_layer(input)
        h_e = ot * torch.tanh(c_e)
        # print(h_e.size())
        # print(c_e.size())
        return c_e, h_e
