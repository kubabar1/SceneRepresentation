import torch.nn as nn
import torch

from nn.lstm.LSTMcellGQN import LSTMcellGQN


class InterferenceLSTMcell(LSTMcellGQN):
    def __init__(self):
        super(InterferenceLSTMcell, self).__init__()
        self.down_sample = nn.Conv2d(256 + 7 + 2 * 128 + 3, 128, (5, 5), stride=(1, 1), padding=(2, 2))
        self.down_sample_x_q = nn.Conv2d(3, 3, (4, 4), stride=(4, 4))

    def forward(self, x_q, v_q, r, c_e, h_e, h_g, u):
        # print("*********************************")
        # print(r.size())
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

        input = self._cat_input(x_q, v_q, r, h_e, h_g, u)
        self.forget_layer(input)
        zl = self.conv(prev_ht)
        v_q = self.v_q.expand(36, 7, 16, 16)
        r = self.r
        h_combined = torch.cat([prev_ht, v_q, r, zl], dim=1)
        ft = self.forget_layer(self.conv1(h_combined))
        candidate = self.candidate_layer(self.conv1(h_combined))
        it = self.input_layer(self.conv1(h_combined))
        ct = prev_ct * ft + candidate * it
        ot = self.output_layer(self.conv1(h_combined))
        ht = ot * torch.tanh(ct)
        return ht, ct
