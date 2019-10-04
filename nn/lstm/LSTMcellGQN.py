import torch.nn as nn
import torch


class LSTMcellGQN(nn.Module):
    def __init__(self, v_q, r):
        super(LSTMcellGQN, self).__init__()
        self.conv = nn.Conv2d(256, 256, (5, 5), stride=(1, 1), padding=(2, 2))
        self.conv1 = nn.Conv2d(256 + 7 + 256 + 256, 256, (5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(256, 256, (4, 4), stride=(4, 4))
        self.forget_layer = nn.Sigmoid()
        self.input_layer = nn.Sigmoid()
        self.output_layer = nn.Sigmoid()
        self.candidate_layer = nn.Tanh()
        self.n_layers = 12
        self.n_hidden = 256
        self.v_q = v_q
        self.r = r

    def forward(self, prev_ht, prev_ct):
        zl = self.conv(prev_ht)
        v_q = self.v_q.expand(36, 7, 16, 16)
        r = self.r
        # print("#####################")
        print(prev_ht.size())
        print(v_q.size())
        print(r.size())
        print(zl.size())
        # print("#####################")
        h_combined = torch.cat([prev_ht, v_q, r, zl], dim=1)
        # print(h_combined.size())
        # print("#####################")
        ft = self.forget_layer(self.conv1(h_combined))
        # print(ft.size())
        # print("#####################")
        candidate = self.candidate_layer(self.conv1(h_combined))
        it = self.input_layer(self.conv1(h_combined))
        ct = prev_ct * ft + candidate * it
        ot = self.output_layer(self.conv1(h_combined))
        ht = ot * torch.tanh(ct)
        ut = self.conv2(ht)
        # print("#####################")
        # print("#####################")
        # print("#####################")
        # print(ht.size())
        # print(ut.size())
        return ht, ct, ut

