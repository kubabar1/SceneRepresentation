import torch.nn as nn


class LSTMcellGQN(nn.Module):
    def __init__(self):
        super(LSTMcellGQN, self).__init__()
        self.forget_layer = nn.Sigmoid()
        self.input_layer = nn.Sigmoid()
        self.output_layer = nn.Sigmoid()
        self.candidate_layer = nn.Tanh()
        self.up_sample_h_g = nn.ConvTranspose2d(128, 128, (4, 4), stride=(4, 4))
        self.up_sample_v = nn.ConvTranspose2d(7, 7, (16, 16))  # TODO: check
