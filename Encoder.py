import torch.distributions
import math
from nn.lstm.LSTMcellGQN import LSTMcellGQN
from nn.cnn.RepresentationNN import *
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Properties import *


class Encoder:
    def __init__(self):
        self.model = RepresentationNN()

    def estimate_ELBO(self, D, sigma_t):
        r = torch.zeros((36, 256, 16, 16))
        M = len(D[0][0])
        B = len(D)

        o = []
        o_q = []

        for batch in D:
            b = []
            for k in range(M):
                b.append(batch[0][k])
            o.append(b)
            o_q.append(batch[1])

        x_tensors = []
        v_tensors = []
        x_q_tensors = []
        v_q_tensors = []

        for i in range(M):
            tmp_x = []
            tmp_v = []
            for j in range(B):
                tmp_x.append(D[j][0][i][0])
                tmp_v.append(D[j][0][i][1].view(1, 7, 1, 1))
            x_tensors.append(torch.cat(tmp_x))
            v_tensors.append(torch.cat(tmp_v))

        for j in range(B):
            x_q_tensors.append(D[j][1][0])
            v_q_tensors.append(D[j][1][1].view(1, 7, 1, 1))

        for i in range(M):
            r_k = self.model(x_tensors[i], v_tensors[i])
            r = r + r_k

        print("#################################")

        print(r.size())

        print("#################################")

        print(torch.cat(v_q_tensors).size())

        interference = LSTMcellGQN(torch.cat(v_q_tensors), r)
        generator = LSTMcellGQN(torch.cat(v_q_tensors), r)

        # Generator initial state
        c_g = torch.zeros([B, 256, 16, 16])
        h_g = torch.zeros([B, 256, 16, 16])
        u = torch.zeros([B, 256, 16, 16])

        # Interference initial state
        c_e = torch.zeros([B, 256, 16, 16])
        h_e = torch.zeros([B, 256, 16, 16])

        ELBO = torch.zeros([B, 3, 64, 64])

        conv1 = nn.ConvTranspose2d(256, 3, (4, 4), stride=(4, 4))
        conv2 = nn.ConvTranspose2d(256, 3, (4, 4), stride=(4, 4))

        for l in range(L):  # L - number of generative layers
            pi = torch.distributions.Normal(conv1(h_g), F.softplus(conv1(h_g)))
            c_e, h_e, _ = interference(c_e, h_e)
            q = torch.distributions.Normal(conv2(h_e), F.softplus(conv2(h_e)))
            z = q.sample()
            c_g, h_g, u = generator(c_g, h_g)
            ELBO = ELBO - torch.distributions.kl.kl_divergence(pi, q)
        conv3 = nn.ConvTranspose2d(256, 3, (61, 61), stride=(1, 1))
        ttmmpp = torch.distributions.Normal(conv3(u), sigma_t).log_prob(x_q_tensors)
        ELBO = ELBO + ttmmpp
        return ELBO

