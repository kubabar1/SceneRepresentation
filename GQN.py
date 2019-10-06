import torch.distributions
from nn.lstm.GeneratorLSTMcell import GeneratorLSTMcell
from nn.lstm.InferenceLSTMcell import InferenceLSTMcell
from nn.cnn.RepresentationNN import *
import torch
import torch.nn.functional as F
from Properties import *


class GQN(nn.Module):
    def __init__(self):
        super(GQN, self).__init__()
        self.model = RepresentationNN()
        self.inference = InferenceLSTMcell()
        self.generator = GeneratorLSTMcell()
        self.conv1 = nn.ConvTranspose2d(128, 3, (5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.ConvTranspose2d(128, 3, (1, 1), stride=(1, 1))

    def estimate_ELBO(self, D, sigma_t):

        r = torch.zeros((36, 256, 16, 16)).to(DEVICE)
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
            x_tensors.append(torch.cat(tmp_x).to(DEVICE))
            v_tensors.append(torch.cat(tmp_v).to(DEVICE))

        for j in range(B):
            x_q_tensors.append(D[j][1][0].to(DEVICE))
            v_q_tensors.append(D[j][1][1].view(1, 7, 1, 1).to(DEVICE))

        for i in range(M):
            r_k = self.model(x_tensors[i], v_tensors[i])
            r = r + r_k

        # print("#################################")
        # print(r.size())
        # print("#################################")
        # print(torch.cat(v_q_tensors).size())
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        # Generator initial state
        c_g = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)
        h_g = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)
        u = torch.zeros([B, 128, 4 * r.size()[2], 4 * r.size()[3]]).to(DEVICE)

        # Interference initial state
        c_e = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)
        h_e = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)

        ELBO = 0



        for l in range(L):  # L - number of generative layers
            pi = torch.distributions.Normal(self.conv1(h_g), F.softplus(self.conv1(h_g)))
            c_e, h_e = self.inference(torch.cat(x_q_tensors), torch.cat(v_q_tensors), r, c_e, h_e, h_g, u)
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print(h_e.size())
            q = torch.distributions.Normal(self.conv1(h_e), F.softplus(self.conv1(h_e)))
            z = q.sample()
            # print(z.size())
            c_g, h_g, u = self.generator(h_g, torch.cat(v_q_tensors), r, z, c_g, u)
            ELBO -= torch.sum(torch.distributions.kl.kl_divergence(q, pi))
            # print("ELBO=" + str(ELBO))
        # print(torch.sum(torch.distributions.Normal(self.conv2(u), sigma_t).log_prob(torch.cat(x_q_tensors))))
        ELBO += torch.sum(torch.distributions.Normal(self.conv2(u), sigma_t).log_prob(torch.cat(x_q_tensors)))
        # print("ELBO=" + str(ELBO))
        return ELBO
