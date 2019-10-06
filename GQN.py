import torch.distributions
from nn.lstm.GeneratorLSTMcell import GeneratorLSTMcell
from nn.lstm.InterferenceLSTMcell import InterferenceLSTMcell
from nn.cnn.RepresentationNN import *
import torch
import torch.nn.functional as F
from Properties import *


class GQN:
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

        #print("#################################")
        #print(r.size())
        #print("#################################")
        #print(torch.cat(v_q_tensors).size())
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        interference = InterferenceLSTMcell()
        generator = GeneratorLSTMcell()

        # Generator initial state
        c_g = torch.zeros([B, 128, r.size()[2], r.size()[3]])
        h_g = torch.zeros([B, 128, r.size()[2], r.size()[3]])
        u = torch.zeros([B, 128, 4 * r.size()[2], 4 * r.size()[3]])

        # Interference initial state
        c_e = torch.zeros([B, 128, r.size()[2], r.size()[3]])
        h_e = torch.zeros([B, 128, r.size()[2], r.size()[3]])

        ELBO = 0

        conv1 = nn.ConvTranspose2d(128, 3, (5, 5), stride=(1, 1), padding=(2, 2))

        for l in range(L):  # L - number of generative layers
            pi = torch.distributions.Normal(conv1(h_g), F.softplus(conv1(h_g)))
            c_e, h_e = interference(torch.cat(x_q_tensors), torch.cat(v_q_tensors), r, c_e, h_e, h_g, u)
            #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            #print(h_e.size())
            q = torch.distributions.Normal(conv1(h_e), F.softplus(conv1(h_e)))
            z = q.sample()
            #print(z.size())
            c_g, h_g, u = generator(h_g, torch.cat(v_q_tensors), r, z, c_g, u)
            ELBO = ELBO - torch.sum(torch.distributions.kl.kl_divergence(q, pi))
            print("ELBO=" + str(ELBO))
        conv2 = nn.ConvTranspose2d(128, 3, (1, 1), stride=(1, 1))
        print(torch.sum(torch.distributions.Normal(conv2(u), sigma_t).log_prob(torch.cat(x_q_tensors))))
        ELBO = ELBO + torch.sum(torch.distributions.Normal(conv2(u), sigma_t).log_prob(torch.cat(x_q_tensors)))
        print("ELBO=" + str(ELBO))
        return ELBO
