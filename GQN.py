import torch.distributions
from nn.lstm.GeneratorLSTMcell import GeneratorLSTMcell
from nn.lstm.InferenceLSTMcell import InferenceLSTMcell
from nn.cnn.TowerRepresentationNN import *
import torch.nn.functional as F
from Properties import *


class GQN(nn.Module):
    def __init__(self):
        super(GQN, self).__init__()
        self.representation = TowerRepresentationNN()
        self.inference = InferenceLSTMcell()
        self.generator = GeneratorLSTMcell()
        self.conv1 = nn.ConvTranspose2d(128, 3, (5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.ConvTranspose2d(128, 3, (1, 1), stride=(1, 1))

    def estimate_ELBO(self, D, sigma_t):
        r = torch.zeros((36, 256, 16, 16)).to(DEVICE)
        [x_tensors, v_tensors], [x_q_tensors, v_q_tensors] = D
        B = x_q_tensors.size()[0]
        M = len(x_tensors)

        for i in range(M):
            r_k = self.representation(x_tensors[i], v_tensors[i])
            r = r + r_k

        # Generator initial state
        c_g = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)
        h_g = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)
        u = torch.zeros([B, 128, 4 * r.size()[2], 4 * r.size()[3]]).to(DEVICE)

        # Interference initial state
        c_e = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)
        h_e = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)

        ELBO = 0

        # L - number of generative layers
        for l in range(L):
            pi = torch.distributions.Normal(self.conv1(h_g), F.softplus(self.conv1(h_g)))
            c_e, h_e = self.inference(x_q_tensors, v_q_tensors, r, c_e, h_e, h_g, u)
            q = torch.distributions.Normal(self.conv1(h_e), F.softplus(self.conv1(h_e)))
            z = q.sample()
            c_g, h_g, u = self.generator(h_g, v_q_tensors, r, z, c_g, u)
            ELBO -= torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q, pi), dim=[1, 2, 3]))
            # print("ELBO=" + str(ELBO))
        ELBO += torch.mean(
            torch.sum(torch.distributions.Normal(self.conv2(u), sigma_t).log_prob(x_q_tensors), dim=[1, 2, 3]))
        return ELBO

    def generate(self, D, v_q, sigma_t):
        r = torch.zeros((36, 256, 16, 16)).to(DEVICE)
        [x_tensors, v_tensors], _ = D
        B = x_tensors[0].size()[0]
        M = len(x_tensors)

        for i in range(M):
            r_k = self.representation(x_tensors[i], v_tensors[i])
            r = r + r_k

        v_q = v_q.expand(B, v_q.size()[1], v_q.size()[2], v_q.size()[3])

        # Generator initial state
        c_g = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)
        h_g = torch.zeros([B, 128, r.size()[2], r.size()[3]]).to(DEVICE)
        u = torch.zeros([B, 128, 4 * r.size()[2], 4 * r.size()[3]]).to(DEVICE)

        for l in range(L):
            pi = torch.distributions.Normal(self.conv1(h_g), F.softplus(self.conv1(h_g)))
            z = pi.sample()
            c_g, h_g, u = self.generator(h_g, v_q, r, z, c_g, u)
        x_q = torch.distributions.Normal(self.conv2(u), sigma_t).sample()
        return x_q
