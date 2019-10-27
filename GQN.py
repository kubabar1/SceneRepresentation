import torch.distributions
from torch import nn
from torch.distributions import kl_divergence

from nn.lstm.GeneratorLSTMcell import GeneratorLSTMcell
from nn.lstm.InferenceLSTMcell import InferenceLSTMcell
from nn.cnn.TowerRepresentationNN import TowerRepresentationNN
from Properties import *


class GQN(nn.Module):
    def __init__(self):
        super(GQN, self).__init__()
        self.representation = TowerRepresentationNN()
        self.inference = InferenceLSTMcell(R_DEPTH + H_G_DEPTH + X_DEPTH + V_DEPTH + U_DEPTH, H_E_DEPTH)
        self.generator = GeneratorLSTMcell(R_DEPTH + Z_DEPTH + V_DEPTH, H_G_DEPTH)
        # self.down_sample_prior = nn.ConvTranspose2d(H_G_DEPTH, 3, (5, 5), stride=(1, 1), padding=(2, 2))
        self.down_sample_prior = nn.ConvTranspose2d(H_G_DEPTH, Z_DEPTH * 2, (5, 5), stride=(1, 1), padding=(2, 2))
        self.down_sample_posterior = nn.ConvTranspose2d(H_G_DEPTH, X_DEPTH * 2, (5, 5), stride=(1, 1), padding=(2, 2))
        self.down_sample_u_res = nn.ConvTranspose2d(U_DEPTH, X_DEPTH, (1, 1), stride=(1, 1))
        self.up_sample_v = nn.ConvTranspose2d(V_DEPTH, V_DEPTH, (16, 16), stride=(16, 16))
        self.down_sample_x_q = nn.Conv2d(X_DEPTH, X_DEPTH, (4, 4), stride=(4, 4))
        self.down_sample_u = nn.Conv2d(U_DEPTH, U_DEPTH, (4, 4), stride=(4, 4))
        self.up_sample_h_g = nn.ConvTranspose2d(H_G_DEPTH, H_G_DEPTH, (4, 4), stride=(4, 4))

    def estimate_ELBO(self, D, sigma_t):
        [x, v], [x_q, v_q] = D
        M = len(x)
        B = x[0].size()[0]

        r = torch.zeros((B, R_DEPTH, R_DIM, R_DIM)).to(DEVICE)

        for i in range(M):
            r_k = self.representation(x[i], v[i])
            r = r + r_k

        # Generator initial state
        c_g = torch.zeros([B, C_G_DEPTH, R_DIM, R_DIM]).to(DEVICE)
        h_g = torch.zeros([B, H_G_DEPTH, R_DIM, R_DIM]).to(DEVICE)
        u = torch.zeros([B, U_DEPTH, 4 * R_DIM, 4 * R_DIM]).to(DEVICE)

        # Interference initial state
        c_e = torch.zeros([B, C_E_DEPTH, R_DIM, R_DIM]).to(DEVICE)
        h_e = torch.zeros([B, H_E_DEPTH, R_DIM, R_DIM]).to(DEVICE)

        # ELBO initial state
        ELBO = 0

        for l in range(L):
            # pi = torch.distributions.Normal(self.down_sample_prior(h_g), F.softplus(self.down_sample_prior(h_g)))
            z_mu_pi, z_var_pi = torch.chunk(self.down_sample_prior(h_g), 2, dim=1)
            pi = torch.distributions.Normal(z_mu_pi, torch.exp(z_var_pi / 2))

            c_e, h_e = self.inference(torch.cat(
                [self.down_sample_x_q(x_q),
                 self.up_sample_v(v_q),
                 h_g,
                 r,
                 self.down_sample_u(u)],
                dim=1), (h_e, c_e))

            # q = torch.distributions.Normal(self.down_sample_prior(h_e), F.softplus(self.down_sample_prior(h_e)))
            z_mu_q, z_var_q = torch.chunk(self.down_sample_prior(h_e), 2, dim=1)
            q = torch.distributions.Normal(z_mu_q, torch.exp(z_var_q / 2))

            z = q.sample()

            c_g, h_g = self.generator(torch.cat([self.up_sample_v(v_q), r, z], dim=1), (h_g, c_g))
            u = u + self.up_sample_h_g(h_g)

            ELBO -= torch.sum(kl_divergence(q, pi), dim=[1, 2, 3])

        ELBO += torch.sum(torch.distributions.Normal(self.down_sample_u_res(u), sigma_t).log_prob(x_q), dim=[1, 2, 3])

        return ELBO

    def generate(self, D, v_q, sigma_t):
        [x_tensors, v_tensors], _ = D
        M = len(x_tensors)
        B = x_tensors[0].size()[0]

        r = torch.zeros((B, R_DEPTH, R_DIM, R_DIM)).to(DEVICE)

        for i in range(M):
            r_k = self.representation(x_tensors[i], v_tensors[i])
            r = r + r_k

        # Generator initial state
        c_g = torch.zeros([B, C_G_DEPTH, R_DIM, R_DIM]).to(DEVICE)
        h_g = torch.zeros([B, H_G_DEPTH, R_DIM, R_DIM]).to(DEVICE)
        u = torch.zeros([B, U_DEPTH, 4 * R_DIM, 4 * R_DIM]).to(DEVICE)

        for l in range(L):
            z_mu_pi, z_var_pi = torch.chunk(self.down_sample_prior(h_g), 2, dim=1)
            pi = torch.distributions.Normal(z_mu_pi, torch.exp(z_var_pi / 2))

            z = pi.sample()

            c_g, h_g = self.generator(torch.cat([self.up_sample_v(v_q), r, z], dim=1), (h_g, c_g))
            u += self.up_sample_h_g(h_g)

        x_q = torch.distributions.Normal(self.down_sample_u_res(u), sigma_t).sample()
        return torch.clamp(x_q, 0, 1)
