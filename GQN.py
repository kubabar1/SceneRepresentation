import torch.distributions
from torch import nn
from torch.distributions import kl_divergence, Normal

from nn.cnn.RepresentationNNTypes import RepresentationNNTypes
from nn.cnn.TowerRepresentationNN import TowerRepresentationNN
from nn.cnn.PyramidRepresentationNN import PyramidRepresentationNN
from nn.cnn.PoolRepresentationNN import PoolRepresentationNN
from nn.lstm.LSTMcellGQN import LSTMcellGQN


class GQN(nn.Module):
    def __init__(self, properties):
        super(GQN, self).__init__()
        self.properties = properties
        self.tmp = RepresentationNNTypes.POOL
        if properties.representation[0] is RepresentationNNTypes.POOL:
            self.representation = PoolRepresentationNN()
        elif properties.representation[0] is RepresentationNNTypes.PYRAMID:
            self.representation = PyramidRepresentationNN()
        else:
            self.representation = TowerRepresentationNN()
        self.inference = LSTMcellGQN(
            properties.R_depth + properties.H_g_depth + properties.X_depth + properties.V_depth + properties.U_depth,
            properties.H_e_depth)
        self.generator = LSTMcellGQN(properties.R_depth + properties.Z_depth + properties.V_depth,
                                           properties.H_g_depth)
        # self.down_sample_prior = nn.ConvTranspose2d(H_G_DEPTH, 3, (5, 5), stride=(1, 1), padding=(2, 2))
        self.down_sample_prior = nn.ConvTranspose2d(properties.H_g_depth, properties.Z_depth * 2, (5, 5), stride=(1, 1),
                                                    padding=(2, 2))
        self.down_sample_posterior = nn.ConvTranspose2d(properties.H_g_depth, properties.X_depth * 2, (5, 5),
                                                        stride=(1, 1), padding=(2, 2))
        self.down_sample_u_res = nn.ConvTranspose2d(properties.U_depth, properties.X_depth, (1, 1), stride=(1, 1))
        self.up_sample_v = nn.ConvTranspose2d(properties.V_depth, properties.V_depth, (16, 16), stride=(16, 16))
        self.up_sample_r = nn.ConvTranspose2d(properties.R_depth, properties.R_depth, (16, 16), stride=(16, 16))
        self.down_sample_x_q = nn.Conv2d(properties.X_depth, properties.X_depth, (4, 4), stride=(4, 4))
        self.down_sample_u = nn.Conv2d(properties.U_depth, properties.U_depth, (4, 4), stride=(4, 4))
        self.up_sample_h_g = nn.ConvTranspose2d(properties.H_g_depth, properties.H_g_depth, (4, 4), stride=(4, 4))

    def estimate_ELBO(self, D, sigma_t):
        device = self.properties.device
        [x, v], [x_q, v_q] = D
        M = len(x)
        B = x[0].size()[0]

        r = torch.zeros((B, self.properties.R_depth, self.properties.R_dim, self.properties.R_dim)).to(device)

        for i in range(M):
            r_k = self.representation(x[i], v[i])
            r += r_k

        # Generator initial state
        c_g = torch.zeros([B, self.properties.C_g_depth, self.properties.Z_dim, self.properties.Z_dim]).to(device)
        h_g = torch.zeros([B, self.properties.H_g_depth, self.properties.Z_dim, self.properties.Z_dim]).to(device)
        u = torch.zeros([B, self.properties.U_depth, 4 * self.properties.Z_dim, 4 * self.properties.Z_dim]).to(device)

        # Interference initial state
        c_e = torch.zeros([B, self.properties.C_e_depth, self.properties.Z_dim, self.properties.Z_dim]).to(device)
        h_e = torch.zeros([B, self.properties.H_e_depth, self.properties.Z_dim, self.properties.Z_dim]).to(device)

        # ELBO initial state
        ELBO = 0

        for l in range(self.properties.L):
            # pi = torch.distributions.Normal(self.down_sample_prior(h_g), F.softplus(self.down_sample_prior(h_g)))
            z_mu_pi, z_var_pi = torch.chunk(self.down_sample_prior(h_g), 2, dim=1)
            pi = torch.distributions.Normal(z_mu_pi, torch.exp(z_var_pi / 2))

            c_e, h_e = self.inference(torch.cat(
                [self.down_sample_x_q(x_q),
                 self.up_sample_v(v_q),
                 h_g,
                 r if h_e.size()[2] == r.size()[2] else self.up_sample_r(r),
                 self.down_sample_u(u)],
                dim=1), (h_e, c_e))

            # q = torch.distributions.Normal(self.down_sample_prior(h_e), F.softplus(self.down_sample_prior(h_e)))
            z_mu_q, z_var_q = torch.chunk(self.down_sample_prior(h_e), 2, dim=1)
            q = torch.distributions.Normal(z_mu_q, torch.exp(z_var_q / 2))

            z = q.rsample()  # TODO check difference between rsample and sample

            c_g, h_g = self.generator(torch.cat(
                [self.up_sample_v(v_q),
                 r if h_g.size()[2] == r.size()[2] else self.up_sample_r(r),
                 z], dim=1), (h_g, c_g))
            u = u + self.up_sample_h_g(h_g)

            ELBO -= torch.sum(kl_divergence(q, pi), dim=[1, 2, 3])

        ELBO += torch.sum(Normal(self.down_sample_u_res(u), sigma_t).log_prob(x_q), dim=[1, 2, 3])

        return ELBO

    def generate(self, D, v_q, sigma_t):
        device = self.properties.device
        [x_tensors, v_tensors], _ = D
        M = len(x_tensors)
        B = x_tensors[0].size()[0]

        r = torch.zeros((B, self.properties.R_depth, self.properties.R_dim, self.properties.R_dim)).to(device)

        for i in range(M):
            r_k = self.representation(x_tensors[i], v_tensors[i])
            r = r + r_k

        # Generator initial state
        c_g = torch.zeros([B, self.properties.C_g_depth, self.properties.Z_dim, self.properties.Z_dim]).to(device)
        h_g = torch.zeros([B, self.properties.H_g_depth, self.properties.Z_dim, self.properties.Z_dim]).to(device)
        u = torch.zeros([B, self.properties.U_depth, 4 * self.properties.Z_dim, 4 * self.properties.Z_dim]).to(device)

        for l in range(self.properties.L):
            z_mu_pi, z_var_pi = torch.chunk(self.down_sample_prior(h_g), 2, dim=1)
            pi = torch.distributions.Normal(z_mu_pi, torch.exp(z_var_pi / 2))

            z = pi.sample()

            c_g, h_g = self.generator(torch.cat(
                [self.up_sample_v(v_q),
                 r if h_g.size()[2] == r.size()[2] else self.up_sample_r(r),
                 z], dim=1), (h_g, c_g))
            u += self.up_sample_h_g(h_g)

        x_q = torch.distributions.Normal(self.down_sample_u_res(u),
                                         sigma_t).rsample()  # TODO check difference between rsample and sample
        return torch.clamp(x_q, 0, 1)
