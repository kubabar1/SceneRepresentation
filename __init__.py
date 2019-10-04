from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions

from nn.cnn.RepresentationNN import *
from torchvision import transforms, datasets
import torch
import math
import matplotlib.pyplot as plt

import torch.nn.functional as F

from nn.lstm.LSTMcellGQN import LSTMcellGQN

EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_DATA_PATH = 'data'

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
# , transforms.Normalize((0.5,), (0.5,))

images_set = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)

images_loader = DataLoader(images_set, shuffle=False)
images_iterator = iter(images_loader)

w = [[10, -10, 2], [10, 0, 2], [10, 10, 2], [0, 10, 2], [-10, 10, 2], [-10, 0, 2], [-10, -10, 2], [0, -10, 2]]
y = [45, 90, 135, 180, 225, 270, 315, 0]
p = [15, 15, 15, 15, 15, 15, 15, 15]

v_k = [(w_i[0], w_i[1], w_i[2], math.cos(y_i), math.sin(y_i), math.cos(p_i), math.sin(p_i)) for w_i, y_i, p_i in
       zip(w, y, p)]

v_k_dataset = TensorDataset(torch.FloatTensor(v_k))
v_k_dataloader = DataLoader(v_k_dataset)
v_k_iterator = iter(v_k_dataloader)

full_data = [[image[0], v] for image, v in zip(images_loader, v_k)]


def data_generator():
    yield [next(images_iterator)[0][0], next(v_k_iterator)]


def estimate_ELBO(D, sigma_t):
    r = torch.zeros((1, 256, 16, 16))
    model = RepresentationNN()
    v_q_tmp = None  # TODO delete
    x_q_tmp = None  # TODO delete
    for image, v in D:
        r_k = model(image, torch.tensor(v))
        print(r_k.size())
        r = r + r_k
        v_q_tmp = v  # TODO delete
        x_q_tmp = image  # TODO delete
        plt.imshow(torch.squeeze(image).permute(1, 2, 0))
        # plt.show()

    interference = LSTMcellGQN(torch.tensor(v_q_tmp), r)
    generator = LSTMcellGQN(torch.tensor(v_q_tmp), r)

    print(r)
    print(r.size())
    print(interference)

    # Generator initial state
    c_g = torch.zeros([1, 256, 16, 16])
    h_g = torch.zeros([1, 256, 16, 16])
    u = torch.zeros([1, 256, 16, 16])

    # Interference initial state
    c_e = torch.zeros([1, 256, 16, 16])
    h_e = torch.zeros([1, 256, 16, 16])

    ELBO = torch.zeros((1, 3, 64, 64))
    for l in range(L):  # L - number of generative layers
        print("test")
        # print(h_g.size())
        # print(h_g[0][0])
        # print(h_g[0][0].std(dim=0).size())
        # print(h_g[0][0].std(dim=1).size())
        conv1 = nn.ConvTranspose2d(256, 3, (4, 4), stride=(4, 4))
        conv2 = nn.ConvTranspose2d(256, 3, (4, 4), stride=(4, 4))

        print(h_g.size())
        # print(conv1(h_g).size())
        print(h_e.size())
        # print(conv1(h_g).size())

        pi = torch.distributions.Normal(conv1(h_g), F.softplus(conv1(h_g)))

        print(c_e.size())
        print(h_e.size())
        c_e, h_e, _ = interference(c_e, h_e)

        q = torch.distributions.Normal(conv2(h_e), F.softplus(conv2(h_e)))

        # print(F.softplus(h_e))

        
        print(torch.distributions.kl.kl_divergence(pi, q))

        z = q.sample()
        # print("z:")
        # print(z)

        c_g, h_g, u = generator(c_g, h_g)
        # print("ELBO:")
        print(torch.distributions.kl.kl_divergence(pi, q).size())

        ELBO = ELBO - torch.distributions.kl.kl_divergence(pi, q)
        # print(ELBO)

    conv3 = nn.ConvTranspose2d(256, 3, (61, 61), stride=(1, 1))

    print(ELBO)
    print(ELBO.size())
    print(sigma_t)
    print(u.size())
    print(x_q_tmp.size())
    print(conv3(u))
    print(conv3(u).size())
    print(F.relu(conv3(u)))
    print(F.relu(conv3(u)).size())
    # print(conv3(x_q_tmp).size())
    print(torch.distributions.Normal(conv3(u), sigma_t))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    ttmmpp = torch.distributions.Normal(conv3(u), sigma_t).log_prob(x_q_tmp)
    print(ELBO.size())
    print(ttmmpp.size())

    ELBO = ELBO + ttmmpp
    print(ELBO)
    return ELBO


def generate(D, v):
    r = 0
    model = RepresentationNN()
    for image, v in full_data:
        r_k = model(image, torch.tensor(v))
        r = r + r_k
        plt.imshow(torch.squeeze(image).permute(1, 2, 0))
        plt.show()
    print(r)
    print(r.size())

    # Generator initial state
    # (c_g_0, h_g_0, u_0) <- (0,0,0)


sigma_i = 2.0
sigma_f = 0.7
sigma_s = lambda s: max(sigma_f + (sigma_i - sigma_f) * (1 - s / (2 * 10 ** 5)), sigma_f)

mi_i = 5 * 10 ** -4
mi_f = 5 * 10 ** -5
mi_s = lambda s: max(mi_f + (mi_i - mi_f) * (1 - s / (1.6 * 10 ** 6)), mi_f)

epsilon = 10 ** -8

beta_1 = 0.9
beta_2 = 0.999

gamma_s = lambda s: mi_s(s) * math.sqrt(1 - beta_2) / (1 - beta_1)

L = 12
B = 36
# S_max = 2 * 10 ** 6
S_max = 100

if __name__ == '__main__':
    # image, v = next(data_generator())
    # print(image)
    # print(image.shape)
    # print(v)
    # print(v.shape)
    # plt.imshow(image.permute(1, 2, 0))
    # plt.show()

    model = RepresentationNN()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), betas=[beta_1, beta_2], lr=gamma_s(0))

    M = 8
    K = 8

    print('#############')
    print(sigma_s(0))
    print('#############')

    estimate_ELBO(full_data, sigma_s(0))

    # for t in range(S_max):
    # for param_group in optimizer.param_groups:
    #    param_group['lr'] = gamma_s(t)
    # print(t)
    # optimizer.zero_grad()

    # D = sample_batch(B, M, K)
    # ELBO = estimate_ELBO(D, sigma_s(t))

    # log_ps = model(data)
    # loss = criterion(log_ps, labels)
    # loss.backward()
    # optimizer.step()

'''
    a = [1., 2., 3.]

    b = [[1., 2., 3.],
         [4., 5., 6.],
         [7., 8., 9.]]

    c = [[[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]],
         [[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]],
         [[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]]]

    a = torch.tensor(a)
    b = torch.tensor(b)
    c = torch.tensor(c)

    print(a)
    print(b)
    print(c)

    print(a.std(dim=0))
    print(a.mean(dim=0))

    print(b.std(dim=0))
    print(b.mean(dim=0))
    print(b.std(dim=1))
    print(b.mean(dim=1))

    print(c.std(dim=0))
    print(c.mean(dim=0))
    print(c.std(dim=1))
    print(c.mean(dim=1))
    print(c.std(dim=2))
    print(c.mean(dim=2))'''
