import math
import torch
import matplotlib.pyplot as plt
import torchvision


def mi(mi_I, mi_F, mi_N, t):
    return max(mi_F + (mi_I - mi_F) * (1 - t / mi_N), mi_F)


def sigma(sigma_F, sigma_I, sigma_N, t):
    return max(sigma_F + (sigma_I - sigma_F) * (1 - t / sigma_N), sigma_F)


def gamma(beta_1, beta_2, t):
    return mi(t) * (math.sqrt(1 - beta_2)) / (1 - beta_1)


def tensor_to_image(tensor_image):
    return torch.squeeze(tensor_image).permute(1, 2, 0)


def show_image_comparation(generated_x, reverence_x):
    plt.figure()
    plt.axis('off')
    plt.imshow(torchvision.utils.make_grid(generated_x, nrow=int(math.sqrt(generated_x.size()[0]))).cpu().detach().numpy().transpose((1, 2, 0)))

    plt.figure()
    plt.axis('off')
    plt.imshow(torchvision.utils.make_grid(reverence_x, nrow=int(math.sqrt(generated_x.size()[0]))).cpu().detach().numpy().transpose((1, 2, 0)))

    plt.show()


def test(text: str, align: bool = True) -> str:
    print("test")
