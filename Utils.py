import math
import torch
import matplotlib.pyplot as plt


def mi(mi_I, mi_F, mi_N, t):
    return max(mi_F + (mi_I - mi_F) * (1 - t / mi_N), mi_F)


def sigma(sigma_F, sigma_I, sigma_N, t):
    return max(sigma_F + (sigma_I - sigma_F) * (1 - t / sigma_N), sigma_F)


def gamma(beta_1, beta_2, t):
    return mi(t) * (math.sqrt(1 - beta_2)) / (1 - beta_1)


def tensor_to_image(tensor_image):
    return torch.squeeze(tensor_image).permute(1, 2, 0)


def show_image_comparation(reference_img, reverence_v, result_img):
    fig = plt.figure()
    fig.suptitle('v = [' + ', '.join(map(str, reverence_v.cpu().view(7).numpy())) + ']')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(tensor_to_image(reference_img.cpu()))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(tensor_to_image(result_img.cpu().detach()))
    plt.show()


def test(text: str, align: bool = True) -> str:
    print("test")
