import math
import torch
import matplotlib.pyplot as plt

from Properties import MI_F, MI_I, MI_N, SIGMA_F, SIGMA_I, SIGMA_N, BETA_1, BETA_2


def mi(t): return max(MI_F + (MI_I - MI_F) * (1 - t / MI_N), MI_F)


def sigma(t):
    return max(SIGMA_F + (SIGMA_I - SIGMA_F) * (1 - t / SIGMA_N), SIGMA_F)


def gamma(t):
    return mi(t) * (math.sqrt(1 - BETA_2)) / (1 - BETA_1)


def gamma_NOT_WORKING(t):
    return mi(t) * (math.sqrt(1 - BETA_2 ** t)) / (1 - BETA_1 ** t)


def tensor_to_image(tensor_image):
    return torch.squeeze(tensor_image).permute(1, 2, 0)


def show_image_comparation(reference_img, reverence_v, result_img):
    fig = plt.figure()
    fig.suptitle('v = [' + ', '.join(map(str, reverence_v.cpu().view(7).numpy())) + ']')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(tensor_to_image(reference_img.cpu()))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(tensor_to_image(result_img.cpu()))
    plt.show()


def test(text: str, align: bool = True) -> str:
    print("test")
