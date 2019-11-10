import os.path
import math
import torch
import matplotlib.pyplot as plt
import torchvision
from torch import optim

from GQN import GQN


def mi(mi_I, mi_F, mi_N, t):
    return max(mi_F + (mi_I - mi_F) * (1 - t / mi_N), mi_F)


def sigma(sigma_F, sigma_I, sigma_N, t):
    return max(sigma_F + (sigma_I - sigma_F) * (1 - t / sigma_N), sigma_F)


def gamma(beta_1, beta_2, t):
    return mi(t) * (math.sqrt(1 - beta_2)) / (1 - beta_1)


def tensor_to_image(tensor_image):
    return torch.squeeze(tensor_image).permute(1, 2, 0)


def show_image_comparation(generated_x, reverence_x):
    plt.figure(1)
    plt.axis('off')
    plt.imshow(
        torchvision.utils.make_grid(generated_x, nrow=int(math.sqrt(generated_x.size()[0]))).cpu().numpy().transpose(
            (1, 2, 0)))

    plt.figure(2)
    plt.axis('off')
    plt.imshow(
        torchvision.utils.make_grid(reverence_x, nrow=int(math.sqrt(generated_x.size()[0]))).cpu().numpy().transpose(
            (1, 2, 0)))
    plt.show()


def save_image_comparation(generated_x, generated_x2, generated_x3, reference_x, representation, epoch, generated_images_path, generated_images2_path, generated_images3_path, referenced_images_path, representation_images_path):
    if not os.path.exists(generated_images_path):
        os.makedirs(generated_images_path)
    if not os.path.exists(generated_images2_path):
        os.makedirs(generated_images2_path)
    if not os.path.exists(generated_images3_path):
        os.makedirs(generated_images3_path)
    if not os.path.exists(referenced_images_path):
        os.makedirs(referenced_images_path)
    if not os.path.exists(representation_images_path):
        os.makedirs(representation_images_path)
    torchvision.utils.save_image(generated_x,
                                 os.path.join(generated_images_path, "generated_" + str(epoch) + ".png"),
                                 nrow=int(math.sqrt(generated_x.size()[0])))
    torchvision.utils.save_image(generated_x2,
                                 os.path.join(generated_images2_path, "generated2_" + str(epoch) + ".png"),
                                 nrow=int(math.sqrt(generated_x2.size()[0])))
    torchvision.utils.save_image(generated_x3,
                                 os.path.join(generated_images3_path, "generated3_" + str(epoch) + ".png"),
                                 nrow=int(math.sqrt(generated_x3.size()[0])))
    torchvision.utils.save_image(reference_x,
                                 os.path.join(referenced_images_path, "referenced_" + str(epoch) + ".png"),
                                 nrow=int(math.sqrt(reference_x.size()[0])))
    torchvision.utils.save_image(representation,
                                 os.path.join(representation_images_path, "representation_" + str(epoch) + ".png"),
                                 nrow=int(math.sqrt(representation.size()[0])))


def save_model(model, epoch, optimizer, loss, sigma_t, save_model_path):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'sigma_t': sigma_t
    }, os.path.join(save_model_path, "model_" + str(epoch) + ".pt"))


def load_model(model_path, properties):
    model = GQN(properties).to(properties.device)
    optimizer = optim.Adam(model.parameters(), lr=properties.mi_I, betas=(properties.beta_1, properties.beta_2),
                           eps=properties.epsilon)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    sigma_t = checkpoint['sigma_t']
    return model, epoch, optimizer, loss, sigma_t


def test(text: str, align: bool = True) -> str:
    print("test")
