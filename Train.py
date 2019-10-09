from torchvision import transforms

from GQN import GQN
from Properties import IMAGE_RESIZE, DATA_PATH, JSON_PATH, EPSILON, B, DEVICE, S_MAX, BETA_1, BETA_2
from Utils import gamma, sigma, mi, show_image_comparation
from dataset.ScenesDataset import ScenesDataset, sample_batch

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

transform = transforms.Compose([transforms.Resize(IMAGE_RESIZE)])
scenes_dataset = ScenesDataset(dataset_root_path=DATA_PATH, json_path=JSON_PATH, transform=transform)

if __name__ == '__main__':
    model = GQN()
    model.to(DEVICE)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=gamma(0), betas=(BETA_1, BETA_2), eps=EPSILON)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma(0))

    mi_t = mi(0)
    sigma_t = sigma(0)

    for t in range(S_MAX):  # tqdm(range(S_max)):
        D = sample_batch(scenes_dataset, B=B, device=DEVICE)
        optimizer.zero_grad()
        ELBO_loss = model.estimate_ELBO(D, sigma_t)
        ELBO_loss.backward()
        optimizer.step()
        scheduler.step()
        mi_t = mi(t)
        sigma_t = sigma(t)
        print("#" + str(t) + " ELBO_loss=" + str(ELBO_loss.item()) + ", lr=" + str(scheduler.get_lr()[0]))

    batch = sample_batch(scenes_dataset, B, device=DEVICE)
    [x_tensor_ref, v_tensor_ref], _ = sample_batch(scenes_dataset, 1, device=DEVICE)
    x_tensor_ref = x_tensor_ref[0]
    v_tensor_ref = v_tensor_ref[0]
    x_q = model.generate(batch, v_tensor_ref, sigma_t)
    for x in x_q:
        show_image_comparation(x_tensor_ref, v_tensor_ref, x)
