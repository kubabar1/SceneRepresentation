from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms

from GQN import GQN
from Properties import IMAGE_RESIZE, DATA_PATH, JSON_PATH, EPSILON, B, DEVICE, S_MAX, BETA_1, BETA_2, MI_I
from Utils import gamma, sigma, mi, show_image_comparation
from dataset.ScenesDataset import ScenesDataset, sample_batch

from torch import nn
from torch import optim

from nn.optim.Scheduler import Scheduler

transform = transforms.Compose([transforms.Resize(IMAGE_RESIZE)])
scenes_dataset = ScenesDataset(dataset_root_path=DATA_PATH, json_path=JSON_PATH, transform=transform)

if __name__ == '__main__':
    model = GQN().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=MI_I, betas=(BETA_1, BETA_2), eps=EPSILON)
    scheduler = Scheduler(optimizer)

    sigma_t = sigma(0)

    for t in range(S_MAX):
        D = sample_batch(scenes_dataset, B=B, device=DEVICE)
        ELBO_loss = model.estimate_ELBO(D, sigma_t)
        (-ELBO_loss.mean()).backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        sigma_t = sigma(t)
        print("#" + str(t) + " ELBO_loss=" + str(-ELBO_loss.mean()) + ", lr=" + str(scheduler.get_lr()[0]))

    batch = sample_batch(scenes_dataset, B, device=DEVICE)
    [x_tensor_ref, v_tensor_ref], _ = sample_batch(scenes_dataset, 1, device=DEVICE)
    x_tensor_ref = x_tensor_ref[0]
    v_tensor_ref = v_tensor_ref[0]
    x_q = model.generate(batch,
                         v_tensor_ref.expand(B, v_tensor_ref.size()[1], v_tensor_ref.size()[2], v_tensor_ref.size()[3]),
                         sigma_t)
    for x in x_q:
        show_image_comparation(x_tensor_ref, v_tensor_ref, x)
