from torchvision import transforms

from GQN import GQN
from Properties import Properties
from Utils import sigma, mi, show_image_comparation
from dataset.ScenesDataset import ScenesDataset, sample_batch

from torch import optim

from nn.optim.Scheduler import Scheduler

if __name__ == '__main__':
    properties = Properties()

    transform = transforms.Compose([transforms.Resize(properties.image_resize)])
    scenes_dataset = ScenesDataset(dataset_root_path=properties.data_path, json_path=properties.json_path,
                                   transform=transform)

    model = GQN(properties).to(properties.device)

    optimizer = optim.Adam(model.parameters(), lr=properties.mi_I, betas=(properties.beta_1, properties.beta_2),
                           eps=properties.epsilon)
    scheduler = Scheduler(optimizer, mi=mi, mi_I=properties.mi_I, mi_F=properties.mi_F, mi_N=properties.mi_N)

    sigma_t = properties.sigma_I

    for t in range(properties.s_max):
        D = sample_batch(scenes_dataset, B=properties.B, device=properties.device)
        ELBO_loss = model.estimate_ELBO(D, sigma_t)

        # TODO: test flow and save model

        (-ELBO_loss.mean()).backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        sigma_t = sigma(properties.sigma_F, properties.sigma_I, properties.sigma_N, t)
        print("#" + str(t) + " ELBO_loss=" + str(-ELBO_loss.mean().item()) + ", lr=" + str(scheduler.get_lr()[0]))

    # TODO: save model

    batch = sample_batch(scenes_dataset, properties.B, device=properties.device)
    [x_tensor_ref, v_tensor_ref], _ = sample_batch(scenes_dataset, 1, device=properties.device)
    x_tensor_ref = x_tensor_ref[0]
    v_tensor_ref = v_tensor_ref[0]
    x_q = model.generate(batch,
                         v_tensor_ref.expand(properties.B, v_tensor_ref.size()[1], v_tensor_ref.size()[2],
                                             v_tensor_ref.size()[3]), sigma_t)
    for x in x_q:
        show_image_comparation(x_tensor_ref, v_tensor_ref, x)
