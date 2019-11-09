import math
import torchvision
from torchvision import transforms

from GQN import GQN
from Properties import Properties
from dataset.DatasetType import DatasetType
from Reader import sample_batch_deepmind
from Utils import sigma, show_image_comparation
from data_reader import DataReader
from dataset.ScenesDataset import ScenesDataset, sample_batch

from torch import optim

from nn.optim.Scheduler import Scheduler

if __name__ == '__main__':
    properties = Properties()

    transform = transforms.Compose([transforms.Resize(properties.image_resize)])
    scenes_dataset = ScenesDataset(dataset_root_path=properties.data_path,
                                   json_path=properties.json_path,
                                   transform=transform)
    deepmind_train_dataset_reader = DataReader(dataset=properties.deepmind_dataset,
                                               context_size=properties.deepmind_dataset_context_size,
                                               root=properties.deepmind_dataset_root_path,
                                               mode='train')

    deepmind_test_dataset_reader = DataReader(dataset=properties.deepmind_dataset,
                                              context_size=properties.deepmind_dataset_context_size,
                                              root=properties.deepmind_dataset_root_path,
                                              mode='test')

    model = GQN(properties).to(properties.device)

    optimizer = optim.Adam(model.parameters(), lr=properties.mi_I, betas=(properties.beta_1, properties.beta_2),
                           eps=properties.epsilon)
    scheduler = Scheduler(optimizer, properties=properties)

    sigma_t = properties.sigma_I

    for t in range(properties.s_max):
        D = sample_batch(scenes_dataset, B=properties.B, device=properties.device) \
            if properties.dataset_type is DatasetType.LOCAL \
            else sample_batch_deepmind(deepmind_train_dataset_reader.read(batch_size=properties.B), B=properties.B,
                                       device=properties.device)

        ELBO_loss, _ = model.estimate_ELBO(D, sigma_t)

        # TODO: test flow and save model

        (-ELBO_loss.mean()).backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        sigma_t = sigma(properties.sigma_F, properties.sigma_I, properties.sigma_N, t)
        print("#" + str(t) + " ELBO_loss=" + str(-ELBO_loss.mean().item()) + ", lr=" + str(scheduler.get_lr()[0]))

    # TODO: save model

    D_test = sample_batch(scenes_dataset, B=properties.B, device=properties.device) \
        if properties.dataset_type is DatasetType.LOCAL \
        else sample_batch_deepmind(deepmind_test_dataset_reader.read(batch_size=properties.B), B=properties.B,
                                   device=properties.device)

    _, [x_tensor_ref, v_tensor_ref] = D_test

    x_q = model.generate(D_test, v_tensor_ref, sigma_t)

    show_image_comparation(x_q, x_tensor_ref)
