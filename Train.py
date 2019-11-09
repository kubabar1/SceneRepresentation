import os.path
import torch
from torchvision import transforms
from GQN import GQN
from Properties import Properties
from dataset.DatasetType import DatasetType
from Utils import sigma, save_model, save_image_comparation, load_model
from dataset.ScenesDataset import ScenesDataset, sample_batch
from torch import optim
from dataset.deepmind_dataset.Reader import sample_batch_deepmind
from dataset.deepmind_dataset.data_reader import DataReader
from nn.optim.Scheduler import Scheduler
import logging


class Train:
    def __init__(self):
        self.properties = Properties()
        self.transform = transforms.Compose([transforms.Resize(self.properties.image_resize)])

    def train(self):
        model = GQN(self.properties).to(self.properties.device)
        optimizer = optim.Adam(model.parameters(), lr=self.properties.mi_I,
                               betas=(self.properties.beta_1, self.properties.beta_2),
                               eps=self.properties.epsilon)
        scheduler = Scheduler(optimizer, properties=self.properties)
        sigma_t = self.properties.sigma_I
        self._train(model=model, optimizer=optimizer, scheduler=scheduler, start_epoch=0, sigma_t=sigma_t)

    def train_loaded_model(self, model_path):
        model, start_epoch, optimizer, loss, sigma_t, scheduler = load_model(model_path, self.properties)
        model = model.to(self.properties.device)
        optimizer = optimizer.to(self.properties.device)
        scheduler = scheduler.to(self.properties.device)
        self._train(model=model, optimizer=optimizer, scheduler=scheduler, start_epoch=start_epoch + 1, sigma_t=sigma_t)

    def _train(self, model, optimizer, scheduler, start_epoch, sigma_t):
        if not os.path.exists(os.path.dirname(self.properties.log_file_path)):
            os.makedirs(os.path.dirname(self.properties.log_file_path))
        logging.basicConfig(filename=self.properties.log_file_path,
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        scenes_dataset = ScenesDataset(dataset_root_path=self.properties.data_path,
                                       json_path=self.properties.json_path,
                                       transform=self.transform)
        deepmind_train_dataset_reader = DataReader(dataset=self.properties.deepmind_dataset,
                                                   context_size=self.properties.deepmind_dataset_context_size,
                                                   root=self.properties.deepmind_dataset_root_path,
                                                   mode='train')

        deepmind_test_dataset_reader = DataReader(dataset=self.properties.deepmind_dataset,
                                                  context_size=self.properties.deepmind_dataset_context_size,
                                                  root=self.properties.deepmind_dataset_root_path,
                                                  mode='test')

        for epoch in range(start_epoch, self.properties.s_max):
            if self.properties.dataset_type is DatasetType.LOCAL:
                D = sample_batch(scenes_dataset, B=self.properties.B, device=self.properties.device)
            else:
                D = sample_batch_deepmind(deepmind_train_dataset_reader.read(batch_size=self.properties.B),
                                          B=self.properties.B,
                                          device=self.properties.device)

            if epoch % self.properties.test_interval == 0:
                with torch.no_grad():
                    D_test = sample_batch(scenes_dataset, B=self.properties.B, device=self.properties.device) \
                        if self.properties.dataset_type is DatasetType.LOCAL \
                        else sample_batch_deepmind(deepmind_test_dataset_reader.read(batch_size=self.properties.B),
                                                   B=self.properties.B,
                                                   device=self.properties.device)
                    [x_tensor_test, v_tensor_test], [x_q_tensor_test, v_q_tensor_test] = D_test
                    ELBO_loss_test, kl = model.estimate_ELBO(D_test, sigma_t)
                    x_q_generated = model.generate([x_tensor_test, v_tensor_test], v_q_tensor_test)

                    log_test_text = "TEST_" + str(epoch) + " ELBO_loss_test=" \
                                    + str(-ELBO_loss_test.mean().item()) + ", kl=" + str(kl.mean().item())
                    print(log_test_text)
                    logger.info(log_test_text)

                    if epoch % self.properties.save_images == 0:
                        save_image_comparation(generated_x=x_q_generated,
                                               reference_x=x_q_tensor_test,
                                               generated_images_path=self.properties.generated_images_path,
                                               referenced_images_path=self.properties.referenced_images_path,
                                               epoch=epoch)

            ELBO_loss, _ = model.estimate_ELBO(D, sigma_t)
            (-ELBO_loss.mean()).backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            sigma_t = sigma(self.properties.sigma_F, self.properties.sigma_I, self.properties.sigma_N, epoch)

            log_train_text = "TRAIN_" + str(epoch) + " ELBO_loss=" + str(-ELBO_loss.mean().item()) + ", lr=" \
                             + str(str(scheduler.get_lr()[0]))
            print(log_train_text)
            logger.info(log_train_text)

            if epoch % self.properties.save_model_interval == 0:
                save_model(model, epoch, optimizer, ELBO_loss, sigma_t, scheduler, self.properties.save_model_path)

        save_model(model, epoch, optimizer, ELBO_loss, sigma_t, scheduler, self.properties.save_model_path)
