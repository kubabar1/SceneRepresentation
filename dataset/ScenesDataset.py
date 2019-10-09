import math
import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import os
import json
from pathlib import Path
from torchvision.datasets import VisionDataset
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from Properties import IMAGE_RESIZE


class ScenesDataset(VisionDataset):
    def __init__(self, dataset_root_path, json_path, transform=None):
        super(ScenesDataset, self).__init__(dataset_root_path, transform=transform)
        self.dataset_root_path = dataset_root_path
        self.json_path = json_path
        self.scenes = []

        with open(Path(json_path), "r") as json_file:
            observations_json = json.loads(json_file.read())
            for scene in observations_json['scenes']:
                scene_tmp = []
                for observation in scene:
                    obs = self._get_observation_from_json_obs(observation)
                    scene_tmp.append(obs)
                self.scenes.append(scene_tmp)

    def _get_observation_from_json_obs(self, observation):
        image_path = os.path.join(self.dataset_root_path, Path(observation['file_path']))
        camera_position = observation['camera_position']
        yaw = observation['yaw']
        pitch = observation['pitch']
        viewpoint = torch.tensor([camera_position[0], camera_position[1], camera_position[2], yaw, pitch])
        im = Image.open(image_path)
        if self.transform is not None:
            im = self.transform(im)
        im = torchvision.transforms.functional.to_tensor(im)
        return [im, viewpoint]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return self.scenes[idx]


def sample_batch(scenes_set, B, M=None, K=None, device=torch.cuda.current_device()):
    """
    Return batch of images and their viewpoints, converted to tensors.

    :param scenes_set: ScenesDataset object
    :param B: batch size
    :param M: number of scene observations (default: random.randint(1, K))
    :param K: number of recorded views of each scene (default: len(scenes_set[0]))
    :return: [[images_batch, viewpoints_batch], [query_image_batch, query_viewpoint_batch]] of shape:
        [
            [torch.tensor([B, 3, 64, 64], torch.tensor([B, 7, 1, 1]))],
            [torch.tensor([1, 3, 64, 64], torch.tensor([1, 7, 1, 1]))]
        ]
    """
    N = len(scenes_set)
    K = len(scenes_set[0]) if K is None else K
    M = random.randint(1, K) if M is None else M
    x_tensors = [torch.Tensor().to(device) for _ in range(M)]
    v_tensors = [torch.Tensor().to(device) for _ in range(M)]
    x_q_tensor = torch.Tensor().to(device)
    v_q_tensor = torch.Tensor().to(device)
    for _ in range(B):
        x_tensors_tmp = []
        v_tensors_tmp = []
        i = random.randint(0, N - 1)
        scene = scenes_set[i]
        view_loader = DataLoader(scene, sampler=RandomSampler(range(len(scene))))
        query_view_loader = DataLoader(scene, sampler=RandomSampler(range(len(scene))))
        query_view_iterator = iter(query_view_loader)
        view_iterator = iter(view_loader)
        for _ in range(M):
            x_i, v_i = next(view_iterator)
            v_i = torch.tensor(
                [v_i[0][0], v_i[0][1], v_i[0][2],
                 math.cos(v_i[0][3]), math.sin(v_i[0][3]),
                 math.cos(v_i[0][4]), math.sin(v_i[0][4])]).view(1, 7, 1, 1)
            x_tensors_tmp.append(x_i)
            v_tensors_tmp.append(v_i)
        x_q, v_q = next(query_view_iterator)
        v_q = torch.tensor(
            [v_q[0][0], v_q[0][1], v_q[0][2], math.cos(v_q[0][3]), math.sin(v_q[0][3]), math.cos(v_q[0][4]),
             math.sin(v_q[0][4])]).view(1, 7, 1, 1)

        for i in range(M):
            x_tensors[i] = torch.cat([x_tensors[i], x_tensors_tmp[i].to(device)])
            v_tensors[i] = torch.cat([v_tensors[i], v_tensors_tmp[i].to(device)])
        x_q_tensor = torch.cat([x_q_tensor, x_q.to(device)])
        v_q_tensor = torch.cat([v_q_tensor, v_q.to(device)])
    return [x_tensors, v_tensors], [x_q_tensor, v_q_tensor]


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(IMAGE_RESIZE)])
    dataset = ScenesDataset('../data', '../data/observations.json', transform=transform)
    scenes = iter(dataset)
    for scene in scenes:
        for observation in scene:
            print(observation[1])

    [x_tensors, v_tensors], [x_q_tensor, v_q_tensor] = sample_batch(dataset, 36)
    M = len(x_tensors)

    print('####################################')
    for i in range(M):
        print(v_tensors[i].size())
