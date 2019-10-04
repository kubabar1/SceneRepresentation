import math
import torch
from PIL import Image
import torchvision
from torchvision.transforms.functional import to_tensor
import os
import json
from pathlib import Path
from torchvision.datasets import VisionDataset
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


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
                    obs = self._get_observation_from_json(observation)
                    scene_tmp.append(obs)
                self.scenes.append(scene_tmp)

    def _get_observation_from_json(self, observation):
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


def sample_batch(scenes_set, B, M=None, K=None):
    """
        batch => [[[[x1,v1], [x2,v2], ...], [x_q1, v_q1]], ...]
    :param scenes_set:
    :param B:
    :param M:
    :param K:
    :return:
    """
    N = len(scenes_set)
    K = len(scenes_set[0]) if K is None else K
    M = random.randint(1, K) if M is None else M
    D = []
    for _ in range(B):
        i = random.randint(0, N - 1)
        scene = scenes_set[i]
        views = []
        view_loader = DataLoader(scene, sampler=RandomSampler(range(len(scene))))
        query_view_loader = DataLoader(scene, sampler=RandomSampler(range(len(scene))))
        query_view_iterator = iter(query_view_loader)
        view_iterator = iter(view_loader)
        for _ in range(M):
            x_i, v_i = next(view_iterator)
            v_i = torch.tensor(
                [v_i[0][0], v_i[0][1], v_i[0][2], math.cos(v_i[0][3]), math.sin(v_i[0][3]), math.cos(v_i[0][4]),
                 math.sin(v_i[0][4])])
            views.append([x_i, v_i])
        x_q, v_q = next(query_view_iterator)
        v_q = torch.tensor(
            [v_q[0][0], v_q[0][1], v_q[0][2], math.cos(v_q[0][3]), math.sin(v_q[0][3]), math.cos(v_q[0][4]),
             math.sin(v_q[0][4])])
        D.append([views, [x_q, v_q]])
    return D


if __name__ == '__main__':
    dataset = ScenesDataset('../data', '../data/observations.json')
    scenes = iter(dataset)
    for scene in scenes:
        for observation in scene:
            print(observation[1])

    print('####################################')

    D = sample_batch(dataset, 36)

    for batch in D:
        obs = batch[0]
        obs_q = batch[1]
        x_q = obs_q[0]
        v_q = obs_q[1]
        print(v_q)
        for x, v in obs:
            print(v)
