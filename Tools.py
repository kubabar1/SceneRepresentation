import torch
from pandas.tests.extension.numpy_.test_numpy_nested import np
from torchvision import transforms

from Properties import *
from dataset.ScenesDataset import ScenesDataset, sample_batch


def tensor_to_image(tensor_image):
    return torch.squeeze(tensor_image).permute(1, 2, 0)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(IMAGE_RESIZE)])

    scenes_set = ScenesDataset(dataset_root_path=DATA_PATH, json_path=JSON_PATH, transform=transform)

    batch = sample_batch(scenes_set, 36)

    a = np.array(batch)

    #print(a[0].shape)

    #print(batch[0])

    # print(list(BatchSampler(RandomSampler(range(10)), batch_size=3, drop_last=True)))
    # print(range(10))
    # print(RandomSampler(range(10)))
