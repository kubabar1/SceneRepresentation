import torch

class Observation:
    def __init__(self, image: torch.Tensor, viewpoint):
        self.image = image
        self.viewpoint = viewpoint

    def to_array(self):
        return [self.image, self.viewpoint]

    def to_list(self):
        return list([self.image, self.viewpoint])

    def __str__(self):
        return str(self.__dict__)
