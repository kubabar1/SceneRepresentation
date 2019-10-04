import torch

class Observation:
    def __init__(self, image, viewpoint):
        self.check(image)
        self.image = image
        self.viewpoint = viewpoint

    def check(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Image needs to be a torch.Tensor")

    def to_array(self):
        return [self.image, self.viewpoint]

    def to_list(self):
        return list([self.image, self.viewpoint])

    def __str__(self):
        return str(self.__dict__)
