from dataset.utils.Observation import Observation


class Scene(list):
    def __init__(self):
        super().__init__()
        self.observations = list()

    def check(self, elem):
        if not isinstance(elem, Observation):
            raise TypeError(elem)

    def append(self, scene):
        self.check(scene)
        self.observations.append(scene)

    def __setitem__(self, i, scene):
        self.check(scene)
        self.observations[i] = scene

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, i):
        return self.observations[i]

    def __delitem__(self, i):
        del self.observations[i]

    def insert(self, i, scene):
        self.check(scene)
        self.observations.insert(i, scene)

    def __str__(self):
        return str(self.observations)