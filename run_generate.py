import tkinter as tk

import PIL
import numpy as np
import torch
from PIL import Image, ImageTk
import math
from GQN.Properties import Properties
from GQN.Utils import load_model
from dataset.deepmind_dataset.Reader import sample_batch_deepmind
from dataset.deepmind_dataset.data_reader import DataReader


class GQNCamera:
    def __init__(self, x, y, z, yaw, pitch, x_test, v_test, sigma_t, properties, step=0.01, angle=5):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.step = step
        self.angle = angle
        self.x_test = x_test
        self.v_test = v_test
        self.sigma_t = sigma_t
        self.properties = properties

    def go_right(self):
        self.x += self.step
        return self.generate_view()

    def go_left(self):
        self.x -= self.step
        return self.generate_view()

    def go_forward(self):
        self.y += self.step
        return self.generate_view()

    def go_backward(self):
        self.y -= self.step
        return self.generate_view()

    def go_up(self):
        self.z += self.step
        return self.generate_view()

    def go_down(self):
        self.z -= self.step
        return self.generate_view()

    def turn_up(self):
        self.pitch += self.angle
        self.pitch %= 360
        return self.generate_view()

    def turn_down(self):
        self.pitch -= self.angle
        self.pitch %= 360
        return self.generate_view()

    def turn_left(self):
        self.yaw -= self.angle
        self.yaw %= 360
        return self.generate_view()

    def turn_right(self):
        self.yaw += self.angle
        self.yaw %= 360
        return self.generate_view()

    def generate_view(self):
        v = [self.x, self.y, self.z, math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)]
        v_tensor = torch.tensor(v)
        v_tensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(v_tensor, dim=1), dim=1), dim=0).to(
            self.properties.device)
        print(v)
        print(self.yaw)
        print(self.pitch)
        x_q_generated, _, x_q_generated3, _ = model.generate([self.x_test, self.v_test], v_tensor, self.sigma_t)
        img_rescaled = self._rescale(x_q_generated3[0].cpu().detach().numpy().transpose((1, 2, 0)))
        img = PIL.Image.fromarray(img_rescaled)
        img = img.resize((500, 500), Image.ANTIALIAS)
        return ImageTk.PhotoImage(image=img)

    @staticmethod
    def _rescale(img_array):
        return (255.0 / img_array.max() * (img_array - img_array.min())).astype(np.uint8)


def download_sample_data_from_deepmind_dataset(properties):
    deepmind_test_dataset_reader = DataReader(dataset=properties.deepmind_dataset,
                                              context_size=properties.deepmind_dataset_context_size,
                                              root=properties.deepmind_dataset_root_path,
                                              mode='test')
    D_test = sample_batch_deepmind(deepmind_test_dataset_reader.read(batch_size=1),
                                   B=1,
                                   device=properties.device)
    [x, v], [_, v_q] = D_test
    return x, v, v_q


if __name__ == "__main__":
    model_path = 'results/models/model_16000.pt'
    properties = Properties(
        data_path='dataset/local_dataset/blender_dataset',
        json_path='dataset/local_dataset/blender_dataset/observations.json',
        deepmind_dataset='rooms_ring_camera',
        deepmind_dataset_root_path='dataset/deepmind_dataset/datasets')
    model, _, _, _, sigma_t = load_model(model_path, properties)
    model = model.to(properties.device)
    x, v, v_q = download_sample_data_from_deepmind_dataset(properties)
    v_q = v_q.cpu().detach()

    root = tk.Tk()

    camera = GQNCamera(x=v_q[0][0][0][0],
                       y=v_q[0][1][0][0],
                       z=v_q[0][2][0][0],
                       yaw=math.acos(v_q[0][3][0][0]),
                       pitch=math.acos(v_q[0][5][0][0]),
                       x_test=x,
                       v_test=v,
                       sigma_t=sigma_t,
                       properties=properties)

    start_view = camera.generate_view()

    # root.attributes("-fullscreen", True)
    vlabel = tk.Label(root, image=start_view)
    vlabel.pack(side="bottom", fill="both", expand="yes")


    def key(event):
        if event.keysym.upper() == 'D':
            img2 = camera.go_right()
            vlabel.configure(image=img2)
            vlabel.image = img2
        elif event.keysym.upper() == 'A':
            img2 = camera.go_left()
            vlabel.configure(image=img2)
            vlabel.image = img2
        elif event.keysym.upper() == 'W':
            img2 = camera.go_forward()
            vlabel.configure(image=img2)
            vlabel.image = img2
        elif event.keysym.upper() == 'S':
            img2 = camera.go_backward()
            vlabel.configure(image=img2)
            vlabel.image = img2
        elif event.keysym.upper() == 'RIGHT':
            img2 = camera.turn_right()
            vlabel.configure(image=img2)
            vlabel.image = img2
        elif event.keysym.upper() == 'LEFT':
            img2 = camera.turn_left()
            vlabel.configure(image=img2)
            vlabel.image = img2
        elif event.keysym.upper() == 'DOWN':
            img2 = camera.turn_down()
            vlabel.configure(image=img2)
            vlabel.image = img2
        elif event.keysym.upper() == 'UP':
            img2 = camera.turn_up()
            vlabel.configure(image=img2)
            vlabel.image = img2
        else:
            print("pressed", repr(event.keysym))


    root.bind("<Key>", key)
    root.mainloop()
