import pyglet
import math
import numpy as np
from pyglet.gl import *
import time

# A library used to create a virtual environment for experiments


class Environment:

    # object_list entries take te form of a list of the object name, and its draw parameters e.g.
    # ['Fob', (250, 250), 0, 'black', 3]

    def __init__(self, window_dimensions, clear=True):
        self.object_lists = []
        self.window_dimensions = window_dimensions
        self.window = pyglet.window.Window(width=self.window_dimensions[0], height=self.window_dimensions[1])
        self.on_draw = self.window.event(self.on_draw)
        self.clear = clear

    def on_draw(self):
        if self.clear:
            glClearColor(1, 1, 1, 1)
            self.window.clear()
        batch = pyglet.graphics.Batch()
        for object_list in self.object_lists[:len(self.object_lists) - 1]:
            Objects.translate(object_list, batch)
        batch.draw()
        Objects.translate(self.object_lists[len(self.object_lists) - 1], batch)  # Last object list is text

    @staticmethod
    def run():
        pyglet.app.run()


class Objects:

    fob_cords = np.loadtxt('./Fob/Fob.txt')
    bar_cords = np.loadtxt('./Fob/Bars.txt')
    circle_cords = np.loadtxt('./Circle/Circle.txt')

    class Fob:
        @staticmethod   # Sprites are unable to be added to the batch object
        def draw(batch, x, y, rotation=0, col=(50, 50, 50, 255), stat1=0., stat2=0.):
            size = 10
            rotation_radians = rotation * math.pi / 180
            c = math.cos(rotation_radians)
            s = math.sin(rotation_radians)
            rot = np.array([[c, s], [-s, c]])
            cords = np.matmul(rot, Objects.fob_cords.transpose()).transpose() * size
            cords = (cords + np.array([x, y])[None, :]).flatten()

            num_cords = int(cords.size / 2)

            batch.add(num_cords, GL_TRIANGLES, None, ('v2f', cords.tolist()),
                      ('c4B', (0, 0, 0, 255) * int(num_cords / 2) + col * int(num_cords / 2)))

            bar_cords = Objects.bar_cords * size
            bar_cords[26:29, 0] += stat1 * (Objects.bar_cords[9, 0] - Objects.bar_cords[6, 0]) * size
            bar_cords[20:23, 0] += stat2 * (Objects.bar_cords[9, 0] - Objects.bar_cords[6, 0]) * size
            bar_cords = (bar_cords + np.array([x, y])[None, :]).flatten()

            batch.add(30, GL_TRIANGLES, None, ('v2f', bar_cords.tolist()),
                      ('c4B', (0, 0, 0, 255) * 6 + (255, 255, 255, 255) * 12 + (50, 50, 50, 255) * 6 +
                       (255, 0, 0, 255) * 6))

    class FobWithVision:
        @staticmethod   # Sprites are unable to be added to the batch object
        def draw(batch, vision_bins, vision_hist, x, y, rotation=0, col=(50, 50, 50, 255), stat1=0., stat2=0.,
                 vision_len=50.):
            size = 10
            rotation_radians = rotation * math.pi / 180
            c = math.cos(rotation_radians)
            s = math.sin(rotation_radians)
            rot = np.array([[c, s], [-s, c]])
            cords = np.matmul(rot, Objects.fob_cords.transpose()).transpose() * size
            cords = (cords + np.array([x, y])[None, :]).flatten()

            num_cords = int(cords.size / 2)

            batch.add(num_cords, GL_TRIANGLES, None, ('v2f', cords.tolist()),
                      ('c4B', (0, 0, 0, 255) * int(num_cords / 2) + col * int(num_cords / 2)))

            bar_cords = Objects.bar_cords * size
            bar_cords[26:29, 0] += stat1 * (Objects.bar_cords[9, 0] - Objects.bar_cords[6, 0]) * size
            bar_cords[20:23, 0] += stat2 * (Objects.bar_cords[9, 0] - Objects.bar_cords[6, 0]) * size
            bar_cords = (bar_cords + np.array([x, y])[None, :]).flatten()

            batch.add(30, GL_TRIANGLES, None, ('v2f', bar_cords.tolist()),
                      ('c4B', (0, 0, 0, 255) * 6 + (255, 255, 255, 255) * 12 + (50, 50, 50, 255) * 6 +
                       (255, 0, 0, 255) * 6))

            xy2 = np.empty((len(vision_bins), 2), dtype=np.float32)
            xy2[:, 0] = vision_len * np.sin(vision_bins + rotation * np.pi / 180.)
            xy2[:, 1] = vision_len * np.cos(vision_bins + rotation * np.pi / 180.)
            xy2 = np.add(xy2, np.array([x, y], dtype=np.float32)[None, :])

            for i in range(len(vision_bins)):
                Objects.Line.draw(batch, x, y, *xy2[i, :].tolist())

    class Rectangle:
        @staticmethod
        def draw(batch, x, y, rotation=0, size=[100, 100], col=(0, 0, 0, 255)):
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            size = np.array(size) / 2
            rotation_radians = rotation * math.pi / 180
            c = math.cos(rotation_radians)
            s = math.sin(rotation_radians)
            rot = np.array([[c, -s], [s, c]])
            coordinates = np.array([-size, [-size[0], size[1]], size, -size, [size[0], -size[1]], size]).transpose()
            coordinates = (np.matmul(rot, coordinates).transpose() + np.array([x, y])).flatten()

            batch.add(6, GL_TRIANGLES, None, ('v2f', coordinates.tolist()), ('c4B', col * 6))

    class Line:
        @staticmethod
        def draw(batch, x1, y1, x2, y2, thickness=1, col=(0, 0, 0, 255)):
            rotation = np.angle(x2 - x1 + (y2 - y1) * 1j) * 180 / np.pi
            Objects.Rectangle.draw(batch, *np.multiply(np.add([x1, y1], [x2, y2]), 0.5).tolist(), rotation,
                                   [np.sqrt(np.array([x2 - x1, y2 - y1]).dot(np.array([x2 - x1, y2 - y1]))), thickness],
                                   col=col)

    class Text:
        @staticmethod
        def draw(batch_unused, x, y, text, size=12):
            label = pyglet.text.Label(text, font_name='Times New Roman', font_size=size, x=x//2, y=y//2,
                                      anchor_x='center', anchor_y='center', color=(0, 0, 0, 255))
            label.draw()

    class Circle:
        @staticmethod
        def draw(batch, x, y, radius=1, col=(0, 0, 0, 255)):

            cords = Objects.circle_cords * radius
            cords = (cords + np.array([x, y])[None, :]).flatten()

            num_cords = int(cords.size / 2)

            batch.add(num_cords, GL_TRIANGLES, None, ('v2f', cords.tolist()), ('c4B', col * num_cords))

    @staticmethod
    def translate(object_list, batch):
        object_dict = {
            "Fob": Objects.Fob,
            "Rectangle": Objects.Rectangle,
            "Line": Objects.Line,
            "Text": Objects.Text,
            "Fob With Vision": Objects.FobWithVision,
            "Circle": Objects.Circle
        }

        for i in range(len(object_list)):
            object_dict[object_list[i][0]].draw(batch, *object_list[i][1:])
