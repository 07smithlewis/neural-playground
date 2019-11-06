import pyglet
import math
import numpy as np
from pyglet.gl import *
import time

# A library used to create a virtual environment for experiments


class Environment:

    # object_list entries take te form of a list of the object name, and its draw parameters e.g.
    # ['Fob', (250, 250), 0, 'black', 3]

    def __init__(self, window_dimensions):
        self.object_lists = []
        self.window_dimensions = window_dimensions
        self.window = pyglet.window.Window(width=self.window_dimensions[0], height=self.window_dimensions[1])
        self.on_draw = self.window.event(self.on_draw)

    def on_draw(self):
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

    class Fob:
        @staticmethod   # Sprites are unable to be added to the batch object
        def draw(batch, x, y, rotation=0, col=(50, 50, 50), stat1=0., stat2=0.):
            size = 10
            rotation_radians = rotation * math.pi / 180
            c = math.cos(rotation_radians)
            s = math.sin(rotation_radians)
            rot = np.array([[c, s], [-s, c]])
            cords = np.matmul(rot, Objects.fob_cords.transpose()).transpose() * size
            cords = (cords + np.array([x, y])[None, :]).flatten()

            num_cords = int(cords.size / 2)
            batch.add(num_cords, GL_TRIANGLES, None, ('v2f', cords), ('c3B', (0, 0, 0) * int(num_cords / 2) +
                                                                      col * int(num_cords / 2)))

            bar_cords = Objects.bar_cords * size
            bar_cords[18:20, 0] += stat1 * (4 - 2 * (Objects.bar_cords[4, 0] - Objects.bar_cords[0, 0])) * size
            bar_cords[14:16, 0] += stat2 * (4 - 2 * (Objects.bar_cords[4, 0] - Objects.bar_cords[0, 0])) * size
            bar_cords = (bar_cords + np.array([x, y])[None, :]).flatten()
            batch.add(20, GL_QUADS, None, ('v2f', bar_cords), ('c3B', (0, 0, 0) * 4 + (255, 255, 255) * 8 +
                                                               (50, 50, 50) * 4 + (255, 0, 0) * 4))

    class FobWithVision:
        @staticmethod   # Sprites are unable to be added to the batch object
        def draw(batch, vision_bins, vision_hist, x, y, rotation=0, col=(50, 50, 50), stat1=0., stat2=0.,
                 vision_len=50.):
            size = 10
            rotation_radians = rotation * math.pi / 180
            c = math.cos(rotation_radians)
            s = math.sin(rotation_radians)
            rot = np.array([[c, s], [-s, c]])
            cords = np.matmul(rot, Objects.fob_cords.transpose()).transpose() * size
            cords = (cords + np.array([x, y])[None, :]).flatten()

            num_cords = int(cords.size / 2)
            batch.add(num_cords, GL_TRIANGLES, None, ('v2f', cords), ('c3B', (0, 0, 0) * num_cords))

            bar_cords = Objects.bar_cords * size
            bar_cords[18:20, 0] += stat1 * (4 - 2 * (Objects.bar_cords[4, 0] - Objects.bar_cords[0, 0])) * size
            bar_cords[14:16, 0] += stat2 * (4 - 2 * (Objects.bar_cords[4, 0] - Objects.bar_cords[0, 0])) * size
            bar_cords = (bar_cords + np.array([x, y])[None, :]).flatten()
            batch.add(20, GL_QUADS, None, ('v2f', bar_cords), ('c3B', (0, 0, 0) * 4 + (255, 255, 255) * 8 +
                                                               (50, 50, 50) * 4 + (255, 0, 0) * 4))

            xy2 = np.empty((len(vision_bins), 2), dtype=np.float32)
            xy2[:, 0] = vision_len * np.sin(vision_bins + rotation * np.pi / 180.)
            xy2[:, 1] = vision_len * np.cos(vision_bins + rotation * np.pi / 180.)
            xy2 = np.add(xy2, np.array([x, y], dtype=np.float32)[None, :])

            xy3 = np.add(xy2[:len(vision_hist), :], xy2[1:, :])

            for i in range(len(vision_bins)):
                Objects.Line.draw(batch, x, y, *xy2[i, :].tolist())
            
            # for i in range(len(vision_hist)):
            #     Objects.Text.draw(batch, *xy3[i, :].tolist(), '{}'.format(int(vision_hist[i])))

    class Rectangle:
        @staticmethod
        def draw(batch, x, y, rotation=0, size=[100, 100], col=(0, 0, 0)):
            size = np.array(size) / 2
            rotation_radians = rotation * math.pi / 180
            c = math.cos(rotation_radians)
            s = math.sin(rotation_radians)
            rot = np.array([[c, -s], [s, c]])
            coordinates = np.array([-size, [-size[0], size[1]], size, [size[0], -size[1]]]).transpose()
            coordinates = (np.matmul(rot, coordinates).transpose() + np.array([x, y])).flatten()

            batch.add(4, GL_QUADS, None, ('v2f', coordinates), ('c3B', col * 4))

    class Line:
        @staticmethod
        def draw(batch, x1, y1, x2, y2, thickness=1, col=(0, 0, 0, 255)):
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glLineWidth(thickness)
            batch.add(2, gl.GL_LINES, None,
                      ('v2f', (x1, y1, x2, y2)),
                      ('c4B', (*col, *col)))

    class Text:
        @staticmethod
        def draw(batch_unused, x, y, text, size=12):
            label = pyglet.text.Label(text, font_name='Times New Roman', font_size=size, x=x//2, y=y//2,
                                      anchor_x='center', anchor_y='center', color=(0, 0, 0, 255))
            label.draw()

    @staticmethod
    def translate(object_list, batch):
        object_dict = {
            "Fob": Objects.Fob,
            "Rectangle": Objects.Rectangle,
            "Line": Objects.Line,
            "Text": Objects.Text,
            "Fob With Vision": Objects.FobWithVision
        }

        for i in range(len(object_list)):
            object_dict[object_list[i][0]].draw(batch, *object_list[i][1:])
