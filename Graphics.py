import pyglet
import math
import numpy as np
from pyglet.gl import *

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
        for object_list in self.object_lists:
            batch = pyglet.graphics.Batch()
            Objects.translate(object_list, batch)
            batch.draw()

    @staticmethod
    def run():
        pyglet.app.run()


class Objects:
    colors = ['Black', 'Grey']
    fob = []
    fob_picker = {}
    j = 0
    for color in colors:
        fob.append(pyglet.image.load('Fob/{}/Fob.png'.format(color)))
        fob_picker[color] = fob[j]
        j += 1

    bars = [[], []]
    for i in range(2):
        for j in range(10):
            bars[i].append(pyglet.image.load('Fob/Bar{}/{}.png'.format(i + 1, j)))

    class Fob:
        @staticmethod   # Sprites are unable to be added to the batch object
        def draw(batch_unused, x, y, rotation=0, col='Black', stat1=0., stat2=0.):
            batch = pyglet.graphics.Batch()
            layer_0 = pyglet.graphics.OrderedGroup(0)
            layer_1 = pyglet.graphics.OrderedGroup(1)

            sprites = [pyglet.sprite.Sprite(Objects.fob_picker[col], x, y, batch=batch, group=layer_0),
                       pyglet.sprite.Sprite(Objects.bars[0][int(9.9 * stat1)], x, y, batch=batch, group=layer_1),
                       pyglet.sprite.Sprite(Objects.bars[1][int(9.9 * stat2)], x, y, batch=batch, group=layer_1)]

            for sprite in sprites:
                sprite.image.anchor_x = sprites[1].image.width / 2
                sprite.image.anchor_y = sprites[1].image.height / 3

            sprites[0].rotation = rotation
            sprites[1].rotation = 0
            batch.draw()

    class FobWithVision:
        @staticmethod   # Sprites are unable to be added to the batch object
        def draw(batch, vision_bins, vision_hist, x, y, rotation=0, col='Black', stat1=0., stat2=0., vision_len=50.):
            batch_sprites = pyglet.graphics.Batch()
            layer_0 = pyglet.graphics.OrderedGroup(0)
            layer_1 = pyglet.graphics.OrderedGroup(1)

            sprites = [pyglet.sprite.Sprite(Objects.fob_picker[col], x, y, batch=batch_sprites, group=layer_0),
                       pyglet.sprite.Sprite(Objects.bars[0][int(9.9 * stat1)], x, y, batch=batch_sprites, group=layer_1),
                       pyglet.sprite.Sprite(Objects.bars[1][int(9.9 * stat2)], x, y, batch=batch_sprites, group=layer_1)]

            for sprite in sprites:
                sprite.image.anchor_x = sprites[1].image.width / 2
                sprite.image.anchor_y = sprites[1].image.height / 3

            sprites[0].rotation = rotation
            sprites[1].rotation = 0
            batch_sprites.draw()

            xy2 = np.empty((len(vision_bins), 2), dtype=np.float32)
            xy2[:, 0] = vision_len * np.sin(vision_bins + rotation * np.pi / 180.)
            xy2[:, 1] = vision_len * np.cos(vision_bins + rotation * np.pi / 180.)
            xy2 = np.add(xy2, np.array([x, y], dtype=np.float32)[None, :])

            xy3 = np.add(xy2[:len(vision_hist), :], xy2[1:, :])

            for i in range(len(vision_bins)):
                Objects.Line.draw(batch, x, y, *xy2[i, :].tolist())

            for i in range(len(vision_hist)):
                Objects.Text.draw(batch, *xy3[i, :].tolist(), '{}'.format(int(vision_hist[i])))

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
