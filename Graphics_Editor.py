import numpy as np

definition = 10
size = 1.
angle = [i * 2 * np.pi / definition for i in range(definition)]
fob_cords = [[size * np.cos(angle[i]), size * np.sin(angle[i])] for i in range(definition)]
fob_cords_ = []
for i in range(definition):
    fob_cords_.extend([[0, size * 2.5], fob_cords[i], fob_cords[(i + 1) % definition]])
fob_cords = np.array(fob_cords_)

fob_cords = np.concatenate((fob_cords, fob_cords * 0.8))

b = 0.2
c = 3
bar_coords = np.array(
    [[-2, c], [-2, c + 1], [2, c + 1], [-2, c], [2, c], [2, c + 1],

     [-2 + b, c + b], [-2 + b, c + 0.5 - b / 2.], [2 - b, c + 0.5 - b / 2.], [2 - b, c + b], [2 - b, c + 0.5 - b / 2.], [-2 + b, c + b],

     [-2 + b, c + 1 - b], [-2 + b, c + 0.5 + b / 2.], [2 - b, c + 0.5 + b / 2.], [2 - b, c + 1 - b], [2 - b, c + 0.5 + b / 2.], [-2 + b, c + 1 - b],

     [-2 + b, c + b], [-2 + b, c + 0.5 - b / 2.], [-2 + b, c + 0.5 - b / 2.], [-2 + b, c + b], [-2 + b, c + 0.5 - b / 2.], [-2 + b, c + b],

     [-2 + b, c + 1 - b], [-2 + b, c + 0.5 + b / 2.], [-2 + b, c + 0.5 + b / 2.], [-2 + b, c + 1 - b], [-2 + b, c + 0.5 + b / 2.], [-2 + b, c + 1 - b]]) * size

circle_cords = [[size * np.cos(angle[i]), size * np.sin(angle[i])] for i in range(definition)]
circle_cords_ = []
for i in range(definition):
    circle_cords_.extend([[0, 0], circle_cords[i], circle_cords[(i + 1) % definition]])
circle_cords = np.array(circle_cords_)

np.savetxt('./Fob/Fob.txt', fob_cords)
np.savetxt('./Fob/Bars.txt', bar_coords)
np.savetxt('./Circle/Circle.txt', circle_cords)
