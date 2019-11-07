import numpy as np

definition = 10
size = 1.
angle = [i * 2 * np.pi / definition for i in range(definition)]
cords = [[size * np.cos(angle[i]), size * np.sin(angle[i])] for i in range(definition)]
cords_ = []
for i in range(definition):
    cords_.extend([[0, size * 2.5], cords[i], cords[(i + 1) % definition]])
cords = np.array(cords_)

cords = np.concatenate((cords, cords * 0.8))


b = 0.2
c = 3
bar_coords = np.array(
    [[-2, c], [-2, c + 1], [2, c + 1], [-2, c], [2, c], [2, c + 1],

     [-2 + b, c + b], [-2 + b, c + 0.5 - b / 2.], [2 - b, c + 0.5 - b / 2.], [2 - b, c + b], [2 - b, c + 0.5 - b / 2.], [-2 + b, c + b],

     [-2 + b, c + 1 - b], [-2 + b, c + 0.5 + b / 2.], [2 - b, c + 0.5 + b / 2.], [2 - b, c + 1 - b], [2 - b, c + 0.5 + b / 2.], [-2 + b, c + 1 - b],

     [-2 + b, c + b], [-2 + b, c + 0.5 - b / 2.], [-2 + b, c + 0.5 - b / 2.], [-2 + b, c + b], [-2 + b, c + 0.5 - b / 2.], [-2 + b, c + b],

     [-2 + b, c + 1 - b], [-2 + b, c + 0.5 + b / 2.], [-2 + b, c + 0.5 + b / 2.], [-2 + b, c + 1 - b], [-2 + b, c + 0.5 + b / 2.], [-2 + b, c + 1 - b]]) * size

np.savetxt('./Fob/Fob.txt', cords)
np.savetxt('./Fob/Bars.txt', bar_coords)