import os
import platform


class Bar:
    def __init__(self, description, bar_length=20):

        self.description = description
        self.bar_length = bar_length
        self.bar = []

        for i in range(bar_length + 1):
            self.bar.append('[')
            for j in range(bar_length):
                if j < i:
                    self.bar[i] += '#'
                else:
                    self.bar[i] += ' '
            self.bar[i] += ']'

    def show(self, fraction):

        os.system('cls' if platform.system() == "Windows" else 'clear')
        print(self.description)
        print(self.bar[int(fraction * self.bar_length + 0.5)] + ' %.2f' % (fraction * 100) + '%')
