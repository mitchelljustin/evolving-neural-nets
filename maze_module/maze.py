import numpy as np
from PIL import Image

ZERO_PADDING = 20


class Maze():
    _maze_array = []

    def __init__(self, width, height, image):
        im = Image.open(image)
        pix = im.load()
        self.width, self.height = width, height
        self._maze_array = np.ones([height + ZERO_PADDING * 2, width + ZERO_PADDING * 2], dtype=np.bool)
        for i in range(height):
            for j in range(width):
                val = (pix[j, i] == (0, 0, 0, 255))
                self._maze_array[i + ZERO_PADDING, j + ZERO_PADDING] = val

    def isSolid(self, x, y):
        x = round(x)
        y = round(y)
        return self._maze_array[y + ZERO_PADDING, x + ZERO_PADDING]

    def draw(self, display_surf):
        for y in range(len(self._maze_array) - ZERO_PADDING * 2):
            for x in range(len(self._maze_array[y]) - ZERO_PADDING * 2):
                if self._maze_array[y + ZERO_PADDING, x + ZERO_PADDING]:
                    display_surf.set_at((x, y), (0, 0, 0))
