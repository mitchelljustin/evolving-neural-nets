from PIL import Image
class Maze():
    _maze_array = []
    def __init__(self,x, y, image):
        im = Image.open(image)
        pix = im.load()
        self._maze_array = [[]]*y
        print("do once")
        for i in range(y):
            self._maze_array[i] = []
            for j in range(x):
                if pix[j,i] == (0,0,0,255):
                    self._maze_array[i].append(1)
                else:
                    self._maze_array[i].append(0)
        return

    def isSolid(self, x, y):
        x = math.floor(x)
        y = math.floor(y)
        return self._maze_array[y][x]


    def draw(self,display_surf):
        for y in range(len(self._maze_array)):
            for x in range(len(self._maze_array[y])):
                if self._maze_array[y][x] == 1:
                    display_surf.set_at((x,y), (0,0,0))
