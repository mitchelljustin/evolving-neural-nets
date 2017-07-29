import math


class Robot():
    _rotation = 0
    speed = 10
    MODIFIER = 10
    # 5 sensors at front at 180/5=36 degrees away from each other
    # one sensor[5] at back 90 degrees
    sensor_angles = [0, 0, 0, 0, 0, 0]
    x = 0
    y = 0
    old_places = []
    sensor_vals = [None]*6
    maze = None

    def __init__(self, maze):
        self.maze = maze
        return

    def move(self, direction):
        self.old_places.append((self.x, self.y))
        for i in range(self.speed):
            new_x = self.x + direction * math.sin(math.radians(self._rotation))
            new_y = self.y + direction * math.cos(math.radians(self._rotation))
            if not self.maze.isSolid(new_x, new_y):
                self.x = new_x
                self.y = new_y
            else:
                break

    def _get_sensor_angle(self, index):
        # the first 5 [0..4] sensors are front ones, each separated by 36 degrees
        if index < 5:
            return (180 - (45 * index) + self._rotation) % 360
        return (270 + self._rotation) % 360

    def _get_pie_start_end(self, index):
        return ((index * 90 + 45 + self._rotation) % 360, (index * 90 - 45 + self._rotation) % 360)

    def update_sensor_angles(self):
        for i in range(len(self.sensor_angles)):
            self.sensor_angles[i] = self._get_sensor_angle(self, i)

    def rotate(self, direction):
        self._rotation += direction * self.MODIFIER

    def getPos(self):
        return (self.x, self.y)

    def getRotation(self):
        return self._rotation % 360

    def getSensorValues(self):
        # returns sensor array
        for i in range(len(self.sensor_angles)):
            self.sensor_angles[i] = self._get_sensor_angle(i)
        for i in range(len(self.sensor_vals)):
            angle = self.sensor_angles[i]
            x1 = self.x
            y1 = self.y
            cos = math.cos(math.radians(angle))
            sin = math.sin(math.radians(angle))

            # check old vals first

            found = False
            if self.sensor_vals[i]:
                for j in range(self.sensor_vals[i] - 10, 1000):
                    x2 = x1 + j * cos
                    y2 = y1 + j * sin
                    if x2 < -20 or y2 < -20:
                        break
                    if y2 > 619 or x2 > 819:
                        break
                    if self.maze.isSolid(x2, y2):
                        self.sensor_vals[i] = j
                        found = True
                        break
            if not found:
                for j in range(1000):
                    x2 = x1 + j * cos
                    y2 = y1 + j * sin
                    if self.maze.isSolid(x2, y2):
                        self.sensor_vals[i] = j
                        break
        return self.sensor_vals

    def getPieValues(self, goal_x, goal_y):
        dx = goal_x - self.x
        dy = goal_y - self.y
        degs = math.degrees(math.atan2(-dy, dx)) % 360
        # pie indicates which pie is lit toward the goal
        # 0: right, 1: up, 2: left, 3: down
        pies = [0] * 4
        for i in range(len(pies)):
            d_s, d_e = self._get_pie_start_end(i)
            if d_s <= degs and d_e >= degs:
                pies[i] = 1
        return pies
