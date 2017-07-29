import math

class Robot():
    _rotation = 0
    speed = 10
    MODIFIER = 10
    # 5 sensors at front at 180/5=36 degrees away from each other
    # one sensor[5] at back 90 degrees
    sensor_angles = [0,0,0,0,0,0]
    x = 0
    y = 0
    old_places = []
    maze = None
    def __init__(self, maze):
        self.maze = maze
        return

    def move(self, direction):
        self.old_places.append((self.x,self.y))
        for i in range(self.speed):
            new_x = self.x + direction*math.sin(math.radians(self._rotation))
            new_y = self.y + direction*math.cos(math.radians(self._rotation))
            if not self.maze.isSolid(new_x, new_y):
                self.x = new_x
                self.y = new_y
            else:
                break

    def _get_sensor_angle(self, index):
        # the first 5 [0..4] sensors are front ones, each separated by 36 degrees
        if index < 5:
            return (180-(45*index) + self._rotation) % 360
        return (270+self._rotation) % 360

    def _get_pie_start_end(self, index):
        return ((index*90 + 45 + self._rotation)%360, (index*90 - 45 + self._rotation)%360)

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
        sensor_values = [0]*6
        for i in range(len(sensor_values)):
            angle = self.sensor_angles[i]
            x1 = self.x
            y1 = self.y
            for j in range(1000):
                x2 = x1 + j * math.cos(math.radians(angle))
                y2 = y1 + j * math.sin(math.radians(angle))
                if self.maze.isSolid(x2, y2):
                    sensor_values[i] = j
                    break
        return sensor_values

    def getPieValues(self, goal_x, goal_y):
        dx = goal_x - self.x
        dy = goal_y - self.y
        degs = math.degrees(math.atan2(-dy, dx)) % 360
        # pie indicates which pie is lit toward the goal
        # 0: right, 1: up, 2: left, 3: down
        pies = [0]*4
        for i in range(len(pies)):
            d_s, d_e = self._get_pie_start_end(i)
            if d_s <= degs and d_e >= degs:
                pies[i] = 1
        return pies




