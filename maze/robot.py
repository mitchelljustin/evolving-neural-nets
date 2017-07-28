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

    def __init__(self):
        return


    def moveRight(self):
        self.x = self.x + self.speed

    def moveLeft(self):
        self.x = self.x - self.speed

    def moveUp(self):
        self.y = self.y - self.speed

    def moveDown(self):
        self.y = self.y + self.speed

    def _get_sensor_angle(self, index):
        # the first 5 [0..4] sensors are front ones, each separated by 36 degrees
        if index < 5:
            return (180-(36*i) + self._rotation) % 360
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

    def getSensorValues(self, maze):
        # returns sensor array
        for i in range(len(self.sensor_angles)):
            self.sensor_angles[i] = _get_sensor_angle(self, i)
        sensor_values = [0]*6
        for i in rangel(len(sensor_values)):
            angle = self.sensor_angles[i]
            x1 = self.x
            y1 = self.y
            for j in range(10000):
                x2 = x1 + j * math.cos(math.radians(angle))
                y2 = y1 + j * math.sin(math.radians(angle))
                if maze.isSolid(x2, y2):
                    sensor_values[i] = j
        return sensors

    def getPieValues(self, goal_x, goal_y):
        dx = goal_x - self.x
        dy = goal_y - self.y
        degs = math.degrees(math.atan2(-dy, dx)) % 360
        print(degs)
        # pie indicates which pie is lit toward the goal
        # 0: right, 1: up, 2: left, 3: down
        pies = [0]*4
        for i in range(len(pies)):
            d_s, d_e = self._get_pie_start_end(i)
            if d_s <= degs and d_e >= degs:
                pies[i] = 1
        return pies




