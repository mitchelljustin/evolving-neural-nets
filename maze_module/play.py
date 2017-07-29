import numpy as np
import pygame
from scipy.spatial import distance

from maze_module.maze import Maze
from maze_module.robot import Robot

END_STEP_NUM = 400
BEHAVIOUR_SAMPLE_STEP_NUMS = [
  (i + 1) * END_STEP_NUM // 3 - 1
  for i in range(3)
]


class App:
  windowWidth = 800
  windowHeight = 600
  robot = []
  render = True

  def __init__(self, render=True):
    self._running = True
    self._display_surf = None
    self._image_surfs = [None] * 100
    self.goalPos = (400, 500)
    self.startPos = (400, 200)
    self.maze = Maze(800, 600, "./maze_module/maze.png")
    self.render = render

  def reset(self):
    self.robot = Robot(self.maze)
    self.robot.x = self.startPos[0]
    self.robot.y = self.startPos[1]

  def on_init(self):
    if self.render:
      pygame.init()
      self._display_surf = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)

      pygame.display.set_caption('Evolved neural nets')
      self._running = True
      self._image_surf = pygame.image.load("./maze_module/robot.png").convert()

  def on_render(self):
    self._display_surf.fill((255, 255, 255))
    self.maze.draw(self._display_surf)
    self._display_surf.fill((217, 33, 33), ((self.startPos[0] - 10, self.startPos[1] - 10), (20, 20)))
    self._display_surf.fill((66, 217, 33), ((self.goalPos[0] - 10, self.goalPos[1] - 10), (20, 20)))
    self._display_surf.blit(self._image_surf, (self.robot.x, self.robot.y))
    for j in range(len(self.robot.old_places) - 1):
      x, y = self.robot.old_places[j]
      x2, y2 = self.robot.old_places[j + 1]
      x = round(x)
      y = round(y)
      x2 = round(x2)
      y2 = round(y2)
      pygame.draw.line(self._display_surf, (217, 0, 0), (x, y), (x2, y2))

    pygame.display.flip()

  def on_cleanup(self):
    if self.render:
      pygame.quit()
    self.robot = Robot(self.maze)
    self.robot.x = self.startPos[0]
    self.robot.y = self.startPos[1]

  def on_execute(self, neuralNet):
    if not self.on_init():
      self._running = False
    behaviour = []
    for step in range(END_STEP_NUM):
      if self.render:
        pygame.event.pump()
      inputs = self.robot.getSensorValues() + self.robot.getPieValues(*self.goalPos)
      left_right, fwd_back = neuralNet.activate(inputs)
      self.robot.rotate(round(left_right))
      self.robot.move(round(fwd_back))
      if step in BEHAVIOUR_SAMPLE_STEP_NUMS:
        behaviour.append((self.robot.x, self.robot.y))
      if self.render and step % 20 == 0:
        self.on_render()
    self.on_cleanup()
    objective_value = 640 - distance.euclidean((self.robot.x, self.robot.y), self.goalPos)
    behaviour = np.array(behaviour).reshape([len(BEHAVIOUR_SAMPLE_STEP_NUMS) * 2])
    return np.concatenate([behaviour, [objective_value]], -1)


if __name__ == "__main__":
  theApp = App()
  theApp.on_execute()
