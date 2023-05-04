import sys

import numpy as np
import pygame as pg
from pygame.locals import *


class App:

    def __init__(self):
        self.camera = Camera()

        pg.init()
        pg.display.set_mode((640, 360))

        self.clock = pg.time.Clock()
        self.fps = 30

        self.main_loop()

    def main_loop(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == QUIT:
                    running = False
                    break

            keys = pg.key.get_pressed()

            if keys[K_w]:
                self.camera.move(z=-0.01)
            if keys[K_s]:
                self.camera.move(z=0.01)
            if keys[K_a]:
                self.camera.move(x=-0.01)
            if keys[K_d]:
                self.camera.move(x=0.01)
            if keys[K_LSHIFT]:
                self.camera.move(y=-0.01)
            if keys[K_SPACE]:
                self.camera.move(y=0.01)

            if keys[K_i]:
                self.camera.rotate([-1, 0, 0])
            if keys[K_k]:
                self.camera.rotate([1, 0, 0])
            if keys[K_j]:
                self.camera.rotate([0, -1, 0])
            if keys[K_l]:
                self.camera.rotate([0, 1, 0])
            if keys[K_u]:
                self.camera.rotate([0, 0, -1])
            if keys[K_o]:
                self.camera.rotate([0, 0, 1])

            if keys[K_q]:
                self.camera.change_fov(-1)
            if keys[K_e]:
                self.camera.change_fov(1)

            pg.display.flip()

            self.clock.tick(self.fps)

        self.quit()

    def quit(self):
        pg.quit()
        sys.exit()


# TODO: move models to a seperate file/make a general object with operations and reading vertices from file
class Cube:

    def __init__(self):
        self.vertices = (
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1)
        )


class Camera:

    def __init__(self):
        self.position = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.fov = 90

    def move(self, x=.0, y=.0, z=.0):
        # TODO: translate from relative coords to world coords and translate position vector
        self.position = [0, 0, 0]

    def rotate(self, v):
        if len(v) != 3:
            raise ValueError("Expected a 3-dimensional rotation vector")

        for i in range(3):
            if v[i] < -360 or v[i] > 360:
                raise ValueError("Rotation must be in range of [-360, 360]")

            self.rotation[i] += v[i]
            if self.rotation[i] > 360:
                self.rotation[i] -= 360
            if self.rotation[i] < 0:
                self.rotation[i] += 360

    def change_fov(self, val):
        self.fov += val

        # TODO: pick reasonable values for FOV range
        if self.fov < 0:
            self.fov = 0
        if self.fov > 360:
            self.fov = 360


if __name__ == "__main__":
    app = App()
