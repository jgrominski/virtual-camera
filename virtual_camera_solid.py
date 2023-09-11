import sys
from math import atan2, cos, pi, sin, sqrt, tan

import numpy as np
import pygame as pg
import pygame.gfxdraw as gfx
from pygame.locals import *


def to_pygame(coords: tuple[float, float]) -> tuple[int, int]:
    return (round(coords[0] * 640 + 640), round(360 - coords[1] * 360))


def draw_polygon(surface: pg.Surface, points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
    pygame_points = []
    for p in points:
        pygame_points.append(to_pygame(p))

    gfx.aapolygon(surface, pygame_points, color)
    gfx.filled_polygon(surface, pygame_points, color)


def in_range(point: np.ndarray) -> bool:
    if abs(point[0]) <= 1 and abs(point[1]) <= 1 and abs(point[2]) <= 1:
        return True
    else:
        return False


class App:

    def __init__(self) -> None:
        self.camera = Camera()
        self.scene = []

        pg.init()
        pg.display.set_caption('Virtual Camera App')
        self.screen = pg.display.set_mode((1280, 720))
        self.background_color = (197, 186, 173)
        self.clock = pg.time.Clock()
        self.fps = 30

        self.build_scene()
        self.main_loop()

    def build_scene(self) -> None:
        transform = Transformation()
        transform.scale((10, 10, 10))
        transform.translate((0, 0, -75))

        colors = [
            [
                (44, 62, 80),
                (44, 62, 80),
                (52, 73, 94),
                (52, 73, 94),
                (52, 152, 219),
                (52, 152, 219),
                (241, 196, 15),
                (241, 196, 15),
                (70, 99, 127),
                (70, 99, 127),
                (230, 126, 34),
                (230, 126, 34)
            ],
            [
                (46, 204, 113),
                (46, 204, 113),
                (52, 73, 94),
                (52, 73, 94),
                (44, 62, 80),
                (44, 62, 80),
                (241, 196, 15),
                (241, 196, 15),
                (70, 99, 127),
                (70, 99, 127),
                (230, 126, 34),
                (230, 126, 34)
            ],
            [
                (44, 62, 80),
                (44, 62, 80),
                (236, 240, 241),
                (236, 240, 241),
                (52, 152, 219),
                (52, 152, 219),
                (52, 73, 94),
                (52, 73, 94),
                (70, 99, 127),
                (70, 99, 127),
                (230, 126, 34),
                (230, 126, 34)
            ],
            [
                (46, 204, 113),
                (46, 204, 113),
                (236, 240, 241),
                (236, 240, 241),
                (44, 62, 80),
                (44, 62, 80),
                (52, 73, 94),
                (52, 73, 94),
                (70, 99, 127),
                (70, 99, 127),
                (230, 126, 34),
                (230, 126, 34)
            ],
            [
                (44, 62, 80),
                (44, 62, 80),
                (52, 73, 94),
                (52, 73, 94),
                (52, 152, 219),
                (52, 152, 219),
                (241, 196, 15),
                (241, 196, 15),
                (231, 76, 60),
                (231, 76, 60),
                (70, 99, 127),
                (70, 99, 127)
            ],
            [
                (46, 204, 113),
                (46, 204, 113),
                (52, 73, 94),
                (52, 73, 94),
                (44, 62, 80),
                (44, 62, 80),
                (241, 196, 15),
                (241, 196, 15),
                (231, 76, 60),
                (231, 76, 60),
                (70, 99, 127),
                (70, 99, 127)
            ],
            [
                (44, 62, 80),
                (44, 62, 80),
                (236, 240, 241),
                (236, 240, 241),
                (52, 152, 219),
                (52, 152, 219),
                (52, 73, 94),
                (52, 73, 94),
                (231, 76, 60),
                (231, 76, 60),
                (70, 99, 127),
                (70, 99, 127)
            ],
            [
                (46, 204, 113),
                (46, 204, 113),
                (236, 240, 241),
                (236, 240, 241),
                (44, 62, 80),
                (44, 62, 80),
                (52, 73, 94),
                (52, 73, 94),
                (231, 76, 60),
                (231, 76, 60),
                (70, 99, 127),
                (70, 99, 127)
            ]
        ]

        positions = [
            (-15, -15, -15),
            (15, -15, -15),
            (-15, -15, 15),
            (15, -15, 15),
            (-15, 15, -15),
            (15, 15, -15),
            (-15, 15, 15),
            (15, 15, 15)
        ]

        for i in range(8):
            position = Transformation()
            position.translate(positions[i])

            cube = Cube(colors[i])
            cube.transform(transform.get_t_matrix())
            cube.transform(position.get_t_matrix())

            polygons = cube.polygons
            for p in polygons:
                self.scene.append(p)

    def main_loop(self) -> None:
        self.screen.fill(self.background_color)

        move_speed = 0.75
        rotation_speed = .025
        fov_change_speed = .025

        running = True
        while running:
            for event in pg.event.get():
                if event.type == QUIT:
                    running = False
                    break
                if event.type == KEYDOWN:
                    if event.key == K_BACKSPACE:
                        self.camera.reset()

            keys = pg.key.get_pressed()

            t_vector = np.array([0., 0., 0., 1.])
            if keys[K_w]:
                t_vector[2] -= move_speed
            if keys[K_s]:
                t_vector[2] += move_speed
            if keys[K_a]:
                t_vector[0] -= move_speed
            if keys[K_d]:
                t_vector[0] += move_speed
            if keys[K_LSHIFT]:
                t_vector[1] -= move_speed
            if keys[K_SPACE]:
                t_vector[1] += move_speed
            self.camera.move(t_vector)

            rotation = np.array([0., 0., 0., 1.])
            if keys[K_i]:
                rotation[0] += rotation_speed
            if keys[K_k]:
                rotation[0] -= rotation_speed
            if keys[K_j]:
                rotation[1] += rotation_speed
            if keys[K_l]:
                rotation[1] -= rotation_speed
            if keys[K_u]:
                rotation[2] += rotation_speed
            if keys[K_o]:
                rotation[2] -= rotation_speed
            self.camera.rotate(rotation)

            if keys[K_q]:
                self.camera.change_fov(fov_change_speed)
            if keys[K_e]:
                self.camera.change_fov(-fov_change_speed)

            self.screen.fill(self.background_color)

            self.scene.sort(key=lambda x: x.camera_distance(
                self.camera), reverse=True)

            for polygon in self.scene:
                polygon.draw(self.screen, self.camera)

            pg.display.flip()

            self.clock.tick(self.fps)

        self.quit()

    def quit(self) -> None:
        pg.quit()
        sys.exit()


class Transformation:

    def __init__(self) -> None:
        self.t_matrix = np.identity(4)

    def translate(self, t_vector: tuple[float, float, float]) -> None:
        T = np.array([[1., 0., 0., t_vector[0]],
                      [0., 1., 0., t_vector[1]],
                      [0., 0., 1., t_vector[2]],
                      [0., 0., 0., 1.]])
        self.t_matrix = T.dot(self.t_matrix)

    def rotate_x(self, angle: float) -> None:
        R = np.array([[1., 0., 0., 0.],
                      [0., cos(angle), -sin(angle), 0.],
                      [0., sin(angle), cos(angle), 0.],
                      [0., 0., 0., 1.]])
        self.t_matrix = R.dot(self.t_matrix)

    def rotate_y(self, angle: float) -> None:
        R = np.array([[cos(angle), 0., sin(angle), 0.],
                      [0., 1., 0., 0.],
                      [-sin(angle), 0., cos(angle), 0.],
                      [0., 0., 0., 1.]])
        self.t_matrix = R.dot(self.t_matrix)

    def rotate_z(self, angle: float) -> None:
        R = np.array([[cos(angle), -sin(angle), 0., 0.],
                      [sin(angle), cos(angle), 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        self.t_matrix = R.dot(self.t_matrix)

    def scale(self, scale: tuple[float, float, float]) -> None:
        S = np.array([[scale[0], 0., 0., 0.],
                      [0., scale[1], 0., 0.],
                      [0., 0., scale[2], 0.],
                      [0., 0., 0., 1.]])
        self.t_matrix = S.dot(self.t_matrix)

    def get_t_matrix(self) -> np.ndarray:
        return self.t_matrix


class Camera:

    def __init__(self) -> None:
        self.position = np.array([0., 0., 0., 1.])
        self.rotation = np.array([0., 0., 0., 1.])
        self.fov = pi / 2
        self.transform = Transformation()

    def move(self, t_vector: np.ndarray) -> None:
        self.transform.translate([-x for x in self.position[:3]])
        t_vector = self.get_t_matrix().dot(t_vector)

        for i in range(3):
            self.position[i] += t_vector[i]

        self.transform.translate(self.position)

    def rotate(self, angles: np.ndarray) -> None:
        self.transform.translate([-x for x in self.position[:3]])
        angles = self.get_t_matrix().dot(angles)

        self.transform.rotate_x(angles[0])
        self.transform.rotate_y(angles[1])
        self.transform.rotate_z(angles[2])

        self.transform.translate(self.position)

    def change_fov(self, val: float) -> None:
        self.fov += val

        if self.fov < pi / 6:
            self.fov = pi / 6
        if self.fov > 2 * pi / 3:
            self.fov = 2 * pi / 3

    def reset(self) -> None:
        self.position = np.array([0., 0., 0., 1.])
        self.rotation = np.array([0., 0., 0., 1.])
        self.fov = pi / 2
        self.transform = Transformation()

    def get_t_matrix(self) -> np.ndarray:
        return self.transform.get_t_matrix()

    def get_projection(self) -> np.ndarray:
        n = .1
        f = 300
        e = 1 / tan(self.fov / 2)
        aspect = 16 / 9

        return np.array([[e / aspect, 0., 0., 0.],
                         [0., e, 0., 0.],
                         [0., 0., (f + n) / (n - f), (2 * f * n) / (n - f)],
                         [0., 0., -1., 0.]])


class Polygon:

    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, midpoint: np.ndarray, color: tuple[int, int, int]) -> None:
        self.vertices = [v1, v2, v3]
        self.midpoint = midpoint
        self.color = color

    def transform(self, t_matrix: np.ndarray) -> None:
        for i in range(len(self.vertices)):
            self.vertices[i] = t_matrix.dot(self.vertices[i])
        self.midpoint = t_matrix.dot(self.midpoint)

    def camera_distance(self, camera: Camera) -> float:
        return sqrt(
            (self.midpoint[0] - camera.position[0]) ** 2 +
            (self.midpoint[1] - camera.position[1]) ** 2 +
            (self.midpoint[2] - camera.position[2]) ** 2
        )

    def draw(self, surface: pg.Surface, camera: Camera) -> None:
        adjusted_vertices = []

        t_matrix = np.linalg.inv(camera.get_t_matrix())

        projection = camera.get_projection()

        for v in self.vertices:
            adjusted_vertex = projection.dot(t_matrix.dot(v))
            adjusted_vertex /= adjusted_vertex[3]
            adjusted_vertices.append(adjusted_vertex[:3])

        if in_range(adjusted_vertices[0]) and in_range(adjusted_vertices[1]) and in_range(adjusted_vertices[2]):
            point_a = adjusted_vertices[0][:2]
            point_b = adjusted_vertices[1][:2]
            point_c = adjusted_vertices[2][:2]

            draw_polygon(surface, [point_a, point_b, point_c], self.color)


class Cube:

    def __init__(self, polygon_colors: list[tuple[int, int, int]]) -> None:
        self.vertices = [np.array([1., 1., 1., 1.]),
                         np.array([1., 1., -1., 1.]),
                         np.array([1., -1., 1., 1.]),
                         np.array([1., -1., -1., 1.]),
                         np.array([-1., 1., 1., 1.]),
                         np.array([-1., 1., -1., 1.]),
                         np.array([-1., -1., 1., 1.]),
                         np.array([-1., -1., -1., 1.])]
        self.polygon_connections = [(0, 1, 2),
                                    (1, 2, 3),
                                    (0, 4, 6),
                                    (0, 2, 6),
                                    (4, 5, 7),
                                    (4, 6, 7),
                                    (1, 3, 5),
                                    (3, 5, 7),
                                    (0, 1, 5),
                                    (0, 4, 5),
                                    (2, 3, 6),
                                    (3, 6, 7)]
        self.polygon_midpoints = [np.array([1., 0., 0., 1.]),
                                  np.array([1., 0., 0., 1.]),
                                  np.array([0., 0., 1., 1.]),
                                  np.array([0., 0., 1., 1.]),
                                  np.array([-1., 0., 0., 1.]),
                                  np.array([-1., 0., 0., 1.]),
                                  np.array([0., 0., -1., 1.]),
                                  np.array([0., 0., -1., 1.]),
                                  np.array([0., 1., 0., 1.]),
                                  np.array([0., 1., 0., 1.]),
                                  np.array([0., -1., 0., 1.]),
                                  np.array([0., -1., 0., 1.])]
        self.polygon_colors = polygon_colors
        self.polygons = []
        self.define_polygons()

    def define_polygons(self) -> None:
        for i in range(len(self.polygon_connections)):
            self.polygons.append(Polygon(self.vertices[self.polygon_connections[i][0]],
                                         self.vertices[self.polygon_connections[i][1]],
                                         self.vertices[self.polygon_connections[i][2]],
                                         self.polygon_midpoints[i],
                                         self.polygon_colors[i]))

    def transform(self, t_matrix: np.ndarray) -> None:
        for p in self.polygons:
            p.transform(t_matrix)


if __name__ == "__main__":
    app = App()
