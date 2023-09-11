import sys
from math import atan2, cos, pi, sin, sqrt, tan

import numpy as np
import pygame as pg
import pygame.gfxdraw as gfx
from pygame.locals import *

# Define region codes
INSIDE = 0  # 0000
LEFT = 1    # 0001
RIGHT = 2   # 0010
BOTTOM = 4  # 0100
TOP = 8     # 1000


def to_pygame(coords: tuple[float, float]) -> tuple[float, float]:
    return (round(coords[0] * 640 + 640), round(360 - coords[1] * 360))


def draw_line(surface: pg.Surface, start_pos: tuple[float, float], end_pos: tuple[float, float],
              color: tuple[int, int, int], width: int, rounded=False) -> None:
    start_pos = to_pygame(start_pos)
    end_pos = to_pygame(end_pos)

    center = ((start_pos[0] + end_pos[0]) / 2,
              (start_pos[1] + end_pos[1]) / 2)

    length = sqrt((start_pos[0] - end_pos[0]) ** 2 +
                  (start_pos[1] - end_pos[1]) ** 2)
    angle = atan2(start_pos[1] - end_pos[1], start_pos[0] - end_pos[0])

    ul = (center[0] + (length/2.) * cos(angle) - (width/2.) * sin(angle),
          center[1] + (width/2.) * cos(angle) + (length/2.) * sin(angle))
    ur = (center[0] - (length/2.) * cos(angle) - (width/2.) * sin(angle),
          center[1] + (width/2.) * cos(angle) - (length/2.) * sin(angle))
    bl = (center[0] + (length/2.) * cos(angle) + (width/2.) * sin(angle),
          center[1] - (width/2.) * cos(angle) + (length/2.) * sin(angle))
    br = (center[0] - (length/2.) * cos(angle) + (width/2.) * sin(angle),
          center[1] - (width/2.) * cos(angle) - (length/2.) * sin(angle))

    gfx.aapolygon(surface, [ul, ur, br, bl], color)
    gfx.filled_polygon(surface, [ul, ur, br, bl], color)

    if rounded:
        ends = ((int((ul[0] + bl[0]) / 2), int((ul[1] + bl[1]) / 2)),
                (int((ur[0] + br[0]) / 2), int((ur[1] + br[1]) / 2)))
        gfx.aacircle(surface, ends[0][0], ends[0][1], int(width / 2), color)
        gfx.filled_circle(surface, ends[0][0],
                          ends[0][1], int(width / 2), color)
        gfx.aacircle(surface, ends[1][0], ends[1][1], int(width / 2), color)
        gfx.filled_circle(surface, ends[1][0],
                          ends[1][1], int(width / 2), color)


def in_range(point: np.ndarray) -> bool:
    if abs(point[0]) <= 1 and abs(point[1]) <= 1 and abs(point[2]) <= 1:
        return True
    else:
        return False


def line_line_intersection(point_a: tuple[float, float], point_b: tuple[float, float], point_c: tuple[float, float], point_d: tuple[float, float]) -> tuple[float, float]:
    x = ((point_a[0] * point_b[1] - point_a[1] * point_b[0]) * (point_c[0] - point_d[0]) - (point_a[0] - point_b[0]) * (point_c[0] * point_d[1] -
         point_c[1] * point_d[0])) / ((point_a[0] - point_b[0]) * (point_c[1] - point_d[1]) - (point_a[1] - point_b[1]) * (point_c[0] - point_d[0]))
    y = ((point_a[0] * point_b[1] - point_a[1] * point_b[0]) * (point_c[1] - point_d[1]) - (point_a[1] - point_b[1]) * (point_c[0] * point_d[1] -
         point_c[1] * point_d[0])) / ((point_a[0] - point_b[0]) * (point_c[1] - point_d[1]) - (point_a[1] - point_b[1]) * (point_c[0] - point_d[0]))
    return (x, y)


def compute_region_code(point: tuple[float, float]) -> int:
    x, y = point
    code = INSIDE

    if x < -1:
        code |= LEFT
    elif x > 1:
        code |= RIGHT

    if y < -1:
        code |= BOTTOM
    elif y > 1:
        code |= TOP

    return code


def cohen_sutherland(point_a: tuple[float, float], point_b: tuple[float, float]) -> tuple[float, float]:
    code_a = compute_region_code(point_a)
    code_b = compute_region_code(point_b)

    if (code_a & code_b) != 0:
        return None, None

    if code_a != INSIDE:
        if (code_a & LEFT) != 0:
            point_a = line_line_intersection(
                point_a, point_b, (-1, -1), (-1, 1))
        if (code_a & RIGHT) != 0:
            point_a = line_line_intersection(point_a, point_b, (1, -1), (1, 1))
        if (code_a & BOTTOM) != 0:
            point_a = line_line_intersection(
                point_a, point_b, (-1, -1), (1, -1))
        if (code_a & TOP) != 0:
            point_a = line_line_intersection(point_a, point_b, (-1, 1), (1, 1))

    if code_b != INSIDE:
        if (code_b & LEFT) != 0:
            point_b = line_line_intersection(
                point_a, point_b, (-1, -1), (-1, 1))
        if (code_b & RIGHT) != 0:
            point_b = line_line_intersection(point_a, point_b, (1, -1), (1, 1))
        if (code_b & BOTTOM) != 0:
            point_b = line_line_intersection(
                point_a, point_b, (-1, -1), (1, -1))
        if (code_b & TOP) != 0:
            point_b = line_line_intersection(point_a, point_b, (-1, 1), (1, 1))

    return point_a, point_b


class App:

    def __init__(self) -> None:
        self.camera = Camera()
        self.scene = []

        pg.init()
        pg.display.set_caption('Virtual Camera App')
        self.screen = pg.display.set_mode((1280, 720))
        self.background_color = (44, 62, 80)
        self.clock = pg.time.Clock()
        self.fps = 30

        self.build_scene()
        self.main_loop()

    def build_scene(self) -> None:
        transform = Transformation()
        transform.scale((10, 10, 10))
        transform.translate((0, 0, -75))

        colors = [
            (26, 188, 156),
            (155, 89, 182),
            (230, 126, 34),
            (236, 240, 241),
            (52, 152, 219),
            (231, 76, 60),
            (149, 165, 166),
            (241, 196, 15)
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

            cube = Cube(colors[i], 3)
            cube.transform(transform.get_t_matrix())
            cube.transform(position.get_t_matrix())

            self.scene.append(cube)

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

            for cube in self.scene:
                cube.draw(self.screen, self.camera)

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


class Cube:

    def __init__(self, color: tuple[int, int, int], width: int) -> None:
        self.origin = np.array([0., 0., 0., 1.])
        self.vertices = [np.array([1., 1., 1., 1.]),
                         np.array([1., 1., -1., 1.]),
                         np.array([1., -1., 1., 1.]),
                         np.array([1., -1., -1., 1.]),
                         np.array([-1., 1., 1., 1.]),
                         np.array([-1., 1., -1., 1.]),
                         np.array([-1., -1., 1., 1.]),
                         np.array([-1., -1., -1., 1.])]
        self.edges = [(0, 1),
                      (0, 2),
                      (0, 4),
                      (1, 3),
                      (1, 5),
                      (2, 3),
                      (2, 6),
                      (3, 7),
                      (4, 5),
                      (4, 6),
                      (5, 7),
                      (6, 7)]
        self.color = color
        self.width = width

    def transform(self, t_matrix: np.ndarray) -> None:
        for i in range(len(self.vertices)):
            self.vertices[i] = t_matrix.dot(self.vertices[i])
        self.origin = t_matrix.dot(self.origin)

    def camera_distance(self, camera: Camera) -> float:
        return sqrt(
            (self.origin[0] - camera.position[0]) ** 2 +
            (self.origin[1] - camera.position[1]) ** 2 +
            (self.origin[2] - camera.position[2]) ** 2
        )

    def draw(self, surface: pg.Surface, camera: Camera) -> None:
        adjusted_vertices = []

        t_matrix = np.linalg.inv(camera.get_t_matrix())

        projection = camera.get_projection()

        for v in self.vertices:
            adjusted_vertex = projection.dot(t_matrix.dot(v))
            adjusted_vertex /= adjusted_vertex[3]
            adjusted_vertices.append(adjusted_vertex[:3])

        for e in self.edges:
            if in_range(adjusted_vertices[e[0]]) or in_range(adjusted_vertices[e[1]]):
                point_a = adjusted_vertices[e[0]][:2]
                point_b = adjusted_vertices[e[1]][:2]

                if abs(adjusted_vertices[e[0]][2]) > 1 or abs(adjusted_vertices[e[1]][2]) > 1:
                    if abs(point_a[0]) != 1 and abs(point_a[1]) != 1 and abs(point_b[0]) != 1 and abs(point_b[1]) != 1:
                        continue

                start_pos, end_pos = cohen_sutherland(point_a, point_b)

                if start_pos is not None and end_pos is not None:
                    draw_line(surface, start_pos, end_pos,
                              self.color, self.width, rounded=True)


if __name__ == "__main__":
    app = App()
