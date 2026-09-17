import time
import math
import json
from pathlib import Path
import sys
import pygame
import numpy as np
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from control import (  # noqa: E402
        CONTROL_DT,
        CONTROL_HZ,
        FixedRateScheduler,
        PID,
        approach,
        load_poses,
        load_settings,
        save_json,
    )
except ModuleNotFoundError as error:
    if error.name != "control":
        raise

    CONTROL_HZ = 50.0
    CONTROL_DT = 1.0 / CONTROL_HZ
    DEFAULT_SETTINGS = {
        "control_hz": CONTROL_HZ,
        "max_joint_speed": 90.0,
        "posture_bias": 0.08,
        "angle_weight": 0.1,
        "minimum_target_y": -1.0,
        "pid": {
            "p": [5.0, 6.0, 6.0, 4.0],
            "i": [0.0, 0.0, 0.0, 0.0],
            "d": [0.35, 0.45, 0.45, 0.30],
            "integral_limit": 10.0,
            "output_limit": 180.0,
        },
    }
    DEFAULT_POSES = [{
        "name": "home",
        "target": {"x": 0.0, "y": 1.0, "z": -9.0},
        "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "angles": [0.0, 0.0, 0.0, 0.0],
    }]

    def _fallback_copy(value):
        return json.loads(json.dumps(value))

    def _fallback_load(path, defaults):
        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, ValueError, TypeError):
            return _fallback_copy(defaults)
        return value if isinstance(value, type(defaults)) else _fallback_copy(defaults)

    def load_settings(path):
        settings = _fallback_load(path, DEFAULT_SETTINGS)
        settings["control_hz"] = CONTROL_HZ
        settings["max_joint_speed"] = max(1.0, float(settings["max_joint_speed"]))
        settings["posture_bias"] = max(0.0, float(settings["posture_bias"]))
        return settings

    def load_poses(path):
        poses = _fallback_load(path, DEFAULT_POSES)
        return poses if isinstance(poses, list) and poses else _fallback_copy(DEFAULT_POSES)

    def save_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")

    class PID:
        def __init__(self, kp, ki, kd, integral_limit=10.0, output_limit=180.0):
            self.kp = float(kp)
            self.ki = float(ki)
            self.kd = float(kd)
            self.integral_limit = abs(float(integral_limit))
            self.output_limit = abs(float(output_limit))
            self.reset()

        def reset(self):
            self.integral = 0.0
            self.previous_error = None

        def update(self, target, measured, dt=CONTROL_DT):
            error = float(target) - float(measured)
            self.integral += error * dt
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
            derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
            self.previous_error = error
            output = self.kp * error + self.ki * self.integral + self.kd * derivative
            return max(-self.output_limit, min(self.output_limit, output))

    class FixedRateScheduler:
        def __init__(self, hz=CONTROL_HZ, max_steps=5):
            self.dt = 1.0 / float(hz)
            self.max_steps = int(max_steps)
            self.accumulator = 0.0

        def advance(self, elapsed):
            self.accumulator += max(0.0, elapsed)
            steps = 0
            while self.accumulator + 1e-12 >= self.dt and steps < self.max_steps:
                self.accumulator -= self.dt
                steps += 1
            return steps

    def approach(current, target, maximum_delta):
        delta = float(target) - float(current)
        if abs(delta) <= maximum_delta:
            return float(target)
        return float(current) + math.copysign(maximum_delta, delta)

class Material:
    def __init__(self):
        self.textures = {}

    def set_texture(self, face, tex_id):
        self.textures[face] = tex_id

    def bind(self, face):
        tex = self.textures.get(face)
        if tex is None:
            glBindTexture(GL_TEXTURE_2D, 0)
        else:
            glBindTexture(GL_TEXTURE_2D, tex)

class Mesh:
    def __init__(self, vertices, faces, edges, uvs=None):
        self.vertices = vertices
        self.faces = faces
        self.edges = edges
        self.uvs = uvs  

class Object3D:
    def __init__(self, mesh, material=None):
        self.mesh = mesh
        self.material = material or Material()
        self.x = self.y = self.z = 0
        self.rx = self.ry = self.rz = 0
        self.pivot = (0.0, 0.0, 0.0)
        self.edge_color = (1, 1, 1)
        self.radius = self._calc_radius()
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

    def _local_matrix(self):
        T = _translation_matrix(self.x, self.y, self.z)
        px, py, pz = self.pivot
        P = _translation_matrix(px, py, pz)
        Pinv = _translation_matrix(-px, -py, -pz)
        R = _rotation_matrix(self.rx, self.ry, self.rz)
        return T @ P @ R @ Pinv

    def _calc_radius(self):
        if not self.mesh.vertices:
            return 1
        return max(math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in self.mesh.vertices)

    def translate(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def rotate(self, rx, ry, rz):
        self.rx, self.ry, self.rz = rx, ry, rz
        
    def rotate_around_point(self, px, py, pz, rx, ry, rz):
        self.pivot = (px, py, pz)
        self.rx, self.ry, self.rz = rx, ry, rz

    def set_pivot(self, px, py, pz):
        self.pivot = (px, py, pz)

    def render(self, planes=None, parent_matrix=None):
        if parent_matrix is None:
            parent_matrix = np.identity(4)

        local_matrix = self._local_matrix()
        world_matrix = parent_matrix @ local_matrix

        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)

        px, py, pz = self.pivot
        glTranslatef(px, py, pz)
        glRotatef(self.rx, 1, 0, 0)
        glRotatef(self.ry, 0, 1, 0)
        glRotatef(self.rz, 0, 0, 1)
        glTranslatef(-px, -py, -pz)

        visible = True
        if planes is not None:
            world_pos = (world_matrix[0, 3], world_matrix[1, 3], world_matrix[2, 3])
            visible = sphere_in_frustum(world_pos, self.radius, planes)

        if visible:
            v = self.mesh.vertices

            glEnable(GL_TEXTURE_2D)
            glDisable(GL_COLOR_MATERIAL)

            for name, face in self.mesh.faces:
                tex = self.material.textures.get(name)
                glBindTexture(GL_TEXTURE_2D, tex if tex else 0)

                glBegin(GL_QUADS)
                for i, vi in enumerate(face):
                    if self.mesh.uvs:
                        glTexCoord2fv(self.mesh.uvs[vi])
                    else:
                        glTexCoord2f([0,1,1,0][i], [0,0,1,1][i])
                    glVertex3fv(v[vi])
                glEnd()

        for child in self.children:
            child.render(planes, world_matrix)

        glPopMatrix()
        
    def render_skybox(self):
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)

        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)

        v = self.mesh.vertices
        glEnable(GL_TEXTURE_2D)
        glColor3f(1, 1, 1)

        for name, face in self.mesh.faces:
            val = self.material.textures.get(name)

            if val is None:
                continue
            elif isinstance(val, tuple):
                if len(val) == 3:
                    tex, rot, flip = val
                else:
                    tex, rot = val
                    flip = None
                glBindTexture(GL_TEXTURE_2D, tex)
            else:
                glBindTexture(GL_TEXTURE_2D, val)
                rot = 0
                flip = None

            uvs = [(0,0),(1,0),(1,1),(0,1)]

            rot_steps = (rot // 90) % 4
            uvs = uvs[rot_steps:] + uvs[:rot_steps]

            if flip == "h":
                uvs = [uvs[1], uvs[0], uvs[3], uvs[2]]
            elif flip == "v":
                uvs = [uvs[3], uvs[2], uvs[1], uvs[0]]

            reversed_face = face[::-1]

            glBegin(GL_QUADS)
            for i, vi in enumerate(reversed_face):
                glTexCoord2f(*uvs[i])
                glVertex3fv(v[vi])
            glEnd()

        glPopMatrix()

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

def create_prism_mesh(l, w, h):
    l, w, h = l/2, w/2, h/2

    vertices = (
        ( w,  l,  h),
        (-w,  l,  h),
        ( w, -l,  h),
        ( w,  l, -h),
        (-w, -l,  h),
        (-w,  l, -h),
        ( w, -l, -h),
        (-w, -l, -h),
    )

    faces = (
        ("front",  (0, 1, 4, 2)),
        ("back",   (3, 5, 7, 6)),
        ("top",    (0, 1, 5, 3)),
        ("bottom", (2, 4, 7, 6)),
        ("right",  (0, 2, 6, 3)),
        ("left",   (1, 4, 7, 5)),
    )

    edges = (
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7)
    )

    return Mesh(vertices, faces, edges)

def create_plane_mesh(l, w):
    l, w = l/2, w/2

    vertices = (
        ( w,  l, 0),
        (-w,  l, 0),
        ( w, -l, 0),
        (-w, -l, 0),
    )

    faces = (
        ("front", (0, 1, 3, 2)),
    )

    edges = (
        (0, 1), (0, 2), (3, 1), (3, 2)
    )

    return Mesh(vertices, faces, edges)

def create_sphere_mesh(radius, stacks, slices):
    vertices = []
    uvs = []
    faces = []
    edges = []

    for i in range(stacks + 1):
        lat = math.pi * (-0.5 + i / stacks)
        y = radius * math.sin(lat)
        xz = radius * math.cos(lat)

        for j in range(slices + 1):
            lon = 2 * math.pi * j / slices
            x = xz * math.cos(lon)
            z = xz * math.sin(lon)
            vertices.append((x, y, z))
            uvs.append((j / slices, i / stacks))  

    for i in range(stacks):
        for j in range(slices):
            top_left     = i * (slices + 1) + j
            top_right    = top_left + 1
            bottom_left  = top_left + (slices + 1)
            bottom_right = bottom_left + 1

            faces.append((f"face_{i}_{j}", (top_left, top_right, bottom_right, bottom_left)))
            edges.append((top_left, top_right))
            edges.append((top_left, bottom_left))

    return Mesh(tuple(vertices), tuple(faces), tuple(edges), uvs=tuple(uvs))

def create_prism(l, w, h, tex_id=None):
    mesh = create_prism_mesh(l, w, h)
    obj = Object3D(mesh)
    if tex_id:
        for face_name in ("front", "back", "top", "bottom", "left", "right"):
            obj.material.set_texture(face_name, tex_id)
    return obj

def create_plane(l, w, tex_id=None):
    mesh = create_plane_mesh(l, w)
    obj = Object3D(mesh)
    if tex_id:
        obj.material.set_texture("front", tex_id)
    return obj

def create_sphere(radius, stacks, slices, tex_id=None):
    mesh = create_sphere_mesh(radius, stacks, slices)
    obj = Object3D(mesh)
    if tex_id:
        for i in range(stacks):
            for j in range(slices):
                obj.material.set_texture(f"face_{i}_{j}", tex_id)
    return obj
def create_skybox(size, front=None, back=None, top=None, bottom=None, left=None, right=None):
    mesh = create_prism_mesh(size, size, size)
    obj = Object3D(mesh)

    def set(face, tex):
        if tex:
            obj.material.textures[face] = (tex, 0)  

    set("back",   front)
    set("front",  back)
    set("top",    top)
    set("bottom", bottom)
    set("right",  left)
    set("left",   right)
    obj.edge_color = (0, 0, 0)
    return obj

def start():
    pygame.init()
    display = (display_x, display_y)
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.display.set_caption(display_name)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov, aspect_ratio, cliping_plane_min, cliping_plane_max)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    glEnable(GL_TEXTURE_2D)

    glClearColor(66/255, 125/255, 219/255, 1)

    return screen

def load_image(path):
    try:
        texture_surface = pygame.image.load(path)
    except FileNotFoundError:
        texture_surface = pygame.Surface((1, 1))
        texture_surface.fill((180, 180, 180))

    if path.lower().endswith(".png"):
        texture_data = pygame.image.tostring(texture_surface, "RGBA", True)
        fmt = GL_RGBA
    else:
        texture_data = pygame.image.tostring(texture_surface, "RGB", True)
        fmt = GL_RGB

    width, height = texture_surface.get_rect().size

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, fmt,
                 width, height, 0,
                 fmt, GL_UNSIGNED_BYTE,
                 texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return tex_id

def load_image_skybox(path):
    try:
        texture_surface = pygame.image.load(path)
    except FileNotFoundError:
        texture_surface = pygame.Surface((1, 1))
        texture_surface.fill((110, 160, 210))

    if path.lower().endswith(".png"):
        texture_data = pygame.image.tostring(texture_surface, "RGBA", True)
        fmt = GL_RGBA
    else:
        texture_data = pygame.image.tostring(texture_surface, "RGB", True)
        fmt = GL_RGB

    width, height = texture_surface.get_rect().size

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, fmt,
                 width, height, 0,
                 fmt, GL_UNSIGNED_BYTE,
                 texture_data)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return tex_id

#Optimisation
def _translation_matrix(x, y, z):
    m = np.identity(4)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m

def _rotation_matrix(rx, ry, rz):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]])
    Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]])
    Rz = np.array([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]])
    return Rx @ Ry @ Rz

def sphere_in_frustum(center, radius, planes):
    x, y, z = center
    for p in planes:
        distance = p[0]*x + p[1]*y + p[2]*z + p[3]
        if distance < -radius:
            return False
    return True

def get_frustum_planes():
    proj = glGetDoublev(GL_PROJECTION_MATRIX).flatten()
    modl = glGetDoublev(GL_MODELVIEW_MATRIX).flatten()

    clip = [0] * 16

    clip[ 0] = modl[ 0]*proj[ 0] + modl[ 1]*proj[ 4] + modl[ 2]*proj[ 8] + modl[ 3]*proj[12]
    clip[ 1] = modl[ 0]*proj[ 1] + modl[ 1]*proj[ 5] + modl[ 2]*proj[ 9] + modl[ 3]*proj[13]
    clip[ 2] = modl[ 0]*proj[ 2] + modl[ 1]*proj[ 6] + modl[ 2]*proj[10] + modl[ 3]*proj[14]
    clip[ 3] = modl[ 0]*proj[ 3] + modl[ 1]*proj[ 7] + modl[ 2]*proj[11] + modl[ 3]*proj[15]

    clip[ 4] = modl[ 4]*proj[ 0] + modl[ 5]*proj[ 4] + modl[ 6]*proj[ 8] + modl[ 7]*proj[12]
    clip[ 5] = modl[ 4]*proj[ 1] + modl[ 5]*proj[ 5] + modl[ 6]*proj[ 9] + modl[ 7]*proj[13]
    clip[ 6] = modl[ 4]*proj[ 2] + modl[ 5]*proj[ 6] + modl[ 6]*proj[10] + modl[ 7]*proj[14]
    clip[ 7] = modl[ 4]*proj[ 3] + modl[ 5]*proj[ 7] + modl[ 6]*proj[11] + modl[ 7]*proj[15]

    clip[ 8] = modl[ 8]*proj[ 0] + modl[ 9]*proj[ 4] + modl[10]*proj[ 8] + modl[11]*proj[12]
    clip[ 9] = modl[ 8]*proj[ 1] + modl[ 9]*proj[ 5] + modl[10]*proj[ 9] + modl[11]*proj[13]
    clip[10] = modl[ 8]*proj[ 2] + modl[ 9]*proj[ 6] + modl[10]*proj[10] + modl[11]*proj[14]
    clip[11] = modl[ 8]*proj[ 3] + modl[ 9]*proj[ 7] + modl[10]*proj[11] + modl[11]*proj[15]

    clip[12] = modl[12]*proj[ 0] + modl[13]*proj[ 4] + modl[14]*proj[ 8] + modl[15]*proj[12]
    clip[13] = modl[12]*proj[ 1] + modl[13]*proj[ 5] + modl[14]*proj[ 9] + modl[15]*proj[13]
    clip[14] = modl[12]*proj[ 2] + modl[13]*proj[ 6] + modl[14]*proj[10] + modl[15]*proj[14]
    clip[15] = modl[12]*proj[ 3] + modl[13]*proj[ 7] + modl[14]*proj[11] + modl[15]*proj[15]

    planes = []

    planes.append([
        clip[3] - clip[0],
        clip[7] - clip[4],
        clip[11] - clip[8],
        clip[15] - clip[12]
    ])

    planes.append([
        clip[3] + clip[0],
        clip[7] + clip[4],
        clip[11] + clip[8],
        clip[15] + clip[12]
    ])

    planes.append([
        clip[3] + clip[1],
        clip[7] + clip[5],
        clip[11] + clip[9],
        clip[15] + clip[13]
    ])

    planes.append([
        clip[3] - clip[1],
        clip[7] - clip[5],
        clip[11] - clip[9],
        clip[15] - clip[13]
    ])

    planes.append([
        clip[3] - clip[2],
        clip[7] - clip[6],
        clip[11] - clip[10],
        clip[15] - clip[14]
    ])

    planes.append([
        clip[3] + clip[2],
        clip[7] + clip[6],
        clip[11] + clip[10],
        clip[15] + clip[14]
    ])

    normalized = []

    for p in planes:
        mag = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)

        normalized.append([
            p[0]/mag,
            p[1]/mag,
            p[2]/mag,
            p[3]/mag
        ])

    return normalized

# 2d functions
def load_image_from_surface(surface):
    texture_data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex_id

def draw_2d_start():
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display_x, display_y, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

def draw_2d_end():
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glEnable(GL_DEPTH_TEST)

def draw_rect(x, y, w, h, color=(1, 1, 1)):
    glDisable(GL_TEXTURE_2D)
    glColor3fv(color)
    glBegin(GL_QUADS)
    glVertex2f(x,     y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x,     y + h)
    glEnd()
    glColor3f(1, 1, 1)
    glEnable(GL_TEXTURE_2D)

def draw_image(tex_id, x, y, w, h):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x,     y)
    glTexCoord2f(1, 0); glVertex2f(x + w, y)
    glTexCoord2f(1, 1); glVertex2f(x + w, y + h)
    glTexCoord2f(0, 1); glVertex2f(x,     y + h)
    glEnd()

def draw_text(text, x, y, size=24, color=(255, 255, 255)):
    font = pygame.font.SysFont(None, size)
    surface = font.render(text, True, color)
    tex_id = load_image_from_surface(surface)
    w, h = surface.get_size()

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor4f(1, 1, 1, 1)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x,     y)
    glTexCoord2f(1, 1); glVertex2f(x + w, y)
    glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
    glTexCoord2f(0, 0); glVertex2f(x,     y + h)
    glEnd()

    glDisable(GL_BLEND)
    glDeleteTextures([tex_id])
    
#moar functions
def set_all_textures(obj, tex_id):
    for name, _ in obj.mesh.faces:
        obj.material.set_texture(name, tex_id)
    
#Tourque functions
def solve_torque(r, f, o):
    return r * f * math.sin(o)

def solve_weight(mass, gravity):
    return mass * gravity

arm_lengths = (4.0, 3.0, 2.0)
joint_names = ("base", "shoulder", "elbow", "wrist")
joint_limits = ((-180.0, 180.0), (-90.0, 90.0), (-135.0, 135.0), (-90.0, 90.0))
minimum_target_y = -1.0
posture_bias = 0.08

def arm_end_position(angles):
    yaw, shoulder, elbow, wrist = [math.radians(value) for value in angles]
    pitches = (shoulder, shoulder + elbow, shoulder + elbow + wrist)
    depth = sum(length * math.cos(pitch) for length, pitch in zip(arm_lengths, pitches))
    height = 1.0 + sum(length * math.sin(pitch) for length, pitch in zip(arm_lengths, pitches))
    return np.array((-math.sin(yaw) * depth, height, -math.cos(yaw) * depth))

def arm_end_orientation(angles):
    return np.array((angles[0], angles[1] + angles[2] + angles[3]))

def angle_difference(target, current):
    return (target - current + 180.0) % 360.0 - 180.0

def rotation_matrix_to_angles(matrix):
    pitch = math.asin(max(-1.0, min(1.0, matrix[0, 2])))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1e-6:
        roll = math.atan2(-matrix[1, 2], matrix[2, 2])
        yaw = math.atan2(-matrix[0, 1], matrix[0, 0])
    else:
        roll = math.atan2(matrix[2, 1], matrix[1, 1])
        yaw = 0.0
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))

def claw_local_orientation(angles, target_orientation):
    parent_rotation = (
        _rotation_matrix(0.0, angles[0], 0.0)
        @ _rotation_matrix(angles[1], 0.0, 0.0)
        @ _rotation_matrix(angles[2], 0.0, 0.0)
        @ _rotation_matrix(angles[3], 0.0, 0.0)
    )
    target_rotation = _rotation_matrix(
        target_orientation["pitch"],
        target_orientation["yaw"],
        target_orientation["roll"],
    )
    return rotation_matrix_to_angles(
        np.linalg.inv(parent_rotation) @ target_rotation
    )

def ik_target_position(target, target_orientation):
    target = np.array(target, dtype=float)
    target_rotation = _rotation_matrix(
        target_orientation["pitch"],
        target_orientation["yaw"],
        target_orientation["roll"],
    )
    claw_offset = target_rotation @ np.array((0.0, 0.0, -1.0, 0.0))
    return target - claw_offset[:3]

def solve_arm_ik(target, orientation, starting_angles, angle_weight):
    if isinstance(target, dict):
        target = (target["x"], target["y"], target["z"])
    target = np.array(target, dtype=float)
    target[1] = max(minimum_target_y, target[1])
    angles = np.array(starting_angles, dtype=float)
    target_orientation = np.array((orientation["yaw"], orientation["pitch"]))
    orientation_weight = max(0.0, min(1.0, float(angle_weight))) * 0.05
    posture = np.array([angles[0], 35.0, -20.0, 0.0])
    for _ in range(120):
        position = arm_end_position(angles)
        orientation_error = angle_difference(
            target_orientation, arm_end_orientation(angles)
        )
        position_error = target - position
        error = np.concatenate((position_error, orientation_error * orientation_weight))
        if (np.linalg.norm(position_error) < 0.001
                and np.linalg.norm(orientation_error) < 0.1):
            break

        jacobian = np.zeros((5, 4))
        for index in range(4):
            probe = angles.copy()
            probe[index] += 0.1
            probe_position = arm_end_position(probe)
            probe_orientation = angle_difference(
                arm_end_orientation(probe), arm_end_orientation(angles)
            )
            jacobian[:, index] = np.concatenate((
                (probe_position - position) / 0.1,
                probe_orientation * orientation_weight / 0.1,
            ))

        damping = 0.15
        correction = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping ** 2 * np.eye(5), error
        )
        null_space = np.eye(4) - np.linalg.pinv(jacobian) @ jacobian
        correction += null_space @ ((posture - angles) * posture_bias)
        angles += correction
    for index, (low, high) in enumerate(joint_limits):
        angles[index] = np.clip(angles[index], low, high)
    return angles

def draw_slider(label, value, low, high, x, y, width=230):
    draw_text(f"{label}: {value:.2f}", x, y - 18, 16, (205, 215, 225))
    draw_rect(x, y, width, 7, color=(0.25, 0.29, 0.35))
    fraction = (value - low) / (high - low) if high > low else 0.0
    draw_rect(x, y, width * max(0.0, min(1.0, fraction)), 7, color=(1.0, 0.72, 0.22))

def draw_control_panel(target, angles, selected_axis, selected_joint, ik_enabled,
                       settings, pose_name, settings_panel):
    panel_x = 15
    panel_y = 70
    panel_width = 360
    panel_height = 930 if settings_panel else 265
    draw_rect(panel_x, panel_y, panel_width, panel_height, color=(0.04, 0.06, 0.10))
    draw_text("ROBOT ARM CONTROL", panel_x + 15, panel_y + 12, 26, (255, 220, 120))
    draw_text("MODE: END EFFECTOR POSITION", panel_x + 15, panel_y + 45, 20, (210, 230, 245))

    target_color = (255, 220, 120) if ik_enabled else (180, 190, 205)
    draw_text(
        f"TARGET {selected_axis.upper()}: {target[selected_axis]: .2f}  [X/Y/Z]",
        panel_x + 15, panel_y + 76, 20, target_color
    )
    draw_text("J/L + I/K: move target", panel_x + 15, panel_y + 102, 18, (180, 190, 205))
    draw_text("Shift + keys: fine adjustment", panel_x + 15, panel_y + 125, 18, (180, 190, 205))
    draw_text(f"JOINT: {joint_names[selected_joint]}  [1-4] select", panel_x + 15, panel_y + 158, 20, (210, 230, 245))
    draw_text(
        f"ANGLES: {angles[0]: .0f} {angles[1]: .0f} {angles[2]: .0f} {angles[3]: .0f}",
        panel_x + 15, panel_y + 184, 18, (180, 190, 205)
    )
    draw_text("[I] IK [R] reset [F2] settings [F5] save [F7] load", panel_x + 15, panel_y + 218, 17, (180, 190, 205))
    draw_text(f"CONTROL: {CONTROL_HZ:.0f} Hz   POSE: {pose_name}", panel_x + 15, panel_y + 242, 17, (180, 190, 205))
    if settings_panel:
        draw_text("SETTINGS (drag sliders)", panel_x + 15, panel_y + 275, 20, (255, 220, 120))
        draw_slider("P", settings["pid"]["p"][selected_joint], 0, 30, panel_x + 15, panel_y + 315)
        draw_slider("I", settings["pid"]["i"][selected_joint], 0, 5, panel_x + 15, panel_y + 355)
        draw_slider("D", settings["pid"]["d"][selected_joint], 0, 5, panel_x + 15, panel_y + 395)
        draw_slider("Max speed", settings["max_joint_speed"], 5, 360, panel_x + 15, panel_y + 435)
        draw_slider("Posture bias", settings["posture_bias"], 0, 1, panel_x + 15, panel_y + 475)
        draw_slider("Angle weight", settings["angle_weight"], 0, 1, panel_x + 15, panel_y + 515)
        draw_slider("Target X", target["x"], -20, 20, panel_x + 15, panel_y + 555)
        draw_slider("Target Y", target["y"], -1, 15, panel_x + 15, panel_y + 595)
        draw_slider("Target Z", target["z"], -20, 10, panel_x + 15, panel_y + 635)
        draw_slider("End roll", target_orientation["roll"], -180, 180, panel_x + 15, panel_y + 675)
        draw_slider("End pitch", target_orientation["pitch"], -180, 180, panel_x + 15, panel_y + 715)
        draw_slider("End yaw", target_orientation["yaw"], -180, 180, panel_x + 15, panel_y + 755)

def reset_pid_controllers():
    for controller in pid_controllers:
        controller.reset()

def save_current_settings():
    settings["posture_bias"] = posture_bias
    settings["angle_weight"] = angle_weight
    settings["minimum_target_y"] = minimum_target_y
    save_json(settings_path, settings)

def save_current_pose(name):
    saved = [pose for pose in poses if pose["name"] != name]
    saved.append({
        "name": name,
        "target": dict(target_position),
        "orientation": dict(target_orientation),
        "angles": list(target_angles),
    })
    save_json(poses_path, saved)
    return saved

def set_slider_value(mouse_x, x, width, low, high):
    fraction = max(0.0, min(1.0, (mouse_x - x) / width))
    return low + fraction * (high - low)

def update_slider_value(slider, mouse_x):
    if slider == "i_selected":
        settings["pid"]["i"][selected_joint] = set_slider_value(mouse_x, 30, 230, 0, 5)
    elif slider == "d_selected":
        settings["pid"]["d"][selected_joint] = set_slider_value(mouse_x, 30, 230, 0, 5)
    elif slider == "p_selected":
        settings["pid"]["p"][selected_joint] = set_slider_value(mouse_x, 30, 230, 0, 30)
    elif slider == "max_speed":
        settings["max_joint_speed"] = set_slider_value(mouse_x, 30, 230, 5, 360)
    elif slider == "posture_bias":
        globals()["posture_bias"] = set_slider_value(mouse_x, 30, 230, 0, 1)
        settings["posture_bias"] = posture_bias
    elif slider == "angle_weight":
        globals()["angle_weight"] = set_slider_value(mouse_x, 30, 230, 0, 1)
        settings["angle_weight"] = angle_weight
    elif slider == "target":
        target_position[selected_axis] = set_slider_value(mouse_x, 30, 230, -10, 10)
    elif slider == "target_x":
        target_position["x"] = set_slider_value(mouse_x, 30, 230, -20, 20)
    elif slider == "target_y":
        target_position["y"] = set_slider_value(mouse_x, 30, 230, -1, 15)
    elif slider == "target_z":
        target_position["z"] = set_slider_value(mouse_x, 30, 230, -20, 10)
    elif slider == "orientation_roll":
        target_orientation["roll"] = set_slider_value(mouse_x, 30, 230, -180, 180)
    elif slider == "orientation_pitch":
        target_orientation["pitch"] = set_slider_value(mouse_x, 30, 230, -180, 180)
    elif slider == "orientation_yaw":
        target_orientation["yaw"] = set_slider_value(mouse_x, 30, 230, -180, 180)
    if slider in ("p_selected", "i_selected", "d_selected"):
        index = selected_joint
        pid_controllers[index].kp = settings["pid"]["p"][index]
        pid_controllers[index].ki = settings["pid"]["i"][index]
        pid_controllers[index].kd = settings["pid"]["d"][index]

# var
display_x = 1500
display_y = 1200
display_name = "You've been 67'd"

fov = 70
aspect_ratio = display_x / display_y
cliping_plane_min = 0.1
cliping_plane_max = 200

cam_x = 0
cam_y = 0
cam_z = 10
cam_rx = 0
cam_ry = 0
cam_move_speed = 0.1
mouse_sensitivity = 0.2

earth_gravity = 9.8

x = 0
setup_pos = True
fps = 60

settings_path = Path(__file__).with_name("settings.json")
poses_path = Path(__file__).with_name("poses.json")
settings = load_settings(settings_path)
posture_bias = settings["posture_bias"]
angle_weight = settings["angle_weight"]
poses = load_poses(poses_path)
pose_name = "home"
settings_panel = False
slider_drag = None
camera_control_enabled = True

clock = pygame.time.Clock()
screen = start()

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

base_yaw = 0
shoulder_pitch = 0
elbow_pitch = 0
wrist_pitch = 0
joint_angles = [base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch]
target_angles = joint_angles.copy()
target_position = {"x": 0.0, "y": 1.0, "z": -9.0}
target_orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
selected_axis = "x"
selected_joint = 0
ik_enabled = True
control_scheduler = FixedRateScheduler(CONTROL_HZ)
pid_controllers = [
    PID(
        settings["pid"]["p"][index],
        settings["pid"]["i"][index],
        settings["pid"]["d"][index],
        settings["pid"]["integral_limit"],
        settings["pid"]["output_limit"],
    )
    for index in range(4)
]
last_frame_time = time.perf_counter()

# textures 
six_seven = load_image("67.png")
tung_tung_sahur = load_image("tung_tung.jpg")
fourty_one = load_image("41.jpg")
pumpkin_cat = load_image("oil.png")
oli = load_image("oil.png")
sky_box = load_image("Daylight_Box.png")
sky_front  = load_image_skybox("Daylight Box_Front.bmp")
sky_back   = load_image_skybox("Daylight Box_Back.bmp")
sky_top    = load_image_skybox("Daylight Box_Top.bmp")
sky_bottom = load_image_skybox("Daylight Box_Bottom.bmp")
sky_left   = load_image_skybox("Daylight Box_Left.bmp")
sky_right  = load_image_skybox("Daylight Box_Right.bmp")

# object
base      = create_prism(2, 2, 2, tex_id=six_seven)
segment_1 = create_prism(1, 1, 4, tex_id=six_seven)
segment_2 = create_prism(1, 1, 3, tex_id=six_seven)
segment_3 = create_prism(1, 1, 2, tex_id=fourty_one)
claw      = create_prism(1.2, 1.0, 0.8, tex_id=six_seven)
orientation_marker = create_prism(0.25, 2.0, 0.25, tex_id=fourty_one)
ground = create_prism(50, 50, 0.1, tex_id=pumpkin_cat)
oli_sphere = create_sphere(2, 16, 16, tex_id=oli)



skybox = create_skybox(90,
    front=sky_front, back=sky_back,
    top=sky_top,     bottom=sky_bottom,
    left=sky_left,   right=sky_right
)

skybox.material.textures["front"]  = (sky_front,   0,   "h")   
skybox.material.textures["top"]    = (sky_top,     180,  None)  
skybox.material.textures["left"]   = (sky_left,    270, None)   
skybox.material.textures["right"]  = (sky_right,   90,   "h")
skybox.material.textures["back"]  = (sky_back,   0,   None)
skybox.material.textures["bottom"]  = (sky_bottom,   0,   None)

#childs
base.add_child(segment_1)
segment_1.add_child(segment_2)
segment_2.add_child(segment_3)
segment_3.add_child(claw)

# main
while True:
    if camera_control_enabled:
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        cam_ry += mouse_dx * mouse_sensitivity
        cam_rx += mouse_dy * mouse_sensitivity
        cam_rx = max(-90, min(90, cam_rx))
    else:
        pygame.mouse.get_rel()

    key = pygame.key.get_pressed()
    if camera_control_enabled:
        if key[K_w]:
            cam_x += math.sin(math.radians(cam_ry)) * cam_move_speed
            cam_z -= math.cos(math.radians(cam_ry)) * cam_move_speed
        if key[K_s]:
            cam_x -= math.sin(math.radians(cam_ry)) * cam_move_speed
            cam_z += math.cos(math.radians(cam_ry)) * cam_move_speed
        if key[K_a]:
            cam_x -= math.cos(math.radians(cam_ry)) * cam_move_speed
            cam_z -= math.sin(math.radians(cam_ry)) * cam_move_speed
        if key[K_d]:
            cam_x += math.cos(math.radians(cam_ry)) * cam_move_speed
            cam_z += math.sin(math.radians(cam_ry)) * cam_move_speed
        if key[K_SPACE]:
            cam_y += cam_move_speed
        if key[K_LCTRL]:
            cam_y -= cam_move_speed

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == K_TAB:
                ik_enabled = True
            elif event.key == K_i:
                ik_enabled = True
            elif event.key == K_F2:
                settings_panel = not settings_panel
            elif event.key == K_F5:
                save_current_settings()
                pose_name = "quick-save"
                poses = save_current_pose(pose_name)
            elif event.key == K_F7:
                loaded = next((pose for pose in poses if pose["name"] == "quick-save"), None)
                if loaded is not None:
                    target_position = dict(loaded["target"])
                    target_orientation = dict(loaded.get("orientation", target_orientation))
                    target_angles = list(loaded["angles"])
                    joint_angles = list(target_angles)
                    reset_pid_controllers()
            elif event.key == K_r:
                joint_angles = [0.0, 0.0, 0.0, 0.0]
                target_angles = joint_angles.copy()
                target_position = {"x": 0.0, "y": 1.0, "z": -9.0}
                target_orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
                reset_pid_controllers()
            elif event.key in (K_x, K_y, K_z):
                selected_axis = pygame.key.name(event.key)
            elif event.key in (K_1, K_2, K_3, K_4):
                selected_joint = event.key - K_1
            elif event.key in (K_j, K_l, K_i, K_k):
                direction = 1 if event.key in (K_l, K_i) else -1
                step = 0.025 if event.mod & KMOD_SHIFT else 0.1
                target_position[selected_axis] += direction * step
            if event.key == K_ESCAPE:
                camera_control_enabled = False
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)
        if event.type == pygame.MOUSEBUTTONDOWN:
            clicked_settings_panel = settings_panel and 15 <= event.pos[0] <= 375 and 70 <= event.pos[1] <= 1000
            if clicked_settings_panel and event.button == 1:
                panel_y = 70
                if panel_y + 297 <= event.pos[1] <= panel_y + 325:
                    slider_drag = "p_selected"
                elif panel_y + 337 <= event.pos[1] <= panel_y + 365:
                    slider_drag = "i_selected"
                elif panel_y + 377 <= event.pos[1] <= panel_y + 405:
                    slider_drag = "d_selected"
                elif panel_y + 417 <= event.pos[1] <= panel_y + 445:
                    slider_drag = "max_speed"
                elif panel_y + 457 <= event.pos[1] <= panel_y + 485:
                    slider_drag = "posture_bias"
                elif panel_y + 497 <= event.pos[1] <= panel_y + 525:
                    slider_drag = "angle_weight"
                elif panel_y + 537 <= event.pos[1] <= panel_y + 565:
                    slider_drag = "target_x"
                elif panel_y + 577 <= event.pos[1] <= panel_y + 605:
                    slider_drag = "target_y"
                elif panel_y + 617 <= event.pos[1] <= panel_y + 645:
                    slider_drag = "target_z"
                elif panel_y + 657 <= event.pos[1] <= panel_y + 685:
                    slider_drag = "orientation_roll"
                elif panel_y + 697 <= event.pos[1] <= panel_y + 725:
                    slider_drag = "orientation_pitch"
                elif panel_y + 737 <= event.pos[1] <= panel_y + 765:
                    slider_drag = "orientation_yaw"
                if slider_drag:
                    update_slider_value(slider_drag, event.pos[0])
                camera_control_enabled = False
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)
            else:
                camera_control_enabled = True
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            slider_drag = None
        if event.type == pygame.MOUSEMOTION and slider_drag:
            update_slider_value(slider_drag, event.pos[0])

    elapsed = time.perf_counter() - last_frame_time
    last_frame_time += elapsed
    for _ in range(control_scheduler.advance(elapsed)):
        if ik_enabled:
            target_angles = solve_arm_ik(
                target_position, target_orientation, target_angles, angle_weight
            ).tolist()
        for index, controller in enumerate(pid_controllers):
            correction = controller.update(
                target_angles[index], joint_angles[index], CONTROL_DT
            )
            correction = max(
                -settings["max_joint_speed"],
                min(settings["max_joint_speed"], correction),
            )
            joint_angles[index] = approach(
                joint_angles[index],
                joint_angles[index] + correction * CONTROL_DT,
                settings["max_joint_speed"] * CONTROL_DT,
            )
            
#run once when setup stuff
    if setup_pos:
        base.translate(0, 0, 0)
        segment_1.rotate(0, 0, 0)
        segment_2.rotate(0, 0, 0)
        segment_3.rotate(0, 0, 0)
        
        segment_1.translate(0, 2, 0)
        segment_2.translate(0, 0, -3.5)
        segment_3.translate(0, 0, -2.5)
        claw.translate(0, 0, -1)
        
        ground.rotate(90, 0, 0)
        ground.translate(0, -1, 0)
        oli_sphere.translate(0, 4, -3)
        oli_sphere.rotate(0, 90, 0)
        
        setup_pos = False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glRotatef(cam_rx, 1, 0, 0)
    glRotatef(cam_ry, 0, 1, 0)
    glTranslatef(-cam_x, -cam_y, -cam_z)

    skybox.translate(cam_x, cam_y, cam_z)

    skybox.render_skybox()

    base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch = joint_angles
    base.rotate_around_point(0, 0, 0, 0, base_yaw, 0)
    segment_1.rotate_around_point(0, 0, 2, shoulder_pitch, 0, 0)
    segment_1.translate(0, 1, -2)
    segment_2.rotate_around_point(0, 0, 1.5, elbow_pitch, 0, 0)
    segment_3.rotate_around_point(0, 0, 1, wrist_pitch, 0, 0)
    claw.rotate(*claw_local_orientation(joint_angles, target_orientation))
    orientation_marker.translate(
        target_position["x"], target_position["y"], target_position["z"]
    )
    orientation_marker.rotate(
        target_orientation["pitch"],
        target_orientation["yaw"],
        target_orientation["roll"],
    )
    
    
    planes = get_frustum_planes()

    base.render(planes)
    ground.render()
    oli_sphere.render()
    orientation_marker.render()

    draw_2d_start()
    draw_rect(10, 10, 200, 40, color=(0.0, 0.0, 0.0))
    draw_text(f"cam pos: {cam_x:.1f} {cam_y:.1f} {cam_z:.1f}", 15, 20, 24, (255, 255, 255))
    draw_control_panel(
        target_position, joint_angles, selected_axis, selected_joint, ik_enabled,
        settings, pose_name, settings_panel
    )
    draw_2d_end()

    pygame.display.flip()
    clock.tick(fps)
