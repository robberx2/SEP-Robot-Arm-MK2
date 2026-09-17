import time
import math
import pygame
import numpy as np
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

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
#meshes
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

#3d object creation functions
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

#general fucntions
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

clock = pygame.time.Clock()
screen = start()

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

base_yaw = 0
shoulder_pitch = 0
elbow_pitch = 0
wrist_pitch = 0

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
claw      = create_sphere(0.5, 16, 16, tex_id=six_seven)
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
    mouse_dx, mouse_dy = pygame.mouse.get_rel()
    cam_ry += mouse_dx * mouse_sensitivity
    cam_rx += mouse_dy * mouse_sensitivity
    cam_rx = max(-90, min(90, cam_rx))

    key = pygame.key.get_pressed()
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
            if event.key == K_ESCAPE:
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)
        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)
            
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
        oli_sphere.translate(0, 4, 5)
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

    base.rotate_around_point(0, 0, 0, 0, base_yaw, 0)
    segment_1.rotate_around_point(0, 0, 2, shoulder_pitch, 0, 0)
    segment_1.translate(0, 1, -2)
    segment_2.rotate_around_point(0, 0, 1.5, elbow_pitch, 0, 0)
    segment_3.rotate_around_point(0, 0, 1, wrist_pitch, 0, 0)
    
    
    planes = get_frustum_planes()

    base.render(planes)
    ground.render()
    oli_sphere.render()

    draw_2d_start()
    draw_rect(10, 10, 200, 40, color=(0.0, 0.0, 0.0))
    draw_text(f"cam pos: {cam_x:.1f} {cam_y:.1f} {cam_z:.1f}", 15, 20, 24, (255, 255, 255))
    draw_2d_end()

    pygame.display.flip()
    clock.tick(fps)
