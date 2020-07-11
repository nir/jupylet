"""
    jupylet/model.py
    
    Copyright (c) 2020, Nir Aides - nir@winpdb.org

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import gltflib
import logging
import weakref
import pyglet
import struct
import math
import glm
import os

import numpy as np
import PIL.Image

from pyglet.graphics import *

from .utils import abspath
from .shader import Shader, ShaderProgram


_shader_program = None


def set_shader_program(sp):

    global _shader_program

    _shader_program = sp


def get_shader_program():

    if _shader_program is None:

        vp = abspath('./shaders/default-vertex-shader.glsl')
        fp = abspath('./shaders/default-fragment-shader.glsl')

        vs = Shader(open(vp).read(), 'vertex')
        fs = Shader(open(fp).read(), 'fragment')

        set_shader_program(ShaderProgram(vs, fs))

    return _shader_program


def flatten(m):
    return sum(m.to_list(), [])

def strip(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


def union(d0, d1):
    
    if d1:
        d0 = dict(d0 or {})
        d0.update(d1)
        
    return d0 or {}

class Object(object):
    
    def __repr__(self):    
        return '%s(%s)' % (type(self).__name__, ', '.join('%s=%s' % i for i in self.__dict__.items() if i[0][0] != '_'))

class Group(pyglet.graphics.Group):

    def __init__(self, drawable, parent=None):
        
        super(Group, self).__init__(parent)
        
        self._drawable = weakref.proxy(drawable)
  
    def rget(self, key):
        
        value = self._drawable.__dict__.get(key)
        if value is not None:
            return value
        
        return self.parent and self.parent.rget(key)
    
    @property
    def shader(self):
        return self.rget('_shader')
    
    def set_state(self, face=GL_FRONT_AND_BACK):
        return self._drawable.set_state()
        
    def unset_state(self):
        return self._drawable.unset_state()


def load_blender_gltf(path):
    
    g0 = gltflib.GLTF.load(path)
    s0 = g0.model.scenes[g0.model.scene]
    
    scene = Scene(s0.name)
    scene._source = g0
    
    for m0 in g0.model.materials:
        material = load_blender_gltf_material(g0, m0)
        scene.add_material(material)
    
    for n0 in s0.nodes:
        n0 = g0.model.nodes[n0]

        if is_blender_gltf_light(g0, n0):
            light = load_blender_gltf_light(g0, n0)
            scene.add_light(light)
            
        elif is_blender_gltf_camera(g0, n0):
            camera = load_blender_gltf_camera(g0, n0)
            scene.add_camera(camera)
        
        elif is_blender_gltf_mesh(g0, n0):
            mesh = load_blender_gltf_mesh(g0, n0, list(scene.materials.values()))
            scene.add_mesh(mesh)
        
    return scene


class Scene(Object):
    
    def __init__(self, name):
        
        self.name = name
        
        self.meshes = {}
        self.lights = {}
        self.cameras = {}
        self.materials = {}
        
        self.cubemap = None
        self.hdri = None
                
        self._batch = pyglet.graphics.Batch()
        self._group = Group(self)
        self._shader = None
        
    def add_material(self, material):
        self.materials[material.name] = material
        
    def add_mesh(self, mesh):
        self.meshes[mesh.name] = mesh
        mesh._group.parent = self._group
        mesh.batch_add(self._batch)
        
    def add_light(self, light):
        light.set_index(len(self.lights))
        self.lights[light.name] = light
        
    def add_camera(self, camera):
        self.cameras[camera.name] = camera
        
    def batch_add(self):
        
        for mesh in self.meshes.values():
            mesh.batch_add(self._batch)
            
    def draw(self):
        
        if not self._shader:
            self._shader = get_shader_program()
            
        self._batch.draw()
        
    def set_state(self):
        
        self.active_texture_orig = ctypes.c_int()
        glGetIntegerv(GL_ACTIVE_TEXTURE, self.active_texture_orig)
        
        self._shader.use()
        self._shader['nlights'] = len(self.lights)
        
        for light in self.lights.values():
            light.set_state(self._shader)
        
        for camera in self.cameras.values():
            camera.set_state(self._shader)
        
    def unset_state(self):
        self._shader.stop()


def load_blender_gltf_material(g0, m0):
    
    lbt = load_blender_gltf_texture
    pbr = m0.pbrMetallicRoughness
    
    c = lbt(g0, pbr.baseColorTexture) if pbr.baseColorTexture else pbr.baseColorFactor
    m = pbr.metallicFactor
    r = lbt(g0, pbr.metallicRoughnessTexture) if pbr.metallicRoughnessTexture else pbr.roughnessFactor
    s = 0.1
    e = lbt(g0, m0.emissiveTexture) if m0.emissiveTexture else m0.emissiveFactor
    o = lbt(g0, m0.occlusionTexture)
    n = lbt(g0, m0.normalTexture)

    material = Material(m0.name, c, m, r, s, e, o, n)
    material._source = m0
    
    return material


def load_blender_gltf_texture(g0, ti):
    
    if ti is None:
        return None
    
    t0 = g0.model.textures[getattr(ti, 'index', ti)]
    i0 = g0.model.images[t0.source]
    r0 = [r for r in g0.resources if r._uri == i0.uri][0]
        
    dirname = os.path.abspath(r0._basepath)
    
    if dirname not in pyglet.resource.path:
        pyglet.resource.path.append(dirname)
        pyglet.resource.reindex()

    texture = pyglet.resource.texture(r0.filename)

    glBindTexture(texture.target, texture.id)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    return texture


def t2i(texture):
    
    i0 = texture.get_image_data()
    return PIL.Image.frombuffer(i0.format, (i0.width, i0.height), i0.get_data())

    
class Material(Object):
    
    def __init__(
        self,
        name,
        color,
        metallic=0,
        roughness=0.5,
        specular=0.1,
        emissive=[0, 0, 0],
        occlusion=None,
        normals=None
    ):
        
        self.name = name
        
        self.color = color
        self.metallic = metallic
        self.roughness = roughness     
        self.specular = specular     
        self.emissive = emissive
        self.occlusion = occlusion
        self.normals = normals
        self.normals_gamma = self.compute_normals_gamma(normals)

    def compute_normals_gamma(self, normals):

        if not isinstance(normals, pyglet.image.Texture):
            return 1.

        na = np.array(t2i(normals))
        nm = na[...,:2].mean() / 255

        return math.log(0.5) / math.log(nm)        

    def set_state(self, shader):
            
        self.active_texture_orig = ctypes.c_int()
        glGetIntegerv(GL_ACTIVE_TEXTURE, self.active_texture_orig)

            
        if isinstance(self.color, pyglet.image.Texture):
            
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(self.color.target, self.color.id) 
            
            shader['material.color_texture'] = 1
            shader['material.color_texture_exists'] = True
        else:
            shader['material.color_texture_exists'] = False
            shader['material.color'] = self.color
                   
        if isinstance(self.normals, pyglet.image.Texture):
            
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(self.normals.target, self.normals.id) 
            
            shader['material.normals_gamma'] = self.normals_gamma
            shader['material.normals_texture'] = 2
            shader['material.normals_texture_exists'] = True
        else:
            shader['material.normals_texture_exists'] = False
                   
        if isinstance(self.roughness, pyglet.image.Texture):
            
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(self.roughness.target, self.roughness.id) 
            
            shader['material.roughness_texture'] = 3
            shader['material.roughness_texture_exists'] = True
        else:
            shader['material.roughness_texture_exists'] = False
            shader['material.roughness'] = self.roughness
            shader['material.metallic'] = self.metallic
                   
        if isinstance(self.emissive, pyglet.image.Texture):
            
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(self.emissive.target, self.emissive.id) 
            
            shader['material.emissive_texture'] = 4
            shader['material.emissive_texture_exists'] = True
        else:
            shader['material.emissive_texture_exists'] = False
            shader['material.emissive'] = self.emissive                   

        shader['material.specular'] = self.specular
        
    def unset_state(self):

        glActiveTexture(self.active_texture_orig.value)
        

def compute_matrix(angle=0, axis=(0., 1., 0.), scale=1., xyz=(0., 0., 0.), matrix=glm.mat4(1.)):
    
    if xyz is not None and xyz != (0, 0, 0):
        matrix = glm.translate(matrix, xyz)
        
    if scale is not None and scale != 1.:
        matrix = glm.scale(matrix, glm.vec3(scale))
        
    if angle:
        matrix = glm.rotate(matrix, angle, axis)

    return matrix


def decompose(
    matrix,
    scale=None,
    rotation=None,
    translation=None,
    skew=None,
    perspective=None,
    ):
    
    status = glm.decompose(
        matrix, 
        scale or glm.vec3(), 
        rotation or glm.quat(), 
        translation or glm.vec3(), 
        skew or glm.vec3(), 
        perspective or glm.vec4()
    )
    
    return scale or rotation or translation or skew or perspective


def q2aa(rotation, deg=False):
    """Transform quaternion to angle+axis."""
    
    if not rotation or rotation == (1., 0., 0., 0.):
        return 0, glm.vec3(0, 0, 1)
    
    c, xs, ys, zs = rotation #glm.conjugate(rotation)

    angle = math.acos(c) * 2
    s = math.sin(angle / 2)

    if deg:
        angle = round(180 * angle / math.pi, 3)
    
    return angle, glm.vec3(xs / s, ys / s, zs / s)


def aa2q(angle, axis):
    
    a = math.cos(angle / 2)
    i, j, k = math.sin(angle / 2) * glm.normalize(axis)
    
    return [round(v, 6) for v in [a, i, j, k]]


def xyzw2wxyz(q):
    
    if q:
        x, y, z, w = q
        return w, x, y, z

class Node(Object):
  
    def __init__(self, name, rotation=None, scale=None, translation=None):
        
        self.name = name
        
        angle, axis = q2aa(rotation)
        
        self._matrix = compute_matrix(angle, axis, scale, translation)
        
    @property
    def scale(self):
        return decompose(self._matrix, scale=glm.vec3())
        
    @scale.setter
    def scale(self, scale):        
        self._matrix = glm.scale(self._matrix, glm.vec3(scale / self.scale))
        
    @property
    def position(self):
        return self._matrix[3].xyz
    
    @position.setter
    def position(self, xyz):
        self._matrix[3].xyz = xyz
        
    def move_global(self, xyz):
        self.position += xyz

    def move_local(self, xyz):
        self._matrix = glm.translate(self._matrix, xyz)
        
    def set_rotation_global(self, angle, axis=(0., 1., 0.)):
        self._matrix = compute_matrix(self.scale, angle, axis, self.position)
        
    def rotate_local(self, angle, axis=(0., 1., 0.)):
        self._matrix = glm.rotate(self._matrix, angle, axis)
            
    def rotate_global(self, angle, axis=(0., 1., 0.)):
        
        pos = self.position
        self.position = 0, 0, 0
        self._matrix = glm.rotate(glm.mat4(), angle, axis) * self._matrix
        self.position = pos
        
    def get_rotation(self, deg=False):
        """Extract rotation angle and axis from rotation matrix. 
        
        This method does not separate global and local rotations. 
        
        For that it is neccessary to keep track of separate model transformations.
        
        References:
        https://stackoverflow.com/questions/17918033/glm-decompose-mat4-into-translation-and-rotation
        and https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        rotation = decompose(self._matrix, rotation=glm.quat())

        return q2aa(rotation, deg=deg)
    
    @property
    def up(self):
        """Return the local up (+y) axis."""
        return (self._matrix * glm.vec4(0, 1, 0, 0)).xyz

    @property
    def front(self):
        """Return the local front (+z) axis."""
        return (self._matrix * glm.vec4(0, 0, 1, 0)).xyz


def is_blender_gltf_light(g0, n0):
    
    if not n0.children:
        return False
    
    nc = g0.model.nodes[n0.children[0]]
    if not nc.extensions:
        return False
    
    return nc.extensions.get('KHR_lights_punctual') is not None

    
def load_blender_gltf_light(g0, n0):
    
    nc = g0.model.nodes[n0.children[0]]
    
    l0 = nc.extensions.get('KHR_lights_punctual')['light']
    l1 = g0.model.extensions['KHR_lights_punctual']['lights'][l0]
    
    m0 = glm.mat4_cast(glm.quat(xyzw2wxyz(nc.rotation)))
    m1 = glm.mat4_cast(glm.quat(xyzw2wxyz(n0.rotation)))  
    qq = glm.quat_cast(m1 * m0)
    
    light = Light(
        n0.name, 
        qq, 
        n0.scale, 
        n0.translation,
        l1['type'],
        l1['color'],
        l1['intensity'] / 4.,
        l1.get('spot')
    )
    light._source = n0
    
    return light


LIGHT_TYPE = {
    'directional': 0,
    'point': 1,
    'spot': 2,
}


class Light(Node):
    
    def __init__(
        self,       
        name,
        rotation=None, 
        scale=None, 
        translation=None,
        type='point',
        color=glm.vec3(1.),
        intensity=500,
        spot=None,
        **kwargs
    ):
        
        super(Light, self).__init__(name, rotation, scale, translation)
        
        self.index = -1
        
        self.type = type
        
        self.spot = spot
        self.color = [round(c, 3) for c in color]
        self.intensity = round(intensity, 3)

    def set_index(self, index):  
        self.index = index
        
    def get_uniform_name(self, key):
        return 'lights[%s].%s' % (self.index, key)
    
    def set_state(self, shader):
        
        prefix = self.get_uniform_name('')
                        
        shader[prefix + 'type'] = LIGHT_TYPE[self.type]
        shader[prefix + 'color'] = self.color
        shader[prefix + 'intensity'] = self.intensity
        
        shader[prefix + 'position'] = self.position
        shader[prefix + 'direction'] = self.front
        
        

def is_blender_gltf_camera(g0, n0):
    
    if not n0.children:
        return False
    
    nc = g0.model.nodes[n0.children[0]]
    
    return nc.camera is not None

    
def load_blender_gltf_camera(g0, n0):
    
    nc = g0.model.nodes[n0.children[0]]
    c0 = g0.model.cameras[nc.camera]

    m0 = glm.mat4_cast(glm.quat(xyzw2wxyz(nc.rotation)))
    m1 = glm.mat4_cast(glm.quat(xyzw2wxyz(n0.rotation)))  
    qq = glm.quat_cast(m1 * m0)
    
    camera = Camera(
        n0.name, 
        qq, 
        n0.scale, 
        n0.translation,
        c0.type,
        **vars(c0.perspective or c0.orthographic)
    )
    camera._source = n0
    
    return camera


class Camera(Node):
    
    def __init__(
        self, 
        name, 
        rotation=None, 
        scale=None, 
        translation=None,
        type='perspective',
        znear=100,
        zfar=100,
        yfov=glm.radians(60),
        xmag=1.,
        ymag=1.,
        **kwargs
    ):
        
        super(Camera, self).__init__(name, rotation, scale, translation)
        
        self.type=type
        
        self.znear=round(znear, 3)
        self.zfar=round(zfar, 3)
        self.yfov=round(yfov, 3)
        self.xmag=round(xmag, 3)
        self.ymag=round(ymag, 3)
        
    def set_state(self, shader):

        x0, y0, w0, h0 = pyglet.image.get_buffer_manager().get_viewport()
        aspect = w0 / h0
                    
        shader['view'] = flatten(glm.lookAt(self.position, self.position - self.front, self.up))
        shader['projection'] = flatten(glm.perspective(
            self.yfov, 
            aspect, 
            self.znear, 
            self.zfar
        ))
        shader['camera.position'] = self.position




def is_blender_gltf_mesh(g0, n0):
    
    if n0.mesh is not None:
        return True
    
    for nc in n0.children or []:
        if is_blender_gltf_mesh(g0, g0.model.nodes[nc]):
            return True
        
    return False

    
def load_blender_gltf_mesh(g0, n0, materials, parent=None):
    
    mesh = Mesh(
        n0.name, 
        xyzw2wxyz(n0.rotation), 
        n0.scale, 
        n0.translation,
        parent
    )
    mesh._source = n0
    
    m0 = g0.model.meshes[n0.mesh]
    
    for p0 in m0.primitives:
        mesh.add_primitive(load_blender_gltf_primitive(g0, p0, materials))
    
    for cn in n0.children or []:
        n1 = g0.model.nodes[cn]
        mesh.children[n1.name] = load_blender_gltf_mesh(g0, n1, materials, mesh)
        
    return mesh

    
class Mesh(Node):
    
    def __init__(self, name, rotation=None, scale=None, translation=None, parent=None):
        
        super(Mesh, self).__init__(name, rotation, scale, translation)
        
        self.primitives = []
        self.children = {}
        
        self._parent = weakref.proxy(parent) if parent else None
        self._group = Group(self, parent._group if parent else None)

    def add_primitive(self, primitive):
        self.primitives.append(primitive)
        primitive._group.parent = self._group
        
    def batch_add(self, batch):
        
        for primitive in self.primitives:
            primitive.batch_add(batch)
            
        for mesh in self.children.values():
            mesh.batch_add(batch)
            
    def composed_matrix(self):
        
        if self._parent is None:
            return self._matrix

        return self._matrix * self._parent.composed_matrix()

    def set_state(self):
        self._group.shader['model'] = flatten(self.composed_matrix())
        
    def unset_state(self):
        pass

def load_blender_gltf_primitive(g0, p0, materials):
    
    primitive = Primitive(
        material=materials[p0.material], 
        indices=get_buffer0(g0, p0.indices, '')[-1],
        vertices=get_buffer0(g0, p0.attributes.POSITION),
        tangents=get_buffer0(g0, p0.attributes.TANGENT),
        normals=get_buffer0(g0, p0.attributes.NORMAL),
        coords=get_buffer0(g0, p0.attributes.TEXCOORD_0)
    )
    primitive._source = p0
    
    return primitive


def get_buffer0(g0, ai, prefix=''):
    
    if ai is not None:
        a0 = g0.model.accessors[ai]
        fmt, data = get_buffer(g0, a0)  
        return prefix + fmt, data
    
    
def get_buffer(g0, a0):
    
    t2f = {
        'FLOAT': 'f',
        'UNSIGNED_SHORT': 'HS',
        'UNSIGNED_INT': 'I',
    }
    
    t2n = {
        'SCALAR': '1',
        'VEC2': '2',
        'VEC3': '3',
        'VEC4': '4',
    }
    
    ctn = gltflib.ComponentType._value2member_map_[a0.componentType].name
    fmt = t2n[a0.type] + t2f[ctn][-1]
    
    v0 = g0.model.bufferViews[a0.bufferView]
    b0 = g0.model.buffers[v0.buffer]
    r0 = [r for r in g0.resources if r._uri == b0.uri][0]
    
    if not r0.loaded:
        r0.load()
        
    ao = a0.byteOffset or 0
    d0 = r0.data[ao + v0.byteOffset: ao + v0.byteOffset + v0.byteLength]

    data = [s[0] for s in struct.Struct(t2f[ctn][0]).iter_unpack(d0)]   
    
    return fmt, data

    
class Primitive(Object):
    
    def __init__(self, material, indices, vertices, tangents=None, normals=None, coords=None):
        
        self.material = material

        self._indices = indices
        self._vertices = vertices
        self._tangents = tangents
        self._normals = normals
        self._coords = coords
        
        self.nvertices = len(self._vertices[1]) // 3
        self.has_uvs = self._coords is not None
       
        self._group = Group(self)
        
    def batch_add(self, batch):

        vl = [self._vertices, self._normals, self._coords, self._tangents]
        vl = [('%sg%s' % (i, fv[0]), fv[1]) for i, fv in enumerate(vl) if fv]
        
        batch.add_indexed(self.nvertices, GL_TRIANGLES, self._group, self._indices, *vl)
        
    def set_state(self):

        shader = self._group.shader
        shader['has_tangents'] = int(self._tangents is not None)

        self.material.set_state(shader)
        
    def unset_state(self):
        self.material.unset_state()

