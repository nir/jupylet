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
import glob
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


class LRU(object):
    
    def __init__(self, max_items):
        
        self.step = max_items
        self.items = {i: [i, i, i, 0] for i in range(max_items)}
        
    def allocate(self, lid=None):
        
        self.step += 1
        
        if lid is None:
            lid = self.step
            
        r = self.items.get(lid)
        
        if r is None:
            
            lid0, slot = min(self.items.values())[1:3]
            self.items.pop(lid0)
            self.items[lid] = [self.step, lid, slot, 0]
            
            return self.step, lid, slot, 1
            
        r[0] = self.step

        return r


_MIN_TEXTURES = 4
_MAX_TEXTURES = 32
_lru_textures = LRU(_MAX_TEXTURES - _MIN_TEXTURES)

_MAX_MATERIALS = 32
_lru_materials = LRU(_MAX_MATERIALS)


class Object(object):
    
    def __init__(self, dirty=True):
        
        self._items = {}
        self._dirty = set([True]) if dirty else set()

    def __dir__(self):
        return list(self._items.keys()) + super().__dir__()
    
    def __getattr__(self, k):
        
        if k not in self._items:
            raise AttributeError(k)
            
        return self._items[k]
    
    def __setattr__(self, k, v):
        
        if k == '_items' or k not in self._items:
            return super(Object, self).__setattr__(k, v)
        
        self._items[k] = v
        self._dirty.add(k)

    def __repr__(self):    
        return '%s(%s)' % (type(self).__name__, ', '.join(
            '%s=%s' % i for i in list(self.__dict__.items()) + list(self._items.items()) if i[0][0] != '_'
        ))


class Group(pyglet.graphics.Group):

    def __init__(self, drawable, parent=None):
        
        super(Group, self).__init__(parent)
        
        self._drawable = weakref.proxy(drawable)
  
    def rbool(self, key):
        
        value = self._drawable.__dict__.get(key)
        if bool(value):
            return value
        
        return self.parent and self.parent.rbool(key)
    
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
    
    def __init__(self, name, shadows=False):
        
        super(Scene, self).__init__()

        self.name = name
        
        self.meshes = {}
        self.lights = {}
        self.cameras = {}
        self.materials = {}
        
        self.shadows = shadows
        self.shadowmap_renderer = ShadowMapRenderer()

        self.cubemap = None
        self.hdri = None
                
        self._batch = pyglet.graphics.Batch()
        self._group = Group(self)
        self._shader = None

        self._width = None
        self._height = None

    def add_cubemap(self, path, intensity=1.0):

        self.cubemap = CubeMap(path, intensity)
        self.cubemap._group.parent = self._group
        self.cubemap.cube.batch_add(self._batch)

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
            
    def get_viewport(self):
        """Get viewport dimensions. 

        This call takes about ~5us, therefore the code should minimize calling it.
        """
        viewport = (GLint * 4)()
        glGetIntegerv(GL_VIEWPORT, viewport)
        return viewport
        
    def draw(self, width=None, height=None):
        
        if width is None or height is None:
            width, height = self.get_viewport()[2:]

        self._width = width
        self._height = height

        self._active_texture_orig = ctypes.c_int()
        glGetIntegerv(GL_ACTIVE_TEXTURE, self._active_texture_orig)
        
        if not self._shader:
            self._shader = get_shader_program()
            
        with self._shader:

            self._shader['nlights'] = len(self.lights)
            
            if self.shadows:
                self.shadowmap_renderer.draw(self, self._shader, self._width, self._height)

            # TODO: optimize with dirty mechanism.
            self._shader['shadowmap_pass'] = 2 if self.shadows else 0

            for light in self.lights.values():
                light.set_state(self._shader, self.shadows)
            
            for camera in self.cameras.values():
                camera.set_state(self._shader, self._width, self._height)
            
            self.batch_draw()
        
        glActiveTexture(self._active_texture_orig.value)

    def batch_draw(self):
        
        if self._batch._draw_list_dirty:
            self._batch._update_draw_list()

        r0 = None
        
        for func in self._batch._draw_list:
            
            if r0 is None:
                r0 = func()

            elif getattr(func, '__self__', None) is r0:
                r0 = None
                func()
                
    def set_state(self):
        pass
        
    def unset_state(self):
        pass


class ShadowMapRenderer(object):

    def __init__(self, width=1024, height=1024):
        
        self.width = width
        self.height = height
        
        self.fbo = GLuint()
        glGenFramebuffers(1, ctypes.byref(self.fbo))        
              
    def draw(self, scene, shader, width, height):

        if not any(light.shadows for light in scene.lights.values()):
            return

        if scene.cubemap:
            scene.cubemap.hide = True
            
        glViewport(0, 0, self.width, self.height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glDisable(GL_CULL_FACE)
        #glCullFace(GL_FRONT)

        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        shader._extra['shadowmap_pass'] = 1
        shader['shadowmap_pass'] = 1

        camera = list(scene.cameras.values())[0]
        camera_position = glm.vec4(camera.position, 1.0)

        zfar = camera.zfar
        yfov = math.cos(camera.yfov) * zfar
        xfov = yfov * width / height

        camera_screen = camera._matrix * glm.mat4(
            xfov, yfov, -zfar, 1., 
            -xfov, yfov, -zfar, 1., 
            -xfov, -yfov, -zfar, 1., 
            xfov, -yfov, -zfar, 1.
        )

        for i, light in enumerate(scene.lights.values()):

            if light.shadows:
    
                shader['shadowmap_light'] = i

                for j in range(light.shadowmaps_count):

                    shadowmap = light.set_state_shadowmap(
                        j, shader, camera_position, camera_screen
                    )

                    glFramebufferTexture2D(
                        GL_FRAMEBUFFER, 
                        GL_DEPTH_ATTACHMENT, 
                        GL_TEXTURE_2D, 
                        shadowmap, 
                        0
                    )

                    glClear(GL_DEPTH_BUFFER_BIT)
                    scene.batch_draw()
        
        shader._extra['shadowmap_pass'] = 0

        glEnable(GL_CULL_FACE)
        #glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, width, height)
    
        if scene.cubemap:
            scene.cubemap.hide = False
            

def fix_pyglet_image_decoders():
    
    dext = pyglet.image.codecs._decoder_extensions
    dext['.jpg'][:] = sorted(dext['.jpg'], key=lambda d: d.__class__.__name__ != 'PILImageDecoder')


def load_blender_gltf_material(g0, m0):
    
    fix_pyglet_image_decoders()

    lbt = load_blender_gltf_texture
    pbr = m0.pbrMetallicRoughness
    
    c = lbt(g0, pbr.baseColorTexture) if pbr.baseColorTexture else pbr.baseColorFactor
    m = pbr.metallicFactor
    r = lbt(g0, pbr.metallicRoughnessTexture) if pbr.metallicRoughnessTexture else pbr.roughnessFactor
    s = 0.1
    e = lbt(g0, m0.emissiveTexture) if m0.emissiveTexture else m0.emissiveFactor
    o = lbt(g0, m0.occlusionTexture)
    n = lbt(g0, m0.normalTexture)
    
    ns = getattr(m0.normalTexture, 'scale', 1.)

    material = Material(m0.name, c, m, r, s, e, o, n, ns)
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
        normals=None,
        normals_scale=1,
    ):
        
        super(Material, self).__init__()
        
        self.name = name

        self._items = dict(
            color = color,
            metallic = metallic,
            roughness = roughness,     
            specular = specular,     
            emissive = emissive,
            occlusion = occlusion,
            normals = normals,
            normals_scale = normals_scale or 1,
            normals_gamma = self.compute_normals_gamma(normals),
        )

        self._mlid, self._mslot = _lru_materials.allocate()[1:3]
        self._clid, self._cslot = self.allocate_texture(self._items['color'])[1:3]
        self._nlid, self._nslot = self.allocate_texture(self._items['normals'])[1:3]
        self._elid, self._eslot = self.allocate_texture(self._items['emissive'])[1:3]
        self._rlid, self._rslot = self.allocate_texture(self._items['roughness'])[1:3]
        
    def allocate_texture(self, t, lid=None):
        if isinstance(t, pyglet.image.Texture):
            return _lru_textures.allocate(lid)
        return None, None, None, None

    def compute_normals_gamma(self, normals):

        if not isinstance(normals, pyglet.image.Texture):
            return 1.

        na = np.array(t2i(normals))
        nm = na[...,:2].mean() / 255

        return math.log(0.5) / math.log(nm)        

    def set_state(self, shader):
            
        shader['material'] = self._mslot

        _, _, self._mslot, mnew = _lru_materials.allocate(self._mlid)
        _, _, self._cslot, cnew = self.allocate_texture(self._items['color'], self._clid)
        _, _, self._nslot, nnew = self.allocate_texture(self._items['normals'], self._nlid)
        _, _, self._eslot, enew = self.allocate_texture(self._items['emissive'], self._elid)
        _, _, self._rslot, rnew = self.allocate_texture(self._items['roughness'], self._rlid)

        dirty = self._dirty or mnew or cnew or nnew or enew or rnew

        if dirty:

            self._dirty.clear()

            material = 'materials[%s].' % self._mslot

            if isinstance(self.color, pyglet.image.Texture):
                
                glActiveTexture(GL_TEXTURE0 + self._cslot + _MIN_TEXTURES)
                glBindTexture(self.color.target, self.color.id) 
                
                shader['textures[%s].t' % self._cslot] = self._cslot + _MIN_TEXTURES
                shader[material + 'color_texture'] = self._cslot
            else:
                shader[material + 'color_texture'] = -1
                shader[material + 'color'] = self.color
                    
            if isinstance(self.normals, pyglet.image.Texture):
                
                shader[material + 'normals_scale'] = self.normals_scale
                shader[material + 'normals_gamma'] = self.normals_gamma

                glActiveTexture(GL_TEXTURE0 + self._nslot + _MIN_TEXTURES)
                glBindTexture(self.normals.target, self.normals.id) 
                
                shader['textures[%s].t' % self._nslot] = self._nslot + _MIN_TEXTURES
                shader[material + 'normals_texture'] = self._nslot
            else:
                shader[material + 'normals_texture'] = -1
                    
            if isinstance(self.emissive, pyglet.image.Texture):
                
                glActiveTexture(GL_TEXTURE0 + self._eslot + _MIN_TEXTURES)
                glBindTexture(self.emissive.target, self.emissive.id) 

                shader['textures[%s].t' % self._eslot] = self._eslot + _MIN_TEXTURES
                shader[material + 'emissive_texture'] = self._eslot
            else:
                shader[material + 'emissive_texture'] = -1
                shader[material + 'emissive'] = self.emissive                   

            if isinstance(self.roughness, pyglet.image.Texture):
                
                glActiveTexture(GL_TEXTURE0 + self._rslot + _MIN_TEXTURES)
                glBindTexture(self.roughness.target, self.roughness.id) 
                
                shader['textures[%s].t' % self._rslot] = self._rslot + _MIN_TEXTURES
                shader[material + 'roughness_texture'] = self._rslot
            else:
                shader[material + 'roughness_texture'] = -1
                shader[material + 'roughness'] = self.roughness
                shader[material + 'metallic'] = self.metallic
                    
            shader[material + 'specular'] = self.specular
                    
    def unset_state(self):
        pass
        

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
        
        super(Node, self).__init__()
        
        self.name = name
        
        angle, axis = q2aa(rotation)
        
        self._matrix = compute_matrix(angle, axis, scale, translation)
        
    @property
    def scale(self):
        return decompose(self._matrix, scale=glm.vec3())
        
    @scale.setter
    def scale(self, scale):
        self._dirty.add('_matrix')
        self._matrix = glm.scale(self._matrix, glm.vec3(scale / self.scale))
        
    @property
    def position(self):
        return self._matrix[3].xyz
    
    @position.setter
    def position(self, xyz):
        self._dirty.add('_matrix')
        self._matrix[3].xyz = xyz
        
    def move_global(self, xyz):
        self._dirty.add('_matrix')
        self.position += xyz

    def move_local(self, xyz):
        self._dirty.add('_matrix')
        self._matrix = glm.translate(self._matrix, xyz)
        
    def set_rotation_global(self, angle, axis=(0., 1., 0.)):
        self._dirty.add('_matrix')
        self._matrix = compute_matrix(self.scale, angle, axis, self.position)
        
    def rotate_local(self, angle, axis=(0., 1., 0.)):
        self._dirty.add('_matrix')
        self._matrix = glm.rotate(self._matrix, angle, axis)
            
    def rotate_global(self, angle, axis=(0., 1., 0.)):
        
        self._dirty.add('_matrix')
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
    m1 = glm.mat4_cast(glm.quat(xyzw2wxyz(n0.rotation))) if n0.rotation else glm.mat4(1.0) 
    qq = glm.quat_cast(m1 * m0)

    attenuation = {
        'spot': 1 / 10,
        'point': 1 / 10,
        'directional': 5 / 4,
    }

    ambient = 0.001

    light = Light(
        n0.name, 
        qq, 
        n0.scale, 
        n0.translation,
        l1['type'],
        l1['color'],
        l1['intensity'] * attenuation.get(l1['type'], 1.),
        ambient,
        l1.get('spot', {}).get('outerConeAngle', math.pi / 4),
        l1.get('spot', {}).get('innerConeAngle', math.pi / 4 * 0.9),
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
        ambient=0.001,
        outer_cone=math.pi/4,
        inner_cone=math.pi/4 * 0.9,
        shadows=True,
        **kwargs
    ):
        
        super(Light, self).__init__(name, rotation, scale, translation)
        
        self.index = -1
        
        self._items = dict(

            type = type,
            
            color = [round(c, 3) for c in color],
            intensity = round(intensity, 3),
            ambient = ambient,

            swidth = 32.,
            snear = 0.01,
            sfar = 100.,

            inner_cone = inner_cone,
            outer_cone = outer_cone,

            pcf = 3,
            bias = 0.005,
            shadows = shadows,
        )

        size = 1024
        self.shadowmaps_size = size

        self.shadowmaps_depths = [1.0, 0.6, 0.3, 0.1, 0.]

        self.shadowmaps = [dict(
            t = self.create_shadowmap_texture(size, size),
            lid = None,
            slot = None,
        ) for _ in self.shadowmaps_depths[:-1]]
    
    @property
    def shadowmaps_count(self):
        return len(self.shadowmaps_depths) - 1 if self._items['type'] == 'directional' else 1

    def create_shadowmap_texture(self, width=1024, height=1024):

        shadowmap = GLuint()
        
        glGenTextures(1, ctypes.byref(shadowmap))
        glBindTexture(GL_TEXTURE_2D, shadowmap)
        
        blank = (GLubyte * (width * height * 4))()
        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_DEPTH_COMPONENT, 
            width, height, 
            0, 
            GL_DEPTH_COMPONENT, 
            GL_FLOAT, 
            blank
        )
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, (GLfloat * 4)(1., 1., 1., 1.));  

        return shadowmap
 
    def set_index(self, index):  
        self.index = index

    def get_uniform_name(self, key):
        return 'lights[%s].%s' % (self.index, key)
 
    def set_state(self, shader, shadows):
        
        prefix = self.get_uniform_name('')

        if shadows and self.shadows:

            for i in range(self.shadowmaps_count):

                sm = self.shadowmaps[i]

                _, sm['lid'], sm['slot'], smnew = _lru_textures.allocate(sm['lid'])
                
                if smnew:

                    glActiveTexture(GL_TEXTURE0 + sm['slot'] + _MIN_TEXTURES)
                    glBindTexture(GL_TEXTURE_2D, sm['t']) 
                    
                    shader['textures[%s].t' % sm['slot']] = sm['slot'] + _MIN_TEXTURES
                    shader[prefix + 'shadowmap_textures[%s].t' % i] = sm['slot']

        if self._dirty:

            self._dirty.clear()
                            
            shader[prefix + 'type'] = LIGHT_TYPE[self.type]
            shader[prefix + 'color'] = self.color
            shader[prefix + 'intensity'] = self.intensity
            shader[prefix + 'ambient'] = self.ambient
            
            shader[prefix + 'position'] = self.position
            shader[prefix + 'direction'] = self.front
            
            shader[prefix + 'inner_cone'] = math.cos(self.inner_cone)
            shader[prefix + 'outer_cone'] = math.cos(self.outer_cone)

            shader[prefix + 'snear'] = self.snear

            shader[prefix + 'shadows'] = self.shadows

            shader[prefix + 'shadowmap_pcf'] = self.pcf
            shader[prefix + 'shadowmap_bias'] = self.bias
            shader[prefix + 'shadowmap_textures_count'] = self.shadowmaps_count
            shader[prefix + 'shadowmap_textures_size'] = self.shadowmaps_size
                        
    def set_state_shadowmap(self, shadowmap_index, shader, camera_position, camera_screen):
        
        si = shadowmap_index

        prefix = self.get_uniform_name('')

        view = glm.lookAt(self.position, self.position - self.front, self.up)

        if self.type == 'directional':
            
            position = view * camera_position
            screen = view * camera_screen

            min0, max0 = compute_plane_minmax(
                position, screen, self.shadowmaps_depths[si]
            )

            min1, max1 = compute_plane_minmax(
                position, screen, self.shadowmaps_depths[si+1]
            )

            min2 = glm.min(min0, min1)
            max2 = glm.max(max0, max1)

            projection = glm.ortho(
                round(min2.x), 
                round(max2.x), 
                round(min2.y), 
                round(max2.y), 
                self.snear, 
                self.sfar
            )
            
            shader[prefix + 'scale'] = max(max2.x - min2.x, max2.y - min2.y)

        elif self.type == 'point':
            projection = glm.perspective(
                math.pi / 2, 
                1., 
                self.snear, 
                self.sfar
            )

        else:
            projection = glm.perspective(
                2 * self.outer_cone, 
                1., 
                self.snear, 
                self.sfar
            )
        
        self._view = view
        self._proj = projection
        
        projection = flatten(projection * view)

        shader[prefix + 'shadowmap_textures[%s].depth' % si] = self.shadowmaps_depths[si]
        shader[prefix + 'shadowmap_textures[%s].projection' % si] = projection
        shader[prefix + 'shadowmap_projection'] = projection

        return self.shadowmaps[si]['t']


def compute_plane_minmax(position, far_screen, split):
    
    pm = glm.mat4(position, position, position, position)
    s0 = (far_screen - pm) * split + pm
    
    min0 = glm.min(s0[0], s0[1], s0[2], s0[3])
    max0 = glm.max(s0[0], s0[1], s0[2], s0[3])
    
    return min0, max0


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
        
        self._items = dict(
            type = type,
            znear = round(znear, 3),
            zfar = round(zfar, 3),
            yfov = round(yfov, 3),
            xmag = round(xmag, 3),
            ymag = round(ymag, 3),
        )

        self._aspect = 0

    def set_state(self, shader, width, height):

        dirty = self._aspect != width / height
        self._aspect = width / height
                    
        if dirty or self._dirty:

            self._view0 = glm.lookAt(self.position, self.position - self.front, self.up)
            self._proj0 = glm.perspective(
                self.yfov, 
                self._aspect, 
                self.znear, 
                self.zfar
            )

            shader['view'] = flatten(self._view0)
            shader['projection'] = flatten(self._proj0)
            shader['camera.position'] = self.position
            shader['camera.zfar'] = self.zfar

            self._dirty.clear()


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
        
        self.shadow_bias = 0
        self.hide = False
        
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

        return self._parent.composed_matrix() * self._matrix 

    def set_state(self):
        
        if self.hide:
            return self._group

        shader = self._group.shader

        if shader._extra.get('shadow_bias') != self.shadow_bias:
            shader._extra['shadow_bias'] = self.shadow_bias
            shader['shadow_bias'] = self.shadow_bias
            
        shader['model'] = flatten(self.composed_matrix())
        
    def unset_state(self):

        if self.hide:
            return

        self._dirty.clear()

def load_blender_gltf_primitive(g0, p0, materials):
    
    primitive = Primitive(
        material=materials[p0.material], 
        indices=get_buffer0(g0, p0.indices, '')[-1],
        vertices=get_buffer0(g0, p0.attributes.POSITION),
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
    
    def __init__(self, material, indices, vertices, normals=None, coords=None):
        
        super(Primitive, self).__init__()
        
        self.material = material

        self._indices = indices
        self._vertices = vertices
        self._normals = normals
        self._coords = coords
        
        self.nvertices = len(self._vertices[1]) // 3
        self.has_uvs = self._coords is not None
       
        self._group = Group(self)
        
    def batch_add(self, batch):

        vl = [self._vertices, self._normals, self._coords]
        vl = [('%sg%s' % (i, fv[0]), fv[1]) for i, fv in enumerate(vl) if fv]
        
        batch.add_indexed(self.nvertices, GL_TRIANGLES, self._group, self._indices, *vl)
        
    def set_state(self):
        shader = self._group.shader
        if shader._extra.get('shadowmap_pass') != 1:
            self.material.set_state(shader)
        
    def unset_state(self):
        if self._group.shader._extra.get('shadowmap_pass') != 1:
            self.material.unset_state()


class CubeMap(Object):
    
    def __init__(self, path, intensity=1.0):
        
        super(CubeMap, self).__init__()
        
        scene = load_blender_gltf(abspath('./assets/cube.gltf'))

        self.cube = scene.meshes['CubeMap']
        self.path = path
        self.hide = False

        self._items = dict(
            intensity = intensity,
            texture_id = self.load(path, False),
        )

        self._group = Group(self)
        self.cube._group.parent = self._group

    def set_state(self):
                
        if self.hide:
            return self._group

        shader = self._group.shader

        shader['cubemap.render_cubemap'] = True

        glCullFace(GL_FRONT)
        glDepthFunc(GL_LEQUAL)   
        
        if self._dirty:
            
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_id)

            shader['cubemap.intensity'] = self.intensity
            shader['cubemap.texture_exists'] = True
            shader['cubemap.texture'] = 0

            self._dirty.clear()

    def unset_state(self):

        if self.hide:
            return

        self._group.shader['cubemap.render_cubemap'] = False

        glDepthFunc(GL_LESS)        
        glCullFace(GL_BACK) 

    def load(self, path, update_shader=True):
        
        paths = set(glob.glob(path))

        tid = GLuint()
        glGenTextures(1, ctypes.byref(tid))
        glBindTexture(GL_TEXTURE_CUBE_MAP, tid.value)

        faces = {
            'right': 'RT',
            'left': 'LF',
            'bottom': 'DN', # Bottom and top should actually be reversed...
            'top': 'UP',
            'front': 'FT',
            'back': 'BK',
        }

        def get_face_path(face, paths):
            return [p for p in paths if faces[face] in p][0]

        for i, face in enumerate(faces):
        
            path = get_face_path(face, paths)
            
            im = PIL.Image.open(path)
            b0 = im.tobytes("raw", "RGB", 0, -1)
            w0, h0 = im.size
            
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, 
                0, 
                GL_RGB, 
                w0, 
                h0, 
                0, 
                GL_RGB, 
                GL_UNSIGNED_BYTE, 
                b0
            )
    
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        if update_shader:
            self.path = path
            self.texture_id = tid

        return tid

