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


import pyglet
import glm
import os

import numpy as np

from pyglet.graphics import *


def flatten(m):
    return sum(m.to_list(), [])


def wavefront2pyglet(material):
    
    formats = material.vertex_format.lower().split('_')
    array = np.array(material.vertices).reshape(-1, material.vertex_size)
    
    c0 = 0
    data = []
    
    for i, fmt in enumerate(formats):
        w0 = int(fmt[1])
        f0 = '%sg%s' % (i, fmt[1:])
        data.append((f0, array[:,c0:c0+w0].flatten().tolist()))
        c0 += w0

    return data


def get_material_texture(texture):
    
    if texture is None or not texture.exists():
        return
    
    path = os.path.abspath(texture.path)
    dirname, basename = os.path.split(path)
    
    if dirname not in pyglet.resource.path:
        pyglet.resource.path.append(dirname)
        pyglet.resource.reindex()

    texture = pyglet.resource.texture(basename)

    glBindTexture(texture.target, texture.id)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    return texture


class Group(pyglet.graphics.Group):

    def __init__(self, shader=None, matrix=None, parent=None):
        
        super(Group, self).__init__(parent)
        
        self.shader = shader
        self.matrix = matrix
        
        if matrix is None and self.parent is None:
            self.matrix = glm.mat4() 
                
    def get_matrix(self):
        
        if self.matrix:
            return self.matrix
        
        if self.parent:
            return self.parent.get_matrix()
        
    def get_shader(self):
        
        if self.shader:
            return self.shader
        
        if self.parent:
            return self.parent.get_shader()
        
    def set_state(self, face=GL_FRONT_AND_BACK):
        
        if self.shader:
            self.shader.use()
            
        if self.matrix:
            self.get_shader()['model'] = flatten(self.matrix)
        
    def unset_state(self):
        
        if self.matrix and self.parent:
            self.get_shader()['model'] = flatten(self.parent.get_matrix())
        
        if self.shader:
            self.shader.stop()


class MaterialGroup(Group):

    def __init__(self, material, shader=None, matrix=None, parent=None):
        
        super(MaterialGroup, self).__init__(shader, matrix, parent)
        
        self.material = material
        
        self.texture = get_material_texture(material.texture)
        self.texture_bump = get_material_texture(material.texture_bump)
        self.texture_specular_highlight = get_material_texture(material.texture_specular_highlight)
        
    def set_state(self, face=GL_FRONT_AND_BACK):
        
        super(MaterialGroup, self).set_state()
        
        shader = self.get_shader()

        if self.texture is not None:
            #glEnable(self.texture.target)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(self.texture.target, self.texture.id)
            shader['material.texture'] = 0
            
        shader['material.texture_exists'] = self.texture is not None

        if self.texture_bump is not None:
            #glEnable(self.texture_bump.target)
            glActiveTexture(GL_TEXTURE1)            
            glBindTexture(self.texture_bump.target, self.texture_bump.id)
            shader['material.texture_bump'] = 1

        shader['material.texture_bump_exists'] = self.texture_bump is not None
        
        if self.texture_specular_highlight is not None:
            #glEnable(self.texture_specular_highlight.target)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(self.texture_specular_highlight.target, self.texture_specular_highlight.id)
            shader['material.texture_specular_highlight'] = 2

        shader['material.texture_specular_highlight_exists'] = self.texture_specular_highlight is not None
        
        #shader['material.ambient'] = self.material.ambient
        shader['material.diffuse'] = self.material.diffuse
        shader['material.specular'] = self.material.specular
        shader['material.shininess'] = self.material.shininess
                
    def __eq__(self, other):
        # Do not consolidate Groups when adding to a Batch.
        # Matrix multiplications requires isolation.
        return False

    def __hash__(self):
        return hash((tuple(self.material.diffuse) + tuple(self.material.ambient) +
                     tuple(self.material.specular) + tuple(self.material.emissive), self.material.shininess))


#@functools.lru_cache(maxsize=4096)
def compute_matrix(scale=1., angle=0, axis=(0., 1., 0.), xyz=(0., 0., 0.), matrix=glm.mat4()):
    
    if xyz != (0, 0, 0):
        matrix = glm.translate(matrix, xyz)
        
    if scale != 1.:
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


class Objection(object):
    
    @property
    def scale(self):
        return decompose(self.matrix, scale=glm.vec3())
        
    @scale.setter
    def scale(self, scale):        
        self.matrix = glm.scale(self.matrix, glm.vec3(scale / self.scale))
        
    @property
    def position(self):
        return self.matrix[3].xyz
    
    @position.setter
    def position(self, xyz):
        self.matrix[3].xyz = xyz
        
    def move_global(self, xyz):
        self.position += xyz

    def move_local(self, xyz):
        self.matrix = glm.translate(self.matrix, xyz)
        
    def set_rotation_global(self, angle, axis=(0., 1., 0.)):
        self.matrix = compute_matrix(self.scale, angle, axis, self.position)
        
    def rotate_local(self, angle, axis=(0., 1., 0.)):
        self.matrix = glm.rotate(self.matrix, angle, axis)
            
    def rotate_global(self, angle, axis=(0., 1., 0.)):
        
        pos = self.position
        self.position = 0, 0, 0
        self.matrix = glm.rotate(glm.mat4(), angle, axis) * self.matrix
        self.position = pos
        
    def get_rotation(self):
        """Extract rotation angle and axis from rotation matrix. 
        
        This method does not separate global and local rotations. 
        
        For that it is neccessary to keep track of separate model transformations.
        
        References:
        https://stackoverflow.com/questions/17918033/glm-decompose-mat4-into-translation-and-rotation
        and https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        rotation = decompose(self.matrix, rotation=glm.vec4())

        c, xs, ys, zs = rotation #glm.conjugate(rotation)

        angle = math.acos(c) * 2
        s = math.sin(angle / 2)

        return angle, glm.vec3(x / s, y / s, z / s)
    
    @property
    def up(self):
        """Return the local up (+y) axis."""
        return (self.matrix * glm.vec4(0, 1, 0, 0)).xyz

    @property
    def front(self):
        """Return the local front (+z) axis."""
        return (self.matrix * glm.vec4(0, 0, 1, 0)).xyz


class Model(Objection):
    
    def __init__(self, wavefront, shader, batch):
        
        self.wavefront = wavefront
        self.shader = shader
        
        self._scale = 1.

        self.group = Group(shader)
        self.vertex_lists = []
        self._groups = {}

        for name, material in wavefront.materials.items():
            
            g0 = MaterialGroup(material, parent=self.group)
            d0 = wavefront2pyglet(material)
            sz = len(material.vertices) // material.vertex_size
            
            self.vertex_lists.append(batch.add(sz, GL_TRIANGLES, g0, *d0))
            self._groups[name] = g0

    @property
    def matrix(self):
        return self.group.matrix

    @matrix.setter
    def matrix(self, m):
        self.group.matrix = m
        

class Camera(Objection):

    def __init__(self, position=None):
        
        #super(Camera, self).__init__(inverted=True)
        
        self.matrix = glm.mat4() 
        
        if position:
            self.position = position
                
    def set_state(self, shader):

        shader['view'] = flatten(glm.lookAt(self.position, self.position + self.front, self.up))
        shader['camera.position'] = self.position

