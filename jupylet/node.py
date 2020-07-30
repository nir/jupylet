"""
    jupylet/node.py
    
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
import copy
import math
import glm

from pyglet.graphics import *


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
        

def compute_matrix(angle=0, axis=(0., 0., 1.), scale=1., xyz=(0., 0., 0.), matrix=glm.mat4(1.)):
    
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

    if s == 0:
        return 0, glm.vec3(0, 0, 1)

    if deg:
        angle = round(180 * angle / math.pi, 3)
    
    return angle, glm.vec3(xs / s, ys / s, zs / s)


def aa2q(angle, axis=glm.vec3(0, 0, 1)):
    return glm.angleAxis(angle, glm.normalize(axis))


_i4 = glm.mat4(1.)


class Node(Object):
  
    def __init__(
        self, 
        name='', 
        anchor=None,
        scale=None, 
        rotation=None, 
        position=None,
    ):
        
        super(Node, self).__init__()
        
        self._itemz = None
        self._items = dict(
            anchor = glm.vec3(0.),
            scale = scale or glm.vec3(1.),
            rotation = rotation or glm.quat(1., 0., 0., 0.),
            position = position or glm.vec3(0.),
        )

        self.name = name

        self._matrix = glm.mat4(1.)

    @property
    def matrix(self):
        
        if self._itemz != self._items:
            self._itemz = copy.deepcopy(self._items)
            self._dirty.clear()

            t0 = glm.translate(_i4, self.position)
            r0 = t0 * glm.mat4_cast(self.rotation)
            s0 = glm.scale(r0, self.scale)
            a0 = glm.translate(s0, -self.anchor)

            self._matrix = a0

        return self._matrix

    def move_local(self, xyz):

        rxyz = glm.mat4_cast(self.rotation) * glm.vec4(xyz, 1.)
        self.position += xyz1.xyz
        
    def rotate_local(self, angle, axis=(0., 0., 1.)):

        axis = glm.mat4_cast(self.rotation) * glm.vec4(axis, 1.)
        self.rotation *= aa2q(angle, axis.xyz)
            
    def move_global(self, xyz):
        self.position += xyz
        
    def rotate_global(self, angle, axis=(0., 0., 1.)):
        self.rotation *= aa2q(angle, axis)
            
    @property
    def up(self):
        """Return the local up (+y) axis."""
        return (self.matrix * glm.vec4(0, 1, 0, 0)).xyz

    @property
    def front(self):
        """Return the local front (+z) axis."""
        return (self.matrix * glm.vec4(0, 0, 1, 0)).xyz

