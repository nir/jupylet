"""
    jupylet/node.py
    
    Copyright (c) 2022, Nir Aides - nir.8bit@gmail.com

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


import copy
import math
import glm

from .utils import glm_dumps, glm_loads


class Object(object):
    
    """Implement an object that tracks changes to its properties.

    It is used for implementing lazy mechanisms that only engage in costly
    computations when necessary.
    """

    def __init__(self, dirty=True):
        
        self._items = {}
        self._dirty = set([True]) if dirty else set()

    def __dir__(self):
        return list(self._items.keys()) + super().__dir__()
    
    def __getattr__(self, k):
        
        if k not in self._items:
            return super().__getattribute__(k)
            
        return self._items[k]
    
    def __setattr__(self, k, v):
        
        if k == '_items' or k not in self._items:
            return super().__setattr__(k, v)
        
        self._items[k] = v
        self._dirty.add(k)

    def __repr__(self):    
        return '%s(%s)' % (type(self).__name__, ', '.join(
            '%s=%s' % i for i in list(self.__dict__.items()) + list(self._items.items()) if i[0][0] != '_'
        ))


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
  
    """Handle and represent geometric operations as matrix transformation.
    
    Handle and represent scaling, rotation, and translation in local and global
    3D coordinates as a single matrix (lazily maintained) transformation.

    Args:
        name (str, optional): Name of object.
        anchor (glm.vec3, optional): 
    """

    def __init__(
        self, 
        name='', 
        anchor=None,
        scale=None, 
        rotation=None, 
        position=None,
    ):
        
        super().__init__()
        
        self.name = name

        self.anchor = glm.vec3(anchor) if anchor else glm.vec3(0.)
        self.scale0 = glm.vec3(scale) if scale else glm.vec3(1.)
        self.rotation = glm.quat(rotation) if rotation else glm.quat(1., 0., 0., 0.)
        self.position = glm.vec3(position) if position else glm.vec3(0.)

        self._itemz = None

        self._matrix = glm.mat4(1.)

    @property
    def scale(self):
        return self.scale0

    @scale.setter
    def scale(self, value):
        self.scale0 = value

    @property
    def matrix(self):
        
        if self._itemz != [self.anchor, self.scale0, self.rotation, self.position]:
            self._itemz = copy.deepcopy([
                self.anchor, 
                self.scale0, 
                self.rotation, 
                self.position
            ])

            t0 = glm.translate(_i4, self.position)
            r0 = t0 * glm.mat4_cast(self.rotation)
            s0 = glm.scale(r0, self.scale0)
            a0 = glm.translate(s0, -self.anchor)

            self._matrix = a0

            self._dirty.add('_matrix')

        return self._matrix

    def move_local(self, xyz):
        """Move by given displacement in local coordinate system.

        Args:
            xyz (glm.vec3): Displacement.
        """
        rxyz = glm.mat4_cast(self.rotation) * glm.vec4(xyz, 1.)
        self.position += rxyz.xyz
        
    def rotate_local(self, angle, axis=(0., 0., 1.)):
        """Rotate counter clockwise by given angle around given axis in local 
        coordinate system.

        Args:
            angle (float): Angle in radians.
            axis (glm.vec3): Rotation axis.
        """
        self.rotation *= aa2q(angle, glm.vec3(axis))
            
    def move_global(self, xyz):
        """Move by given displacement in global coordinate system.

        Args:
            xyz (glm.vec3): Displacement.
        """
        self.position += xyz
        
    def rotate_global(self, angle, axis=(0., 0., 1.)):
        """Rotate counter clockwise by given angle around given axis in global 
        coordinate system.

        Args:
            angle (float): Angle in radians.
            axis (glm.vec3): Rotation axis.
        """
        self.rotation = aa2q(angle, glm.vec3(axis)) * self.rotation
            
    @property
    def up(self):
        """glm.vec3: Return the local up (+y) axis."""
        return (self.matrix * glm.vec4(0, 1, 0, 0)).xyz

    @property
    def front(self):
        """glm.vec3: Return the local front (+z) axis."""
        return (self.matrix * glm.vec4(0, 0, 1, 0)).xyz

    def get_state(self):
        """Get a dictionary of properties defining the object state.
        
        Returns:
            dict: A dictionary of properties.
        """
        return dict(
            rotation = glm_dumps(glm.quat(self.rotation)), 
            position = glm_dumps(glm.vec3(self.position)),
            anchor = glm_dumps(glm.vec3(self.anchor)), 
            scale0 = glm_dumps(glm.vec3(self.scale0)), 
        )

    def set_state(self, s):
        """Set object state from given dictionary of properties.
        
        Args:
            s (dict): A dictionary of properties previously returned by a call 
                to ``get_state()``.
        """
        for k, v in s.items():
            setattr(self, k, glm_loads(v))

