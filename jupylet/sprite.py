"""
    jupylet/sprite.py
    
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


import moderngl
import pathlib
import math
import glm

import PIL.Image

import moderngl_window as mglw
import numpy as np

from moderngl_window.meta import TextureDescription

from .collision import trbl, hitmap_and_outline_from_alpha, compute_collisions
from .resource import texture_load, pil_from_texture
from .state import State
from .node import Node, aa2q, q2aa


_empty_array = np.array([])


class Sprite(Node):
    
    def __init__(self,
        img, 
        x=0, 
        y=0,
        scale=1.0,
        angle=0.0,
        anchor_x='center',
        anchor_y='center',
        flip=True, 
        mipmap=True, 
        autocrop=False,
        anisotropy=8.0, 
        height=None,
        width=None,
        name=None
    ):

        super(Sprite, self).__init__(
            name,
            rotation=aa2q(glm.radians(angle)),
            scale=scale,
            position=glm.vec3(x, y, 0),
        )

        #self._items = dict(
        #)

        self.flip = flip
        self.mipmap = mipmap 
        self.autocrop = autocrop
        self.anisotropy = anisotropy
        
        self.geometry = mglw.geometry.quad_2d(
            size=(1.0, 1.0), 
            pos=(0.5, 0.5)
        )

        self.texture = texture_load(
            img,
            anisotropy=anisotropy, 
            autocrop=autocrop,
            mipmap=mipmap, 
            flip=flip, 
        )

        self.hitmap, self.outline = hitmap_and_outline_from_alpha(self.image)

        self.scale *= glm.vec3(
            glm.vec2(self.texture.size) / max(*self.texture.size), 1
        )

        if width:
            self.width = width
        
        elif height:
            self.height = height

        self.set_anchor(anchor_x, anchor_y)

    def render(self, shader):
        
        shader['model'].write(self.matrix)
        shader['texture_id'] = 0 

        self.texture.use(location=0)
        self.geometry.render(shader)

    @property
    def angle(self):
        angle, axis = q2aa(self.rotation)
        return round(glm.degrees(angle * glm.sign(axis.z)), 4)

    @angle.setter
    def angle(self, angle):
        self.rotation = aa2q(glm.radians(angle))

    def set_anchor(self, ax=None, ay=None):

        if ax == 'left':
            self.anchor.x = 0
        elif ax == 'center':
            self.anchor.x = 0.5
        elif ax == 'right':
            self.anchor.x = 1.
        elif type(ax) in (int, float):
            self.anchor.x = ax / self.width

        if ay == 'bottom':
            self.anchor.y = 0
        elif ay == 'center':
            self.anchor.y = 0.5
        elif ay == 'top':
            self.anchor.y = 1.
        elif type(ay) in (int, float):
            self.anchor.y = ay / self.width

    @property
    def width(self):
        return self.scale.x * max(self.texture.size)

    @width.setter
    def width(self, width):
        self.scale *= width / self.texture.width / self.scale.x

    @property
    def height(self):
        return self.scale.y * max(self.texture.size)

    @height.setter
    def height(self, height):
        self.scale *= height / self.texture.height / self.scale.y

    @property
    def image(self):
        return pil_from_texture(self.texture)
    
    @image.setter
    def image(self, img):
        
        width = self.width
        height = self.height

        self.texture = texture_load(
            img,
            anisotropy=self.anisotropy, 
            autocrop=self.autocrop,
            mipmap=self.mipmap, 
            flip=self.flip, 
        )

        self.hitmap, self.outline = hitmap_and_outline_from_alpha(self.image)

        if width != self.width and height != self.height:
            self.width = width 

    def collisions_with(self, o, debug=False):
        
        #if self.distance_to(o) > self.radius + o.radius:
        #    return

        x0, y0 = self.position.x, self.position.y
        x1, y1 = o.position.x, o.position.y

        t0, r0, b0, l0 = self._trbl()
        t1, r1, b1, l1 = o._trbl()

        if t0 + y0 < b1 + y1 or t1 + y1 < b0 + y0:
            return _empty_array[:0]

        if r0 + x0 < l1 + x1 or r1 + x1 < l0 + x0:
            return _empty_array[:0]
        
        return compute_collisions(o, self, debug=debug)

    def distance_to(self, o=None, pos=None):

        x, y = pos or (o.position.x, o.position.y)
        
        dx = x - self.position.x
        dy = y - self.position.y

        return (dx ** 2 + dy ** 2) ** 0.5
    
    def angle_to(self, o=None, pos=None):
        
        qd = {
            (True, True): 0,
            (True, False): 180,
            (False, False): 180,
            (False, True): 360,
        }
        
        x, y = pos or (o.position.x, o.position.y)
        
        dx = x - self.position.x
        dy = y - self.position.y

        a0 = math.atan(dy / (dx or 1e-7)) / math.pi * 180 + qd[(dy >= 0, dx >= 0)]

        return -a0

    @property
    def top(self):
        t, r, b, l = self._trbl()
        return self.position.y + t
        
    @property
    def right(self):
        t, r, b, l = self._trbl()
        return self.position.x + r
        
    @property
    def bottom(self):
        t, r, b, l = self._trbl()
        return self.position.y + b
        
    @property
    def left(self):
        t, r, b, l = self._trbl()
        return self.position.x + l
        
    @property
    def radius(self):
        t, r, b, l = self._trbl()
        rs = max(t, b) ** 2 + max(r, l) ** 2
        return rs ** .5
        
    def _trbl(self):
        tx = self.texture
        return trbl(
            tx.width, 
            tx.height, 
            self.anchor_x, 
            self.anchor_y, 
            self.angle,
            self.scale
        )

    def wrap_position(self, width, height, margin=50):
        self.position.x = (self.position.x + margin) % (width + 2 * margin) - margin
        self.position.y = (self.position.y + margin) % (height + 2 * margin) - margin

    def clip_position(self, width, height, margin=0):
        self.position.x = max(-margin, min(margin + width, self.position.x))
        self.position.y = max(-margin, min(margin + height, self.position.y))

    def get_state(self):

        return State(
            _items = self._items,
            #opacity = self.opacity,
        )

    def set_state(self, s):
        
        for k, v in s._items.items():
            setattr(k, v)
        
        #self.opacity = s.opacity


"""
def canvas2sprite(c):
    
    a = c.get_image_data()
    b = a.tostring()
    x = ctypes.string_at(id(b)+20, len(b))
    d = pyglet.image.ImageData(a.shape[1], a.shape[0], 'RGBA', x)
    s = Sprite(d)
    
    return s
"""

