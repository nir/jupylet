"""
    jupylet/shadertoy.py
    
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


import webcolors
import datetime
import moderngl
import pathlib
import math
import time
import glm
import re

import scipy.signal
import PIL.Image

import moderngl_window as mglw
import numpy as np

from moderngl_window.meta import ProgramDescription
from moderngl_window.opengl import program

from .audio.device import get_output_as_array
from .audio import FPS

from .resource import load_texture, pil_from_texture, find_path
from .utils import glm_dumps, glm_loads
from .color import c2v
from .state import State
from .node import Node, aa2q, q2aa
from .env import get_window_size
from .lru import SPRITE_TEXTURE_UNIT


def get_shadertoy_audio(start=-512, length=512, amp=1.):
    
    a0 = get_output_as_array(start, length, resample=512)[0]
    if a0 is None:
        a0 = np.zeros((512, 2))
    
    ft = np.fft.rfft(a0, axis=0)
    sa = np.square(np.abs(ft)) + 1e-6
    ps = 10 * np.log10(sa)
    rs = scipy.signal.resample(ps, len(a0))
    ns = (rs + 50) / 100
    
    return np.stack((ns * 256, a0 * amp * 128 + 128)).clip(0, 255)


def load_shadertoy_program(source):

    if '\n' not in source:
        path = find_path(source)
        source = path.open().read()

    path = find_path('shaders/shadertoy-wrapper.glsl')
    
    single = path.open().read()
    single = re.sub(r'void mainImage.* {}', source, single)

    pd = ProgramDescription(path='shadertoy.glsl')
    sd = program.ProgramShaders.from_single(pd, single)
    pg = sd.create()

    return pg


class Shadertoy(Node):

    """A Shadertoy canvas.
    
    Args:
        x (float): The x position for the shadertoy canvas.
        y (float): The y position for the shadertoy canvas.
        angle (float): clockwise rotation of the shadertoy canvas in degrees.
        anchor_x (float or str): either 'left', 'center' or 'right' or a 
            value between 0.0 (for left) and 1.0 (for right) indicating
            the anchor point inside the shadertoy canvas along its x axis.
        anchor_y (float or str): either 'bottom', 'center' or 'top' or a 
            value between 0.0 (for bottom) and 1.0 (for top) indicating
            the anchor point inside the shadertoy canvas along its y axis.
        color (str or 3-tuple): color by which to tint shadertoy canvas image.
            Could be a color name, color hex notation, or a 3-tuple.
    """

    def __init__(
        self,
        code,
        x=0, 
        y=0,
        width=800,
        height=450,
        angle=0.0,
        anchor_x='center',
        anchor_y='center',
        color='white',
        name=None,
    ):
        """"""

        super().__init__(
            name,
            rotation=aa2q(glm.radians(angle)),
            scale=None,
            position=glm.vec3(x, y, 0),
        )

        self.nframe = 0

        w0, h0 = get_window_size()
        
        self.shader = load_shadertoy_program(code)
        self.shader['projection'].write(glm.ortho(
            0, w0, 0, h0, -1, 1
        ))

        self.geometry = mglw.geometry.quad_2d(
            size=(1.0, 1.0), 
            pos=(0.5, 0.5)
        )

        self.channel0 = None
        self.channel1 = None
        self.channel2 = None
        self.channel3 = None
        
        self.channeltime = [0., 0., 0., 0.]

        self.components = 3
        self.color4 = glm.vec4(1., 1., 1., 1.)

        self.width = width
        self.height = height
        
        self.set_anchor(anchor_x, anchor_y)
        self.color = color

    def update(self, shader):
        pass

    def draw(self, ct, dt):
        """Render shadertoy to canvas - this is an alias to shadertoy.render()."""
        return self.render(ct, dt)
        
    def render(self, ct, dt):
        """Render shadertoy to canvas."""
        
        if self._dirty:
            self.update(self.shader)

        self.shader['model'].write(self.matrix)
        self.shader['components'] = self.components
        self.shader['color'].write(self.color4)

        if 'iResolution' in self.shader._members:
            self.shader['iResolution'].write(self.scale)
        
        if 'iTime' in self.shader._members:
            self.shader['iTime'] = ct
        
        if 'iTimeDelta' in self.shader._members:
            self.shader['iTimeDelta'] = dt

        if 'iFrame' in self.shader._members:
            self.shader['iFrame'] = self.nframe
            self.nframe += 1

        for i in [0, 1, 2, 3]:
            
            ch = getattr(self, 'channel%s' % i, None)
            if ch is None:
                continue

            self.shader['iChannel%s' % i] = SPRITE_TEXTURE_UNIT + i
            ch.use(location=SPRITE_TEXTURE_UNIT+i)

            if 'iChannelTime[%s]' % i in self.shader._members:
                self.shader['iChannelTime[%s]' % i] = self.channeltime[i]

            if 'iChannelResolution[%s]' % i in self.shader._members:
                self.shader['iChannelResolution[%s]' % i].write(
                    glm.vec3(ch.width, ch.height, ch.components)
                )

        if 'iDate' in self.shader._members:
            dt = datetime.datetime.now()
            self.shader['iDate'].write(glm.vec4(dt.year, dt.month, dt.day, time.time()))

        if 'iSampleRate' in self.shader._members:
            self.shader['iSampleRate'] = FPS

        self.geometry.render(self.shader)

    @property
    def x(self):
        """float: The x coordinate of the anchor position."""
        return self.position.x
        
    @x.setter
    def x(self, value):
        self.position.x = value
        
    @property
    def y(self):
        """float: The y coordinate of anchor position."""
        return self.position.y
        
    @y.setter
    def y(self, value):
        self.position.y = value
        
    @property
    def angle(self):
        """float: The rotation angle around the anchor in degrees."""
        angle, axis = q2aa(self.rotation)
        return round(glm.degrees(angle * glm.sign(axis.z)), 4)

    @angle.setter
    def angle(self, angle):
        self.rotation = aa2q(glm.radians(angle))

    def set_anchor(self, ax=None, ay=None):
        """Set the anchor point of the sprite.

        The anchor is a point in the sprite that is used for rotation and 
        positioning. Imagine a pin going through the sprite and that you use
        this pin to position the sprite on the canvas and to rotate it. The
        point at which the pin goes through the texture is the anchor point.

        The anchor point is set separately for the x axis, and for the y axis.

        Args:
            ax (str or float): the x anchor can be one of 'left', 'center', 
                'right', or a float value between 0.0 (left) and 1.0 (right).

            ay (str or float): the y anchor can be one of 'bottom', 'center', 
                'top', or a float value between 0.0 (bottom) and 1.0 (top).
        """
        self._ax = ax
        self._ay = ay

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
        """float: Width in pixels."""
        return self.scale0.x

    @width.setter
    def width(self, width):
        self.scale0.x = width

    @property
    def height(self):
        """float: Height in pixels."""
        return self.scale0.y

    @height.setter
    def height(self, height):
        self.scale0.y = height

    def set_channel(
        self, 
        data,
        channel, 
        channeltime=None,
        mipmap=True, 
        autocrop=False,
        anisotropy=8.0, 
        ):

        assert channel in [0, 1, 2, 3]

        # Disable mipmaping and anisotropy for audio data.
        if isinstance(data, np.ndarray) and data.shape[0] < 8:
            mipmap = False
            anisotropy = False

        if channeltime:
            self.channeltime[channel] = channeltime

        channel = 'channel%s' % channel

        texture = getattr(self, channel, None)
        if texture is not None:
            texture.release()
            
        texture = load_texture(
            data,
            anisotropy=anisotropy, 
            autocrop=autocrop,
            mipmap=mipmap, 
            flip=False, 
        )
        texture.repeat_x = False
        texture.repeat_y = False

        setattr(self, channel, texture)

    def distance_to(self, o=None, pos=None):
        """Compute the distance to another sprite or coordinate.

        Args:
            o (Sprite, optional): Other sprite to compute distance to.
            pos (tuple, optional): An (x, y) coordinate to compute distance to.

        Returns:
            float: Distance in pixels.
        """
        x, y = pos or (o.position.x, o.position.y)
        
        dx = x - self.position.x
        dy = y - self.position.y

        return (dx ** 2 + dy ** 2) ** 0.5
    
    def angle_to(self, o=None, pos=None):
        """Compute clockwise angle in degrees to another sprite or coordinate.

        Args:
            o (Sprite, optional): Other sprite to compute angle to.
            pos (tuple, optional): An (x, y) coordinate to compute angle to.

        Returns:
            float: Angle in degrees.
        """

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
        """float: Get the top coordinate of the sprite's bounding box."""
        t, r, b, l = self._trbl()
        return self.position.y + t
        
    @property
    def right(self):
        """float: Get the right coordinate of the sprite's bounding box."""
        t, r, b, l = self._trbl()
        return self.position.x + r
        
    @property
    def bottom(self):
        """float: Get the bottom coordinate of the sprite's bounding box."""
        t, r, b, l = self._trbl()
        return self.position.y + b
        
    @property
    def left(self):
        """float: Get the left coordinate of the sprite's bounding box."""
        t, r, b, l = self._trbl()
        return self.position.x + l
        
    @property
    def radius(self):
        """float: Get the radius of a circle containing the sprite's bounding box."""
        t, r, b, l = self._trbl()
        rs = max(t, b) ** 2 + max(r, l) ** 2
        return rs ** .5
        
    def _trbl(self):
        return trbl(
            self.width, 
            self.height, 
            self.anchor.x * self.width, 
            self.anchor.y * self.height, 
            self.angle,
        )

    def wrap_position(self, width, height, margin=50):
        """Wrap sprite's coordinates around given canvas width and height.

        Use this method to make the sprite come back from one side of the 
        canvas if it goes out from the opposite side.

        Args:
            width (float): The canvas width to wrap around.
            height (float): The canvas height to wrap around.
            margin (float, optional): An extra margin to add around canvas 
                before wrapping the sprite around to the opposite side.
        """
        self.position.x = (self.position.x + margin) % (width + 2 * margin) - margin
        self.position.y = (self.position.y + margin) % (height + 2 * margin) - margin

    def clip_position(self, width, height, margin=0):
        """Clip sprite's coordinates to given canvas width and height.

        Use this method to prevent the sprite from going out of the canvas. 

        Args:
            width (float): The canvas width to clip to.
            height (float): The canvas height to clip to.
            margin (float, optional): An extra margin to add around canvas 
                before clipping the sprite's coordinates.
        """
        self.position.x = max(-margin, min(margin + width, self.position.x))
        self.position.y = max(-margin, min(margin + height, self.position.y))

    @property
    def opacity(self):
        """float: Get or set the opacity of the sprite.

        Setting opacity to 0 would render the sprite completely transparent.
        Settint opacity to 1 would render the sprite completely opaque.
        """
        return self.color4.a

    @opacity.setter
    def opacity(self, opacity):
        self.color4.a = opacity
        
    @property
    def color(self):
        """glm.vec4: Get or set the color of the sprite.
        
        The sprite color will be multiplied by the color values of its bitmap
        image. 

        The color can be specified by name (e.g. 'white') or in hex notation
        (e.g '#cc4488') or as a 4-tuple or glm.vec4 value.
        """
        return self.color4

    @color.setter
    def color(self, color):        
        self.color4 = c2v(color, self.color4.a)

