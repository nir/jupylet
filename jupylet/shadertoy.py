"""
    jupylet/shadertoy.py
    
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


import functools
import webcolors
import datetime
import moderngl
import pathlib
import weakref
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

from .resource import load_texture, pil_from_texture, find_path, get_context
from .utils import glm_dumps, glm_loads
from .color import c2v
from .state import State
from .node import Node, aa2q, q2aa
from .env import get_window_size
from .lru import SPRITE_TEXTURE_UNIT


@functools.lru_cache()
def get_indices(size):
    
    i0 = np.arange(size)
    i1 = np.ones((size, size))
    i2 = (i1 * i0 + i0[:,None]).astype('long')
    
    return i2


def get_correlation(a0, size=300, step=2, prev=[]):
    
    size = min(size, len(a0))
    
    if not prev:
        prev.append(a0)
        return 0

    p0 = prev[0]

    s0 = size // 2 // step
    c0 = a0[get_indices(s0) * step] @ p0[::step][:s0]
    
    ix = c0.argmax() * step
    a1 = a0[ix:]
    
    prev[0] = a1

    return ix


def resample_logx(data, num=None):
    
    assert data.ndim == 1
    
    num = num or data.size
    
    x1 = np.exp(np.linspace(0, np.log(data.size), num)) - 1
    x2 = x1.astype('long')
    
    xx = 1 - x1 + x2

    return data[x2] * xx + data[(x2 + 1).clip(0, data.size-1)] * (1 - xx)


def get_shadertoy_audio(
    amp=1., 
    length=512, 
    buffer=500, 
    data=None, 
    channel_time=None,
    correlate=True,
    resample='linear',
    ):
    
    if data is not None:
        a0 = data
        ct = channel_time

    else:
        l0 = length + buffer
        a0, ct = get_output_as_array(-l0, l0)[:2]
        
        if a0 is None:
            a0 = np.zeros((length, 2))

    if correlate:
        ix = get_correlation(a0.mean(-1), buffer)
        a0 = a0[ix:][:length]
            
    if channel_time is not None:
        ct = channel_time

    ft = np.fft.rfft(a0, axis=0)
    sa = ft.conj() * ft
    ps = 10 * np.log10(sa + 1e-6)
    
    if resample == 'linear':
        rs = scipy.signal.resample(ps, len(a0), domain='time')
    else:
        rs = resample_logx(ps, len(a0))
        
    ns = (rs + 50) / 100
    
    return np.stack((ns * 256, a0 * amp * 128 + 128)).clip(0, 255), ct


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
        width (float): The width of the shadertoy canvas.
        height (float): The height of the shadertoy canvas.
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
        width=800,
        height=450,
        x=0, 
        y=0,
        angle=0.0,
        anchor_x='left',
        anchor_y='bottom',
        color='white',
        name=None,
    ):
        """"""

        super().__init__(
            name,
            rotation=aa2q(glm.radians(angle)),
            position=glm.vec3(x, y, 0),
        )

        self.t0 = None
        self.ct = 0
        self.dt = 0

        self.iframe = 0

        w0, h0 = get_window_size()
        
        self.shader = load_shadertoy_program(code)
        self.shader._members['jpl_projection'].write(glm.ortho(
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

        self.width = width
        self.height = height
        self.components = 4
        
        self.color4 = glm.vec4(1., 1., 1., 1.)

        self.set_anchor(anchor_x, anchor_y)
        self.color = color

        self.tx0 = None
        self.tx1 = None
        self.fbo = None

    def __del__(self):
        self.release()

    def release(self):

        if self.tx0 is not None:

            self.tx0.release()
            self.tx0 = None

            self.tx1.release()
            self.tx1 = None

        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None

    def update(self, shader):
        pass

    def use(self, location):
        self.tx0.use(location=location)

    def render2buffer(self, ct, dt, iframe, width, height):

        if self.iframe == iframe:
            return

        ctx = get_context()
        fb0 = ctx.fbo

        if (self.width, self.height) != (width, height):

            self.release()

        if self.tx0 is None:

            self.width = width
            self.height = height

            self.tx0 = ctx.texture((int(self.width), int(self.height)), self.components, dtype='f4')        
            self.tx1 = ctx.texture((int(self.width), int(self.height)), self.components, dtype='f4')

        self.fbo is not None and self.fbo.release()
        self.fbo = ctx.framebuffer(color_attachments=[self.tx1])
        self.fbo.use()
        self.fbo.clear()

        self.shader._members['jpl_projection'].write(glm.ortho(
            0, self.width, 0, self.height, -1, 1
        ))

        self.render(ct, dt)

        self.tx0, self.tx1 = self.tx1, self.tx0

        fb0.use()

    def draw(self, ct, dt):
        """Render shadertoy to canvas - this is an alias to shadertoy.render()."""
        return self.render(ct, dt)
        
    def render(self, ct, dt):
        """Render shadertoy to canvas."""
        
        if self.t0 is None:
            self.t0 = ct

        self.ct = ct - self.t0
        self.dt = dt

        self.iframe += 1

        if self._dirty:
            self.update(self.shader)

        for i in [0, 1, 2, 3]:
            ch = getattr(self, 'channel%s' % i, None)
            if isinstance(ch, Shadertoy):
                ch.render2buffer(ct, dt, self.iframe, self.width, self.height)

        self.shader._members['jpl_model'].write(self.matrix)
        self.shader._members['jpl_components'].value = self.components
        self.shader._members['jpl_color'].write(self.color4)

        if 'iResolution' in self.shader._members:
            self.shader._members['iResolution'].write(self.scale)
        
        if 'iTime' in self.shader._members:
            self.shader._members['iTime'].value = self.ct
        
        if 'iTimeDelta' in self.shader._members:
            self.shader._members['iTimeDelta'].value = dt

        if 'iFrame' in self.shader._members:
            self.shader._members['iFrame'].value = self.iframe

        for i in [0, 1, 2, 3]:
            
            ch = getattr(self, 'channel%s' % i, None)
            if ch is None:
                continue

            if 'iChannel%s' % i in self.shader._members:
                self.shader._members['iChannel%s' % i].value = SPRITE_TEXTURE_UNIT + i
                ch.use(location=SPRITE_TEXTURE_UNIT+i)

            if 'iChannelTime[%s]' % i in self.shader._members:
                self.shader._members['iChannelTime[%s]' % i].value = self.channeltime[i]

            if 'iChannelResolution[%s]' % i in self.shader._members:
                self.shader._members['iChannelResolution[%s]' % i].write(
                    glm.vec3(ch.width, ch.height, ch.components)
                )

        if 'iDate' in self.shader._members:
            dt = datetime.datetime.now()
            self.shader._members['iDate'].write(glm.vec4(dt.year, dt.month, dt.day, time.time()))

        if 'iSampleRate' in self.shader._members:
            self.shader._members['iSampleRate'].value = FPS

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
        """Set the anchor point of the shadertoy canvas.

        The anchor is a point in the shadertoy canvas that is used for 
        rotation and positioning. Imagine a pin going through the canvas and 
        that you use this pin to position the canvas and to rotate it. The
        point at which the pin goes through the canvas is the anchor point.

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
        channel, 
        data,
        channeltime=None,
        ):

        assert channel in [0, 1, 2, 3]

        if isinstance(data, Shadertoy):
            if data is self:
                data = weakref.proxy(data)
            setattr(self, 'channel%s' % channel, data)
            return

        if channeltime:
            self.channeltime[channel] = channeltime

        channel = 'channel%s' % channel

        texture = getattr(self, channel, None)
        if texture is not None:
            texture.release()
            
        texture = load_texture(
            data,
            anisotropy=1., 
            autocrop=False,
            mipmap=False, 
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

