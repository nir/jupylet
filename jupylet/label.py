"""
    jupylet/label.py
    
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


import functools
import webcolors
import pyglet
import math
import os

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

import numpy as np

from .color import color2rgb
from .state import State


def rtl(s):
    return str(s[::-1])


@functools.lru_cache(maxsize=32)
def load_font(path, size):
    return PIL.ImageFont.truetype(path, size)


@functools.lru_cache(maxsize=2048)
def draw_chr(c, path, size):
    
    font = load_font(path, size)
    w, h = font.getsize(c)
    
    im = PIL.Image.new('L', (w, h))
    di = PIL.ImageDraw.Draw(im)
    di.text((0, 0), c, fill='white', font=font)
    
    return np.array(im)


def draw_str(s, path, size, line_height=1.2):
    
    al = []

    lh = math.ceil(size * line_height)

    hh = 0
    ww = 0
    
    mh = 0
    mw = 0
    
    for c in s.rstrip():
        if c == '\n':
            hh += lh
            mw = max(mw, ww)
            ww = 0
            mh = 0
            continue
            
        a = draw_chr(c, path, size)
        al.append((a, (hh, ww)))
        
        h, w = a.shape
        
        mh = max(mh, h)
        ww += w
        
    mh = hh + mh
    mw = max(mw, ww)
    a0 = np.zeros((mh, mw), dtype='uint8')

    for a, (hh, ww) in al:
        
        h, w = a.shape
        a0[hh:hh+h, ww:ww+w] = a
        
    return a0


class Label(pyglet.text.Label):
    
    def __init__(self, text='',
                 font_name=None, font_size=16, bold=False, italic=False,
                 color='white',
                 x=10, y=10, width=None, height=None,
                 anchor_x='left', anchor_y='baseline',
                 align='left',
                 multiline=False, dpi=None, batch=None, group=None):
                
        if type(color) == str:
            color = color2rgb(color)
                    
        super(Label, self).__init__(text,
                 font_name, font_size, bold, italic,
                 color,
                 x, y, width, height,
                 anchor_x, anchor_y,
                 align,
                 multiline, dpi, batch, group)
        
    @property
    def alpha(self):
        return self.color[-1]
        
    @alpha.setter
    def alpha(self, alpha):
        self.color = self.color[:3] + (alpha,)
        
    @property
    def color_name(self):
        return webcolors.rgb_to_name(self.color[:3])

    @property
    def color(self):
        """Text color.
        Color is a 4-tuple of RGBA components, each in range [0, 255].
        :type: (int, int, int, int)
        """
        return self.document.get_style('color')

    @color.setter
    def color(self, color):
        
        if type(color) == str:
            color = color2rgb(color)
        else:
            color = tuple(int(v) for v in color)

        self.document.set_style(0, len(self.document.text), {'color': color})

    def get_state(self):
        
        return State(
            x = self.x,
            y = self.y,
            text = self.text,
            color = self.color,
        )

    def set_state(self, s):
        
        self.x = s.x
        self.y = s.y
        self.text = s.text
        self.color = s.color

