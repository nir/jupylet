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
import pyglet
import math
import io
import os

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

import moderngl_window as mglw
import numpy as np

from moderngl_window.meta import DataDescription

from .sprite import Sprite
from .state import State


def rtl(s):
    return str(s[::-1])


#
# Note: Find free fonts at www.fontsquirrel.com
#

@functools.lru_cache(maxsize=32)
def load_font(path, size):

    ff = mglw.resources.data.load(DataDescription(path=path, kind='binary'))
    return PIL.ImageFont.truetype(io.BytesIO(ff), size)


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
    bl = draw_chr('a', path, size).shape[0]

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

    bl = mh - bl  
    mh = hh + mh
    mw = max(mw, ww)
    a0 = np.zeros((mh, mw), dtype='uint8')

    for a, (hh, ww) in al:
        
        h, w = a.shape
        a0[hh:hh+h, ww:ww+w] = a
        
    return a0, bl


class Label(Sprite):
    
    def __init__(
        self, 
        text='',
        font_path='fonts/SourceSerifPro-Bold.otf', 
        font_size=16,
        line_height=1.2,
        bold=False, 
        italic=False,
        color='white',
        x=0, 
        y=0, 
        width=None, 
        height=None,
        anchor_x='left', 
        anchor_y='baseline',
        align='left'
    ):
                
        image, baseline = draw_str(text, font_path, font_size, line_height)

        super(Label, self).__init__(
            image,
            x, 
            y, 
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            height=height,
            width=width,
            collisions=False,
        )

        self._items = dict(
            text = text,
            font_path = font_path,
            font_size = font_size,
            line_height = line_height,
        )

        self.baseline = baseline / self.texture.height

        self.color = color

    def update(self, shader):

        if self._dirty:
            self._dirty.clear()
            
            self.image, baseline = draw_str(
                self.text, 
                self.font_path, 
                self.font_size, 
                self.line_height
            )

            self.baseline = baseline / self.texture.height

    def get_state(self):
        return dict(
            text = self.text,
            font_path = self.font_path,
            font_size = self.font_size,
            line_height = self.line_height,
            sprite = super(Label, self).get_state(),
        )

    def set_state(self, s):
        
        for k, v in s.items():
            if k == 'sprite':
                super(Label, self).set_state(v)
            else:
                setattr(self, k, v)

