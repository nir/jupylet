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
import os

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

import numpy as np

from .sprite import Sprite
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


class Label(Sprite):
    
    def __init__(
        self, 
        text='',
        font_name=None, 
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
        anchor_y='bottom',
        align='left'
    ):
                
        img = draw_str(text, font_name, font_size, line_height)

        super(Label, self).__init__(
            img,
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
            font_name = font_name,
            font_size = font_size,
            line_height = line_height,
        )

    def update(self, *args):

        if self._dirty:
            self._dirty.clear()
            
            self.image = draw_str(
                self.text, 
                self.font_name, 
                self.font_size, 
                self.line_height
            )

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

