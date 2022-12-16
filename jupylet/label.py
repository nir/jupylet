"""
    jupylet/label.py
    
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


def draw_str(s, path, size, line_height=1.2, align='left'):
    
    al = []
    ll = []

    # Compute line height and baseline height.
    lh = math.ceil(size * line_height)
    bl = draw_chr('a', path, size).shape[0]

    # Coordinates for top-left position for each char.
    hh = 0
    ww = 0
    
    # Maximum accumulated width and height for label. 
    mh = 0
    mw = 0
    
    for c in s.rstrip():
        if c == '\n':
            ll.append(ww)
            hh += lh
            mw = max(mw, ww)
            ww = 0
            mh = 0
            continue
            
        ca = draw_chr(c, path, size)
        al.append((ca, (hh, ww), len(ll)))
        
        h, w = ca.shape
        
        mh = max(mh, h)
        ww += w

    ll.append(ww)

    # Compute final baseline, maximum width and height for label.
    bl = mh - bl  
    mh = hh + mh
    mw = max(mw, ww)

    a0 = np.zeros((mh, mw), dtype='uint8')

    aw = {'left': 0, 'center': 0.5, 'right': 1}[align]

    for ca, (hh, ww), li in al:
        
        a = int((mw - ll[li]) * aw)
        h, w = ca.shape
        a0[hh:hh+h, ww+a:ww+a+w] = ca
        
    return a0, bl


class Label(Sprite):

    """A text label.

    Since a text label is actually implemented as a 2D sprite, it has all the
    functionality and methods of a Sprite.

    Args:
        text (str): text to render as label.
        font (path): path to a true type or open type font.
        font_size (float): font size to use. 
        line_height (float): determines the distance between lines.
        align (str): the desired alignment for the text label. May be one
            of 'left', 'center', and 'right'.
        color (str or 3-tuple): a color name, color hex notation, or a 
            3-tuple. specifying the color for the text label.
        x (float): the x position for the label.
        y (float): the y position for the label.
        angle (float): clockwise rotation of the label in degrees.
        anchor_x (float or str): either 'left', 'center' or 'right' or a 
            value between 0.0 (for left) and 1.0 (for right) indicating
            the anchor point inside the label along its x axis.
        anchor_y (float or str): either 'bottom', 'baseline', 'center' or 
            'top' or a value between 0.0 (for bottom) and 1.0 (for top) 
            indicating the anchor point inside the label along its y axis.
    """                

    def __init__(
        self, 
        text='',
        font_path='fonts/SourceSerifPro-Bold.otf', 
        font_size=16,
        line_height=1.2,
        align='left',
        bold=False, 
        italic=False,
        color='white',
        x=0, 
        y=0, 
        angle=0.0,
        width=None, 
        height=None,
        anchor_x='left', 
        anchor_y='baseline',
    ):
        """"""

        image, baseline = draw_str(text, font_path, font_size, line_height, align)

        super().__init__(
            image,
            x, 
            y, 
            angle=angle,
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
            align = align,
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
                self.line_height,
                self.align,
            )

            self.baseline = baseline / self.texture.height

            if self._ay == 'baseline':
                self.anchor.y = self.baseline

    def get_state(self):
        return dict(
            sprite = super().get_state(),
            text = self.text,
            font_path = self.font_path,
            font_size = self.font_size,
            line_height = self.line_height,
            align = self.align,
        )

    def set_state(self, s):
        
        super().set_state(s['sprite'])

        for k, v in s.items():
            if k != 'sprite':
                setattr(self, k, v)

