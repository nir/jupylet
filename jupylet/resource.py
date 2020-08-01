"""
    jupylet/resource.py
    
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
import io

import PIL.Image

import numpy as np

import moderngl_window as mglw

from moderngl_window.meta import TextureDescription, DataDescription


SHADER_2D = 'shader_2d'
SHADER_3D = 'shader_3d'

_shaders = {}


def set_shader_2d(shader):
    _shaders[SHADER_2D] = shader
    return shader
    

def get_shader_2d():
    return _shaders[SHADER_2D]


def register_dir(path):
    mglw.resources.register_dir(pathlib.Path(path).absolute())


def texture_load(
    o, 
    anisotropy=8.0, 
    autocrop=False,
    mipmap=True, 
    flip=False, 
):
    
    if type(o) is str:
        d0 = mglw.resources.data.load(DataDescription(path=o, kind='binary'))
        im = PIL.Image.open(io.BytesIO(d0))
        im.load()

    if isinstance(o, np.ndarray):
        im = PIL.Image.fromarray(o.astype('uint8'))

    if isinstance(o, PIL.Image.Image):
        im = o
    
    if autocrop:
        pil_autocrop(im)

    return mglw.resources.textures.load(
        TextureDescription(
            path=None, 
            flip=flip, 
            mipmap=mipmap, 
            anisotropy=anisotropy, 
            image=im
        )
    )

    
def pil_from_texture(t):

    return PIL.Image.frombuffer(
        {1: 'L', 3: 'RGB', 4: 'RGBA'}[t.components], 
        (t.width, t.height), 
        t.read()
    )


def pil_autocrop(im):
    return im.crop(im.getbbox())


def pil_resize_to(im, size=128, resample=PIL.Image.BILINEAR):

    w0, h0 = im.size
    wh = max(w0, h0)
    sr = size / wh

    w1 = round(sr * w0)
    h1 = round(sr * h0)

    return im.resize((w1, h1), resample=resample)

