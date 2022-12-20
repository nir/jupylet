"""
    jupylet/resource.py
    
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


import moderngl
import pathlib
import glob
import io
import os
import re

import PIL.Image

import numpy as np

import moderngl_window as mglw

import moderngl_window.loaders.texture.cube

from moderngl_window.meta import TextureDescription, DataDescription
from moderngl_window.conf import settings


_context = None


def set_context(context):

    global _context
    _context = context


def get_context():

    assert _context is not None, 'First create an App() instance!'
    return _context


SHADER_2D = 'shader_2d'
SHADER_3D = 'shader_3d'

_shaders = {
    SHADER_2D: None,
    SHADER_3D: None,
}


def set_shader_3d(shader):
    _shaders[SHADER_3D] = shader
    return shader
    

def get_shader_3d():
    return _shaders[SHADER_3D]


def set_shader_2d(shader):
    _shaders[SHADER_2D] = shader
    return shader
    

def get_shader_2d():
    return _shaders[SHADER_2D]


_dirs = set()
_regd = set()


def register_dir(path, relative_to=''):

    p0 = os.path.join(relative_to, path)
    pp = pathlib.Path(p0).absolute()
    mglw.resources.register_dir(pp)

    _regd.add(str(pp))


def unresolve_path(path):
    """Attempt to unresolve absolute path back to reource relative."""

    path = str(pathlib.Path(path))

    for p in _dirs:
        if path.startswith(p):
            return path[len(p):].lstrip('\\/')


def find_path(path, throw=True):

    assert _regd, "No resource path has been registered yet. Did you remember to create an App() instance?"
    
    dd = DataDescription(path=path, kind='binary')
    mglw.resources.data.resolve_loader(dd)
    pp = dd.loader_cls(dd).find_data(dd.path)
    
    if pp is None and throw:
        raise IOError("Path %r not found." % path)

    if pp is not None:
        p0 = pathlib.Path(path)
        p1 = str(pp)[: -len(str(p0))]
        _dirs.add(p1)

    return pp


def find_glob_path(path):
    
    dirname, basename = os.path.split(path)
    dirname = find_path(dirname)
    path = os.path.join(dirname, basename)

    return glob.glob(path)


class CubeLoader0(moderngl_window.loaders.texture.cube.Loader):

    def _load_texture(self, path):

        switch = self.meta._kwargs['flip_left_right'] and path in [self.meta.pos_y, self.meta.neg_y]
        flip_key = 'flip_y' if 'flip_y' in self.meta._kwargs else 'flip'

        if switch:
            self.meta._kwargs[flip_key], self.meta._kwargs['flip_left_right'] = self.meta._kwargs['flip_left_right'], self.meta._kwargs[flip_key]

        #print(repr(self.meta._kwargs))
        image = super(CubeLoader0, self)._load_texture(path)
        if self.meta._kwargs.get('flip_left_right'):
            image = image.transpose(PIL.Image.Transpose.FLIP_LEFT_RIGHT)

        if switch:
            self.meta._kwargs[flip_key], self.meta._kwargs['flip_left_right'] = self.meta._kwargs['flip_left_right'], self.meta._kwargs[flip_key]

        return image


def _init_loaders():

    loaders = settings.TEXTURE_LOADERS
    if any('CubeLoader0' in l for l in loaders):
        return

    loaders[:] = [l for l in loaders if 'cube' not in l]
    loaders.append('moderngl_window.loaders.texture.cube.CubeLoader0')

    moderngl_window.loaders.texture.cube.CubeLoader0 = CubeLoader0


def load_texture_cube(
    path,
    flip=False,
    mipmap=False,
    anisotropy=1.0,
    flip_left_right=False,
    **kwargs
) -> moderngl.TextureCube:
    """Loads a texture cube.

    Keyword Args:
        path (str): Glob pattern path to images for each face of the cubemap.
        mipmap (bool): Generate mipmaps. Will generate max possible levels unless
                        `mipmap_levels` is defined.
        anisotropy (float): Number of samples for anisotropic filtering
        **kwargs: Additional parameters to TextureDescription
    Returns:
        moderngl.TextureCube: Texture instance
    """

    _init_loaders()

    paths = find_glob_path(path)
    paths = [unresolve_path(p) for p in paths]

    k2k = dict(
        RT = 'pos_x', posx = 'pos_x',
        UP = 'pos_y', posy = 'pos_y',
        BK = 'pos_z', posz = 'pos_z',
        LF = 'neg_x', negx = 'neg_x',
        DN = 'neg_y', negy = 'neg_y',
        FT = 'neg_z', negz = 'neg_z',    
    )

    ptn = r'^(.*?(BK|DN|FT|LF|RT|UP|pos[xyz]|neg[xyz])[^\\/]*)$'
    f2p = [re.match(ptn, p).groups() for p in paths]
    f2p = {k2k[k]:v for v, k in f2p}

    return mglw.resources.textures.load(TextureDescription(
        pos_x=f2p['pos_x'],
        pos_y=f2p['pos_y'],
        pos_z=f2p['pos_z'],
        neg_x=f2p['neg_x'],
        neg_y=f2p['neg_y'],
        neg_z=f2p['neg_z'],
        flip=flip,
        flip_left_right=flip_left_right,
        mipmap=mipmap,
        anisotropy=anisotropy,
        kind='cube',
        **kwargs,
    ))


def load_texture(
    o, 
    anisotropy=8.0, 
    autocrop=False,
    mipmap=True, 
    flip=False, 
):

    return mglw.resources.textures.load(
        TextureDescription(
            path=None, 
            flip=flip, 
            mipmap=mipmap, 
            anisotropy=anisotropy, 
            image=load_image(o, autocrop)
        )
    )


def load_image(
    o, 
    autocrop=False,
    flip=False, 
    copy=True,
):
    
    if type(o) is str:
        d0 = mglw.resources.data.load(DataDescription(path=o, kind='binary'))
        im = PIL.Image.open(io.BytesIO(d0))
        im.load()

    if isinstance(o, np.ndarray):
        im = PIL.Image.fromarray(o.real.astype('uint8'))

    if isinstance(o, PIL.Image.Image):
        if copy and not autocrop and not flip:
            im = o.copy()
        else:
            im = o
    
    if autocrop:
        im = pil_autocrop(im)

    if flip:
        im = im.transpose(PIL.Image.Transpose.FLIP_TOP_BOTTOM)

    return im


def pil_from_texture_array(ta, layer=0):

    b0 = ta.read()
    nl = len(b0) // ta.layers
    b1 = b0[layer * nl: layer * nl + nl]
    
    return PIL.Image.frombuffer(
        {1: 'L', 3: 'RGB', 4: 'RGBA'}[ta.components], 
        (ta.width, ta.height), 
        b1
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

