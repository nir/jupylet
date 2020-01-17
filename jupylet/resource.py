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


import pyglet

import PIL.Image

import numpy as np

from pyglet.image.codecs import ImageDecodeException


__all__ = ['image_from', 'image']


def image_from(o, autocrop=False):
    
    if type(o) is str:
        return image_from_resource(o, autocrop)
    
    if isinstance(o, np.ndarray):
        return image_from_array(o)
   
    if isinstance(o, PIL.Image.Image):
        return image_from_pil(o)
    
    return o

    
def image_from_resource(name, autocrop=False):
    
    try:
        return image(name, autocrop=autocrop)

    except pyglet.resource.ResourceNotFoundException:
        _loader.reindex()
        return image(name, autocrop=autocrop)

    
def image_from_array(array):
    
    array = array.astype('uint8')

    height, width = array.shape[:2]
    
    if len(array.shape) == 2:
        channels = 1
    else:
        channels = array.shape[-1]
        
    mode = ['L', None, 'RGB', 'RGBA'][channels-1]
        
    return pyglet.image.ImageData(width, height, mode, array.reshape(-1).tobytes(), -width*channels)


def image_from_pil(image):
    
    width, height = image.size 
    channels = {'L': 1, 'RGB': 3, 'RGBA': 4}[image.mode]
        
    return pyglet.image.ImageData(width, height, image.mode, image.tobytes(), -width*channels)


def image(name, flip_x=False, flip_y=False, rotate=0, atlas=True, autocrop=False):
    """Load an image with optional transformation.

    This is similar to `texture`, except the resulting image will be
    packed into a :py:class:`~pyglet.image.atlas.TextureBin` if it is an appropriate size for packing.
    This is more efficient than loading images into separate textures.

    :Parameters:
        `name` : str
            Filename of the image source to load.
        `flip_x` : bool
            If True, the returned image will be flipped horizontally.
        `flip_y` : bool
            If True, the returned image will be flipped vertically.
        `rotate` : int
            The returned image will be rotated clockwise by the given
            number of degrees (a multiple of 90).
        `atlas` : bool
            If True, the image will be loaded into an atlas managed by
            pyglet. If atlas loading is not appropriate for specific
            texturing reasons (e.g. border control is required) then set
            this argument to False.

    :rtype: `Texture`
    :return: A complete texture if the image is large or not in an atlas,
        otherwise a :py:class:`~pyglet.image.TextureRegion` of a texture atlas.
    """
    _loader._require_index()
    if name in _loader._cached_images:
        identity = _loader._cached_images[name]
    else:
        identity = _loader._cached_images[name] = _alloc_image(name, atlas=atlas, autocrop=autocrop)

    if not rotate and not flip_x and not flip_y:
        return identity

    return identity.get_transform(flip_x, flip_y, rotate)


def _alloc_image(name, atlas=True, autocrop=False):

    file = _loader.file(name)
    try:
        img = _pil_load(name, file=file, autocrop=autocrop)
    finally:
        file.close()

    if not atlas:
        return img.get_texture(True)

    # find an atlas suitable for the image
    bin = _loader._get_texture_atlas_bin(img.width, img.height)
    if bin is None:
        return img.get_texture(True)

    return bin.add(img)


def _pil_load(filename, file, autocrop=False):

    try:
        image = PIL.Image.open(file)
    except Exception as e:
        raise ImageDecodeException(
            'PIL cannot read %r: %s' % (filename or file, e))

    try:
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    except Exception as e:
        raise ImageDecodeException('PIL failed to transpose %r: %s' % (filename or file, e))

    # Convert bitmap and palette images to component
    if image.mode in ('1', 'P'):
        image = image.convert()

    if image.mode not in ('L', 'LA', 'RGB', 'RGBA'):
        raise ImageDecodeException('Unsupported mode "%s"' % image.mode)

    if autocrop:
        image = image.crop(image.getbbox())
        
    width, height = image.size

    # tostring is deprecated, replaced by tobytes in Pillow (PIL fork)
    # (1.1.7) PIL still uses it
    image_data_fn = getattr(image, "tobytes", getattr(image, "tostring"))
    return pyglet.image.ImageData(width, height, image.mode, image_data_fn())


_loader = pyglet.resource._default_loader

