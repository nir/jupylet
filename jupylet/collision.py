"""
    jupylet/collision.py
    
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
import glm

import PIL.Image
import scipy.signal

import numpy as np

from .resource import pil_resize_to


MAP_SIZE = 128


def affine(a=0, s=1, ax=0, ay=0, dx=0, dy=0):
    
    r = math.radians(a)
    a = math.cos(r) * s
    b = math.sin(r) * s
    
    return np.array([
        [a, b, -a*ax - b*ay + dx],
        [-b, a, b*ax - a*ay + dy],
        [0, 0, 1]
    ])


"""
@functools.lru_cache(1024)
def trbl0(width, height, anchor_x=0, anchor_y=0, angle=0, scale=1):
    
    bb0 = np.array([[width, height, 1], [width, 0, 1], [0, 0, 1], [0, height, 1]])
    bb1 = affine0(angle, scale, anchor_x, anchor_y).dot(bb0.T).T
    bb2 = bb1.tolist()
    
    x = [v[0] for v in bb2]
    y = [v[1] for v in bb2]
    
    t, r, b, l = max(y), max(x), min(y), min(x)
    
    return t, r, b, l
"""

def affine0(a=0, s=1, ax=0, ay=0, dx=0, dy=0):
    
    r = math.radians(a)
    a = math.cos(r) * s
    b = math.sin(r) * s
    
    return glm.mat3(
        a, b, -a*ax - b*ay + dx,
        -b, a, b*ax - a*ay + dy,
        0, 0, 1
    )


@functools.lru_cache(1024)
def trbl(width, height, anchor_x=0, anchor_y=0, angle=0, scale=1):
    
    bb0 = glm.mat4x3(
        width, height, 1, 
        width, 0, 1, 
        0, 0, 1, 
        0, height, 1
    )
    
    bb1 = glm.transpose(bb0) * affine0(angle, scale, anchor_x, anchor_y) 
    
    x = bb1[0]
    y = bb1[1]
    
    t, r, b, l = max(y), max(x), min(y), min(x)
    
    return t, r, b, l


def hitmap_and_outline_from_alpha(im):

    if isinstance(im, np.ndarray):
        assert max(im.shape[:2]) == MAP_SIZE, 'Maximum size of array expected to be %s.' % MAP_SIZE
    else:
        im = pil_resize_to(im, MAP_SIZE)

    a0 = np.array(im)
    
    if len(a0.shape) == 3:
        a0 = a0[...,-1]

    k0 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    
    a2 = (a0 > 128).astype('uint8')[::-1]
    a3 = scipy.signal.convolve2d(a2, k0, 'same') != 4
    a4 = a2 * a3
    a5 = np.stack(a4.nonzero()[::-1], -1)
    a6 = np.concatenate([a5, a5[:,:1] * 0 + 1], -1)
    
    h1, w1 = a2.shape
    xx = max(a2.shape) + 2

    a7 = np.pad(a2, ((1, xx - h1 - 1), (1, xx - w1 - 1)))

    return a7, a6


def collisions_from_hitmap_and_outline(a0, a1):
    a1 = np.core.umath.clip(a1, 0, a0.shape[0]-1)
    a2 = a0[a1[:,1], a1[:,0]]
    a3 = a1[a2.nonzero()]
    return a3


def compute_collisions(o0, o1, debug=False):

    s0 = max(o0.height, o0.width) / MAP_SIZE
    s1 = max(o1.height, o1.width) / MAP_SIZE

    dr = o1.angle - o0.angle
    r0 = o1.angle * math.pi / 180 * -1

    dy0 = o0.y - o1.y
    dx0 = o0.x - o1.x

    dx1 = math.cos(r0) * dx0 - math.sin(r0) * dy0
    dy1 = math.cos(r0) * dy0 + math.sin(r0) * dx0

    dy2 = (dy1 + 1 + o1.anchor.y * o1.height) / s1
    dx2 = (dx1 + 1 + o1.anchor.x * o1.width) / s1

    af = affine(
        dr, 
        s0 / s1, 
        o0.anchor.x * o0.width / s0, 
        o0.anchor.y * o0.height / s0, dx2, 
        dy2
    )

    oo = af.dot(o0.outline.T)[:2].T.astype('int64')
    cl = collisions_from_hitmap_and_outline(o1.hitmap, oo)

    if not debug:
        return cl - (o1.anchor.x * o1.width / s1, o1.anchor.y * o1.height / s1)

    #
    # Use the following code to display debug output in a jupyter notebook:
    #
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    # ax.set_aspect(1.)
    # _ = ax.plot(hm[:,1], hm[:,0], 'y.', oo[:,0], oo[:,1], 'r.', cl[:,0], cl[:,1], 'bo')
    #

    hm = np.argwhere(o1.hitmap > 0)
    return cl, oo, hm

