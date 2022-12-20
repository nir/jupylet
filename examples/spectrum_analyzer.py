"""
    examples/spectrum_analyzer.py
    
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
import _thread
import json
import math
import sys
import io
import os

sys.path.insert(0, os.path.abspath('./..'))

import jupylet.color

from jupylet.app import App
from jupylet.state import State
from jupylet.label import Label
from jupylet.sprite import Sprite
from jupylet.shadertoy import Shadertoy, get_shadertoy_audio

from jupylet.audio.bundle import *

import numpy as np

from scipy.interpolate import interp1d

import matplotlib
import matplotlib.pyplot as plt


STATE_PATH = 'spectrum_analyzer.state'

state = State(
    
    up = False,
    down = False,
    left = False,
    right = False,
    shift = False,
    
    decay = 0.8,
    samples = 4096, 
    
    xmin = 100,
    xmax = 19500,
    ymin = -75,
    ymax = 100,
)


dl = sd.query_devices()
idi = sd.default.device['input']
sample_rate = dl[idi]['default_samplerate']


def implot(*args, xscale='log', xmin=None, xmax=None, ymin=None, ymax=None, figsize=(10, 5), **kwargs):
    
    buf = io.BytesIO()
    
    fig = plt.figure(figsize=figsize, dpi=100)
    
    ax0 = fig.add_subplot(111)
    ax0.grid(True, which='both')
    ax0.set_xscale(xscale)
    
    ax0.set_xlabel('Frequency (Hz)')
    ax0.set_ylabel('Decibel (dB)')

    if xscale == 'log':
        ax0.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
        ax0.get_xaxis().set_minor_formatter(matplotlib.ticker.LogFormatter(minor_thresholds=(10, 0)))
    
    if xmin or xmax:
        ax0.set_xlim(xmin=xmin, xmax=xmax)

    if ymin or ymax:
        ax0.set_ylim(ymin=ymin, ymax=ymax)

    pl0 = ax0.plot(*args, **kwargs)[0] # Returns a tuple of line objects, thus the comma
    
    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close(fig)

    return PIL.Image.open(buf)


@functools.lru_cache(maxsize=64)
def get_plot_frame(    
    xmin=0, xmax=1000, 
    ymin=-50, ymax=50, 
    figsize=(12, 6), 
    cropx=None,
    rgba=False
):
    
    xx = np.arange(xmin, xmax)
    im = implot(
        xx, xx - 1e6, 
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        figsize=figsize
    )
    
    if cropx:
        im = im.crop((0, 0, cropx, im.size[-1]))
        
    if not rgba:
        return im
    
    a0 = np.array(im)
    a1 = np.concatenate((a0 * 0, 255 - a0[...,:1]), -1)

    return PIL.Image.fromarray(a1)


def get_plot_frame0(cropx=None):
    
    return get_plot_frame(
        xmin=state.xmin, xmax=state.xmax, 
        ymin=state.ymin, ymax=state.ymax, 
        figsize=(12, 6),
        cropx=cropx,
        rgba=True,
    )


w, h = get_plot_frame0().size

app = App(width=w, height=h, quality=100)#, log_level=logging.INFO)

plot_frame = Sprite(get_plot_frame0(), x=w, anchor_x='right', anchor_y='bottom', collisions=False)
label0 = Label('Use ← ↑ ↓ → SHIFT and SPACE to control the display', x=74, y=60, color='red')


# The code in the following cell is of a simple shadertoy shader that displays an 
# audio spectrum. [Shadertoy shaders](http://shadertoy.com/) are an easy way to 
# create graphic effects by programming the GPU directly:

st0 = Shadertoy("""

    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        // Normalized pixel coordinates (from 0 to 1)
        vec2 uv = fragCoord / iResolution.xy;

        // Time varying pixel color
        vec3 col = vec3(1., 1., 1.);

        float dst = 1000.;
        float dx0 = 0.00005;
        float uvx = max(0, min(1, (uv.x - 0.07) / (0.99 - 0.07)));
        
        for (int i=0; i < 50; i++) {
        
            float dx1 = dx0 * (i - 25);
            float dy1 = texture(iChannel0, vec2(uvx + dx1, 0.)).r - uv.y; 
            float dxy = dx1 * dx1 + dy1 * dy1;
            
            if (dxy < dst) {
                dst = dxy;
            }
        }
        
        vec3 sig = vec3(0.00033 / max(32 * dst, 1e-6));

        sig *= vec3(1., 1., 1.);

        if (uv.x < 0.07) {
            sig *= 0;
        }
        
        if (uv.x > 0.99) {
            sig *= 0;
        }
        
        col -= sig;

        // Output to screen
        fragColor = vec4(col,1.0);
    }
    
""", w, h, 0, h, 0, 'left', 'top')


def clip(a, low=-math.inf, high=math.inf):
    return min(max(a, low), high)


XMIN = 60
XMAX = 19500


def trans_x(tx):
    
    m0 = clip(XMIN / state.xmin, 0, 1)
    m1 = clip(XMAX / state.xmax, 1, 2)
    
    state.xmin *= clip(tx, m0, m1) 
    state.xmax *= clip(tx, m0, m1) 
    
    
def scale_x(sx):
    
    mid0 = (state.xmax + state.xmin) / 2 - 500
    mid1 = (state.xmax + state.xmin) / 2 + 500
    
    m0 = clip(XMIN / state.xmin, 0, 1)
    m1 = clip(mid0 / state.xmin, 1, 2)
    m2 = clip(mid1 / state.xmax, 0, 1)
    m3 = clip(XMAX / state.xmax, 1, 2)
    
    state.xmin *= clip(sx, m0, m1) 
    state.xmax *= clip(1/sx, m2, m3) 
    
    
YMIN = -90
YMAX = 150


def trans_y(ty):
    
    m0 = clip(YMIN - state.ymin, -99, 0)
    m1 = clip(YMAX - state.ymax, 0, 100)
    
    hh = state.ymax - state.ymin
    
    state.ymin += clip(ty, m0, m1) * hh / 16
    state.ymax += clip(ty, m0, m1) * hh / 16
    
    
def scale_y(sy):
    
    mid0 = (state.ymax + state.ymin) / 2 - 15
    mid1 = (state.ymax + state.ymin) / 2 + 15
    diff = (state.ymax - state.ymin)
    
    m0 = clip(YMIN - state.ymin, -99, 0)
    m1 = clip(mid0 - state.ymin, 0, 100)
    m2 = clip(mid1 - state.ymax, -99, 0)
    m3 = clip(YMAX - state.ymax, 0, 100)
    
    state.ymin += clip(-sy * diff / 16, m0, m1)
    state.ymax += clip(sy * diff / 16, m2, m3) 
    
    
def reset_xy():
    
    state.xmin = 100
    state.xmax = 19500
    
    state.ymin = -75
    state.ymax = 100


@app.event
def key_event(key, action, modifiers):
    logger.info('Enter key_event(key=%r, action=%r, modifiers=%r).', key, action, modifiers)
    
    keys = app.window.keys

    value = action == keys.ACTION_PRESS
    
    state.shift = modifiers.shift
        
    if key == keys.SPACE:
        reset_xy()
        
    if key == keys.UP:
        state.up = value
        
    if key == keys.DOWN:
        state.down = value
        
    if key == keys.LEFT:
        state.left = value
        
    if key == keys.RIGHT:
        state.right = value


@app.run_me_every(1/10)
def modify_display(ct, dt):
    
    d = -1 if state.shift else 1
    s = 2 ** dt
        
    if  state.shift:
        
        if state.up:
            scale_y(-s)

        if state.down:
            scale_y(s)

        if state.right:
            scale_x(s)

        if state.left:
            scale_x(1 / s)

    else:
        
        if state.left:
            trans_x(1 / s)

        if state.right:
            trans_x(s)

        if state.down:
            trans_y(s)

        if state.up:
            trans_y(-s)


def resample_logx(data, num=None):
    
    assert data.ndim == 1
    
    num = num or data.size
    
    x1 = np.exp(np.linspace(0, np.log(data.size), num)) - 1
    x2 = x1.astype('long')
    
    xx = 1 - x1 + x2

    return data[x2] * xx + data[(x2 + 1).clip(0, data.size-1)] * (1 - xx)


def resample_logx2(data, num=None):
    
    assert data.ndim == 1
    
    num = num or data.size
    
    x1 = np.exp(np.linspace(0, np.log(data.size), num)) - 1
    x2 = x1.astype('long')
    
    xx = 1 - x1 + x2
    xx = data[x2] * xx + data[(x2 + 1).clip(0, data.size-1)] * (1 - xx)
    
    idx3 = ((x2[1:] - x2[:-1]) < 2).sum()
    
    if idx3 >= len(x2) - 16:
        return xx
    
    x3 = x2[idx3:]
    
    p0 = np.pad(x3[1:] - x3[:-1], (1, 1)) / 2

    xa = np.ceil(x3 - p0[:-1])[:,None].astype('long')
    xb = np.ceil(x3 + p0[1: ])[:,None].astype('long')

    md = (xb - xa).max()
    ar = np.arange(md)[None,:] / (md - 1)
    
    xe = (xb - xa) * ar + xa
    xi = xe.astype('long')

    xz = np.take(data, xi).max(-1)
    
    xx[idx3:] = xz
    
    return xx


data0 = []
data1 = None
dataz = None


def callback(indata, frames, time, status):

    global data0, data1, dataz

    data0.append(indata[:,0].astype('float'))

    if len(data0) > 24:
        data0.pop(0)

    datax = np.concatenate(data0)[-state.samples:]
        
    a0 = np.fft.rfft(datax)
    a5 = a0.conj() * a0
    a6 = 5 * np.log(a5.real + 1e-6)

    if dataz is None or dataz.shape != a6.shape:
        dataz = a6
    else:
        dataz = dataz * state.decay + a6 * (1 - state.decay)
        
    a6 = dataz
    f0 = np.fft.rfftfreq(len(datax), 1 / sample_rate)

    a7 = resample_logx2(a6, 1024)
    f1 = resample_logx(f0, 1024)

    x0 = (f1 < state.xmin).sum()
    x1 = (f1 < state.xmax).sum()

    data1 = a7[x0:x1].clip(state.ymin, state.ymax)   


async def input_worker():
    
    with sd.InputStream(
        device=idi, channels=1, 
        callback=callback,
        blocksize=2048,
        #latency='low',
        #samplerate=sample_rate,
    ):
        while True:
            await asyncio.sleep(1.)

frame_bot = 26
frame_top = 247

@app.event
def render(ct, dt):
    
    app.window.clear()
    
    if data1 is not None:
        
        data2 = (data1 - state.ymin) / (state.ymax - state.ymin)
        data2 = data2 * (frame_top - frame_bot) + frame_bot
        #data2 = (np.pad(data2, (36, 6))) 
        
        st0.set_channel(0, np.stack((data2, data2)), ct)   
        st0.render(ct, dt)
     
    plot_frame.image = get_plot_frame0(cropx=w)
    plot_frame.draw()
    
    label0.draw()


loop = asyncio.get_event_loop_policy().get_event_loop()
task = loop.create_task(input_worker())


if __name__ == '__main__':

    os.path.exists(STATE_PATH) and app.load_state(STATE_PATH, state)

    app.run(1/30)
    
    app.save_state(STATE_PATH, STATE_PATH, state)

