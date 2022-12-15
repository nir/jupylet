"""
    examples/spectrum_analyzer.py
    
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
import _thread
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

import matplotlib.pyplot as plt


def implot(*args, xscale='log', xmin=None, xmax=None, ymin=None, ymax=None, figsize=(10, 5), **kwargs):
    
    buf = io.BytesIO()
    
    fig = plt.figure(figsize=figsize, dpi=100)
    
    ax0 = fig.add_subplot(111)
    ax0.grid(True, which='both')
    ax0.set_xscale(xscale)

    if xmin or xmax:
        ax0.set_xlim(xmin=xmin, xmax=xmax)

    if ymin or ymax:
        ax0.set_ylim(ymin=ymin, ymax=ymax)

    pl0 = ax0.plot(*args, **kwargs)[0] # Returns a tuple of line objects, thus the comma
    
    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close(fig)

    return PIL.Image.open(buf)


@functools.lru_cache(maxsize=16)
def get_plot_frame(xmin=0, xmax=1000, ymin=-50, ymax=50, figsize=(12, 6), rgba=False):
    
    xx = np.arange(xmin, xmax)
    im = implot(
        xx, xx - 1e6, 
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        figsize=figsize
    )
    
    if not rgba:
        return im
    
    a0 = np.array(im)
    a1 = np.concatenate((a0 * 0, 255 - a0[...,:1]), -1)

    return PIL.Image.fromarray(a1)


state = State(
    
    up = False,
    down = False,
    
    gain = 1.,
    decay = 0.8,
    samples = 4096, 
    
    xmin = 100,
    xmax = 20000,
    ymin = -75,
    ymax = 150,
)


plot_frame_image = get_plot_frame(
    xmin=state.xmin, xmax=state.xmax, 
    ymin=state.ymin, ymax=state.ymax, 
    figsize=(12, 6),
    rgba=True,
)

w, h = plot_frame_image.size


app = App(width=w, height=h, quality=100)#, log_level=logging.INFO)


plot_frame = Sprite(plot_frame_image, anchor_x='left', anchor_y='left')

label0 = Label('gain: %.2f' % state.gain, x=70, y=445, color='red')
label1 = Label('use ↑ ↓ to control gain', x=70, y=470, color='red')


st0 = Shadertoy("""

    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        // Normalized pixel coordinates (from 0 to 1)
        vec2 uv = fragCoord / iResolution.xy;

        // Time varying pixel color
        vec3 col = vec3(1., 1., 1.);

        float dst = 1000.;
        float dx0 = 0.000033;
        
        for (int i=0; i < 100; i++) {
        
            float dx1 = dx0 * (i - 50);
            float dy1 = texture(iChannel0, vec2(uv.x + dx1, 0.)).r - uv.y; 
            float dxy = dx1 * dx1 + dy1 * dy1;
            
            if (dxy < dst) {
                dst = dxy;
            }
        }
        
        vec3 sig = vec3(0.00033 / max(32 * dst, 1e-6));

        sig *= vec3(1., 1., 1.);

        if (uv.x < 0.05) {
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


@app.event
def key_event(key, action, modifiers):
            
    keys = app.window.keys
    value = action == keys.ACTION_PRESS

    if key == keys.UP:
        state.up = value

    if key == keys.DOWN:
        state.down = value


@app.run_me_every(1/24)
def modify_gain(ct, dt):
    
    s = 2 ** dt
    
    if state.up:
        state.gain *= s
        label0.text = 'gain: %.2f' % state.gain

    if state.down:
        state.gain /= s
        label0.text = 'gain: %.2f' % state.gain


data0 = []
data1 = None
dataz = None


@app.event
def render(ct, dt):
    
    app.window.clear()
    
    if data1 is not None:
        
        data2 = (np.pad(data1, (37, 6)) - state.ymin + 0) * 1.15 + 20 
        st0.set_channel(0, np.stack((data2, data2)), ct)   
        st0.render(ct, dt)
     
    plot_frame.draw()
    
    label0.draw()
    label1.draw()


dl = sd.query_devices()
idi = sd.default.device['input']
sample_rate = dl[idi]['default_samplerate']


def resample_logx(data, num=None):
    
    assert data.ndim == 1
    
    num = num or data.size
    
    x1 = np.exp(np.linspace(0, np.log(data.size), num)) - 1
    x2 = x1.astype('long')
    
    xx = 1 - x1 + x2

    return data[x2] * xx + data[(x2 + 1).clip(0, data.size-1)] * (1 - xx)


def callback(indata, frames, time, status):

    global data0, data1, dataz

    data0.append(indata[:,0] * state.gain)

    if len(data0) > 24:
        data0.pop(0)

    datax = np.concatenate(data0)[-state.samples:]
        
    a0 = np.fft.rfft(datax)
    a5 = a0.conj() * a0
    a6 = 10 * np.log(a5.real + 1e-6)

    if dataz is None or dataz.shape != a6.shape:
        dataz = a6
    else:
        dataz = dataz * state.decay + a6 * (1 - state.decay)
        
    a6 = dataz
    f0 = np.fft.rfftfreq(len(datax), 1 / sample_rate)

    a7 = resample_logx(a6, 1024)
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


task = asyncio.get_event_loop().create_task(input_worker())


if __name__ == '__main__':
    app.run(1/30)

