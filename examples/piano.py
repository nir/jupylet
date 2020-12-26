"""
    examples/sounds_demo.py
    
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


import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jupylet.color

from jupylet.app import App
from jupylet.state import State
from jupylet.label import Label
from jupylet.sprite import Sprite
from jupylet.shadertoy import Shadertoy, get_shadertoy_audio

from jupylet.audio.bundle import *

import numpy as np


app = App(width=512, height=420, quality=100)#, log_level=logging.INFO)


#
# Default oscilloscope shader:
# The code in the following cell is of a simple shadertoy shader that 
# displays an audio oscilloscope. Shadertoy (http://shadertoy.com/) are 
# an easy way to create graphic effects by programming the GPU directly:
#
st0 = Shadertoy("""

    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        // Normalized pixel coordinates (from 0 to 1)
        vec2 uv = fragCoord / iResolution.xy;

        // Time varying pixel color
        vec3 col = 0.2 + 0.2 * cos(iTime + uv.xyx + vec3(0, 2, 4));

        float amp = texture(iChannel0, vec2(uv.x, 1.)).r; 

        vec3 sig = vec3(0.00033 / max(pow(amp - uv.y, 2.), 1e-6));

        sig *= vec3(.5, .5, 4.) / 2.;

        col += sig;

        // Output to screen
        fragColor = vec4(col,1.0);
    }
    
""", 512, 256, 0, 420, 0, 'left', 'top')


#
# Audio visualization shader by Alban Fichet:
# The code in the following cell is of a beautiful shadertoy shader by 
# graphics expert and researcher Alban Fichet (https://afichet.github.io/) 
# who kindly allowed its inclusion here under CC BY 4.0 license: 
#
ba1 = Shadertoy("""

    vec3 hue2rgb(in float h) {
        vec3 k = mod(vec3(5., 3., 1.) + vec3(h*360./60.), vec3(6.));
        return vec3(1.) - clamp(min(k, vec3(4.) - k), vec3(0.), vec3(1.));
    }

    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        vec2 uv = fragCoord/iResolution.xy;

        float aspect = iResolution.x / iResolution.y;
        float blurr = 0.3;
        float sharpen = 1.7;

        vec2 maxWindow = vec2(3., 3./aspect);
        uv = mix(-maxWindow, maxWindow, uv);

        float r = dot(uv, uv);
        float theta = atan(uv.y, uv.x) + 3.14;

        float t = abs(2.*theta / (2.*3.14) - 1.);

        float signal = 2.0*texture(iChannel0,vec2(t,1.)).x;
        float ampl = 2.0*texture(iChannel0,vec2(0.8,.25)).x;

        float v = 1. - pow(smoothstep(0., blurr, abs(r - signal)), 0.02);
        float hue = pow(fract(abs(sin(theta/2.) * ampl)), sharpen);

        fragColor = vec4(v * hue2rgb(fract(hue + iTime/10.)), 1.0);
    }
    
""")

bb1 = Shadertoy("""

    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        vec2 uv = fragCoord/iResolution.xy;

        vec2 center = vec2(.5 + 0.15*sin(iTime));
        float zoom = 1.02;

        vec4 prevParams = texture(iChannel0, (uv - center)/zoom + center);
        vec4 bB = texture(iChannel1, uv);

        fragColor = clamp(mix(prevParams, 4 * bB, 0.1), 0., 1.) ;
    }
    
""")

bb1.set_channel(0, bb1)
bb1.set_channel(1, ba1)

st1 = Shadertoy("""

    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        vec3 bA = texelFetch(iChannel0, ivec2(fragCoord), 0).rgb;
        vec3 bB = texelFetch(iChannel1, ivec2(fragCoord), 0).rgb;

        vec3 col_rgb = bB;

        col_rgb = col_rgb*exp2(3.5);

        fragColor = vec4(pow(col_rgb, vec3(1./2.2)), 1.);
    } 

""", 512, 256, 0, 420, 0, 'left', 'top')


st1.set_channel(0, ba1)
st1.set_channel(1, bb1)

keyboard_layout = Sprite('images/keyboard.png', x=256, y=82, scale=0.5)
label0 = Label('amp: %.2f' % get_master_volume(), anchor_x='right', x=app.width - 10, y=174)
label1 = Label('use ↑ ↓ to control volume', x=10, y=194)
label2 = Label('tap SPACE to change shader', x=10, y=174)

state = State(
    
    up = False,
    down = False,
    shader = 0,
)

keys = app.window.keys

keyboard = {

    keys.Z: note.C,
    keys.S: note.Cs,
    keys.X: note.D,
    keys.D: note.Ds,
    keys.C: note.E,
    keys.V: note.F,
    keys.G: note.Fs,
    keys.B: note.G,
    keys.H: note.Gs,
    keys.N: note.A,
    keys.J: note.As,
    keys.M: note.B,

    keys.Q: note.C5,
    50: note.Cs5,
    keys.W: note.D5,
    51: note.Ds5,
    keys.E: note.E5,
    keys.R: note.F5,
    53: note.Fs5,
    keys.T: note.G5,
    54: note.Gs5,
    keys.Y: note.A5,
    55: note.As5,
    keys.U: note.B5,

    keys.I: note.C6,
    57: note.Cs6,
    keys.O: note.D6,
    48: note.Ds6,
    keys.P: note.E6,
}


_keyd = {}

@app.event
def key_event(key, action, modifiers):
            
    keys = app.window.keys
    value = action == keys.ACTION_PRESS

    if key == keys.UP:
        state.up = value

    if key == keys.DOWN:
        state.down = value

    if key == keys.SPACE and action == keys.ACTION_PRESS:
        state.shader = 1 - state.shader

    if action == keys.ACTION_PRESS and key in keyboard:
        assert key not in _keyd
        _keyd[key] = tb303.play_poly(note=keyboard[key])
        
    if action == keys.ACTION_RELEASE and key in keyboard:
        _keyd.pop(key).play_release()


@app.run_me_every(1/24)
def modify_volume(ct, dt):
    
    s = 2 ** dt
    amp = get_master_volume()
    
    if state.up:
        amp *= s
        set_master_volume(amp)
        label0.text = 'amp: %.2f' % amp

    if state.down:
        amp /= s
        set_master_volume(amp)
        label0.text = 'amp: %.2f' % amp


@app.event
def render(ct, dt):
    
    app.window.clear(color='#555')
    
    keyboard_layout.draw()
    
    if state.shader == 0:
        st0.set_channel(0, *get_shadertoy_audio(amp=5))   
        st0.render(ct, dt)
    else:
        ba1.set_channel(0, *get_shadertoy_audio(amp=5))   
        st1.render(ct, dt)

    label0.draw()
    label1.draw()
    label2.draw()


app.set_midi_sound(tb303)

set_latency('lowest')
set_effects(ConvolutionReverb('./sounds/impulses/MaesHowe.flac'))

xylo = Sample('sounds/VCSL/Xylophone/Xylophone - Medium Mallets.sfz')
xylo.amp = 8


@app.sonic_live_loop
async def loop0():
            
    use(tb303, duration=2, amp=0.15)

    play(note.C2)
    await sleep(3)

    play(note.E2)
    await sleep(3)

    play(note.C2)
    await sleep(6)


@app.sonic_live_loop
async def loop1():
    
    use(xylo, amp=8)
        
    play(note.C5)
    await sleep(1)

    play(note.E5)
    await sleep(1)

    play(note.G5)
    await sleep(1)


if __name__ == '__main__':
    app.run(1/30)

