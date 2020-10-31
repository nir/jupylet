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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jupylet.color

from jupylet.app import App
from jupylet.state import State
from jupylet.label import Label
from jupylet.sprite import Sprite

from jupylet.audio.bundle import *


app = App(width=512, height=420)#, log_level=logging.INFO)

oscilloscope = Sprite(np.zeros((256, 512, 4), 'uint8'), x=256, y=292)

keyboard_layout = Sprite('images/keyboard.png', x=256, y=82, scale=0.5)


state = State(
    
    amp = 1.,
    ms = 50.,
    
    up = False,
    down = False,
    left = False,
    right = False,
)

label0 = Label('amp: %.1f' % state.amp, x=10, y=194)
label1 = Label('span: %.1f ms' % state.ms, x=10, y=174)
label2 = Label('use ← → ↑ ↓ to modify', anchor_x='right', x=app.width - 10, y=174)


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
    value = action != keys.ACTION_RELEASE

    if key == keys.UP:
        state.up = value

    if key == keys.DOWN:
        state.down = value

    if key == keys.RIGHT:
        state.right = value

    if key == keys.LEFT:
        state.left = value
        
    if action == keys.ACTION_PRESS and key in keyboard:
        assert key not in _keyd
        _keyd[key] = tb303.play_poly(note=keyboard[key])
        
    if action == keys.ACTION_RELEASE and key in keyboard:
        _keyd.pop(key).play_release()


@app.run_me_every(1/10)
def modify_oscilloscope(ct, dt):
    
    s = 2 ** dt
    
    if state.up:
        state.amp *= s
        label0.text = 'amp: %.1f' % state.amp

    if state.down:
        state.amp /= s
        label0.text = 'amp: %.1f' % state.amp

    if state.right:
        state.ms *= s
        state.ms = min(256, state.ms)
        label1.text = 'span: %.1f ms' % state.ms

    if state.left:
        state.ms /= s
        label1.text = 'span: %.1f ms' % state.ms


@app.event
def render(ct, dt):
    
    app.window.clear(color='#555')
    
    im, ts, te = get_oscilloscope_as_image(
        1/app.interval,
        ms=state.ms, 
        amp=state.amp, 
        color=255, 
        size=(512, 256)
    )

    oscilloscope.image = im    
    oscilloscope.draw()
    
    keyboard_layout.draw()
    
    label0.draw()
    label1.draw()
    label2.draw()


app.set_midi_sound(tb303)

#set_latency('lowest')
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

