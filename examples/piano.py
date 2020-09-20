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


import functools
import asyncio
import logging
import random
import mido
import sys
import os

import PIL.ImageDraw
import PIL.Image 

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jupylet.color

from jupylet.app import App
from jupylet.sound import Sample, Synth, sleep, get_oscilloscope_as_image
from jupylet.state import State
from jupylet.label import Label
from jupylet.sprite import Sprite


logger = logging.getLogger()


app = App(width=512, height=420)#, log_level=logging.INFO)

a0 = np.zeros((256, 512, 4), 'uint8')
oscilloscope = Sprite(a0, x=256, y=292)
layout = Sprite('images/keyboard.png', x=256, y=82, scale=0.5)

synth = Synth()
#synth.set_envelope(min_duration=0.05, attack=0.0, decay=0.1, sustain=0.7, release=0.5)

keys = app.window.keys
keyboard = {

    keys.Z: 'C',
    keys.S: 'Cs',
    keys.X: 'D',
    keys.D: 'Ds',
    keys.C: 'E',
    keys.V: 'F',
    keys.G: 'Fs',
    keys.B: 'G',
    keys.H: 'Gs',
    keys.N: 'A',
    keys.J: 'As',
    keys.M: 'B',

    keys.Q: 'C5',
    50: 'Cs5',
    keys.W: 'D5',
    51: 'Ds5',
    keys.E: 'E5',
    keys.R: 'F5',
    53: 'Fs5',
    keys.T: 'G5',
    54: 'Gs5',
    keys.Y: 'A5',
    55: 'As5',
    keys.U: 'B5',

    keys.I: 'C6',
    57: 'Cs6',
    keys.O: 'D6',
    48: 'Ds6',
    keys.P: 'E6',
}


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


pk = {}

@app.event
def key_event(key, action, modifiers):
            
    keys = app.window.keys
    value = action == keys.ACTION_PRESS

    if key == keys.UP:
        state.up = value

    if key == keys.DOWN:
        state.down = value

    if key == keys.RIGHT:
        state.right = value

    if key == keys.LEFT:
        state.left = value
        
    if action == keys.ACTION_PRESS and key in keyboard:
        assert key not in pk
        pk[key] = synth.play_new(note=keyboard[key])
           
    if action == keys.ACTION_RELEASE and key in keyboard:
        pk.pop(key).play_release()


@app.run_me_every(1/24)
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
    
    layout.draw()
    
    label0.draw()
    label1.draw()
    label2.draw()


xylo = Sample(
    'sounds/VCSL/Xylophone/Xylophone - Medium Mallets.sfz',
    loop=False,
)

xylo.amp = 16

_keyd = {}

def midi_Callback(msg):
    
    if msg.type == 'note_on':
        if msg.velocity != 0:
            _keyd[msg.note] = xylo.play_new(key=msg.note, velocity=msg.velocity)
        else:
            _keyd[msg.note].play_release()


_port = mido.open_input(callback=midi_Callback)


if __name__ == '__main__':
    app.run(1/30)

