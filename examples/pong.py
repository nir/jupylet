"""
    examples/pong.py
    
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

"""
    * pong sound is by [freesound](https://freesound.org/people/NoiseCollector/sounds/4359/).
    * Commodore 64 font is by [KreativeKorp](https://www.kreativekorp.com/software/fonts/c64.shtml).
"""


import math
import time
import sys
import os

import PIL.Image

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jupylet.color

from jupylet.app import App
from jupylet.state import State
from jupylet.label import Label
from jupylet.sprite import Sprite

from jupylet.audio.sample import Sample

import moderngl_window.timers.clock as _clock


app = App()


background = '#3e32a2'
foreground = '#7c71da'


a0 = np.ones((32, 32)) * 255
a1 = np.ones((128, 16)) * 255
a2 = np.ones((app.height * 9 // 10, app.width * 9 // 10, 3)) * 255

ball = Sprite(a0, y=app.height/2, x=app.width/2)

padl = Sprite(a1, y=app.height/2, x=48)
padr = Sprite(a1, y=app.height/2, x=app.width-48)

field = Sprite(a2, y=app.height/2, x=app.width/2, color=background) 

pong_sound = Sample('sounds/pong-blip.wav', amp=0.2).load()


scorel = Label(
    '0', font_size=42, color=foreground, 
    x=64, y=app.height/2, 
    anchor_y='center', anchor_x='left',
    font_path='fonts/PetMe64.ttf'
)

scorer = Label(
    '0', font_size=42, color=foreground, 
    x=app.width-64, y=app.height/2, 
    anchor_y='center', anchor_x='right',
    font_path='fonts/PetMe64.ttf'
)


@app.event
def render(ct, dt):
    
    app.window.clear(color=foreground)
    
    field.draw()
    
    scorel.draw()
    scorer.draw()
    
    ball.draw()
    padl.draw()
    padr.draw()


state = State(
    
    sl = 0,
    sr = 0,
    
    bvx = 192,
    bvy = 192,
    
    vyl = 0,
    pyl = app.height/2,

    vyr = 0,
    pyr = app.height/2,

    left = False,
    right = False,

    key_a = False,
    key_d = False,
)


@app.event
def key_event(key, action, modifiers):
        
    keys = app.window.keys
    
    if action == keys.ACTION_PRESS:
        
        if key == keys.LEFT:
            state.left = True

        if key == keys.RIGHT:
            state.right = True

        if key == keys.A:
            state.key_a = True

        if key == keys.D:
            state.key_d = True

    if action == keys.ACTION_RELEASE:

    
        if key == keys.LEFT:
            state.left = False

        if key == keys.RIGHT:
            state.right = False

        if key == keys.A:
            state.key_a = False

        if key == keys.D:
            state.key_d = False


@app.run_me_every(1/120)
def update_pads(ct, dt):
        
    if state.right:
        state.pyr = min(app.height, state.pyr + dt * 512)
        
    if state.left:
        state.pyr = max(0, state.pyr - dt * 512)
        
    if state.key_a:
        state.pyl = min(app.height, state.pyl + dt * 512)
        
    if state.key_d:
        state.pyl = max(0, state.pyl - dt * 512)
        
    ayl = 200 * (state.pyl - padl.y)
    ayr = 200 * (state.pyr - padr.y)

    state.vyl = state.vyl * 0.9 + (ayl * dt)
    state.vyr = state.vyr * 0.9 + (ayr * dt)
    
    padl.y += state.vyl * dt
    padr.y += state.vyr * dt
    
    padr.clip_position(app.width, app.height)
    padl.clip_position(app.width, app.height)


@app.run_me_every(1/60)
def update_ball(ct, dt):
    
    bs0 = state.bvx ** 2 + state.bvy ** 2
    
    ball.angle += 200 * dt
    
    ball.x += state.bvx * dt
    ball.y += state.bvy * dt
    
    if ball.top >= app.height:
        pong_sound.play(pan=2*max(.25, min(.75, ball.x / app.width))-1)
        ball.y -= ball.top - app.height
        state.bvy = -state.bvy
        
    if ball.bottom <= 0:
        pong_sound.play(pan=2*max(.25, min(.75, ball.x / app.width))-1)
        ball.y -= ball.bottom
        state.bvy = -state.bvy
        
    if ball.right >= app.width:
        pong_sound.play(pan=2*max(.25, min(.75, ball.x / app.width))-1)
        ball.x -= ball.right - app.width
        
        state.bvx = -192
        state.bvy = 192 * np.sign(state.bvy)
        bs0 = 0
        
        state.sl += 1
        scorel.text = str(state.sl)
        
    if ball.left <= 0:
        pong_sound.play(pan=2*max(.25, min(.75, ball.x / app.width))-1)
        ball.x -= ball.left
        
        state.bvx = 192
        state.bvy = 192 * np.sign(state.bvy)
        bs0 = 0
        
        state.sr += 1
        scorer.text = str(state.sr)
        
    if state.bvx > 0 and ball.top >= padr.bottom and padr.top >= ball.bottom: 
        if 0 < ball.right - padr.left < 10:
            pong_sound.play(pan=2*max(.25, min(.75, ball.x / app.width))-1)
            ball.x -= ball.right - padr.left
            state.bvx = -state.bvx
            state.bvy += state.vyr / 2
            
    if state.bvx < 0 and ball.top >= padl.bottom and padl.top >= ball.bottom: 
        if 0 < padl.right - ball.left < 10:
            pong_sound.play(pan=2*max(.25, min(.75, ball.x / app.width))-1)
            ball.x += ball.left - padl.right
            state.bvx = -state.bvx
            state.bvy += state.vyl / 2
            
    bs1 = state.bvx ** 2 + state.bvy ** 2
    
    if bs1 < 0.9 * bs0:
        state.bvx = (bs0 - state.bvy ** 2) ** 0.5 * np.sign(state.bvx)

    ball.wrap_position(app.width, app.height)


@app.run_me()
def highlights(ct, dt):
    
    sl0 = state.sl
    sr0 = state.sr
    
    slc = np.array(scorel.color)
    src = np.array(scorer.color)
    
    while True:
        
        ct, dt = yield 1/24
        
        r0 = 0.9 ** (120 * dt)
        
        scorel.color = np.array(scorel.color) * r0 + (1 - r0) * slc
        scorer.color = np.array(scorer.color) * r0 + (1 - r0) * src
        
        if sl0 != state.sl:
            sl0 = state.sl
            scorel.color = 'white'

        if sr0 != state.sr:
            sr0 = state.sr
            scorer.color = 'white'


def step(player0=[0, 0, 0, 0, 0], player1=[0, 0, 0, 0, 0], n=1):
    
    state.key_a, state.key_d = player0[:2]
    
    state.left, state.right = player1[:2]
    
    sl0 = state.sl
    sr0 = state.sr
    
    if app.mode == 'hidden': 
        app.step(n)
        
    reward = (state.sl - sl0) - (state.sr - sr0)

    return observe(reward)


def observe(reward=0):

    return {
        'screen0': app.observe(),
        'player0': {'score': state.sl, 'reward': reward},
        'player1': {'score': state.sr, 'reward': -reward},
    }


START = 'pong-start.state'


def reset():
    load(START)
    return observe()
    
    
def load(path):
    app.load_state(path, state, ball, padl, padr, scorel, scorer)
    return observe()
    

def save(path=None):
    app.save_state('pong', path, state, ball, padl, padr, scorel, scorer)


if __name__ == '__main__':
    app.run()

