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


import pyglet
import math
import time
import sys
import os

import PIL.Image

import numpy as np

import pyglet.window.key as key

p0 = os.path.abspath(os.path.dirname(__file__))
p1 = os.path.abspath(os.path.join(p0, '..'))

sys.path.insert(0, p1)
os.chdir(p0)

import jupylet.color

from jupylet.app import App, State
from jupylet.label import Label
from jupylet.sprite import Sprite


if __name__ == '__main__':
    mode = 'window'
else:
    mode = 'hidden'

app = App(mode=mode)

window = app.window

WIDTH = app.width
HEIGHT = app.height


background = '#3e32a2'
foreground = '#7c71da'

app.set_window_color(foreground)


a0 = np.ones((32, 32)) * 255
a1 = np.ones((128, 16)) * 255
a2 = np.ones((HEIGHT * 9 // 10, WIDTH * 9 // 10, 3)) * jupylet.color.color2rgb(background)[:3]

ball = Sprite(a0, y=HEIGHT/2, x=WIDTH/2, autocrop=True)

padl = Sprite(a1, y=HEIGHT/2, x=48)
padr = Sprite(a1, y=HEIGHT/2, x=WIDTH-48)

field = Sprite(a2, y=HEIGHT/2, x=WIDTH/2) 

sound = pyglet.media.load('sounds/pong-blip.wav', streaming=False)


pyglet.font.add_file('fonts/PetMe64.ttf')

scorel = Label(
    '0', font_name='Pet Me 64', font_size=42, color=foreground, 
    x=64, y=HEIGHT/2, anchor_y='center', anchor_x='left'
)

scorer = Label(
    '0', font_name='Pet Me 64', font_size=42, color=foreground, 
    x=WIDTH-64, y=HEIGHT/2, anchor_y='center', anchor_x='right'
)


@app.event
def on_draw():
    
    window.clear()
    
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
    pyl = HEIGHT/2,

    vyr = 0,
    pyr = HEIGHT/2,

    left = False,
    right = False,

    key_a = False,
    key_d = False,
)

@app.event
def on_key_press(symbol, modifiers):
        
    if symbol == key.LEFT:
        state.left = True
        
    if symbol == key.RIGHT:
        state.right = True
        
    if symbol == key.A:
        state.key_a = True
        
    if symbol == key.D:
        state.key_d = True
        

@app.event
def on_key_release(symbol, modifiers):
    
    if symbol == key.LEFT:
        state.left = False
        
    if symbol == key.RIGHT:
        state.right = False

    if symbol == key.A:
        state.key_a = False
        
    if symbol == key.D:
        state.key_d = False
        

@app.run_me_again_and_again(1/120)
def update_pads(dt):
        
    if state.right:
        state.pyr = min(HEIGHT, state.pyr + dt * 512)
        
    if state.left:
        state.pyr = max(0, state.pyr - dt * 512)
        
    if state.key_a:
        state.pyl = min(HEIGHT, state.pyl + dt * 512)
        
    if state.key_d:
        state.pyl = max(0, state.pyl - dt * 512)
        
    ayl = 200 * (state.pyl - padl.y)
    ayr = 200 * (state.pyr - padr.y)

    state.vyl = state.vyl * 0.9 + (ayl * dt)
    state.vyr = state.vyr * 0.9 + (ayr * dt)
    
    padl.y += state.vyl * dt
    padr.y += state.vyr * dt
    
    padr.clip_position(WIDTH, HEIGHT)
    padl.clip_position(WIDTH, HEIGHT)


@app.run_me_again_and_again(1/120)
def update_ball(dt):
    
    bs0 = state.bvx ** 2 + state.bvy ** 2
    
    ball.rotation += 200 * dt
    
    ball.x += state.bvx * dt
    ball.y += state.bvy * dt
    
    if ball.top >= HEIGHT:
        app.play_once(sound)
        ball.y -= ball.top - HEIGHT
        state.bvy = -state.bvy
        
    if ball.bottom <= 0:
        app.play_once(sound)
        ball.y -= ball.bottom
        state.bvy = -state.bvy
        
    if ball.right >= WIDTH:
        app.play_once(sound)
        ball.x -= ball.right - WIDTH
        
        state.bvx = -192
        state.bvy = 192 * np.sign(state.bvy)
        bs0 = 0
        
        state.sl += 1
        scorel.text = str(state.sl)
        
    if ball.left <= 0:
        app.play_once(sound)
        ball.x -= ball.left
        
        state.bvx = 192
        state.bvy = 192 * np.sign(state.bvy)
        bs0 = 0
        
        state.sr += 1
        scorer.text = str(state.sr)
        
    if state.bvx > 0 and ball.top >= padr.bottom and padr.top >= ball.bottom: 
        if 0 < ball.right - padr.left < 10:
            app.play_once(sound)
            ball.x -= ball.right - padr.left
            state.bvx = -state.bvx
            state.bvy += state.vyr / 2
            
    if state.bvx < 0 and ball.top >= padl.bottom and padl.top >= ball.bottom: 
        if 0 < padl.right - ball.left < 10:
            app.play_once(sound)
            ball.x += ball.left - padl.right
            state.bvx = -state.bvx
            state.bvy += state.vyl / 2
            
    bs1 = state.bvx ** 2 + state.bvy ** 2
    
    if bs1 < 0.9 * bs0:
        state.bvx = (bs0 - state.bvy ** 2) ** 0.5 * np.sign(state.bvx)

    ball.wrap_position(WIDTH, HEIGHT)


@app.run_me_now()
def highlights(dt):
    
    sl0 = state.sl
    sr0 = state.sr
    
    slc = np.array(scorel.color)
    src = np.array(scorer.color)
    
    while True:
        
        dt = yield 1/30
        
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

    return {
        'screen0': app.array0,
        'player0': {'score': state.sl, 'reward': reward},
        'player1': {'score': state.sr, 'reward': -reward},
    }


START = 'pong-start.state'


def reset():
    return load(START)
    
    
def load(path):
    return app.load_state(path, state, ball, padl, padr, scorel, scorer)
    

def save(path=None):
    return app.save_state('pong', path, state, ball, padl, padr, scorel, scorer)


if __name__ == '__main__':
    app.run()

