"""
    examples/spaceship.py
    
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
import math
import sys
import os


p0 = os.path.abspath('.')
p1 = os.path.abspath(os.path.join(p0, '..'))

sys.path.insert(0, p1)

import jupylet.color

from jupylet.app import App
from jupylet.label import Label
from jupylet.sprite import Sprite

import pyglet.window.key as key


if __name__ == '__main__':
    mode = 'window'
else:
    mode = 'hidden'

app = App(mode=mode)

window = app.window

WIDTH = app.width
HEIGHT = app.height

stars = Sprite('stars.png', scale=2.5)
alien = Sprite('alien.png', scale=0.5)
ship = Sprite('ship1.png', x=WIDTH/2, y=HEIGHT/2, scale=0.5)
moon = Sprite('moon.png', x=WIDTH-70, y=HEIGHT-70, scale=0.5)

circle = Sprite('yellow-circle.png')
circle.opacity = 0
circle.width = 184

label = Label('Hello World!', color='cyan', font_size=16, x=10, y=10)


@app.event
def on_draw():
    
    window.clear()
    
    stars.draw()
    moon.draw()
    
    label.draw()

    circle.draw()
    alien.draw()
    ship.draw()


@app.event
def on_mouse_motion(x, y, dx, dy):
    
    alien.x = x
    alien.y = y
    
    circle.x = x
    circle.y = y    
    

@app.run_me_again_and_again(1/36)
def update_alien(dt):
    alien.rotation += dt * 36


vx = 0
vy = 0

up = 0
left = 0
right = 0


@app.run_me_again_and_again(1/120)
def update_ship(dt):
    
    global vx, vy

    if left:
        ship.rotation -= 2
        
    if right:
        ship.rotation += 2
        
    if up:
        vx += 3 * math.cos((90 - ship.rotation) / 180 * math.pi)
        vy += 3 * math.sin((90 - ship.rotation) / 180 * math.pi)

    ship.x += vx * dt
    ship.y += vy * dt
    
    ship.wrap_position(WIDTH, HEIGHT)
    
    if len(ship.collisions_with(alien)) > 0:
        circle.opacity = 128
    else:
        circle.opacity = 0
        

@app.event
def on_key_press(symbol, modifiers):
    
    global up, left, right
    
    if symbol == key.UP:
        ship.image = 'ship2.png'
        up = True

    if symbol == key.LEFT:
        left = True
        
    if symbol == key.RIGHT:
        right = True
        

@app.event
def on_key_release(symbol, modifiers):
    
    global up, left, right
    
    if symbol == key.UP:
        ship.image = 'ship1.png'
        up = False
        
    if symbol == key.LEFT:
        left = False
        
    if symbol == key.RIGHT:
        right = False


if __name__ == '__main__':
    app.run()

