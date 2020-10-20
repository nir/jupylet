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


import logging
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jupylet.sprite import Sprite
from jupylet.label import Label
from jupylet.app import App


logger = logging.getLogger()


app = App()


stars = Sprite('images/stars.png', scale=2.5)
alien = Sprite('images/alien.png', scale=0.5)
ship = Sprite('images/ship1.png', x=app.width/2, y=app.height/2, scale=0.5)
moon = Sprite('images/moon.png', x=app.width-70, y=app.height-70, scale=0.5)

circle = Sprite('images/yellow-circle.png', width=184)
circle.opacity = 0.

label = Label('hello, world', color='cyan', font_size=32, x=10, y=10)


@app.event
def mouse_position_event(x, y, dx, dy):
    logger.info('Enter mouse_position_event(%r, %r, %r, %r).', x, y, dx, dy)
    
    alien.x = x
    alien.y = y
    
    circle.x = x
    circle.y = y    


vx = 0
vy = 0

up = 0
left = 0
right = 0


@app.run_me_every(1/60)
def update_ship(ct, dt):
    
    global vx, vy

    if left:
        ship.angle += 128 * dt
        
    if right:
        ship.angle -= 128 * dt
        
    if up:
        vx += 3 * math.cos(math.radians(90 + ship.angle))
        vy += 3 * math.sin(math.radians(90 + ship.angle))

    ship.x += vx * dt
    ship.y += vy * dt
    
    ship.wrap_position(app.width, app.height)
    
    if len(ship.collisions_with(alien)) > 0:
        circle.opacity = 0.5
    else:
        circle.opacity = 0.0


@app.run_me_every(1/60)
def rotate(ct, dt):
    
    alien.angle += 64 * dt


@app.event
def key_event(key, action, modifiers):
    logger.info('Enter key_event(key=%r, action=%r, modifiers=%r).', key, action, modifiers)
    
    global up, left, right
    
    keys = app.window.keys
    
    if action == keys.ACTION_PRESS:

        if key == keys.UP:
            ship.image = 'images/ship2.png'
            up = True

        if key == keys.LEFT:
            left = True

        if key == keys.RIGHT:
            right = True

    if action == keys.ACTION_RELEASE:
    
        if key == keys.UP:
            ship.image = 'images/ship1.png'
            up = False

        if key == keys.LEFT:
            left = False

        if key == keys.RIGHT:
            right = False


@app.event
def render(ct, dt):
    #logger.debug('Enter render(%r, %r).', ct, dt)
    
    app.window.clear()
    
    stars.draw()
    moon.draw()

    circle.draw()
    alien.draw()
    ship.draw()
    
    label.draw()


if __name__ == '__main__':
    app.run()

