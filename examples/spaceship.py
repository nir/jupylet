import pyglet
import math
import sys
import os


p0 = os.path.abspath('.')
p1 = os.path.abspath(os.path.join(p0, '..'))

sys.path.insert(0, p1)

import jupylet

from jupylet import *

import pyglet.window.key as key


app = App(mode='window')

window = app.window

WIDTH = app.width
HEIGHT = app.height

stars = Sprite('stars.png', scale=2.5)
alien = Sprite('alien.png', scale=0.5)
ship = Sprite('ship1.png', x=WIDTH/2, y=HEIGHT/2, scale=0.5)
moon = Sprite('moon.png', x=WIDTH-70, y=HEIGHT-70, scale=0.5)

label = Label('Hello World!', color='cyan', font_size=16, x=10, y=10)


@app.event
def on_draw():
    
    window.clear()
    
    stars.draw()
    moon.draw()
    
    label.draw()
    alien.draw()
    ship.draw()


@app.event
def on_mouse_motion(x, y, dx, dy):
    alien.x = x
    alien.y = y
    
    
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


app.run()