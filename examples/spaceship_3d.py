"""
    examples/hello-opengl.py
    
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


import random
import struct
import time
import sys
import os

sys.path.insert(0, os.path.abspath('./..'))

from jupylet.label import Label
from jupylet.app import App, State

from jupylet.model import load_blender_gltf, t2i

import pywavefront
import glm

import numpy as np

import PIL.Image

import matplotlib.pyplot as plt

import pyglet

from pyglet.graphics import *

import pyglet.window.key as key


if __name__ == '__main__':
    mode = 'window'
else:
    mode = 'hidden'

app = App(768, 512, mode=mode)

scene = load_blender_gltf('./scenes/moon/alien-moon.gltf')

scene.add_cubemap('./scenes/moon/nebula/nebula*.png', 2.)

camera = scene.cameras['Camera']

mesh = random.choice(list(scene.meshes.values()))


state = State(
    
    capslock = False,
    shift = False,
    
    up = False,
    down = False,
    right = False,
    left = False,
    
    key_w = False,
    key_s = False,
    key_a = False,
    key_d = False
)


@app.event
def on_key_press(symbol, modifiers):
    on_key(symbol, modifiers, True)


@app.event
def on_key_release(symbol, modifiers):
    on_key(symbol, modifiers, False)
    

def on_key(symbol, modifiers, value):
    
    if symbol == key.CAPSLOCK and value:
        state.capslock = not state.capslock
        
    if symbol == key.LSHIFT:
        state.shift = value
        
    if symbol == key.UP:
        state.up = value
        
    if symbol == key.DOWN:
        state.down = value
        
    if symbol == key.LEFT:
        state.left = value
        
    if symbol == key.RIGHT:
        state.right = value
        
    if symbol == key.W:
        state.key_w = value
        
    if symbol == key.S:
        state.key_s = value    
        
    if symbol == key.A:
        state.key_a = value
        
    if symbol == key.D:
        state.key_d = value


@app.run_me_again_and_again(1/48)
def move_object(dt):
        
    obj = mesh if state.capslock else camera
    
    sign = -1 if mesh is camera else 1
    
    if state.right and state.shift:
        obj.rotate_local(sign * 2. * dt, (0, 0, 1))
            
    if state.right and not state.shift:
        obj.rotate_local(-dt / 2., (0, 1, 0))
            
    if state.left and state.shift:
        obj.rotate_local(sign * 2. * -dt, (0, 0, 1))
            
    if state.left and not state.shift:
        obj.rotate_local(dt / 2., (0, 1, 0))
            
    if state.up:
        obj.rotate_local(sign * dt / 2., (1, 0, 0))
        
    if state.down:
        obj.rotate_local(sign * -dt / 2., (1, 0, 0))
        
    if state.key_w and state.shift:
        obj.move_local((0, dt / 0.1, 0))
                
    if state.key_w and not state.shift:
        obj.move_local((0, 0, sign * dt / 0.1))
                
    if state.key_s and state.shift:
        obj.move_local((0, -dt / 0.1, 0))
        
    if state.key_s and not state.shift:
        obj.move_local((0, 0, sign * -dt / 0.1))
        
    if state.key_a:
        obj.move_local((sign * dt / 0.1, 0, 0))
        
    if state.key_d:
        obj.move_local((sign * -dt / 0.1, 0, 0))
        

label0 = Label('Hello World!', color='white', font_size=12, x=10, y=74)
label1 = Label('Hello World!', color='white', font_size=12, x=10, y=52)
label2 = Label('Hello World!', color='white', font_size=12, x=10, y=30)
label3 = Label('Hello World!', color='white', font_size=12, x=10, y=8)


dtl = [0]

@app.event
def on_draw():
        
    app.window.clear()
    app.set3d()    
    
    t0 = time.time()
    
    scene.draw()
    
    dtl.append(0.98 * dtl[-1] + 0.02 * (time.time() - t0))
    dtl[:] = dtl[-256:]
    
    app.set2d()
            
    label0.text = 'time to draw - %.2f ms' % (1000 * dtl[-1])
    label1.text = 'up - %r' % mesh.up
    label2.text = 'front - %r' % mesh.front
    label3.text = 'position - %r' % mesh.position
    
    label0.draw()
    label1.draw()  
    label2.draw()  
    label3.draw()  


@app.schedule_interval(1/30)
def spin(dt):
    scene.meshes['Alien'].rotate_local(-0.5 * dt, (0, 0, 1))


if __name__ == '__main__':
    app.run()

