"""
    examples/spaceship_3d.py
    
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
import random
import struct
import time
import glm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jupylet.label import Label
from jupylet.app import App
from jupylet.state import State
from jupylet.model import Skybox
from jupylet.loader import load_blender_gltf


logger = logging.getLogger()


if __name__ == '__main__':
    mode = 'window'
else:
    mode = 'hidden'

app = App(768, 512, mode=mode)#, log_level=logging.INFO)

scene = load_blender_gltf('./scenes/moon/alien-moon.gltf')

scene.skybox = Skybox('./scenes/moon/nebula/nebula*.png', intensity=2., flip_left_right=True)

scene.shadows = True


sun = scene.lights['Light.Sun']
sun.shadowmaps_depths = [1., 0.12, 0.04, 0.015, 0.0]

moon = scene.meshes['Moon']
moon.shadow_bias = 0.2

camera = scene.cameras['Camera']


state = State(
    
    capslock = False,
    shift = False,
    alt = False,
    
    up = False,
    down = False,
    right = False,
    left = False,
    
    key_w = False,
    key_s = False,
    key_a = False,
    key_d = False,
    
    lv = glm.vec3(0),
    av = glm.vec3(0),
)


@app.event
def key_event(key, action, modifiers):
    logger.info('Enter key_event(key=%r, action=%r, modifiers=%r).', key, action, modifiers)
    
    keys = app.window.keys

    value = action != keys.ACTION_RELEASE
    
    if key == keys.CAPS_LOCK and value:
        state.capslock = not state.capslock
        
    state.alt = modifiers.alt        
    state.shift = modifiers.shift
        
    if key == keys.SPACE:
        state.lv *= 0.
        state.av *= 0.
        
    if key == keys.UP:
        state.up = value
        
    if key == keys.DOWN:
        state.down = value
        
    if key == keys.LEFT:
        state.left = value
        
    if key == keys.RIGHT:
        state.right = value
        
    if key == keys.W:
        state.key_w = value
        
    if key == keys.S:
        state.key_s = value    
        
    if key == keys.A:
        state.key_a = value
        
    if key == keys.D:
        state.key_d = value


obj = moon if state.capslock else camera

linear_acceleration = 1 / 2
angular_acceleration = 1 / 24


@app.run_me_every(1/48)
def move_object(ct, dt):
        
    global obj
    
    obj = moon if state.capslock else camera
    sign = -1 if obj is camera else 1
    
    if state.right and state.shift:
        state.av.z += angular_acceleration * sign
            
    if state.right and not state.shift:
        state.av.y -= angular_acceleration
        
    if state.left and state.shift:
        state.av.z -= angular_acceleration * sign
            
    if state.left and not state.shift:
        state.av.y += angular_acceleration
            
    if state.up:
        state.av.x -= angular_acceleration
        
    if state.down:
        state.av.x += angular_acceleration
        
    if state.key_w and state.alt:
        state.lv.y += linear_acceleration
                
    if state.key_w and not state.alt:
        state.lv.z += linear_acceleration * sign
                
    if state.key_s and state.alt:
        state.lv.y -= linear_acceleration
        
    if state.key_s and not state.alt:
        state.lv.z -= linear_acceleration * sign
        
    if state.key_a:
        state.lv.x += linear_acceleration * sign
        
    if state.key_d:
        state.lv.x -= linear_acceleration * sign
        
    state.lv = glm.clamp(state.lv, -64, 64)
    state.av = glm.clamp(state.av, -64, 64)
    
    obj.move_local(dt * state.lv)
    
    obj.rotate_local(dt * state.av.x, (1, 0, 0))
    obj.rotate_local(dt * state.av.y, (0, 1, 0))
    obj.rotate_local(dt * state.av.z, (0, 0, 1))
    
    state.lv *= 0.67 ** dt
    state.av *= 0.67 ** dt


label0 = Label('Hello World!', color='white', font_size=14, x=10, y=74)
label1 = Label('Hello World!', color='white', font_size=14, x=10, y=52)
label2 = Label('Hello World!', color='white', font_size=14, x=10, y=30)
label3 = Label('Hello World!', color='white', font_size=14, x=10, y=8)

hello_world = Label('hello, world 3D!', color='cyan', font_size=24, x=575, y=10)


@app.event
def render(ct, dt):
        
    app.window.clear(blue=0.3)
    
    scene.draw()
    
    label0.text = 'time to draw - %0.3f ms' % (1000 * app._time2draw_rm)
    label1.text = 'up - %0.3f, %0.3f, %0.3f' % tuple(obj.up)
    label2.text = 'front - %0.3f, %0.3f, %0.3f' % tuple(obj.front)
    label3.text = 'position - %0.3f, %0.3f, %0.3f' % tuple(obj.position)
    
    label0.draw()
    label1.draw()  
    label2.draw()  
    label3.draw()  

    hello_world.draw()


@app.schedule_interval(1/30)
def spin(ct, dt):
    scene.meshes['Alien'].rotate_local(-0.5 * dt, (0, 0, 1))


if __name__ == '__main__':
    app.run()

