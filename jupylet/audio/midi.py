"""
    jupylet/audio/midi.py
    
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
import _thread
import logging

try:
    import rtmidi
    import mido

except:
    rtmidi = None
    mido = None


logger = logging.getLogger(__name__)


_port = None


def midi_port_handler(*args):
    
    global _port
    
    if mido is None or rtmidi is None:
        return
    
    input_names = mido.get_input_names()
    
    if not input_names or not _callback:
        if _port is not None:
            _port.close()
            _port = None 
        return
          
    name = input_names[0]
    
    if _port is not None and _port.name != name:
        _port.close()
    
    try:
        _port = mido.open_input(name=name, callback=_callback)
    except rtmidi._rtmidi.SystemError:
        pass


_callback = None


def set_midi_callback(callback):

    global _callback
    global _port

    _callback = callback

    if _port is not None:
        _port.callback = callback


_sound = None


def set_midi_sound(s):

    global _sound
    _sound = s


_keyd = {}


def simple_midi_callback(msg):

    if msg.type == 'note_on':

        if msg.velocity != 0 and _sound is not None:
            _keyd[msg.note] = _sound.play_new(key=msg.note, velocity=msg.velocity)
            
        elif msg.note in _keyd:
            _keyd[msg.note].play_release()

