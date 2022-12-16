"""
    jupylet/audio/midi.py
    
    Copyright (c) 2022, Nir Aides - nir.8bit@gmail.com

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

#
# python-rtmidi requires the following command on ubuntu:
# sudo apt-get install libasound2-dev libjack-dev build-essentials
#

import logging

try:
    import rtmidi
    import mido
except:
    rtmidi = None
    mido = None


logger = logging.getLogger(__name__)


def test_rtmidi():

    if rtmidi:
        return True

    logger.warning(
        'Module python-rtmidi is not installed. To install it on Ubuntu ' + 
        'Linux run these commands:\n' +
        '$ sudo apt-get install libasound2-dev libjack-dev build-essentials\n' +
        '$ pip install python-rtmidi'
    )


def get_input_names():
    return mido.get_input_names()


def get_input_name():
    
    if _input_name:
        return _input_name

    inl = mido.get_input_names()
    if inl:
        return inl[0]
        

def set_input_name(name=None):

    global _input_name
    _input_name = name


_input_name = None
_port = None


def midi_port_handler(*args):
    logger.info('Enter midi_port_handler(*args=%r).', args)
    
    global _input_name
    global _port
    
    if mido is None or rtmidi is None:
        return
    
    name = get_input_name()

    input_names = mido.get_input_names()
    
    if name not in input_names or not _callback:
        if _port is not None:
            logger.info('Close midi port.')
            _port.close()
            _port = None 
        return
    
    if _port is not None:
        if _port.name == name:
            return

        logger.info('Close midi port.')
        _port.close()
    
    try:
        logger.info('Call mido.open_input(name=%r, callback=%r).', name, _callback)
        _port = mido.open_input(name=name, callback=_callback)
    except rtmidi._rtmidi.SystemError:
        pass


_callback = None


def set_midi_callback(callback):
    logger.info('Enter set_midi_callback(callback=%r).', callback)

    global _callback
    global _port

    _callback = callback

    if _port is not None:
        _port.callback = callback


_sound = None


def set_midi_sound(s):
    logger.info('Enter set_midi_sound(s=%r).', s)

    global _sound
    _sound = s


_keyd = {}


def simple_midi_callback(msg):
    #logger.debug('Enter simple_midi_callback(msg=%r).', msg)

    if _sound is None:
        return

    if msg.type == 'note_on':

        if msg.note not in _keyd and msg.velocity != 0:
            _keyd[msg.note] = _sound.play_poly(key=msg.note, velocity=msg.velocity)

        elif msg.note in _keyd:
            _keyd.pop(msg.note).play_release()
            
    elif getattr(msg, 'note', None) in _keyd:
        _keyd.pop(msg.note).play_release()

