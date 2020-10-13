"""
    jupylet/audio/__init__.py
    
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


import asyncio
import pathlib
import time
import os

from ..utils import callerframe, callerpath


def sonic_py(resource_dir='.'):

    from ..app import App

    red = os.path.join(callerpath(), resource_dir)
    red = pathlib.Path(red).absolute()

    app = App(32, 32, resource_dir=str(red))
    app.run(0)
    
    return app


FPS = 44100


def t2frames(t):
    """Convert time in seconds to frames at 44100 frames per second.
    
    Args:
        t (float): The time duration in seconds.

    Returns:
        int: The number of frames.
    """
    return int(FPS * t)


def frames2t(frames):
    """Convert frames at 44100 frames per second to time in seconds.
    
    Args:
        frames (int): The number of frames.

    Returns:
        float: The time duration in seconds.
    """
    return frames  / FPS


def get_time():
    return time.time()
  

_note_value = 4


def set_note_value(v=4):
    global _note_value
    _note_value = v


def get_note_value():
    return _note_value


_bpm = 240


def set_bpm(bpm=4):
    global _bpm
    _bpm = bpm


def get_bpm():
    return _bpm


dtd = {}
syd = {}


def use(synth, **kwargs):

    if kwargs:
        synth = synth.copy().set(**kwargs)

    cf = callerframe()
    cn = cf.f_code.co_name
    hh = cn if cn == '<module>' else hash(cf) 

    syd[hh] = synth


PLAY_EXTRA_LATENCY = 0.150


def play(note, *args, **kwargs):

    cf = callerframe()
    cn = cf.f_code.co_name
    hh = cn if cn == '<module>' else hash(cf) 

    sy = syd[hh]
    
    tt = dtd.get(hh) or get_time()
    tt += PLAY_EXTRA_LATENCY

    return sy.play_new(note, t=tt, *args, **kwargs)


def sleep(dt=0):
    
    tt = get_time()

    cf = callerframe()
    cn = cf.f_code.co_name
    hh = cn if cn == '<module>' else hash(cf) 

    sy = syd.get(hh)
    if sy is not None:
        dt = dt * get_note_value() * 60 / get_bpm()

    t0 = dtd.get(hh)
    if not t0 or t0 + 1 < tt:
        t0 = tt

    t1 = dtd[hh] = max(t0 + dt, tt)

    return asyncio.sleep(t1 - tt)


def stop():
    
    from .device import stop_sound
    stop_sound()

