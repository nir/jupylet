"""
    jupylet/audio/__init__.py
    
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


import asyncio
import pathlib
import time
import os

from ..utils import callerframe, callerpath


def sonic_py(resource_dir='.'):
    """Start an audio application.

    An audio application is need to run live loops.
    
    Args:
        resource_dir (str): Path to root of resource dir, for samples, etc...

    Returns:
        App: A running application object.
    """
    from ..app import App

    red = os.path.join(callerpath(), resource_dir)
    red = pathlib.Path(red).absolute()

    app = App(32, 32, resource_dir=str(red))
    app.run(0)
    
    return app


DEFAULT_AMP = 0.5

MIDDLE_C = 261.63

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
    """Set the note value representing one beat.
    
    Args:
        v (float): Note value.
    """
    global _note_value
    _note_value = v


def get_note_value():
    return _note_value


_bpm = 240


def set_bpm(bpm=240):
    """Set the tempo to the given beats per minute.
    
    Args:
        bpm (float): Beats per minute.
    """
    global _bpm
    _bpm = bpm


def get_bpm():
    return _bpm


dtd = {}
syd = {}


def use(sound, **kwargs):
    """Set the instrument to use in subsequent calls to :func:`play`.
    
    You can supply key/value pairs of properties to modify in the given 
    instrument. If you do, the instrument will be copied first, and
    the modifications will be applied to the new copy.

    Args:
        sound (GatedSound): Instrument to use.
        **kwargs: Properties of intrument to modify.
    """
    if kwargs:
        sound = sound.copy().set(**kwargs)

    cf = callerframe()
    cn = cf.f_code.co_name

    if cn in ['<module>', 'async-def-wrapper']:
        hh = '<module>'
    elif cn.startswith('<cell line'):
        hh = '<cell line'
    else:
        hh = hash(cf) 

    syd[hh] = sound


PLAY_EXTRA_LATENCY = 0.150


def play(note, duration=None, **kwargs):
    """Play given note polyphonically with the instrument previously set by 
    call to :func:`use`.
    
    You can supply key/value pairs of properties to modify in the given 
    instrument.

    Args:
        note (float): Note to play in units of semitones 
            where 60 is middle C.
        duration (float, optional): Duration to play note, in whole notes.
        **kwargs: Properties of intrument to modify.
    """

    cf = callerframe()
    cn = cf.f_code.co_name
    
    if cn in ['<module>', 'async-def-wrapper']:
        hh = '<module>'
    elif cn.startswith('<cell line'):
        hh = '<cell line'
    else:
        hh = hash(cf) 

    sy = syd[hh]
    
    tt = dtd.get(hh) or get_time()
    tt += PLAY_EXTRA_LATENCY

    return sy.play_poly(note, duration, t=tt, **kwargs)


def sleep(duration=0):
    """Get some sleep.
    
    Example: 
        ::
    
            @app.sonic_live_loop2
            async def boom_pam():
                        
                use(tb303, resonance=8, decay=1/8, cutoff=48, amp=1)
                
                play(C2, 1/8)
                await sleep(1/4)
                
                play(C3, 1/8)
                await sleep(1/4)

    Args:
        duration (float): Duration to sleep in whole notes.

    Returns:
        coroutine: A sleep coroutine to use with `await`.
    """
    tt = get_time()
    dt = duration

    cf = callerframe()
    cn = cf.f_code.co_name

    if cn in ['<module>', 'async-def-wrapper']:
        hh = '<module>'
    elif cn.startswith('<cell line'):
        hh = '<cell line'
    else:
        hh = hash(cf) 

    #sy = syd.get(hh)
    #if sy is not None:
    dt = dt * get_note_value() * 60 / get_bpm()

    t0 = dtd.get(hh)
    if not t0 or t0 + 1 < tt:
        t0 = tt

    t1 = dtd[hh] = max(t0 + dt, tt)

    return asyncio.sleep(t1 - tt)


def stop():
    
    from .device import stop_sound
    stop_sound()

