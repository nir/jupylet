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

import numpy as np

from ..utils import callerframe, callerpath

from .note import note2key


def sonic_py(resource_dir='.', **kwargs):
    """Start an audio application.

    An audio application is needed to run live loops.
    
    Args:
        resource_dir (str): Path to root of resource dir, for samples, etc...

    Returns:
        App: A running application object.
    """
    from ..app import App

    red = os.path.join(callerpath(), resource_dir)
    red = pathlib.Path(red).absolute()

    app = App(32, 32, resource_dir=str(red), **kwargs)
    app.run(0)
    
    return app


DEFAULT_AMP = 0.5

MIDDLE_C = 261.63

FS = FPS = 44100 # FPS is the old name and is kept for backward compatibility.


def t2frames(t):
    """Convert time in seconds to frames at 44100 frames per second.
    
    Args:
        t (float): The time duration in seconds.

    Returns:
        int: The number of frames.
    """
    return int(FS * t)


def frames2t(frames):
    """Convert frames at 44100 frames per second to time in seconds.
    
    Args:
        frames (int): The number of frames.

    Returns:
        float: The time duration in seconds.
    """
    return frames  / FS


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


_bpm = 120


def set_bpm(bpm=120):
    """Set the tempo to the given beats per minute.
    
    Args:
        bpm (float): Beats per minute.
    """
    global _bpm
    _bpm = bpm


def get_bpm():
    return _bpm


_safety_limit = 3.


def set_safety_limit(limit):
    """Set the highest peak of samples that jupylet will play.

    An array of samples that peaks past this limit, for example the output
    of a filter that ran away, is refused rather than played through the
    speakers. Full scale is 1.

    Args:
        limit (float): The highest peak allowed, or None to turn the check
            off.
    """
    global _safety_limit
    _safety_limit = limit


def get_safety_limit():
    return _safety_limit


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


def _is_note(s):
    """Tell whether a string names a note, like 'C4' or 'Eb'."""

    try:
        note2key(s)
        return True
    except (KeyError, ValueError, IndexError):
        return False


def play(note, duration=None, **kwargs):
    """Play given note polyphonically with the instrument previously set by
    call to :func:`use`, or play a sample.

    You can supply key/value pairs of properties to modify in the given
    instrument.

    Instead of a note, you can give a sample to play: a :class:`Sample`
    object, an array of samples at the sampling frequency FS, for example a
    sound computed in a notebook, or the path to an audio file of type WAV,
    OGG or FLAC. It plays as it is, with no need for :func:`use`. An array
    or a file plays at full amplitude, amp=1, unless given another amp.

    Args:
        note (float or str): Note to play in units of semitones
            where 60 is middle C, or as a string like 'C4'; or a sample.
        duration (float, optional): Duration to play note, in whole notes.
        **kwargs: Properties of intrument, or of the sample, to modify.

    Returns:
        GatedSound: The sound object representing the playing note or
            sample.
    """
    #
    # Imported here rather than at the top: importing it loads the whole
    # audio stack, which this module, imported by all of jupylet, avoids.
    # It also imports this module itself.
    #
    from .sample import Sample

    #
    # Arrays and audio files play at full amplitude by default, as they
    # would with sounddevice, since they carry their own levels.
    #
    if isinstance(note, (np.ndarray, pathlib.PurePath)) or (
        type(note) is str and not _is_note(note)
    ):
        note = Sample(note)
        kwargs.setdefault('amp', 1.)

    cf = callerframe()
    cn = cf.f_code.co_name
    
    if cn in ['<module>', 'async-def-wrapper']:
        hh = '<module>'
    elif cn.startswith('<cell line'):
        hh = '<cell line'
    else:
        hh = hash(cf)

    tt = dtd.get(hh) or get_time()
    tt += PLAY_EXTRA_LATENCY

    if isinstance(note, Sample):
        note.play(None, duration, t=tt, **kwargs)
        return #note

    sy = syd.get(hh)

    if sy is None:
        raise RuntimeError(
            'No instrument to play: call use(instrument) first, '
            'for example use(tb303).'
        )

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

