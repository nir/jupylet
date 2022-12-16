"""
    jupylet/audio/device.py
    
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


import platform
import _thread
import logging
import queue
import copy
import time

import scipy.signal
import PIL.Image

try:
    import sounddevice as sd
except:
    sd = None

import numpy as np

from ..audio import FPS
from ..env import is_sphinx_build


logger = logging.getLogger(__name__)
    

def disable_audio():
    global sd
    sd = None

    
_schedule = None


def set_schedule(t):
    global _schedule
    _schedule = t


def get_schedule():
    return _schedule


_worker_tid = None


def _init_worker_thread():
    logger.info('Enter _init_worker_thread().')

    global _worker_tid
    
    if not _worker_tid and sd is not None:
        _worker_tid = _thread.start_new_thread(_start_sound_stream, ())
        
#
# This queue is only used to keep the sound stream running.
#
_workerq = queue.Queue()
_sp = {}


def _start_sound_stream():
    """Start the sound device output stream handler."""
    logger.info('Enter _start_sound_stream().')
    
    #if platform.system() == 'Windows':
    #    _sp['latency'] = 'low'
    #else:
    #    _sp['latency'] = None

    while True:
        with sd.OutputStream(samplerate=FPS, channels=2, callback=_stream_callback, **_sp):
            _workerq.get()
        
    global _worker_tid
    _worker_tid = None
    

LOWEST_LATENCY = 0.050


def set_device_latency(latency='high'):

    assert latency in ['high', 'low', 'lowest', 'minimal']

    if latency in ['lowest', 'minimal']:
        _set_stream_params(latency=LOWEST_LATENCY, blocksize=1024)
        return

    _set_stream_params(latency=latency, blocksize=None)


def get_device_latency_ms(latency='high'):

    assert latency in ['high', 'low', 'lowest', 'minimal']

    if latency in ['lowest', 'minimal']:
        return LOWEST_LATENCY * 1000

    if sd is None or is_sphinx_build():
        return 100

    dd = sd.query_devices(sd.default.device[-1])

    return dd['default_%s_output_latency' % latency] * 1000


def _set_stream_params(**kwargs):
    _sp.update(kwargs)
    _workerq.put(1)


_al_seconds = 1
_al = []


def start_recording(limit=60):

    global _al_seconds
    global _al
    
    _al_seconds = limit    
    _al.clear()


def stop_recording():

    global _al_seconds

    a0 = np.concatenate(_al)
    _al_seconds = 1
    return a0


_dt = []
_safety_event0 = 0


def _stream_callback(outdata, frames, _time, status):
    """Compute and set the sound data to be played in a few milliseconds.

    Args:
        outdata (ndarray): The sound device output buffer.
        frames (int): The number of frames to compute and set into the output
            buffer.
        _time (struct): A bunch of clocks.
    """
    global _safety_event0

    t0 = time.time()
    dt = _time.outputBufferDacTime - _time.currentTime

    set_schedule(t0 + dt)
    
    if status:
        logger.warning('Stream callback called with status: %r.', status)
        _safety_event0 = t0

    #
    # Get all sound objects currently playing, mix them, and set the mixed
    # array into the output buffer.
    #

    sounds = _get_sounds()
    
    if not sounds:
        a0 = np.zeros_like(outdata)        
    else: 
        a0 = _mix_sounds(sounds, frames)

    a0 = _apply_effects(_effects, a0)

    a0 = (a0 * _master_volume).clip(-1, 1)
    
    if t0 < _safety_event0 + 1:
        a0 *= max(0.01, min(1, t0 - _safety_event0))

    outdata[:] = a0

    # 
    # Aggregate the output data and timers for the oscilloscope.
    #

    while len(_al) * frames > _al_seconds * FPS:
        _al.pop(0)
        _dt.pop(0)

    _al.append(a0)
    _dt.append((
        t0, 
        frames, 
        _time.inputBufferAdcTime,
        _time.outputBufferDacTime,
        _time.currentTime,
    ))


_master_volume = 0.5


def get_master_volume():
    return _master_volume


def set_master_volume(amp):
    global _master_volume
    _master_volume = amp


_effects = []

 
def get_effects():
    return _effects


def set_effects(*effects):

    global _effects

    if effects and effects[0] is None:
        _effects = []
    
    elif effects and type(effects[0]) in (list, tuple):
        _effects = effects[0]

    else:
        _effects = effects


#
# A list of all currently playing sound objects.
#
_sounds0 = []
_sounds1 = []


def stop_sound():
    _sounds0.clear()
    _sounds1.clear()


def add_sound(sound):
    """Add sound to the set of currently playing sound objects."""

    if sd is None:
        return

    if _worker_tid is None:
        _init_worker_thread()

    _sounds0.append(sound)

         
def _get_sounds():
    """Get currently playing sound objects."""

    sd = set()
    sl = []

    while _sounds0:
        s = _sounds0.pop(0)
        if s not in sd and not s.done:
            sl.append(s)
            sd.add(s)

    while _sounds1:
        s = _sounds1.pop(0)
        if s not in sd and not s.done:
            sl.append(s)
            sd.add(s)

    _sounds1[:] = sl

    return sl
      

def _mix_sounds(sounds, frames):
    """Mix sound data from given sounds into a single numpy array.

    Args:
        sounds: Currently playing sound objects.
        frames (int): Number of frames to consume from each sound object.

    Returns:
        ndarray
    """
    es = {}

    for s0 in sounds:
        el = s0.get_effects()
        es.setdefault(el, []).append(s0)

    al = []

    for el, sl in es.items():
        a0 = _mix_sounds0(sl, frames)
        a0 = _apply_effects(el, a0)
        al.append(a0)
        
    return np.stack(al).sum(0)


def _mix_sounds0(sounds, frames):

    al = [s.consume(frames) for s in sounds]
    return np.stack(al).sum(0)


def _apply_effects(effects, a):

    for e in effects or []:
        a = e(a)
    return a


def get_output_as_array(start=-FPS, length=None, mono=False, resample=None):

    _dt0, _al0 = _dt, _al
    
    if not _dt0 or not _al0:
        return None, None, None

    t0, st, _, da, ct = _dt0[-1]
    t1 = t0 + da - ct + st / FPS

    if start >= 0:
        start = max(start - t1, -1)

    if start >= 0:
        return None, None, None

    if type(start) is float:
        start = max(start, -1)
        start = int(start * FPS)

    else:
        start = int(start)

    if length is None:
        end = None

    elif length <= 0:
        return None, None, None

    elif type(length) is float:
        length = min(length, 1)
        length = int(length * FPS)

    end = start + length

    if end > 0:
        return None, None, None

    if end == 0:
        end = None

    ll = 0
    _al1 = []

    for a0 in _al0[::-1]:
        _al1.append(a0)
        ll += len(a0)
        if ll >= -start:
            break

    _al2 = _al1[::-1]

    a0 = np.concatenate(_al2)[start:end]
    
    tend = t1 + (end or 0) / FPS
    tstart = tend - len(a0) / FPS

    if mono:
        a0 = a0.mean(-1, keepdims=True)
        
    if resample and len(a0) != resample:
        a0 = scipy.signal.resample(a0, resample)
    
    if not mono and a0.shape[-1] == 1:
        a0 = np.stack((a0, a0), -1)
        
    return a0, tstart, tend

