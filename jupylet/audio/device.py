"""
    jupylet/audio/device.py
    
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


import platform
import _thread
import logging
import queue
import copy
import time

import skimage.draw
import scipy.signal
import PIL.Image

try:
    import sounddevice as sd
except:
    sd = None

import numpy as np


logger = logging.getLogger(__name__)


FPS = 44100
    

_worker_tid = None


def _init_worker_thread():
    logger.info('Enter _init_worker_thread().')

    global _worker_tid
    if not _worker_tid:
        _worker_tid = _thread.start_new_thread(_start_sound_stream, ())
        
#
# This queue is only used to keep the sound stream running.
#
_workerq = queue.Queue()


def _start_sound_stream():
    """Start the sound device output stream handler."""
    logger.info('Enter _start_sound_stream().')
    
    if platform.system() == 'Windows':
        latency = 'low'
    else:
        latency = None

    with sd.OutputStream(channels=2, callback=_stream_callback, latency=latency):
        _workerq.get()
        
    global _worker_tid
    _worker_tid = None
    

_al_seconds = 1
_al = []
_dt = []


def _stream_callback(outdata, frames, _time, status):
    """Compute and set the sound data to be played in a few milliseconds.

    Args:
        outdata (ndarray): The sound device output buffer.
        frames (int): The number of frames to compute and set into the output
            buffer.
        _time (struct): A bunch of clocks.
    """
    t0 = time.time()

    if status:
        logger.warning('Stream callback called with status: %r.', status)
    
    #
    # Get all sound objects currently playing, mix them, and set the mixed
    # array into the output buffer.
    #

    sounds = _get_sounds()
    
    if not sounds:
        a0 = np.zeros_like(outdata)
        outdata[:] = a0

        #_workerq.put('QUIT')
        #raise sd.CallbackStop
        
    else: 
        a0 = _mix_sounds(sounds, frames)
        outdata[:] = a0

    # 
    # Aggregate the output data and timers for the oscilloscope.
    #

    if len(_al) * frames > _al_seconds * FPS:
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


#
# A set of all currently playing sound objects.
#
_sounds = set()


def _add_sound(sound):
    """Add sound to the set of currently playing sound objects."""

    if sd is None:
        return

    if _worker_tid is None:
        _init_worker_thread()

    _sounds.add(sound)

         
def _get_sounds():
    """Get currently playing sound objects."""

    for s in list(_sounds):
        if s.done:
            _sounds.discard(s)

    return list(_sounds)
      

def _mix_sounds(sounds, frames):
    """Mix sound data from given sounds into a single numpy array.

    Args:
        sounds: Currently playing sound objects.
        frames (int): Number of frames to consume from each sound object.

    Returns:
        ndarray
    """
    d = np.stack([s._consume(frames) for s in sounds])
    return np.sum(d, 0).clip(-1, 1)


def get_oscilloscope_as_image(
    fps,
    ms=256, 
    amp=1., 
    color=255, 
    size=(512, 256),
    scale=2.
):
    """Get an oscilloscope image of sound data sent to the sound device.

    Args:
        fps (float): frame rate (in frames per second) of the monitor on which 
            the oscilloscope is to be displayed. The given rate will affect the
            visual stability of the displayed wave forms.
        ms (float): Span of the oscilloscope x-axis in milliseconds.
        amp (float): Scale of the y-axis (amplification of the signal).
        color (int or tuple): Color of the drawn sound wave.
        size (tuple): size of oscilloscope in pixels given as (width, height).

    Returns:
        Image, float, float: a 3-tuple with the oscilloscope image, and the 
            timestamps of the leftmost and rightmost drawn samples.
    """
    w0, h0 = size
    w1, h1 = int(w0 // scale), int(h0 // scale)

    a0, ts, te = get_oscilloscope_as_array(fps, ms, amp, color, (w1, h1))
    im = PIL.Image.fromarray(a0).resize(size).transpose(PIL.Image.FLIP_TOP_BOTTOM)

    return im, ts, te


def get_oscilloscope_as_array(
    fps,
    ms=256, 
    amp=1., 
    color=255, 
    size=(512, 256)
):
    """Get an oscilloscope array of sound data sent to the sound device.

    Args:
        fps (float): frame rate (in frames per second) of the monitor on which 
            the oscilloscope is to be displayed. The given rate will affect the
            visual stability of the displayed wave forms.
        ms (float): Span of the oscilloscope x-axis in milliseconds.
        amp (float): Scale of the y-axis (amplification of the signal).
        color (int or tuple): Color of the drawn sound wave.
        size (tuple): size of oscilloscope in pixels given as (width, height).

    Returns:
        ndarray, float, float: a 3-tuple with the oscilloscope array, and the 
            timestamps of the leftmost and rightmost drawn samples.        
    """
    
    ms = min(ms, 256)
    w0, h0 = size

    oa = get_output_as_array()
    if not oa or len(oa[0]) < 1000:
        return np.zeros((h0, w0, 4), dtype='uint8'), 0, 0

    a0, tstart, tend = oa

    t0 = int((time.time() + 0.01) * fps) / fps
    te = t0 + min(0.025, ms / 2000)
    ts = te - ms / 1000

    s1 = int((te - tend) * FPS)
    s0 = int((ts - tend) * FPS)

    a0 = a0[s0: s1] * amp
    a0 = a0.clip(-1., 1.)

    if len(a0) < 100:
        return np.zeros((h0, w0, 4), dtype='uint8'), 0, 0

    a0 = scipy.signal.resample(a0, w0)

    a1 = np.arange(len(a0))
    a2 = ((a0 + 1) * h0 / 2).clip(0, h0 - 1).astype(a1.dtype)
    a3 = np.stack((a2, a1), -1)

    a4 = np.concatenate((a3[:-1], a3[1:]), -1)
    a5 = np.zeros((h0, w0, 4), dtype='uint8')
    a5[...,:3] = color

    for cc in a4:
        y, x, c = skimage.draw.line_aa(*cc)
        a5[y, x, -1] = c * 255
    
    return a5, ts, te


def get_output_as_array(start=-FPS, end=None, resample=None):

    if not _dt or not _al:
        return

    t0, st, _, da, ct = _dt[-1]
    t1 = t0 + da - ct + st / FPS

    a0 = np.concatenate(_al).mean(-1)[start:end]
    
    tend = t1 + (end or 0) / FPS
    tstart = tend - len(a0) / FPS

    if resample:
        a0 = scipy.signal.resample(a0, resample)
    
    return a0, tstart, tend

