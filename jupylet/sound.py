"""
    jupylet/sound.py
    
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
import asyncio
import inspect
import logging
import random
import queue
import copy
import loky
import math
import time
import sys
import os

import loky.backend.context
import loky.backend.queues
import skimage.draw
import scipy.signal
import PIL.Image

try:
    import sounddevice as sd
except:
    sd = None

import soundfile as sf

import numpy as np

from .resource import find_path
from .utils import o2h, callerframe, trimmed_traceback
from .utils import setup_basic_logging, get_logging_level
from .env import get_app_mode, is_remote


logger = logging.getLogger(__name__)


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


#
# The sound server runs in a dedicated worker process 
# (living in a pool of size 1).
#
_pool0 = None

#
# This queue can be used to communicate information asynchronously
# from the sound server to the client process.
#
_queue = None


def _submit(foo, *args, **kwargs):
    """Submit a job to the sound server.

    Args:
        foo (function): The function to run at the sound server process.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        A future object representing the submitted job.
    """
    logger.debug('Enter _submit(foo=%r, *args=%r, **kwargs=%r).', foo, args, kwargs)

    global _pool0
    global _queue

    assert not _is_worker, '_submit() should not be called by worker process.'
        
    if sd is None:
        return MockFuture()
        
    if _pool0 is None:
        _ctx00 = loky.backend.context.get_context('loky')
        _queue = loky.backend.queues.Queue(ctx=_ctx00)
        _pool0 = loky.get_reusable_executor(
            max_workers=1, 
            context = _ctx00,
            timeout=2**20,
            initializer=_init_worker,
            initargs=(_queue, get_logging_level())
        )

    return _pool0.submit(foo, *args, **kwargs)


class MockFuture(object):
    """Mock future objects for remote headless servers. 

    We don't want them to start playing sounds in their remote server rack.
    """
    def result(self):
        pass


# 
# This flag is used to determine if code is running at server or
# client process.
#
_is_worker = False


def _init_worker(q, logging_level=logging.WARNING):
    """Init the sound server worker."""

    global _is_worker
    global _queue

    _is_worker = True
    _queue = q

    setup_basic_logging(logging_level)
    _init_worker_thread()
    

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
    
    #
    # Latency is set to 66ms since on Windows it appears to invoke the callback 
    # regularly at nearly 10ms which is the effective resolution of sleep 
    # "wakeups" on Windows.
    # 
    # The default 'low' latency appears to be 100ms (which is a little high)
    # and alternates between calling the callback at 10ms and 20ms intervals.
    # 
    # Such alternation would cause (ear) noticable delays with starting the 
    # playing of repeating sounds. This problem could and should be fixed by 
    # adding a mechanism to schedule the start of new sounds to a particular 
    # frame in the output stream buffer.
    #
    with sd.OutputStream(channels=2, callback=_stream_callback, latency='low'):#0.066):
        _workerq.get()
        
    global _worker_tid
    _worker_tid = None
    

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
        _put_sounds(sounds)

    # 
    # Aggregate the output data and timers for the oscilloscope.
    #

    if len(_al) * frames > FPS:
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


#def _quit_sound_worker():
#    _soundsq.put('QUIT')


#
# A server-side queue for all currently playing sound objects.
#
_soundsq = queue.Queue()


def _get_sounds():
    """Get all sound objects from the queue of currently playing sounds."""

    sounds = []
    
    while not _soundsq.empty():
        
        sound = _soundsq.get()
        if sound == 'QUIT':
            return
        
        if not sound.done:
            sounds.append(sound)
        else:
            sound.playing = False

    return sounds
      
    
def _put_sounds(sounds):
    """Put sound objects into the queue of currently playing sounds."""
    
    for sound in sounds:

        if not sound.done:
            _soundsq.put(sound)
        else:
            sound.playing = False

     
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


def proxy(wait=False, write_only=False, return_self=False, return_async=False):
    """Decorate function or sound object method to transparently run it in the 
    sound server process.

    Args:
        wait (bool): Emulate a regular function call.
        write_only (bool): Submit function to run in the server and return None
            immediately. Should take about 50-100 usec to run.
        return_self (bool): Submit function to run in the server and return self
            immediately. Should take about 50-100 usec to run.
        return_async (bool): Submit function to run in the server and return a
            waitable future object.
    """
    assert write_only or wait or return_async or return_self

    def proxy1(foo):

        #
        # Figure out if foo is a regular function or a sound object method.
        #
        s0 = inspect.getfullargspec(foo)
        is_class_method = s0.args and s0.args[0] == 'self'

        @functools.wraps(foo)
        def proxy0(*args, **kwargs):

            #
            # If on sound server, call the function and return its result.
            # 
            if _is_worker:
                return foo(*args, **kwargs)

            if is_class_method:
                self, args = args[0], args[1:]
                self_uid = self.uid
            else:
                self = None
                self_uid = None
            
            #
            # On remote servers or in headless mode bypass the sound module,
            # since we do not want sound in these scenarios.
            #
            if is_remote() or get_app_mode() == 'hidden':
                return self if return_self else None

            fuu = _submit(_proxy_server, self_uid, foo.__name__, *args, **kwargs)

            if return_self:
                return self

            if write_only:
                return

            if wait:
                return fuu.result()

            return asyncio.wrap_future(fuu)
        
        return proxy0

    return proxy1


#
# A server-side dictionary for all sound objects currently referenced by the 
# client process.
#
_soundsd = {}


_debug_level_names = {
    'get_output_as_array': 0,
}


def _proxy_server(_uid, name, *args, **kwargs):
    """Run submitted function by its name (and object uid).
    
    If _uid is not None but curresponding object is not found in the sound 
    objects dictionary, silently ignore the request.
    """
    if name not in _debug_level_names:
        logging.info('Enter _proxy_server(_uid=%r, name=%r, *args=%r, **kwargs=%r).', _uid, name, args, kwargs)
    else:
        logging.debug('Enter _proxy_server(_uid=%r, name=%r, *args=%r, **kwargs=%r).', _uid, name, args, kwargs)
        if not _debug_level_names[name]:
            _debug_level_names[name] = 1
            logging.info('Messages for %s() will only be shown in DEBUG logging level.' % name)

    if _uid is None:
        foo = globals().get(name)
    elif _uid in _soundsd:
        foo = getattr(_soundsd[_uid], name)
    else:
        return

    if callable(foo):
        return foo(*args, **kwargs) 
    
    return foo


@proxy(wait=True)
def _eval(x, _repr_=True):
    """Debug function to evaluate arbitrary expressions in the sound server."""
    logging.info('Enter _eval(x=%r, _repr_=%r).', x, _repr_)
    
    try:
        if _repr_:
            return repr(eval(x))
        else:
            return eval(x)
    except:
        return trimmed_traceback()


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


@proxy(wait=True)
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


@functools.lru_cache(maxsize=128)
def load_sound(path, channels=2):
    """Load sound file as a numpy array with given number of sound channels."""
    logger.info('Enter load_sound(path=%r, channels=%r).', path, channels)
    
    data, fs = sf.read(path, dtype='float32') 

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
            
    if data.shape[-1] == 1 and channels > 1:
        data = data.repeat(channels, -1)

    if data.shape[-1] > channels:
        data = data[:, :channels]

    return data, fs


def _create_sound(classname, *args, **kwargs):
    logger.info('Enter _create_sound(classname=%r, *args=%r, **kwargs=%r).', classname, args, kwargs)

    _cls = globals()[classname]
    _sound = _cls(*args, **kwargs)
    _soundsd[_sound.uid] = _sound


def _delete_sound(uid):
    logger.info('Enter _delete_sound().')
    _soundsd.pop(uid, None)


def _getattr(uid, key):
    logger.info('Enter _getattr(uid=%r, key=%r).', uid, key)
    return getattr(_soundsd[uid], key)
    
    
def _setattr(uid, key, value):
    logger.info('Enter _setattr(uid=%r, key=%r, value=%r).', uid, key, value)
    return setattr(_soundsd[uid], key, value)
    

def _generate_uid():
    """Generate unique sound object id."""
    logger.debug('Enter _generate_uid().')
    return o2h((random.random(), time.time()))


class Sample(object):
    
    def __init__(
        self, 
        path=None, 
        amp=0.5, 
        pan=0., 
        loop=False,
        duration=0.,
        attack=0.,
        decay=0.,
        sustain=1.,
        release=0.,
        **kwargs,
    ):
        """A sound sample.

        Args:
            path (str): Path to a WAV, OGG, or FLAC sound file.
            amp (float): Sound volume with a value between 0. and 1.
            pan (float): Balance between left (-1.) and right (1.) speakers.
            loop (bool): Play sound in an endless loop (default False).
            duration (float): Duration in seconds to play sound (0. plays entire 
                sound sample).
            attack (float): Time in seconnds to reach maximum volume.
            decay (float): Time in seconnds to decay volume to sustain level.
            sustain (float): Volume level at which sound is sustained for its 
                duration.
            release (float): Time in seconds to decay volume to 0. once play
                duration is up.
        """
        uid = kwargs.pop('uid', None)
        
        #
        # This code runs on the client, when the server is about to create
        # a copy of an existing sound. See the copy() method below.
        #
        if not _is_worker and uid is not None:
            self.__dict__['uid'] = uid
            return

        #
        # This code runs on the client to create a new sound object on the 
        # sound server. The client sound object will serve as its proxy.
        #
        if not _is_worker:
            path = str(find_path(path)) if path else None
            self.__dict__['uid'] = _generate_uid()
            _submit(
                _create_sound, 
                'Sample',
                path, 
                amp, 
                pan, 
                loop,
                duration,
                attack,
                decay,
                sustain,
                release,
                uid=self.__dict__['uid'],
            )
            return

        #
        # This code runs on the sound server
        #

        self.uid = uid
        self.path = path

        self.amp = amp
        self.pan = pan
        
        self.loop = loop or duration > 0
        self.duration = duration

        self.min_duration = 0
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release

        self.adsr0 = None
        self.adsr1 = None

        self.playing = False
        self.reset = False
        self._stop = False

        self.channels = 0
        self.buffer = []
        self.dtype = 'float32'

        #self.level0 = 0
        self.index = 0
        self.freq = FPS

    def __del__(self):

        try:
            if not _is_worker and self.__dict__.get('uid', -1) != -1:
                _submit(_delete_sound, self.uid)
        
        except RuntimeError:
            pass

        except TypeError:
            pass

    def __getattr__(self, key):
        
        if key in self.__dict__:
            return self.__dict__[key]
            
        if not _is_worker:
            return _submit(_getattr, self.uid, key).result()

        return super(Sample, self).__getattribute__(key)

    def __setattr__(self, key, value):

        if key in self.__dict__:
            self.__dict__[key] = value
            return

        if not _is_worker:
            _submit(_setattr, self.uid, key, value)
            return
            
        return super(Sample, self).__setattr__(key, value)

    @proxy(wait=True)
    def __dir__(self):
        return super(Sample, self).__dir__()
    
    @proxy(write_only=True)
    def set_envelope(
        self, 
        min_duration=None, 
        attack=None, 
        decay=None, 
        sustain=None, 
        release=None
    ):

        if min_duration is not None:
            self.min_duration = min_duration

        if attack is not None:
            self.attack = attack

        if decay is not None:
            self.decay = decay

        if sustain is not None:
            self.sustain = sustain

        if release is not None:
            self.release = release

    @proxy(wait=True)
    def get_adsr(self):
        
        #if self.duration <= 0:
        #    return

        adsr = (
            self.duration, 
            self.min_duration, 
            self.attack, 
            self.decay, 
            self.sustain, 
            self.release
        )

        if self.adsr0 != adsr:
            self.adsr0 = adsr
            
            dn, md, a0, d0, sv, r0 = adsr
            
            ds = int(FPS * max(dn, md))

            a0 = max(a0, 0.01)
            a1 = np.linspace(0., 1., int(FPS * a0), dtype='float32')
            a2 = a1[:max(1, ds)]
            av = a2[-1]

            d0 = max(d0, 0.01)
            d1 = np.linspace(av, sv, int(FPS * d0), dtype='float32')
            d2 = d1[:max(1, ds - len(a2))]
            dv = d2[-1]

            s0 = ds + 128 if dv > 0 else 1
            s1 = np.linspace(dv, dv, s0, dtype='float32')
            s2 = s1[:max(1, ds - len(a2) - len(d2))]

            r0 = max(r0, 0.01)
            r1 = np.linspace(dv, 0., int(FPS * r0), dtype='float32')
            p0 = np.linspace(0., 0., 4096, dtype='float32')
            
            self.adsr1 = np.concatenate((a2, d2, s2, r1, p0)).reshape(-1, 1)

        return self.adsr1

    @proxy(write_only=True)
    def play_release(self, release=None):

        if release is not None:
            self.release = release
            
        self.duration = frames2t(self.index)
        
    def play_new(self, note=None, **kwargs):
        """Play new copy of sound.

        If sound is already playing it will play the new copy in parallel. 
        This function returns the new sound object.
        """
        return self.copy().play(note, **kwargs)

    @proxy(return_self=True)
    def play(self, note=None, **kwargs):
        logger.info('Enter Sample.play(note=%r, **kwargs=%r).', note, kwargs)
        
        if note is not None:
            self.note = note

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)

        self.load()

        #
        # Set the reset flag to avoid synchronization / consistency problems
        # with the playback thread.
        #
        if self.playing:  
            self.reset = True
            return

        self.playing = True
        _soundsq.put(self)
            
    def copy(self, **kwargs):

        uid = _generate_uid()
        self._copy(uid, **kwargs)
        return type(self)(uid=uid)

    @proxy(write_only=True)
    def _copy(self, uid, **kwargs):

        o = copy.copy(self)

        o.playing = False
        o.reset = False
        o._stop = False
        o.index = 0
        o.uid = uid
        
        _soundsd[o.uid] = o

        for k, v in kwargs.items():
            if k in o.__dict__:
                setattr(o, k, v)

    @proxy(return_self=True)
    def load(self, channels=2):
        
        self.buffer, self.freq = load_sound(self.path, channels)
        self.channels = int(self.buffer.shape[-1])
        self.dtype = self.buffer.dtype

        self.reset = False
        self._stop = False
        self.index = 0

    @proxy(write_only=True)
    def stop(self):
        self._stop = True

    @property
    @proxy(wait=True)
    def done(self):
        
        if self._stop:
            return True
        
        if not self.loop:
            return self.index >= len(self.buffer)
    
        if not self.duration:
            return False

        adsr = self.get_adsr()
        if adsr is None:
            return False

        if len(adsr) <= self.index:
            return True

        if self.index < 8:
            return False

        return adsr[self.index].max() == 0

    def _consume0(self, frames):
        
        if self.reset:
            self.reset = False
            self.index = 0

        index = self.index % len(self.buffer)
        data0 = self.buffer[index: index + frames]

        self.index += len(data0)
        
        return data0
    
    @proxy(return_async=True)
    def _consume(self, frames):
        
        ix = self.index
        bl = []
        
        while frames > 0:
                        
            if self.done:
                bl.append(np.zeros((frames, self.channels), dtype=self.dtype))
            else:
                bl.append(self._consume0(frames))
                
            frames -= len(bl[-1])
            
        b0 = np.concatenate(bl, 0)
        b0 = b0 * _ampan(self.amp, self.pan)

        if self.duration:
            b0 = b0 * self.get_adsr()[ix: ix + len(b0)]
        
        #
        # Interpolate start of play with previous last amplitude level.
        #
        #if self.index == 0:
        #    il = min(32, len(data0))
        #    ip = np.linspace(0, 1, il).reshape(-1, 1)
        #    data0 = data0 * 1.
        #    data0[:il] = ip * data0[:il] + (1. - ip) * self.level0
        # ?  self.level0 = data0[-1]
        #

        return b0


@functools.lru_cache(maxsize=1024)
def _ampan(amp, pan, dtype='float32'):
    a0 = np.array([1 - pan, 1 + pan]) * (amp / 2)
    return a0.astype(dtype)


dtd = {}
syd = {}


def use(synth, **kwargs):

    if kwargs:
        synth = synth.copy(**kwargs)

    cf = hash(callerframe())
    syd[cf] = synth


def play(note, **kwargs):

    cf = hash(callerframe())
    sy = syd[cf]

    return sy.play_new(note, **kwargs)


def sleep(dt=0):
    
    dt = max(0, dt)
    cf = hash(callerframe())

    tt = time.time()
    t0 = dtd.get(cf) or tt
    t1 = dtd[cf] = max(t0 + dt, tt)
    
    return asyncio.sleep(t1 - tt)


@functools.lru_cache(maxsize=32)
def _get_sine_wave(cycles=1, frames=256):
    a0 = np.sin(np.linspace(0, 2 * np.pi, frames, dtype='float32'))
    return np.concatenate([a0] * cycles)


@functools.lru_cache(maxsize=32)
def _get_saw_wave(cycles=1, frames=256):
    a0 = np.linspace(-1, 1, frames, dtype='float32')
    return np.concatenate([a0] * cycles)    


@functools.lru_cache(maxsize=32)
def _get_triangle_wave(cycles=1, frames=256):
    a0 = np.linspace(-1, 1, frames//2, dtype='float32')
    a1 = np.linspace(1, -1, frames - frames//2, dtype='float32')
    a2 = np.concatenate((a0, a1)) 
    return np.concatenate([a2] * cycles)    


@functools.lru_cache(maxsize=32)
def _get_pulse_wave(cycles=1, frames=256):
    a0 = np.ones((frames,), dtype='float32')
    a0[:frames//2] *= -1.
    return np.concatenate([a0] * cycles)


_notes = dict(
    C = 1,
    Cs = 2, Db = 2,
    D = 3,
    Ds = 4, Eb = 4,
    E = 5,
    F = 6,
    Fs = 7, Gb = 7,
    G = 8, 
    Gs = 9, Ab = 9,
    A = 10,
    As = 11, Bb = 11,
    B = 12,
)


class note(object):
    pass


for o in range(8):
    for k in _notes:
        ko = (k + str(o)).rstrip('0')
        setattr(note, ko, ko)


def get_note_key(note):
    
    if note[-1].isdigit():
        octave = int(note[-1])
        note = note[:-1]
    else:
        octave = 4
        
    note = note.replace('#', 's')
    return _notes[note] + octave * 12 + 11


def get_note_freq(note):
    return get_key_freq(get_note_key(note))


def get_key_freq(key, a4=440):
    return a4 * 2 ** ((key - 69) / 12)


def get_freq_cycles(f, fps=44100, max_cycles=32):
    
    f0 = fps / f
    mr = 1000
    mc = 1
    
    for i in range(1, 1 + max_cycles):
        
        fi = f0 * i
        fr = fi % 1
        fr = min(fr, 1 - fr)
        #print(fr, fi)
        
        if mr > fr / fi / 0.06:
            mr = fr / fi / 0.06
            mc = i
            
            if mr < 0.003:# or f0 * mc > 2000:
                break
            
    return mc, round(f0 * mc), round(mr, 4)


_shape_foo = {
    'sine': _get_sine_wave,
    'saw': _get_saw_wave,
    'tri': _get_triangle_wave,
    'pulse': _get_pulse_wave,
}


def get_note_wave(note, shape='sine', channels=2):
    
    if freq is not None:
        return 
    if isinstance(note, np.generic):
        note = float(note)

    if type(note) in (int, float):
        freq = get_key_freq(note) if note < 100 else note
    else:
        freq = get_note_freq(note)
        
    return get_freq_wave(freq, shape, channels)


@functools.lru_cache(maxsize=1024)
def get_freq_wave(freq, shape='sine', channels=2):

    cycles, frames = get_freq_cycles(freq)[:2]
    
    foo = _shape_foo[shape]

    wave = foo(cycles)
    wave = scipy.signal.resample(wave, frames)

    data = wave.reshape(-1, 1)
    data = data.repeat(channels, -1)

    return data.astype('float32')


class Synth(Sample):
    
    def __init__(
        self,
        shape='sine',
        amp=0.5, 
        pan=0., 
        duration=0.5,
        attack=0.,
        decay=0.,
        sustain=1.,
        release=0.,
        note='C',
        **kwargs,
    ):
        """A simple single voice synthesizer.

        Args:
            shape (str): Waveform shape - one of sine, tri, saw, and pulse.
            amp (float): Sound volume with a value between 0. and 1.
            pan (float): Balance between left (-1.) and right (1.) speakers.
            duration (float): Duration in seconds to play sound (0. plays entire 
                sound sample).
            attack (float): Time in seconnds to reach maximum volume.
            decay (float): Time in seconnds to decay volume to sustain level.
            sustain (float): Volume level at which sound is sustained for its 
                duration.
            release (float): Time in seconds to decay volume to 0. once play
                duration is up.
            note (str or float):  
        """
        uid = kwargs.pop('uid', None)

        if not _is_worker and uid is not None:
            self.__dict__['uid'] = uid
            return

        if not _is_worker:
            self.__dict__['uid'] = _generate_uid()
            _submit(
                _create_sound, 
                'Synth',
                shape, 
                amp, 
                pan, 
                duration,
                attack,
                decay,
                sustain,
                release,
                note,
                uid=self.__dict__['uid'],
            )
            return

        super(Synth, self).__init__(
            amp=amp, 
            pan=pan, 
            loop=True,
            duration=duration,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            uid=uid,
        )

        self.shape = shape
        self.note = note

    @proxy(return_self=True)
    def load(self, channels=2):
        
        self.buffer = get_note_wave(
            note=self.note, 
            shape=self.shape, 
            channels=channels
        )
        
        self.channels = int(self.buffer.shape[-1])
        self.dtype = self.buffer.dtype

        self.reset = False
        self._stop = False
        self.index = 0
 

# ------------------------


FPS = 44100

class Sound(object):
    
    def __init__(self):
                
        self.frames = 1024        
        self.index = 0
                
    def rset(self, key, value, force=False):
        
        if force or self.__dict__.get(key, '__NONE__') != value:
            for s in self.__dict__.values():
                if isinstance(s, Sound):
                    s.rset(key, value, force=True)
            
        self.__dict__[key] = value
            
    def __call__(self, *args, **kwargs):
        
        assert getattr(self, 'frames', None) is not None, 'You must call super() from your sound class constructor'
        
        a0 = self.forward(*args, **kwargs)
        self.index += self.frames
        
        return a0

        
    def forward(self, *args, **kwargs):
        return np.zeros((self.frames, self.channels))

def _expand_channels(a0, channels):

    if len(a0.shape) == 1:
        a0 = np.expand_dims(a0, -1)

    if a0.shape[1] < channels:
        a0 = a0.repeat(channels, 1)

    if a0.shape[1] > channels:
        a0 = a0[:,:channels]

    return a0


class Gate(Sound):
    
    def __init__(self):
        
        super(Gate, self).__init__()
        
        self.states = []
        
    def forward(self):
        
        states = []
        end = self.index + self.frames
        
        while self.states:
            if self.states[0][0] >= end:
                break
            states.append(self.states.pop(0))
        
        return states
        
    def open(self, t):
        self._append(t, 'open')
        
    def release(self, t):
        self._append(t, 'release')
        
    def _append(self, t, event):
        
        lasti = self.index
        if self.states:
            lasti = self.states[-1][0]

        frame = max(int(t * FPS), lasti)
        self.states.append((frame, event))


def get_exponential_adsr_curve(dt, start=0, end=None, th=0.01, eps=1e-6):
    
    df = max(math.ceil(dt * FPS), eps)
    end = min(df + 1, end or 60 * FPS)
        
    a0 = np.arange(start/df, end/df - eps, 1/df, dtype='float32')
    a1 = np.exp(a0 * math.log(th))
    a2 = (1. - a1) / (1. - th)
    
    return a2


def get_linear_adsr_curve(dt, start=0, end=None, eps=1e-6):
    
    df = max(math.ceil(dt * FPS), eps)
    end = min(df + 1, end or 60 * FPS)

    a0 = np.arange(start/df, end/df, 1/df, dtype='float32')
    
    return a0


class Envelope(Sound):
    
    def __init__(
        self, 
        attack=0.,
        decay=0., 
        sustain=1., 
        release=0.,
        linear=True,
    ):
        
        super(Envelope, self).__init__()
        
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.linear = linear
        
        self._state = None
        self._start = 0
        self._valu0 = 0
        self._valu1 = 0
        
    def forward(self, states):
        
        end = self.index + self.frames
        index = self.index
        states = states + [(end, 'continue')]
        curves = []
        
        for frame, event in states:
            #print(frame, event)
            
            while True:
                curves.append(self.get_curve(index, frame))
                index += len(curves[-1])
                if index >= frame:
                    break
                    
            if event == 'open':
                self._state = 'attack'
                self._start = index
                self._valu0 = self._valu1
            
            if event == 'release':
                self._state = 'release'
                self._start = index
                self._valu0 = self._valu1
            
        return np.concatenate(curves)[:,None]
    

    def get_curve(self, start, end):
        
        if self._state in (None, 'sustain'):
            return np.ones((end - start,), dtype='float32') * self._valu0
        
        start = start - self._start
        end = end - self._start
        dt = getattr(self, self._state)
                    
        if self.linear:
            curve = get_linear_adsr_curve(dt, start, end)
        else:
            curve = get_exponential_adsr_curve(dt, start, end)
    
        #print(dt, start, end, len(curve), curve[-1])
        done = curve[-1] >= 0.9999
        
        if self._state == 'attack':
            target = 1.
            next_state = 'decay'
            
        if self._state == 'decay':
            target = self.sustain * self._valu0
            next_state = 'sustain'
            
        if self._state == 'release':
            target = 0.
            next_state = None
            
        curve = (target - self._valu0) * curve  + self._valu0
        
        if done:
            self._state = next_state
            self._start += start + len(curve)
            self._valu0 = curve[-1]
            
        self._valu1 = curve[-1]
        
        return curve
    
    @property
    def done(self):
        return self._state is None and self._start > 0


#
# Do not change this "constant"!
#
_NP_ZERO = np.zeros((1,), dtype='float32')


def get_radians(freq, phase=0, frames=8192):
    
    pt = 2 * math.pi / FPS * freq
    
    if isinstance(pt, np.ndarray):
        pt = pt.reshape(-1)
    else:
        pt = pt * np.ones((frames,), dtype='float32')
            
    p0 = phase % (2 * math.pi) + _NP_ZERO
    p1 = np.concatenate((p0, pt))
    p2 = np.cumsum(p1)
    
    radians = p2[:-1]
    phase_o = p2[-1] % (2 * math.pi)
    
    return radians, phase_o

def get_sine_wave(freq, phase=0, frames=8192, **kwargs):
    
    radians, phase_o = get_radians(freq, phase, frames)
    
    a0 = np.sin(radians)
    
    return a0, phase_o


def get_triangle_wave(freq, phase=0, frames=8192, **kwargs):
    
    radians, phase_o = get_radians(freq, phase, frames)

    a0 = radians % (2 * math.pi)
    a1 = a0 / math.pi - 1
    a2 = a1 * np.sign(-a1)
    a3 = a2 * 2 + 1

    return a3, phase_o


def get_saw_wave(freq, phase=0, frames=8192, sign=1., **kwargs):
    
    radians, phase_o = get_radians(freq, phase, frames)

    a0 = (radians + math.pi) % (2 * math.pi) * sign / math.pi - 1
    
    return a0, phase_o


def get_pulse_wave(freq, phase=0, frames=8192, duty=0.5, **kwargs):
    
    if isinstance(duty, np.ndarray):
        duty = duty.reshape(-1)
        
    radians, phase_o = get_radians(freq, phase, frames)

    a0 = radians % (2 * math.pi) < (2 * math.pi * duty)    
    a1 = a0 * 2 - 1
    
    return a1, phase_o


_LOG_C4 = math.log(262)
_LOG_CC = math.log(2) / 12
_LOG_CX = _LOG_C4 - 60 * _LOG_CC


def key2freq(key):
    
    if isinstance(key, np.ndarray):
        return np.exp(key * _LOG_CC + _LOG_CX)
    else:
        return math.exp(key * _LOG_CC + _LOG_CX)
    
 
def freq2key(freq):
    
    if isinstance(freq, np.ndarray):
        return (np.log(freq) - _LOG_CX) / _LOG_CC
    else:
        return (math.log(freq) - _LOG_CX) / _LOG_CC
        

class Oscillator(Sound):
    
    def __init__(self, shape='sine', freq=262., phase=0., duty=0.5, sign=1., key=None):
        
        super(Oscillator, self).__init__()
        
        self.shape = shape
        self.phase = phase
        self.freq = freq
        
        if key is not None:
            self.key = key
        
        self.duty = duty
        self.sign = sign
        
    def forward(self, osc=None, duty=None):
        
        if duty is None:
            duty = self.duty
            
        if osc is not None:
            freq = key2freq(self.key + osc)
        else:
            freq = self.freq
        
        get_wave = dict(
            sine = get_sine_wave,
            tri = get_triangle_wave,
            saw = get_saw_wave,
            pulse = get_pulse_wave,
        )[self.shape]
        
        a0, self.phase = get_wave(
            freq, 
            self.phase, 
            self.frames, 
            duty=duty, 
            sign=self.sign
        )
        
        return a0[:,None]
    
    @property
    def key(self):
        return freq2key(self.freq)
    
    @key.setter
    def key(self, value):
        self.freq = key2freq(value)


class ButterFilter(Sound):
    
    def __init__(self, order=15, freq=8192, btype='lowpass'):
        
        super(ButterFilter, self).__init__()
        
        self.order = order
        self.freq = freq
        self.btype = btype
        
        self._watch = None
        
        self.b = None
        self.a = None
        self.z = None
        
    def forward(self, x):
        
        if self._watch != (self.order, self.freq, self.btype):
            
            self._watch = (self.order, self.freq, self.btype)
            self.b, self.a = signal.butter(self.order, self.freq / FPS * 2, self.btype)
            self.z = signal.lfilter_zi(self.b, self.a)[:,None]
           
            if self.z.shape[1] != x.shape[1]:
                self.z = self.z.repeat(x.shape[1], -1)[:,:x.shape[1]]
            
            # Warmup
            self.z = signal.lfilter(self.b, self.a, x * 0, 0, zi=self.z)[-1]
            
        x, self.z = signal.lfilter(self.b, self.a, x, 0, zi=self.z)
            
        return x.astype('float32')
    

class PhaseModulator(Sound):
    
    def __init__(self, beta=1.):
        """A sort of phase modulator.
        
        It can be used for aproximate frequency modulation by using the 
        normalized cumsum of the modulated signal, but the signal should be 
        balanced so its cumsum does not drift.
        """
        
        super(PhaseModulator, self).__init__()
        
        self.beta = beta
        
        self._buffer = None
        
    def forward(self, c, s):
        
        beta = int(self.beta) + 1
        
        if self._buffer is None:
            self._buffer = np.zeros((2 * beta, c.shape[1]), dtype=c.dtype)
            
        t1 = np.arange(beta, beta + len(c), dtype='float32') + self.beta * s.mean(-1).clip(-1, 1)
        t2 = t1.astype('int32')
        t3 = (t1 - t2.astype('float32'))[:, None]
        
        a0 = np.concatenate((self._buffer, c))
        a1 = a0[t2]
        a2 = a0[t2 + 1]
        a3 = a2 * t3 + a1 * (1 - t3)
        
        self._buffer = a0[-2 * beta:]
        
        return a3

        