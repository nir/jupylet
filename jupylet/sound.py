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
import hashlib
import inspect
import logging
import random
import queue
import copy
import loky
import time
import sys
import os

import loky.backend.context
import loky.backend.queues
import concurrent.futures
import skimage.draw
import scipy.signal
import PIL.Image

import multiprocessing.process as process

try:
    import sounddevice as sd
except:
    sd = None

import soundfile as sf

import numpy as np

from .resource import find_path
from .utils import o2h, callerframe, trimmed_traceback
from .env import get_app_mode, is_remote, in_python_script


logger = logging.getLogger(__name__)


FPS = 44100


def t2frames(t):
    return int(FPS * t)


def frames2t(frames):
    return frames  / FPS


_pool0 = None
_queue = None


def submit(foo, *args, **kwargs):

    global _pool0
    global _queue

    assert not _is_worker, 'submit() should not be called by worker process.'
        
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
            initargs=(_queue,)
        )

    return _pool0.submit(foo, *args, **kwargs)


class MockFuture(object):
    def result(self):
        pass


_is_worker = False

def _init_worker(q):

    global _is_worker
    global _queue

    _is_worker = True
    _queue = q

    _init_sound_worker()
    

_worker_tid = None

def _init_sound_worker():
    
    global _worker_tid
    if not _worker_tid:
        _worker_tid = _thread.start_new_thread(_sound_worker, ())
        

_workerq = queue.Queue()

def _sound_worker():
    
    with sd.OutputStream(channels=2, callback=_stream_callback, latency=0.066):
        _workerq.get()
        
    global _worker_tid
    _worker_tid = None
    

#def _quit_sound_worker():
#    _soundsq.put('QUIT')
    

_soundsq = queue.Queue()
_soundsd = {}


def get_sounds():
    
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
      
    
def put_sounds(sounds):
    
    for sound in sounds:

        if not sound.done:
            _soundsq.put(sound)
        else:
            sound.playing = False

     
def mix_sounds(sounds, frames):
    
    d = np.stack([s._consume(frames) for s in sounds])
    return np.sum(d, 0).clip(-1, 1)


_al = []
_dt = []


def _stream_callback(outdata, frames, _time, status):
        
        t0 = time.time()

        if status:
            print(status, file=sys.stderr)
        
        sounds = get_sounds()
        
        if not sounds:
            a0 = np.zeros_like(outdata)
            outdata[:] = a0

            #_workerq.put('QUIT')
            #raise sd.CallbackStop
            
        else: 
            a0 = mix_sounds(sounds, frames)
            outdata[:] = a0
            put_sounds(sounds)

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


def proxy(write_only=False, wait=False, return_async=False, return_self=False):

    assert write_only or wait or return_async or return_self

    def proxy1(foo):

        s0 = inspect.getfullargspec(foo)
        is_class_method = s0.args and s0.args[0] == 'self'

        @functools.wraps(foo)
        def proxy0(*args, **kwargs):

            if _is_worker:
                return foo(*args, **kwargs)

            if is_class_method:
                self, args = args[0], args[1:]
                self_uid = self.uid
            else:
                self = None
                self_uid = None
            
            if is_remote() or get_app_mode() == 'hidden':
                return self if return_self else None

            fuu = submit(proxy_server, self_uid, foo.__name__, *args, **kwargs)

            if return_self:
                return self

            if write_only:
                return

            if wait:
                return fuu.result()

            return asyncio.wrap_future(fuu)
        
        return proxy0

    return proxy1


def proxy_server(_uid, name, *args, **kwargs):

    if _uid is None:
        foo = globals().get(name)
    elif _uid in _soundsd:
        foo = getattr(_soundsd[_uid], name)
    else:
        return

    if callable(foo):
        return foo(*args, **kwargs) 
    
    return foo


@proxy(write_only=True)
def _queue_put(x):
    _queue.put(x)


@proxy(wait=True)
def _eval(x, _repr_=True):

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
    
    w0, h0 = size
    w1, h1 = int(w0 // scale), int(h0 // scale)

    a0, ts, te = get_oscilloscope_as_array(fps, ms, amp, color, (w1, h1))
    #a0[:,w1//2:w1//2+1] = 255
    
    im = PIL.Image.fromarray(a0).resize(size)

    return im, ts, te


def get_oscilloscope_as_array(
    fps,
    ms=256, 
    amp=1., 
    color=255, 
    size=(512, 256)
):
    
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
    
    #while t0 - time.time() > 0:
    #    np.random.randn(1024)

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


@proxy(wait=True)  
def get_dt():
    return _dt


@functools.lru_cache(maxsize=128)
def load_sound(path, channels=2):

    data, fs = sf.read(path, dtype='float32') 

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
            
    if data.shape[-1] == 1 and channels > 1:
        data = data.repeat(channels, -1)

    if data.shape[-1] > channels:
        data = data[:, :channels]

    return data, fs


def _create_sound(classname, *args, **kwargs):

    _cls = globals()[classname]
    _sound = _cls(*args, **kwargs)
    _soundsd[_sound.uid] = _sound


def _delete_sound(uid):
    _soundsd.pop(uid, None)


def _getattr(uid, key):
    return getattr(_soundsd[uid], key)
    
    
def _setattr(uid, key, value):
    return setattr(_soundsd[uid], key, value)
    

def _create_uid():
    return o2h((random.random(), time.time()))


class Sample(object):
    
    def __init__(
        self, 
        path=None, 
        amp=1., 
        pan=0., 
        loop=False,
        duration=0.,
        attack=0.,
        decay=0.,
        sustain=1.,
        release=0.,
        uid=None,
    ):
        
        if not _is_worker and uid is not None:
            self.__dict__['uid'] = uid
            return

        if not _is_worker:
            path = str(find_path(path)) if path else None
            self.__dict__['uid'] = _create_uid()
            submit(
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
                self.__dict__['uid'],
            )
            return

        self.uid = uid
        self.path = path

        self.amp = amp
        self.pan = pan
        
        self.loop = loop or bool(duration)
        self.duration = duration

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

        self.index = 0
        self.indez = 0
        self.freq = FPS

    def __del__(self):

        try:
            if not _is_worker and self.__dict__.get('uid', -1) != -1:
                submit(_delete_sound, self.uid)
        
        except RuntimeError:
            pass

        except TypeError:
            pass

    def __getattr__(self, key):
        
        if key in self.__dict__:
            return self.__dict__[key]
            
        if not _is_worker:
            return submit(_getattr, self.uid, key).result()

        return super(Sample, self).__getattribute__(key)

    def __setattr__(self, key, value):

        if key in self.__dict__:
            self.__dict__[key] = value
            return

        if not _is_worker:
            submit(_setattr, self.uid, key, value)
            return
            
        return super(Sample, self).__setattr__(key, value)

    @proxy(wait=True)
    def __dir__(self):
        return super(Sample, self).__dir__()
    
    @proxy(write_only=True)
    def set_envelope(
        self, 
        duration=None, 
        attack=None, 
        decay=None, 
        sustain=None, 
        release=None
    ):

        if duration is not None:
            self.duration = duration

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
        
        if not self.duration:
            return

        adsr = (self.duration, self.attack, self.decay, self.sustain, self.release)
        if self.adsr0 != adsr:
            self.adsr0 = adsr
            
            dn, a0, d0, s0, r0 = adsr
            
            a0 = max(min(a0, dn), 0.01)
            d0 = max(min(d0, dn - a0), 0.01)
            dn = max(min(dn, dn - a0 - d0), 0.01)
            r0 = max(r0, 0.01)

            a1 = np.linspace(0., 1., int(FPS * a0), dtype='float32')
            d1 = np.linspace(1., s0, int(FPS * d0), dtype='float32')
            s1 = np.linspace(s0, s0, int(FPS * dn), dtype='float32')
            r1 = np.linspace(s0, 0., int(FPS * r0), dtype='float32')
            p1 = np.linspace(0., 0., 4096, dtype='float32')
            
            self.adsr1 = np.concatenate((a1, d1, s1, r1, p1)).reshape(-1, 1)

        return self.adsr1

    @proxy(write_only=True)
    def play_release(self, release=None):

        if release is not None:
            self.release = release
            
        self.duration = frames2t(self.indez)
        
    def play_new(self, note=None, **kwargs):
        """Play new copy of sound.

        If sound is already playing it will play the new copy in parallel. 
        This function returns the new sound object.
        """
        return self.copy().play(note, **kwargs)

    @proxy(return_self=True)
    def play(self, note=None, **kwargs):

        if note is not None:
            self.note = note

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)

        self.load()

        if self.playing:  
            self.reset = True
            return

        self.playing = True
        _soundsq.put(self)
            
    def copy(self, **kwargs):

        uid = _create_uid()
        self._copy(uid, **kwargs)
        return type(self)(uid=uid)

    @proxy(write_only=True)
    def _copy(self, uid, **kwargs):

        o = copy.copy(self)

        o.playing = False
        o.reset = False
        o._stop = False
        o.index = 0
        o.indez = 0
        o.uid = uid
        
        _soundsd[o.uid] = o

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)

    @proxy(return_self=True)
    def load(self, channels=2):
        
        self.buffer, self.freq = load_sound(self.path, channels)
        self.channels = int(self.buffer.shape[-1])
        self.dtype = self.buffer.dtype

        self.reset = False
        self._stop = False
        self.index = 0
        self.indez = 0

    @proxy(write_only=True)
    def stop(self):
        self._stop = True

    @property
    @proxy(wait=True)
    def done(self):
        
        if self._stop:
            return True
        
        if not self.loop:
            return self.remains <= 0
    
        if not self.duration:
            return False

        return frames2t(self.indez) >= self.duration + max(self.release, 0.01)

    @property
    @proxy(wait=True)
    def remains(self):
        return len(self.buffer) - self.index
        
    def _consume0(self, frames):
        
        if self.reset:
            self.reset = False
            self.index = 0
            self.indez = 0

        frames = min(frames, self.remains)
        
        data = self.buffer[self.index: self.index + frames]
        
        self.index += frames
        self.indez += frames
        
        if self.loop:
            self.index = self.index % len(self.buffer)
            
        return data
    
    @proxy(return_async=True)
    def _consume(self, frames):
        
        iz = self.indez
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
            b0 = b0 * self.get_adsr()[iz: iz + len(b0)]
        
        return b0


@functools.lru_cache(maxsize=1024)
def _ampan(amp, pan, dtype='float32'):
    a0 = np.array([1 - pan, 1 + pan]) * (amp / 2)
    return a0.astype(dtype)


dth = {}

def sleep(dt=0):
    
    dt0 = max(0, dt)
    fid = hash(callerframe())
    t00 = dth.get(fid) or time.time()
    dth[fid] = t00 + dt0
    
    return asyncio.sleep(dt0)


#play(C4, amp=2)
#sleep(0.5)


@functools.lru_cache(maxsize=32)
def get_sine_wave(cycles=1, frames=256):
    a0 = np.sin(np.linspace(0, 2 * np.pi, frames, dtype='float32'))
    return np.concatenate([a0] * cycles)


@functools.lru_cache(maxsize=32)
def get_saw_wave(cycles=1, frames=256):
    a0 = np.linspace(-1, 1, frames, dtype='float32')
    return np.concatenate([a0] * cycles)    


@functools.lru_cache(maxsize=32)
def get_triangle_wave(cycles=1, frames=256):
    a0 = np.linspace(-1, 1, frames//2, dtype='float32')
    a1 = np.linspace(1, -1, frames - frames//2, dtype='float32')
    a2 = np.concatenate((a0, a1)) 
    return np.concatenate([a2] * cycles)    


@functools.lru_cache(maxsize=32)
def get_pulse_wave(cycles=1, frames=256):
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
    'sine': get_sine_wave,
    'saw': get_saw_wave,
    'tri': get_triangle_wave,
    'pulse': get_pulse_wave,
}


@functools.lru_cache(maxsize=1024)
def get_note_wave(note, shape='sine', channels=2):
    
    foo = _shape_foo[shape]

    if type(note) in (int, float):
        freq = get_key_freq(note) if note < 100 else note
    else:
        freq = get_note_freq(note)
        
    cycles, frames = get_freq_cycles(freq)[:2]
    
    wave = foo(cycles)
    wave = scipy.signal.resample(wave, frames)

    data = wave.reshape(-1, 1)
    data = data.repeat(channels, -1)

    return data.astype('float32')


class Synth(Sample):
    
    def __init__(
        self,
        shape='sine',
        amp=1., 
        pan=0., 
        duration=0.5,
        attack=0.,
        decay=0.,
        sustain=1.,
        release=0.,
        note='C',
        uid=None,
    ):

        if not _is_worker and uid is not None:
            self.__dict__['uid'] = uid
            return

        if not _is_worker:
            self.__dict__['uid'] = _create_uid()
            submit(
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
                self.__dict__['uid'],
            )
            return

        super(Synth, self).__init__(
            amp=amp, 
            pan=pan, 
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
        
        self.buffer = get_note_wave(self.note, self.shape, channels)
        self.channels = int(self.buffer.shape[-1])
        self.dtype = self.buffer.dtype

        self.reset = False
        self._stop = False
        self.index = 0
        self.indez = 0
 
