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
import logging
import random
import queue
import copy
import time
import sys
import os

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
from .utils import o2h, callerframe
from .env import get_app_mode, is_remote, in_python_script


logger = logging.getLogger(__name__)


SPS = 44100


def t2samples(t):
    return SPS * t


def samples2t(samples):
    return samples  / SPS


class Mock(object):
    def result(self):
        pass


_pool = None


def submit(foo, *args, **kwargs):

    global _pool

    if sd is None:
        return Mock()
        
    if _pool is None and getattr(process.current_process(), '_inheriting', False):
        return Mock()

    if _pool is None:
        _pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        
    return _pool.submit(foo, *args, **kwargs)


_worker0 = []
_workerq = queue.Queue()
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

    return sounds
      
    
def put_sounds(sounds):
    
    for sound in sounds:
        if not sound.done:
            _soundsq.put(sound)
        else:
            sound.playing = False

     
def mix_sounds(sounds, frames):
    
    d = np.stack([s._consume_loop(frames) for s in sounds])
    return np.sum(d, 0).clip(-1, 1)


class _dt0(object):

    def reset(self):

        self.init = time.time()
        self.start = 0
        self.callback = 0
        self.out = 0
        self.cbl = []

_dt = _dt0()
_al = []


def callback(outdata, frames, _time, status):
        
        _dt.cbl.append((
            time.perf_counter_ns() / 10e8, 
            repr(frames), 
            repr(_time.inputBufferAdcTime),
            repr(_time.outputBufferDacTime),
            repr(_time.currentTime),
        ))
        _dt.cbl[:] = _dt.cbl[-16:]
        _dt.callback = _dt.callback or time.time()

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

        if len(_al) * frames > SPS:
            _al.pop(0)
        _al.append(a0)

        _dt.out = _dt.out or time.time()


def init_sound_worker0():
    
    if not _worker0:
        _dt.reset()
        _worker0.append(_thread.start_new_thread(sound_worker0, ()))
        

def sound_worker0():
    
    _dt.start = _dt.start or time.time()

    with sd.OutputStream(channels=2, callback=callback, latency=0.066):
        _workerq.get()
        
    _worker0.pop(0)
    

def quit_sound_worker():
    _soundsq.put('QUIT')
    

def get_oscilloscope_as_image(
    ms=1024, 
    amp=1., 
    color=255, 
    size=(512, 256),
    scale=2.
):
    
    w0, h0 = size
    w1, h1 = int(w0//scale), int(h0//scale)

    a0 = get_oscilloscope_as_array(ms, amp, color, (w1, h1))
    im = PIL.Image.fromarray(a0).resize(size)
    return im


def get_oscilloscope_as_array(ms=1024, amp=1., color=255, size=(512, 256)):
    
    w0, h0 = size

    a0 = get_array(SPS * ms // 1000, w0).mean(-1)
    a0 = (a0 * amp).clip(-1., 1.)
    #a0 = scipy.signal.resample(a0, w0)

    a1 = np.arange(len(a0))
    a2 = ((a0 + 1) * h0 / 2).clip(0, h0 - 1).astype(a1.dtype)
    a3 = np.stack((a2, a1), -1)

    a4 = np.concatenate((a3[:-1], a3[1:]), -1)
    a5 = np.zeros((h0, w0, 4))
    a5[...,:3] = color

    for cc in a4:
        y, x, c = skimage.draw.line_aa(*cc)
        a5[y, x, -1] = c * 255
    
    return a5.astype('uint8')


def get_dt():
    return submit(_get_dt).result()


def _get_dt():
    return vars(_dt)


def get_array(steps=SPS, width=None):

    try:
        return submit(_get_array, steps, width).result()
    except:
        return np.zeros((width or SPS, 2))


def _get_array(steps=SPS, width=None):
    a0 = np.concatenate(_al)[int(-steps):]
    if width:
        a0 = scipy.signal.resample(a0, width)
    return a0

   
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


_is_worker = False


def proxy_server(uid, name, *args, **kwargs):

    globals()['_is_worker'] = True

    sound = _soundsd[uid]
    foo = getattr(sound, name)
    
    if callable(foo):
        return foo(*args, **kwargs) 
        
    return foo


def proxy(write_only=False, wait=False, return_self=False):

    def proxy1(foo):

        def proxy0(self, *args, **kwargs):

            if _is_worker:
                return foo(self, *args, **kwargs)

            if is_remote() or get_app_mode() == 'hidden':
                return self if return_self else None

            fuu = submit(proxy_server, self.uid, foo.__name__, *args, **kwargs)

            if return_self:
                return self

            if write_only:
                return

            if wait:
                return fuu.result()

            return asyncio.wrap_future(fuu)
        
        proxy0.__qualname__ = foo.__name__
        proxy0.__doc__ = foo.__doc__
        
        return proxy0

    return proxy1


def _create_sound(classname, *args, **kwargs):

    globals()['_is_worker'] = True

    _cls = globals()[classname]
    _sound = _cls(*args, **kwargs)
    _soundsd[_sound.uid] = _sound

    return _sound.uid


def _delete_sound(uid):

    globals()['_is_worker'] = True

    _soundsd.pop(uid)


def _dir(uid):

    globals()['_is_worker'] = True

    return dir(_soundsd[uid])
    
    
def _getattr(uid, key):

    globals()['_is_worker'] = True

    return getattr(_soundsd[uid], key)
    
    
def _setattr(uid, key, value):

    globals()['_is_worker'] = True

    return setattr(_soundsd[uid], key, value)
    

class Sound(object):
    
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
    ):
        
        if not _is_worker:
            path = str(find_path(path)) if path else None
            self.__dict__['uid'] = submit(
                _create_sound, 
                'Sound',
                path, 
                amp, 
                pan, 
                loop,
                duration,
                attack,
                decay,
                sustain,
                release,
            ).result()
            return

        self.uid = o2h((random.random(), time.time()))
        self.path = path

        self.amp = amp
        self.pan = pan
        
        self.loop = loop or duration
        self.duration = duration

        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release

        self.adsr0 = None
        self.adsr1 = None

        self.playing = False
        self.reset = False
        self.stop = False

        self.channels = 0
        self.buffer = []
        self.dtype = 'float32'
        self.index = 0
        self.indez = 0
        self.freq = SPS

    def __del__(self):

        try:
            if not _is_worker and 'uid' in self.__dict__:
                submit(_delete_sound, self.uid)
        
        except RuntimeError:
            pass

        except TypeError:
            pass

    def __dir__(self):

        if not _is_worker:
            return submit(_dir, self.uid).result()
        
        return super(Sound, self).__dir__()
    
    def __getattr__(self, key):
        
        if key in self.__dict__:
            return self.__dict__[key]
            
        if not _is_worker:
            return submit(_getattr, self.uid, key).result()

        return super(Sound, self).__getattribute__(key)

    def __setattr__(self, key, value):

        if key in self.__dict__:
            self.__dict__[key] = value
            return

        if not _is_worker:
            submit(_setattr, self.uid, key, value)
            return
            
        return super(Sound, self).__setattr__(key, value)

    @proxy(write_only=True)
    def set_envelope(self, duration=None, attack=None, decay=None, sustain=None, release=None):

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
            dn = min(dn, dn - a0 - d0)
            r0 = max(r0, 0.01)

            a1 = np.linspace(0., 1., int(SPS * a0))
            d1 = np.linspace(1., s0, int(SPS * d0))
            s1 = np.linspace(s0, s0, int(SPS * dn))
            r1 = np.linspace(s0, 0., int(SPS * r0))
            p1 = np.linspace(0., 0., 4096)
            
            self.adsr1 = np.concatenate((a1, d1, s1, r1, p1)).reshape(-1, 1)

        return self.adsr1

    @proxy(write_only=True)
    def play_copy(self, **kwargs):

        o = copy.copy(self)
        o.reset = False
        o.index = 0
        o.indez = 0
        o.play(**kwargs)

    @proxy(write_only=True)
    def play(self, **kwargs):
        """Hey."""

        init_sound_worker0()

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)

        self.load()

        if self.playing:  
            self.reset = True
            return

        self.playing = True
        _soundsq.put(self)
            
    @proxy(return_self=True)
    def load(self, channels=2):
        
        self.buffer, self.freq = load_sound(self.path, channels)
        
        self.channels = int(self.buffer.shape[-1])
        self.dtype = self.buffer.dtype
        self.reset = False
        self.index = 0
        self.indez = 0
                
    @property
    @proxy(wait=True)
    def done(self):
        
        if self.stop:
            return True
        
        if not self.loop:
            return self.remains <= 0
    
        if not self.duration:
            return False

        return samples2t(self.indez) >= self.duration + max(self.release, 0.01)

    @property
    @proxy(wait=True)
    def remains(self):
        return len(self.buffer) - self.index
        
    def _consume(self, l):
        
        if self.reset:
            self.reset = False
            self.index = 0
            self.indez = 0

        l = min(l, self.remains)
        
        data = self.buffer[self.index: self.index+l]
        
        self.index += l
        self.indez += l
        
        if self.loop:
            self.index = self.index % len(self.buffer)
            
        return data
    
    @proxy()
    def _consume_loop(self, l):
        
        iz = self.indez
        bl = []
        
        while l > 0:
                        
            if self.done:
                bl.append(np.zeros((l, self.channels), dtype=self.dtype))
            else:
                bl.append(self._consume(l))
                
            l -= len(bl[-1])
            
        b0 = np.concatenate(bl, 0)
        b0 = b0 * self.amp * [1 - self.pan, 1 + self.pan] / 2.

        if self.duration:
            b0 = b0 * self.get_adsr()[iz: iz + len(b0)]
        
        return b0


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
def get_sine_wave(cycles=1, samples=256):
    a0 = np.sin(np.linspace(0, 2 * np.pi, samples, dtype='float32'))
    return np.concatenate([a0] * cycles)


@functools.lru_cache(maxsize=32)
def get_saw_wave(cycles=1, samples=256):
    a0 = np.linspace(-1, 1, samples, dtype='float32')
    return np.concatenate([a0] * cycles)    


@functools.lru_cache(maxsize=32)
def get_triangle_wave(cycles=1, samples=256):
    a0 = np.linspace(-1, 1, samples//2, dtype='float32')
    a1 = np.linspace(1, -1, samples - samples//2, dtype='float32')
    a2 = np.concatenate((a0, a1)) 
    return np.concatenate([a2] * cycles)    


@functools.lru_cache(maxsize=32)
def get_pulse_wave(cycles=1, samples=256):
    a0 = np.ones((samples,), dtype='float32')
    a0[:samples//2] *= -1.
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


def get_freq_cycles(f, sps=44100, max_cycles=32):
    
    f0 = sps / f
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
    
    if type(note) is int:
        freq = get_key_freq(note)
    else:
        freq = get_note_freq(note)
        
    cycles, samples = get_freq_cycles(freq)[:2]
    
    wave = foo(cycles)
    wave = scipy.signal.resample(wave, samples)

    data = wave.reshape(-1, 1)
    data = data.repeat(channels, -1)

    return data.astype('float32')


class Synth(Sound):
    
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
    ):

        if not _is_worker:
            self.__dict__['uid'] = submit(
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
            ).result()
            return

        super(Synth, self).__init__(
            amp=amp, 
            pan=pan, 
            duration=duration,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
        )

        self.shape = shape
        self.note = note

    @proxy(return_self=True)
    def load(self, channels=2):
        
        self.buffer = get_note_wave(self.note, self.shape, channels)
        
        self.channels = int(self.buffer.shape[-1])
        self.dtype = self.buffer.dtype
        self.reset = False
        self.index = 0
        self.indez = 0
 
