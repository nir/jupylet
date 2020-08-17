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
import time
import os

import concurrent.futures

import multiprocessing.process as process

try:
    import sounddevice as sd
except:
    sd = None

import soundfile as sf

import numpy as np

from .resource import find_path
from .utils import o2h
from .env import get_app_mode, is_remote, in_python_script


logger = logging.getLogger(__name__)


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
    
    d = np.stack([
        s._consume_loop(frames) * s.volume * [1 - s.balance, s.balance] 
        for s in sounds
    ])

    return np.sum(d, 0)


def callback(outdata, frames, _time, status):
        
        if status:
            print(status, file=sys.stderr)
        
        sounds = get_sounds()
        
        if not sounds:
            _workerq.put('QUIT')
            raise sd.CallbackStop
            
        else: 
            outdata[:] = mix_sounds(sounds, frames)
            put_sounds(sounds)


def init_sound_worker0():
    
    if not _worker0:
        _worker0.append(_thread.start_new_thread(sound_worker0, ()))
        

def sound_worker0():
    
    with sd.OutputStream(channels=2, callback=callback, latency='low'):
        _workerq.get()
        
    _worker0.pop(0)
    

def quit_sound_worker():
    _soundsq.put('QUIT')
    
    
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


def _create_sound(path, volume, loop, balance):

    globals()['_is_worker'] = True

    sound = Sound(path, volume, loop, balance)
    _soundsd[sound.uid] = sound
    return sound.uid


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
    
    def __init__(self, path, volume=1., loop=False, balance=0.5):
        
        if not _is_worker:
            path = str(find_path(path))
            self.__dict__['uid'] = submit(_create_sound, path, volume, loop, balance).result()
            return

        self.uid = o2h((random.random(), time.time()))
        self.path = path
        self.volume = volume
        self.balance = balance
        
        self.playing = False
        self.reset = False
        self.loop = loop
        self.stop = False

        self.channels = 0
        self.buffer = []
        self.dtype = 'float32'
        self.index = 0
        self.freq = None

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
    def play(self, balance=None):
        """Hey."""

        init_sound_worker0()

        if balance is not None:
            self.balance = balance

        done = self.done

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
                
    @property
    @proxy(wait=True)
    def done(self):
        
        if self.stop:
            return True
        
        if self.loop:
            return False
        
        return self.remains <= 0
    
    @property
    @proxy(wait=True)
    def remains(self):
        return len(self.buffer) - self.index
        
    def _consume(self, l):
        
        if self.reset:
            self.reset = False
            self.index = 0

        l = min(l, self.remains)
        
        data = self.buffer[self.index: self.index+l]
        self.index += l
        
        if self.loop:
            self.index = self.index % len(self.buffer)
            
        return data
    
    @proxy()
    def _consume_loop(self, l):
        
        bl = []
        
        while l > 0:
                        
            if self.done:
                bl.append(np.zeros((l, self.channels), dtype=self.dtype))
            else:
                bl.append(self._consume(l))
                
            l -= len(bl[-1])
            
        return np.concatenate(bl, 0)

