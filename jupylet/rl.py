"""
    jupylet/rl.py
    
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


import functools
import importlib
import itertools
import builtins
import datetime
import platform
import tempfile
import hashlib
import random
import pickle
import mmap
import time
import sys
import os

import multiprocessing as mp
import numpy as np

from .env import set_rl_worker
from .utils import trimmed_traceback
from .audio.device import disable_audio

try:
    from multiprocessing import shared_memory
except:
    shared_memory = None
    

SCALARS = {str, bytes, int, float, bool}


def is_scalar(a):
    return type(a) in SCALARS


def rgetattr(o, name):
    
    nl = name.split('.')
    for n in nl:
        o = getattr(o, n)
        
    return o


def rsetattr(o, name, value):
    
    nn = name.rsplit('.', 1)
    
    if len(nn) > 1:
        o = rgetattr(o, nn[0])
        name = nn[1]
        
    setattr(o, name, value)


@functools.lru_cache()
def get_shared_memory(name):
    return shared_memory.SharedMemory(name=name)


def load(o, depth=1):

    if type(o) in [int, float, str]:
        return o

    if type(o) is tuple and len(o) == 4 and o[0] == '__ndarray__':
        _, shape, dtype, name = o
        return np.ndarray(shape, dtype, buffer=get_shared_memory(name).buf)

    if depth and type(o) in [list, tuple]:
        return type(o)(load(v, depth-1) for v in o)

    if depth and type(o) is dict:
        return {k: load(v, depth-1) for k, v in o.items()}

    return o


class ModuleProcess(object):
    
    def __init__(self, name, debug=False):
        
        self.name = name
        self.debug = debug

        self.c0, c0 = mp.Pipe()
        self.c1 = self.c0

        self.p0 = mp.Process(target=self._worker0, args=(c0,))
        
    def __del__(self):
        self.stop()
        
    def _worker0(self, c0):
        
        set_rl_worker()
        disable_audio()
        
        self.c0 = c0
        self.c1 = c0

        module = importlib.import_module(self.name)
        
        for name, args, kwargs in iter(self.c0.recv, 'STOP'):            
            
            try:
                if kwargs is not None:
                    foo = getattr(builtins, name, None) or rgetattr(module, name)
                    self._c1_send(foo(*args, **kwargs))
                    
                elif args is not None:
                    rsetattr(module, name, args)

                else:
                    self._c1_send(rgetattr(module, name))
             
            except:
                self.c1.send(('ee', trimmed_traceback()))
                   
    """def log(self, msg, *args):
        if self.debug:
            msg = msg % args
            date = datetime.datetime.now().isoformat()[:23].replace('T', ' ')
            self.q3.put('[%s] [%s]  %s' % (date, self.p0.ident, msg))"""

    def start(self):
        
        if self.p0.ident is not None:
            return

        #assert 'pyglet' not in sys.modules, 'Don\'t import pyglet or jupylet modules except jupylet.rl before starting worker processes.'

        self.p0.start()
    
    def get(self, name):
        self.send(name, None, None)
        return self.recv()

    def set(self, name, value):
        self.send(name, value, None)
        
    def call(self, name, *args, **kwargs):
        self.send(name, args, kwargs)
        return self.recv()
    
    def send(self, name, args, kwargs):
        self.c0.send((name, args, kwargs))
    
    def recv(self):
        return self._c0_recv()
    
    def stop(self):

        if self.p0.ident is None or not self.p0.is_alive():
            return

        try:
            self.c0.send('STOP')
            self.p0.join()

        except AssertionError:
            pass

    def _c1_send(self, v):
        return self.c1.send(('vv', v))
     
    def _c0_recv(self, timeout=5.):     
        
        t, v = self.c1.recv()
        
        if t == 'vv':
            return load(v)
        
        if t == 'ee':
            sys.stderr.write(''.join(v))   

            
class GameProcess(ModuleProcess):
    
    def start(self, interval=1/24, size=224):

        super().start()

        self.call('app.start', interval)
        self.call('app.scale_window_to', size)
        self.call('app._redraw_windows', 0, 0)
        self.call('app.use_shared_memory')        

        return self

    def observe(self):
        return self.call('observe')
        
    def step(self, *args, **kwargs):
        return self.call('step', *args, **kwargs)
        
    def reset(self):
        return self.call('reset')
        
    def load(self, path):
        return self.call('load', path)        

    def save(self, path=None):
        return self.call('save', path)


class Games(object):
    
    def __init__(self, games):
        
        if type(games[0]) is str:
            self.games = [ModuleProcess(game) for game in games]
        else:
            self.games = games
        
    def start(self, interval=1/24, size=224):
        
        for g in self.games:
            if type(g) is GameProcess:
                super(GameProcess, g).start()
            else:
                g.start()
                
        self.call('app.start', interval)
        self.call('app.scale_window_to', size)
        self.call('app._redraw_windows', 0, 0)
        self.call('app.use_shared_memory')        

        return self 
        
    def observe(self):
        return self.call('observe') 

    def step(self, *args, **kwargs):
        return self.call('step', *args, **kwargs) 
        
    def reset(self):
        return self.call('reset')
        
    def load(self, path):
        return self.call('load', path)        

    def save(self, path=None):
        return self.call('save', path)

    def get(self, name):
        self.send(name, None, None)
        return self.recv()

    def set(self, name, value):
        self.send(name, value, None)
        
    def call(self, name, *args, **kwargs):
        self.send(name, args, kwargs)
        return self.recv()

    def send(self, name, args, kwargs):

        if not args or all(is_scalar(a) for a in args):
            for g in self.games:
                g.send(name, args, kwargs)
            
        else:
            assert not any(is_scalar(a) or len(a) != len(self.games) for a in args), 'Positional arguments should be batched with a "sample" for each game.'

            for g, args in zip(self.games, zip(*args)):
                g.send(name, args, kwargs)
    
    def recv(self):
        return [g.recv() for g in self.games]
    
