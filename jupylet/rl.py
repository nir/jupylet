"""
    jupylet/rl.py
    
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


import importlib
import itertools
import traceback
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


_has_display = None


def has_display():
    
    global _has_display
    
    if _has_display is not None:
        return _has_display
    
    v = mp.Value('i', 0)

    if 'pyglet' in sys.modules:
        _has_display0(v)

    else:
        p = mp.Process(target=_has_display0, args=(v,))
        p.start()
        p.join()
    
    _has_display = v.value
    
    return _has_display


def _has_display0(v):

    try:
        import pyglet    
        pyglet.canvas.get_display()
        v.value = 1
    except:
        pass


_xvfb = None


def start_xvfb():
    
    global _xvfb

    if platform.system() == 'Linux' and _xvfb is None:

        import xvfbwrapper
        _xvfb = xvfbwrapper.Xvfb()
        _xvfb.start()


def is_xvfb():
    return _xvfb is not None


SCALARS = {str, bytes, int, float, bool}


def is_scalar(a):
    return type(a) in SCALARS


def o2h(o, n=12):
    return hashlib.sha256(pickle.dumps(o)).hexdigest()[:n]


def get_tmp_dir():
    
    if os.path.exists('/dev/shm/'):
        return '/dev/shm'
    else:
        return tempfile.gettempdir()

def get_random_mm_path():
    return os.path.join(get_tmp_dir(), 'jupylet-mm-array-%s.tmp' % o2h(random.random()))


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


class mmcache(object):
    
    def __init__(self):
        
        self._r = {}
        self._w = {}

        self.ison = os.path.exists('/dev/shm')
        
    def __del__(self):

        for cache in [self._r, self._w]:
            while cache:
                mmd = cache.popitem()[1]
                while mmd:
                    path, mm = mmd.popitem()
                    mode = mm.mode
                    del mm
                    if mode == 'w+':
                        os.remove(path)
                        
    def get(self, dtype, shape, path=None):

        if path:

            mmd = self._r.setdefault((dtype, shape), {})

            if path not in mmd:
                mmd[path] = np.memmap(path, dtype=dtype, shape=shape, mode='r')

            return path, mmd[path]

        mmd = self._w.setdefault((dtype, shape), {})
        if mmd:
            return mmd.popitem()

        path = path or get_random_mm_path()
        return path, np.memmap(path, dtype=dtype, shape=shape, mode='w+')
    
    def set(self, ml):
        for path, mm in ml:
            if mm.mode == 'w+':
                self._w.setdefault((mm.dtype, mm.shape), {})[path] = mm
        
    def dump(self, o, ml, size=1024, depth=1):

        if type(o) in [int, float, str]:
            return o

        if depth and type(o) in [list, tuple]:
            return type(o)(self.dump(v, ml, size, depth-1) for v in o)

        if depth and type(o) is dict:
            return {k: self.dump(v, ml, size, depth-1) for k, v in o.items()}

        if isinstance(o, np.ndarray) and o.size > size:
            path, mm = self.get(o.dtype, o.shape)
            mm[:] = o[:]
            mm.flush()
            ml.append((path, mm))
            return ('__ndarray__', o.dtype.name, o.shape, path)

        return o   
    
    def load(self, o, depth=1):

        if type(o) in [int, float, str]:
            return o

        if type(o) is tuple and len(o) == 4 and o[0] == '__ndarray__':
            return np.array(self.get(*o[1:])[1])

        if depth and type(o) in [list, tuple]:
            return type(o)(self.load(v, depth-1) for v in o)

        if depth and type(o) is dict:
            return {k: self.load(v, depth-1) for k, v in o.items()}

        return o


class ModuleProcess(object):
    
    def __init__(self, name, debug=False):
        
        self.name = name
        self.path = get_random_mm_path()
        self.debug = debug

        self.c0, c0 = mp.Pipe()
        self.c1 = self.c0

        self.p0 = mp.Process(target=self._worker0, args=(c0,))

        self.mm = mmcache()
        
    def __del__(self):
        self.stop()
        
    def _worker0(self, c0):
        
        self.c0 = c0
        self.c1 = c0

        module = importlib.import_module(self.name)
        
        for name, args, kwargs in iter(self.c0.recv, 'STOP'):            
            
            try:
                if kwargs is not None:
                    foo = rgetattr(module, name)
                    self._c1_send(foo(*args, **kwargs))
                    
                elif args is not None:
                    rsetattr(module, name, args)

                else:
                    self._c1_send(rgetattr(module, name))
             
            except:
                self.c1.send(('ee', traceback.format_exception(*sys.exc_info())))
                   
    """def log(self, msg, *args):
        if self.debug:
            msg = msg % args
            date = datetime.datetime.now().isoformat()[:23].replace('T', ' ')
            self.q3.put('[%s] [%s]  %s' % (date, self.p0.ident, msg))"""

    def start(self):
        if self.p0.ident is not None:
            return

        assert 'pyglet' not in sys.modules, 'Don\'t import pyglet or jupylet modules except jupylet.rl before starting worker processes.'

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

        if not self.mm.ison:
            return self.c1.send(('vv', v))

        ml = []
        v = self.mm.dump(v, ml)

        if not ml:
            return self.c1.send(('vv', v))

        self.mm.set(ml)
        return self.c1.send(('mm', v))
            
    def _c0_recv(self, timeout=5.):     
        
        t, v = self.c1.recv()
        
        if t == 'vv':
            return v
        
        if t == 'mm':
            return self.mm.load(v)
        
        if t == 'ee':
            sys.stderr.write('\n'.join(v))   

            
class GameProcess(ModuleProcess):
    
    def start(self, interval=1/30, size=224):

        super(GameProcess, self).start()
        
        self.call('app.start', interval)
        self.call('app.scale_window_to', size)
        self.call('app._redraw_windows', 0)
        self.call('app._redraw_windows', 0)

    def get_observation(self):
        return self.get('app.array0')
        
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
        
    def start(self, interval=1/30, size=224):
        
        for g in self.games:
            if type(g) is GameProcess:
                super(GameProcess, g).start()
            else:
                g.start()
                
        self.call('app.start', interval)
        self.call('app.scale_window_to', size)
        self.call('app._redraw_windows', 0)
        self.call('app._redraw_windows', 0)

    def get_observation(self):
        return self.get('app.array0')        

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
    
